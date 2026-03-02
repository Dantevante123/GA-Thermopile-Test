import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS
# -------------------------
CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "cross_surface_ridge_out")
os.makedirs(OUT_DIR, exist_ok=True)

T_MIN_C = 105.0
T_MAX_C = 130.0

V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

TARGET = "y_tc_mean"
NTC_COL = "ntc_mean"
CHANNELS = ["A", "B", "C", "D"]

# Ridge candidates (lambda). Feel free to add more.
RIDGE_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

# Plot styling
DPI = 220
FONT_TITLE = 18
FONT_AX = 15
FONT_TICK = 13
FONT_LEGEND = 12

# -------------------------
# HELPERS
# -------------------------
def extract_surface(fname: str):
    m = re.match(r"^(J[1-6])", str(fname))
    return m.group(1) if m else None

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))

def fit_ols(X, y):
    """OLS with intercept added inside."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta  # (p+1,1)

def fit_ridge(X, y, lam):
    """
    Ridge regression with intercept (intercept NOT regularized).
    Solves: min ||Xd*beta - y||^2 + lam * ||beta_no_intercept||^2
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(X)), X])  # intercept
    p = Xd.shape[1]

    # Regularization matrix: don't penalize intercept
    R = np.eye(p)
    R[0, 0] = 0.0

    # Closed form: (X^T X + lam R)^{-1} X^T y
    A = Xd.T @ Xd + lam * R
    b = Xd.T @ y
    beta = np.linalg.solve(A, b)
    return beta

def predict(beta, X):
    X = np.asarray(X, float)
    Xd = np.column_stack([np.ones(len(X)), X])
    return (Xd @ beta).reshape(-1)

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("Saved:", path)

# -------------------------
# LOAD + PREP
# -------------------------
df = pd.read_csv(CSV_PATH)
df["surface"] = df["file"].apply(extract_surface)

for ch in CHANNELS:
    ucol = f"U_{ch}"
    vcol = f"{ch}_rms"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing {ucol} and {vcol}.")
        df[ucol] = df[vcol].astype(float) - float(V0[ch])

need = ["surface", TARGET, NTC_COL] + [f"U_{c}" for c in CHANNELS]
df = df.dropna(subset=need).copy()
df = df[(df[TARGET] >= T_MIN_C) & (df[TARGET] <= T_MAX_C)].copy()

surfaces = sorted([s for s in df["surface"].unique() if isinstance(s, str)])
print("Surfaces:", surfaces)
print("Rows after filter:", len(df))

# Feature sets
FEATS_SINGLE = lambda ch: [f"U_{ch}", NTC_COL]
FEATS_QUAD = ["U_A", "U_B", "U_C", "U_D", NTC_COL]

# -------------------------
# EVALUATE FOR EACH LAMBDA
# -------------------------
all_lambda_rows = []

for lam in RIDGE_LAMBDAS:
    fold_rows = []

    for s_test in surfaces:
        train = df[df["surface"] != s_test].copy()
        test  = df[df["surface"] == s_test].copy()

        y_train = train[TARGET].to_numpy(float)
        y_test  = test[TARGET].to_numpy(float)

        # Mean baseline
        mean_pred = float(np.mean(y_train))
        yhat_mean = np.full_like(y_test, mean_pred)
        mean_rmse = rmse(y_test, yhat_mean)
        mean_mae  = mae(y_test, yhat_mean)

        # Best single (per fold)
        best = None
        for ch in CHANNELS:
            beta_s = fit_ols(train[FEATS_SINGLE(ch)].to_numpy(float), y_train)
            yhat_s = predict(beta_s, test[FEATS_SINGLE(ch)].to_numpy(float))
            r = (rmse(y_test, yhat_s), mae(y_test, yhat_s), ch)
            if best is None or r[0] < best[0]:
                best = r
        best_rmse, best_mae, best_ch = best

        # Quad OLS
        beta_q = fit_ols(train[FEATS_QUAD].to_numpy(float), y_train)
        yhat_q = predict(beta_q, test[FEATS_QUAD].to_numpy(float))
        quad_rmse = rmse(y_test, yhat_q)
        quad_mae  = mae(y_test, yhat_q)

        # Quad Ridge (if lam==0 -> same as OLS in practice, but we still compute)
        beta_r = fit_ridge(train[FEATS_QUAD].to_numpy(float), y_train, lam=lam)
        yhat_r = predict(beta_r, test[FEATS_QUAD].to_numpy(float))
        ridge_rmse = rmse(y_test, yhat_r)
        ridge_mae  = mae(y_test, yhat_r)

        fold_rows.append({
            "lambda": lam,
            "test_surface": s_test,
            "mean_rmse": mean_rmse,
            "mean_mae": mean_mae,
            "best_single_ch": best_ch,
            "best_single_rmse": best_rmse,
            "best_single_mae": best_mae,
            "quad_rmse": quad_rmse,
            "quad_mae": quad_mae,
            "ridge_rmse": ridge_rmse,
            "ridge_mae": ridge_mae,
            "n_train": len(train),
            "n_test": len(test),
        })

    folds_df = pd.DataFrame(fold_rows)
    all_lambda_rows.append({
        "lambda": lam,
        "mean_rmse": float(folds_df["mean_rmse"].mean()),
        "mean_mae": float(folds_df["mean_mae"].mean()),
        "best_single_rmse": float(folds_df["best_single_rmse"].mean()),
        "best_single_mae": float(folds_df["best_single_mae"].mean()),
        "quad_rmse": float(folds_df["quad_rmse"].mean()),
        "quad_mae": float(folds_df["quad_mae"].mean()),
        "ridge_rmse": float(folds_df["ridge_rmse"].mean()),
        "ridge_mae": float(folds_df["ridge_mae"].mean()),
    })

# Save lambda sweep summary
lam_df = pd.DataFrame(all_lambda_rows).sort_values("ridge_rmse")
lam_path = os.path.join(OUT_DIR, "ridge_lambda_sweep_summary.csv")
lam_df.to_csv(lam_path, index=False)
print("\nSaved:", lam_path)
print("\nTop lambdas by ridge RMSE:\n", lam_df.head(5))

# Pick best lambda by ridge RMSE
best_lam = float(lam_df.iloc[0]["lambda"])
print("\nBEST lambda =", best_lam)

# -------------------------
# Re-run folds for best lambda and make plots
# -------------------------
fold_rows = []
for s_test in surfaces:
    train = df[df["surface"] != s_test].copy()
    test  = df[df["surface"] == s_test].copy()

    y_train = train[TARGET].to_numpy(float)
    y_test  = test[TARGET].to_numpy(float)

    mean_pred = float(np.mean(y_train))
    yhat_mean = np.full_like(y_test, mean_pred)
    mean_rmse = rmse(y_test, yhat_mean)
    mean_mae  = mae(y_test, yhat_mean)

    best = None
    for ch in CHANNELS:
        beta_s = fit_ols(train[FEATS_SINGLE(ch)].to_numpy(float), y_train)
        yhat_s = predict(beta_s, test[FEATS_SINGLE(ch)].to_numpy(float))
        r = (rmse(y_test, yhat_s), mae(y_test, yhat_s), ch)
        if best is None or r[0] < best[0]:
            best = r
    best_rmse, best_mae, best_ch = best

    beta_q = fit_ols(train[FEATS_QUAD].to_numpy(float), y_train)
    yhat_q = predict(beta_q, test[FEATS_QUAD].to_numpy(float))
    quad_rmse = rmse(y_test, yhat_q)
    quad_mae  = mae(y_test, yhat_q)

    beta_r = fit_ridge(train[FEATS_QUAD].to_numpy(float), y_train, lam=best_lam)
    yhat_r = predict(beta_r, test[FEATS_QUAD].to_numpy(float))
    ridge_rmse = rmse(y_test, yhat_r)
    ridge_mae  = mae(y_test, yhat_r)

    fold_rows.append({
        "test_surface": s_test,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "best_single_ch": best_ch,
        "best_single_rmse": best_rmse,
        "best_single_mae": best_mae,
        "quad_rmse": quad_rmse,
        "quad_mae": quad_mae,
        "ridge_rmse": ridge_rmse,
        "ridge_mae": ridge_mae,
        "n_train": len(train),
        "n_test": len(test),
    })

    # per-fold bar plot
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=DPI)
    labels = ["Mean", f"Best single\n({best_ch}+NTC)", "Quad OLS", f"Quad Ridge\n(λ={best_lam:g})"]
    rmse_vals = [mean_rmse, best_rmse, quad_rmse, ridge_rmse]
    mae_vals  = [mean_mae,  best_mae,  quad_mae,  ridge_mae]

    x = np.arange(len(labels))
    w = 0.34
    b1 = ax.bar(x - w/2, rmse_vals, width=w, label="RMSE (°C)")
    b2 = ax.bar(x + w/2, mae_vals,  width=w, label="MAE (°C)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_AX)
    ax.set_ylabel("Error (°C)", fontsize=FONT_AX)
    ax.set_title(f"Cross-surface fold: test {s_test}", fontsize=FONT_TITLE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ymax = max(rmse_vals + mae_vals)
    ax.set_ylim(0, ymax * 1.18)

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03 * ymax,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=FONT_TICK)
    label_bars(b1); label_bars(b2)

    fig.tight_layout(rect=[0, 0, 0.82, 1.0])
    savefig(fig, f"fold_{s_test}_bar_best_lambda.png")

folds_best = pd.DataFrame(fold_rows)
folds_path = os.path.join(OUT_DIR, "cross_surface_best_lambda_folds.csv")
folds_best.to_csv(folds_path, index=False)
print("\nSaved:", folds_path)

# Overall mean plot
m = folds_best.mean(numeric_only=True)

fig, ax = plt.subplots(figsize=(9.0, 5.1), dpi=DPI)
labels = ["Mean", "Best single\n(per fold)", "Quad OLS", f"Quad Ridge\n(λ={best_lam:g})"]
rmse_vals = [float(m["mean_rmse"]), float(m["best_single_rmse"]), float(m["quad_rmse"]), float(m["ridge_rmse"])]
mae_vals  = [float(m["mean_mae"]),  float(m["best_single_mae"]),  float(m["quad_mae"]),  float(m["ridge_mae"])]

x = np.arange(len(labels))
w = 0.34
b1 = ax.bar(x - w/2, rmse_vals, width=w, label="RMSE (°C)")
b2 = ax.bar(x + w/2, mae_vals,  width=w, label="MAE (°C)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT_AX)
ax.set_ylabel("Error (°C)", fontsize=FONT_AX)
ax.set_title("Cross-surface performance (mean over folds)", fontsize=FONT_TITLE)
ax.grid(True, axis="y", alpha=0.25)
ax.tick_params(axis="y", labelsize=FONT_TICK)
ax.legend(fontsize=FONT_LEGEND, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

ymax = max(rmse_vals + mae_vals)
ax.set_ylim(0, ymax * 1.18)

def label_bars(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03 * ymax,
                f"{h:.2f}", ha="center", va="bottom", fontsize=FONT_TICK)
label_bars(b1); label_bars(b2)

fig.tight_layout(rect=[0, 0, 0.82, 1.0])
savefig(fig, "cross_surface_overall_best_lambda.png")

print("\nDONE ✅ Output dir:", OUT_DIR)