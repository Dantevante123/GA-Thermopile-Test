import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS
# -------------------------
CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"  # <- change if needed
OUT_DIR = os.path.join(os.path.dirname(__file__), "cross_surface_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Temperature range to include
T_MIN_C = 105.0
T_MAX_C = 130.0

# If U columns missing, compute U = V_rms - V0
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

TARGET = "y_tc_mean"
NTC_COL = "ntc_mean"
CHANNELS = ["A", "B", "C", "D"]

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

def fit_ls(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(X)), X])  # intercept
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta  # (p+1,1)

def pred_ls(beta, X):
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

# Ensure U cols
for ch in CHANNELS:
    ucol = f"U_{ch}"
    vcol = f"{ch}_rms"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing {ucol} and {vcol}.")
        df[ucol] = df[vcol].astype(float) - float(V0[ch])

need = ["surface", TARGET, NTC_COL] + [f"U_{c}" for c in CHANNELS]
df = df.dropna(subset=need).copy()

# Filter temp range
df = df[(df[TARGET] >= T_MIN_C) & (df[TARGET] <= T_MAX_C)].copy()

surfaces = sorted([s for s in df["surface"].unique() if isinstance(s, str)])
if len(surfaces) < 2:
    raise ValueError("Need at least 2 surfaces for cross-surface validation.")

print("Surfaces found:", surfaces)
print("Rows after filtering:", len(df))

# -------------------------
# CROSS-SURFACE LOOP
# -------------------------
summary_rows = []
detail_rows = []

for s_test in surfaces:
    train = df[df["surface"] != s_test].copy()
    test  = df[df["surface"] == s_test].copy()

    y_train = train[TARGET].to_numpy(float)
    y_test  = test[TARGET].to_numpy(float)

    # ----- Mean baseline -----
    mean_pred = float(np.mean(y_train))
    yhat_mean = np.full_like(y_test, mean_pred, dtype=float)
    mean_rmse = rmse(y_test, yhat_mean)
    mean_mae  = mae(y_test, yhat_mean)

    # ----- Single-channel models (A..D): T = a*U_i + b*ntc + c -----
    single_metrics = []
    for ch in CHANNELS:
        feats = [f"U_{ch}", NTC_COL]
        beta = fit_ls(train[feats].to_numpy(float), y_train)
        yhat = pred_ls(beta, test[feats].to_numpy(float))

        ch_rmse = rmse(y_test, yhat)
        ch_mae  = mae(y_test, yhat)

        single_metrics.append((ch, ch_rmse, ch_mae, beta))
        detail_rows.append({
            "test_surface": s_test,
            "channel": ch,
            "rmse_C": ch_rmse,
            "mae_C": ch_mae,
            "c_intercept": float(beta[0,0]),
            "a_U": float(beta[1,0]),
            "b_ntc": float(beta[2,0]),
            "n_train": len(train),
            "n_test": len(test),
        })

    # best single by RMSE
    best_ch, best_rmse, best_mae, best_beta = sorted(single_metrics, key=lambda t: t[1])[0]

    # ----- Quad-channel model: T = wA*U_A + ... + wD*U_D + wN*ntc + c -----
    feats_q = ["U_A", "U_B", "U_C", "U_D", NTC_COL]
    beta_q = fit_ls(train[feats_q].to_numpy(float), y_train)
    yhat_q = pred_ls(beta_q, test[feats_q].to_numpy(float))

    quad_rmse = rmse(y_test, yhat_q)
    quad_mae  = mae(y_test, yhat_q)

    summary_rows.append({
        "test_surface": s_test,
        "n_train": len(train),
        "n_test": len(test),
        "mean_rmse_C": mean_rmse,
        "mean_mae_C": mean_mae,
        "best_single_channel": best_ch,
        "best_single_rmse_C": best_rmse,
        "best_single_mae_C": best_mae,
        "quad_rmse_C": quad_rmse,
        "quad_mae_C": quad_mae,
    })

    print(f"\n=== Test surface: {s_test} ===")
    print("Mean baseline  RMSE/MAE:", f"{mean_rmse:.2f}", f"{mean_mae:.2f}")
    print(f"Best single ({best_ch}+NTC) RMSE/MAE:", f"{best_rmse:.2f}", f"{best_mae:.2f}")
    print("Quad-channel   RMSE/MAE:", f"{quad_rmse:.2f}", f"{quad_mae:.2f}")

    # ----- Plot: bar comparison for this fold -----
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=DPI)
    labels = ["Mean\nbaseline", f"Best single\n({best_ch}+NTC)", "Quad-channel\nmodel"]
    rmse_vals = [mean_rmse, best_rmse, quad_rmse]
    mae_vals  = [mean_mae,  best_mae,  quad_mae]

    x = np.arange(len(labels))
    w = 0.34
    b1 = ax.bar(x - w/2, rmse_vals, width=w, label="RMSE (°C)")
    b2 = ax.bar(x + w/2, mae_vals,  width=w, label="MAE (°C)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_AX)
    ax.set_ylabel("Error (°C)", fontsize=FONT_AX)
    ax.set_title(f"Cross-surface test: {s_test}", fontsize=FONT_TITLE)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(True, axis="y", alpha=0.25)
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
    savefig(fig, f"fold_{s_test}_errors_bar.png")

# -------------------------
# SAVE RESULTS
# -------------------------
summary = pd.DataFrame(summary_rows)
details = pd.DataFrame(detail_rows)

summary_path = os.path.join(OUT_DIR, "cross_surface_summary.csv")
details_path = os.path.join(OUT_DIR, "cross_surface_channel_details.csv")
summary.to_csv(summary_path, index=False)
details.to_csv(details_path, index=False)

print("\nSaved:", summary_path)
print("Saved:", details_path)

# -------------------------
# OVERALL SUMMARY PLOT (mean over folds)
# -------------------------
mean_mean_rmse = float(summary["mean_rmse_C"].mean())
mean_mean_mae  = float(summary["mean_mae_C"].mean())
mean_best_rmse = float(summary["best_single_rmse_C"].mean())
mean_best_mae  = float(summary["best_single_mae_C"].mean())
mean_quad_rmse = float(summary["quad_rmse_C"].mean())
mean_quad_mae  = float(summary["quad_mae_C"].mean())

fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=DPI)
labels = ["Mean baseline", "Best single\n(per fold)", "Quad-channel"]
rmse_vals = [mean_mean_rmse, mean_best_rmse, mean_quad_rmse]
mae_vals  = [mean_mean_mae,  mean_best_mae,  mean_quad_mae]

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
savefig(fig, "cross_surface_overall_mean_errors.png")

print("\nDONE ✅ Outputs in:", OUT_DIR)