# poster_plots_single_channel_linear.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS
# -------------------------
CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"  # <- change if needed
OUT_DIR = os.path.dirname(__file__)
SEED = 42

TRAIN_MIN_C = 100.0
TRAIN_MAX_C = 130.0
TEST_FRACTION = 0.20

# If U_A..U_D already exist in CSV, they will be used.
# Otherwise compute U = V_rms - V0
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

TARGET = "y_tc_mean"
NTC_COL = "ntc_mean"

CHANNELS = ["A", "B", "C", "D"]

# Poster styling
POSTER_DPI = 220
FONT_TITLE = 18
FONT_AX = 15
FONT_TICK = 13
FONT_LEGEND = 12
POINT_SIZE = 55

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

def fit_least_squares(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    X_design = np.column_stack([np.ones(len(X)), X])  # intercept
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta  # (p+1,1)

def predict_least_squares(beta, X):
    X = np.asarray(X, float)
    X_design = np.column_stack([np.ones(len(X)), X])
    return (X_design @ beta).reshape(-1)

# -------------------------
# LOAD + PREP DATA
# -------------------------
df = pd.read_csv(CSV_PATH)
df["surface"] = df["file"].apply(extract_surface)

# Ensure U columns exist
for ch in CHANNELS:
    vcol = f"{ch}_rms"
    ucol = f"U_{ch}"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing column {vcol} in CSV.")
        if ch not in V0:
            raise ValueError(f"U_{ch} not in CSV and V0[{ch}] not provided.")
        df[ucol] = df[vcol] - float(V0[ch])

need_cols = ["surface", TARGET, NTC_COL] + [f"U_{c}" for c in CHANNELS]
df = df.dropna(subset=need_cols).copy()

# Train/test split inside 105–130C
df_pool = df[(df[TARGET] >= TRAIN_MIN_C) & (df[TARGET] <= TRAIN_MAX_C)].copy()
if len(df_pool) < 10:
    raise ValueError("Too few samples in train pool. Check filters or CSV.")

rng = np.random.default_rng(SEED)
idx = np.arange(len(df_pool))
rng.shuffle(idx)

n_test = int(round(TEST_FRACTION * len(df_pool)))
test_idx = idx[:n_test]
train_idx = idx[n_test:]

train = df_pool.iloc[train_idx].copy()
test = df_pool.iloc[test_idx].copy()

print("\n--- Split sizes ---")
print("Train (105-130):", len(train))
print("Test  (20% of 105-130):", len(test))

y_train = train[TARGET].to_numpy(float)
y_test = test[TARGET].to_numpy(float)

# Mean baseline (constant predictor)
baseline = float(np.mean(y_train))
y_test_base = np.full_like(y_test, baseline, dtype=float)
base_rmse = rmse(y_test, y_test_base)
base_mae  = mae(y_test, y_test_base)

print("\n--- Mean baseline (constant predictor) ---")
print("Test RMSE (°C):", base_rmse)
print("Test MAE  (°C):", base_mae)

# -------------------------
# SINGLE-CHANNEL MODELS: T = a*U_i + b*ntc + c
# -------------------------
results = []  # store per-channel metrics + coefficients
preds = {}    # store per-channel yhat for test

for ch in CHANNELS:
    feats = [f"U_{ch}", NTC_COL]  # exactly your requested form
    X_train = train[feats].to_numpy(float)
    X_test  = test[feats].to_numpy(float)

    beta = fit_least_squares(X_train, y_train)  # [c, a, b] (intercept first)
    y_test_pred = predict_least_squares(beta, X_test)

    r = {
        "channel": ch,
        "intercept_c": float(beta[0, 0]),
        "a_U": float(beta[1, 0]),
        "b_ntc": float(beta[2, 0]),
        "test_rmse_C": rmse(y_test, y_test_pred),
        "test_mae_C": mae(y_test, y_test_pred),
    }
    results.append(r)
    preds[ch] = y_test_pred

    print(f"\n--- Single-channel linear model ({ch}) ---")
    print("Model: T = a*U_{ch} + b*ntc + c")
    print("c =", r["intercept_c"])
    print("a =", r["a_U"])
    print("b =", r["b_ntc"])
    print("Test RMSE (°C):", r["test_rmse_C"])
    print("Test MAE  (°C):", r["test_mae_C"])

# Save table of coefficients + metrics
res_df = pd.DataFrame(results).sort_values("test_rmse_C")
res_csv = os.path.join(OUT_DIR, "single_channel_linear_results.csv")
res_df.to_csv(res_csv, index=False)
print("\nSaved:", res_csv)

best_ch = res_df.iloc[0]["channel"]
best_rmse = float(res_df.iloc[0]["test_rmse_C"])
best_mae  = float(res_df.iloc[0]["test_mae_C"])

# -------------------------
# QUAD-CHANNEL MODEL (same as your regression script)
# -------------------------
FEATURES_QUAD = ["U_A", "U_B", "U_C", "U_D", NTC_COL]
X_train_q = train[FEATURES_QUAD].to_numpy(float)
X_test_q  = test[FEATURES_QUAD].to_numpy(float)

beta_q = fit_least_squares(X_train_q, y_train)
y_test_pred_q = predict_least_squares(beta_q, X_test_q)

quad_rmse = rmse(y_test, y_test_pred_q)
quad_mae  = mae(y_test, y_test_pred_q)

print("\n--- Quad-channel model ---")
print("Features:", ["intercept"] + FEATURES_QUAD)
print("Coefficients:", beta_q.reshape(-1))
print("Test RMSE (°C):", quad_rmse)
print("Test MAE  (°C):", quad_mae)

# -------------------------
# PLOTS
# -------------------------

# Plot 1: Predicted vs True for each single-channel model
for ch in CHANNELS:
    fig, ax = plt.subplots(figsize=(8.0, 6.6), dpi=POSTER_DPI)

    # color by surface
    surfaces = sorted([s for s in test["surface"].unique() if isinstance(s, str)])
    for s in surfaces:
        m = (test["surface"] == s).to_numpy()
        ax.scatter(y_test[m], preds[ch][m], s=POINT_SIZE, label=s, alpha=0.88)

    minv = float(min(y_test.min(), preds[ch].min()))
    maxv = float(max(y_test.max(), preds[ch].max()))
    pad = 2.0
    ax.plot([minv - pad, maxv + pad], [minv - pad, maxv + pad], linewidth=2, alpha=0.45)

    ax.set_xlabel("True temperature (°C)", fontsize=FONT_AX)
    ax.set_ylabel("Predicted temperature (°C)", fontsize=FONT_AX)
    ax.set_title(f"Single-channel linear model ({ch}): Predicted vs True", fontsize=FONT_TITLE)

    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.25)

    # metrics
    row = res_df[res_df["channel"] == ch].iloc[0]
    txt = f"Test RMSE = {float(row['test_rmse_C']):.2f} °C\nTest MAE  = {float(row['test_mae_C']):.2f} °C"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=FONT_LEGEND,
            bbox=dict(boxstyle="round", alpha=0.12))

    ax.legend(title="Surface", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
              loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"fig_single_{ch}_pred_vs_true.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)

# Plot 2: Comparison bars (Mean baseline vs best single-channel vs quad)
fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=POSTER_DPI)

labels = ["Mean\nbaseline", f"Best single\n({best_ch}+NTC)", "Quad-channel\nmodel"]
rmse_vals = [base_rmse, best_rmse, quad_rmse]
mae_vals  = [base_mae,  best_mae,  quad_mae]

x = np.arange(len(labels))
w = 0.34

b1 = ax.bar(x - w/2, rmse_vals, width=w, label="RMSE (°C)")
b2 = ax.bar(x + w/2, mae_vals,  width=w, label="MAE (°C)")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT_AX)
ax.set_ylabel("Error (°C)", fontsize=FONT_AX)
ax.set_title("Prediction error comparison (test set)", fontsize=FONT_TITLE)

ax.tick_params(axis="y", labelsize=FONT_TICK)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(fontsize=FONT_LEGEND, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

# numeric labels
ymax = max(rmse_vals + mae_vals)
ax.set_ylim(0, ymax * 1.18)

def label_bars(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03 * ymax,
                f"{h:.2f}", ha="center", va="bottom", fontsize=FONT_TICK)

label_bars(b1)
label_bars(b2)

fig.tight_layout(rect=[0, 0, 0.82, 1.0])
out = os.path.join(OUT_DIR, "fig_errors_comparison_mean_single_quad.png")
fig.savefig(out, bbox_inches="tight")
print("Saved:", out)
plt.show()