# poster_plots_regression.py
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

# Train/test settings
TRAIN_MIN_C = 105.0
TRAIN_MAX_C = 130.0
TEST_FRACTION = 0.20

# If U_A..U_D already exist in CSV, they will be used.
# Otherwise we compute U = V_rms - V0 using values below.
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,   # replace if you have newer
    "C": 2.571482,   # replace if you have newer
    "D": 2.587497,   # replace if you have newer
}

FEATURES = ["U_A", "U_B", "U_C", "U_D", "ntc_mean"]  # + intercept automatically
TARGET = "y_tc_mean"  # reference TC in °C

# Poster styling1-
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
    X_design = np.column_stack([np.ones(len(X)), X])
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
for ch in ["A", "B", "C", "D"]:
    vcol = f"{ch}_rms"
    ucol = f"U_{ch}"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing column {vcol} in CSV.")
        if ch not in V0:
            raise ValueError(f"U_{ch} not in CSV and V0[{ch}] not provided.")
        df[ucol] = df[vcol] - float(V0[ch])

need_cols = ["surface", TARGET, "ntc_mean", "U_A", "U_B", "U_C", "U_D"]
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

X_train = train[FEATURES].to_numpy()
y_train = train[TARGET].to_numpy()

X_test = test[FEATURES].to_numpy()
y_test = test[TARGET].to_numpy()

# -------------------------
# FIT MODEL
# -------------------------
beta = fit_least_squares(X_train, y_train)
y_train_pred = predict_least_squares(beta, X_train)
y_test_pred = predict_least_squares(beta, X_test)

train_rmse = rmse(y_train, y_train_pred)
train_mae = mae(y_train, y_train_pred)
test_rmse = rmse(y_test, y_test_pred)
test_mae = mae(y_test, y_test_pred)

# Baseline = predict mean of train
baseline = float(np.mean(y_train))
y_test_base = np.full_like(y_test, baseline, dtype=float)
base_rmse = rmse(y_test, y_test_base)
base_mae = mae(y_test, y_test_base)

print("\n--- Linear least squares model ---")
print("Features:", ["intercept"] + FEATURES)
print("Coefficients:", beta.reshape(-1))
print("\nTest  RMSE (°C):", test_rmse)
print("Test  MAE  (°C):", test_mae)
print("\nBaseline Test RMSE (°C):", base_rmse)
print("Baseline Test MAE  (°C):", base_mae)

# Save predictions
pred_out = test[["file", "surface", TARGET]].copy()
pred_out["T_pred_C"] = y_test_pred
pred_csv = os.path.join(OUT_DIR, "temp_model_test_predictions.csv")
pred_out.to_csv(pred_csv, index=False)
print("\nSaved test predictions:", pred_csv)

# -------------------------
# PLOT 1: Predicted vs True (Test)
# -------------------------
fig, ax = plt.subplots(figsize=(8.0, 6.6), dpi=POSTER_DPI)

surfaces = sorted([s for s in test["surface"].unique() if isinstance(s, str)])
for s in surfaces:
    m = (test["surface"] == s).to_numpy()
    ax.scatter(y_test[m], y_test_pred[m], s=POINT_SIZE, label=s, alpha=0.88)

# diagonal, down-toned
minv = float(min(y_test.min(), y_test_pred.min()))
maxv = float(max(y_test.max(), y_test_pred.max()))
pad = 2.0
ax.plot([minv - pad, maxv + pad], [minv - pad, maxv + pad],
        linewidth=2, alpha=0.45)

ax.set_xlabel("True temperature (°C)", fontsize=FONT_AX)
ax.set_ylabel("Predicted temperature (°C)", fontsize=FONT_AX)
ax.set_title("Quad-channel model: Predicted vs True (test set)", fontsize=FONT_TITLE)

ax.tick_params(axis="both", labelsize=FONT_TICK)
ax.grid(True, alpha=0.25)

txt = f"Test RMSE = {test_rmse:.2f} °C\nTest MAE  = {test_mae:.2f} °C"
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=FONT_LEGEND,
        bbox=dict(boxstyle="round", alpha=0.12))

# Legend moved slightly outside to avoid covering data
ax.legend(title="Surface", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
          loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

fig.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig_pred_vs_true_test.png")
fig.savefig(fig1_path, bbox_inches="tight")
print("Saved:", fig1_path)

# -------------------------
# PLOT 2: Error bars comparison (baseline vs model)
# -------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=POSTER_DPI)

labels = ["Single-channel\nbaseline", "Quad-channel\nmodel"]
rmse_vals = [base_rmse, test_rmse]
mae_vals  = [base_mae,  test_mae]

x = np.arange(len(labels))
w = 0.34

b1 = ax.bar(x - w/2, rmse_vals, width=w, label="RMSE (°C)")
b2 = ax.bar(x + w/2, mae_vals,  width=w, label="MAE (°C)")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT_AX)
ax.set_ylabel("Error (°C)", fontsize=FONT_AX)
ax.set_title("Prediction error (test set)", fontsize=FONT_TITLE)

ax.tick_params(axis="y", labelsize=FONT_TICK)
ax.grid(True, axis="y", alpha=0.25)

# --- Put legend OUTSIDE so it never covers anything ---
ax.legend(fontsize=FONT_LEGEND, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

# --- Improvements (RMSE + MAE) ---
rmse_impr = (base_rmse - test_rmse) / base_rmse * 100.0 if base_rmse > 0 else 0.0
mae_impr  = (base_mae  - test_mae)  / base_mae  * 100.0 if base_mae  > 0 else 0.0

badge = f"RMSE improvement: {rmse_impr:.1f}%   |   MAE improvement: {mae_impr:.1f}%"

# Put badge ABOVE the axes (figure coords) so it never overlaps bars/labels
fig.text(
    0.5, 0.98, badge,
    ha="center", va="top",
    fontsize=FONT_LEGEND,
    bbox=dict(boxstyle="round", alpha=0.12)
)

# --- Add numeric labels ABOVE bars, with safe padding ---
ymax = max(rmse_vals + mae_vals)
ax.set_ylim(0, ymax * 1.18)  # extra headroom so numbers never hit the top

def label_bars(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.03 * ymax,
            f"{h:.2f}",
            ha="center", va="bottom",
            fontsize=FONT_TICK
        )

label_bars(b1)
label_bars(b2)

# Tight layout but leave space for top badge + right legend
fig.tight_layout(rect=[0, 0, 0.82, 0.92])

fig2_path = os.path.join(OUT_DIR, "fig_errors_bar.png")
fig.savefig(fig2_path, bbox_inches="tight")
print("Saved:", fig2_path)
plt.show()