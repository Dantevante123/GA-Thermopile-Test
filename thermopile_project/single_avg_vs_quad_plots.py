import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS
# -------------------------
CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"
OUT_DIR = os.path.dirname(__file__)
SEED = 42

TRAIN_MIN_C = 105.0
TRAIN_MAX_C = 130.0
TEST_FRACTION = 0.20

TARGET = "y_tc_mean"      # reference temp (°C)
NTC_COL = "ntc_mean"      # sensor/ambient (°C)

# If U_A..U_D don't exist, compute U = V_rms - V0
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

CHANNELS = ["A", "B", "C", "D"]

# Styling
DPI = 220
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
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta  # (p+1,1)

def predict_least_squares(beta, X):
    X = np.asarray(X, float)
    Xd = np.column_stack([np.ones(len(X)), X])
    return (Xd @ beta).reshape(-1)

# -------------------------
# LOAD + PREP DATA
# -------------------------
df = pd.read_csv(CSV_PATH)
df["surface"] = df["file"].apply(extract_surface)

# Ensure U columns exist
for ch in CHANNELS:
    ucol = f"U_{ch}"
    vcol = f"{ch}_rms"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing column {vcol} in CSV.")
        df[ucol] = df[vcol].astype(float) - float(V0[ch])

need_cols = ["surface", TARGET, NTC_COL] + [f"U_{ch}" for ch in CHANNELS]
df = df.dropna(subset=need_cols).copy()

# Train/test split inside 105–130°C
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
test  = df_pool.iloc[test_idx].copy()

print("\n--- Split sizes ---")
print("Train (105-130):", len(train))
print("Test  (20% of 105-130):", len(test))

# -------------------------
# FIT MODELS
# -------------------------

# Quad-channel model: [U_A,U_B,U_C,U_D,ntc]
quad_features = [f"U_{ch}" for ch in CHANNELS] + [NTC_COL]
X_train_quad = train[quad_features].to_numpy()
y_train = train[TARGET].to_numpy()

X_test_quad = test[quad_features].to_numpy()
y_test = test[TARGET].to_numpy()

beta_quad = fit_least_squares(X_train_quad, y_train)
y_test_pred_quad = predict_least_squares(beta_quad, X_test_quad)

rmse_quad = rmse(y_test, y_test_pred_quad)
mae_quad  = mae(y_test, y_test_pred_quad)

# Single-channel per channel: T = a*U_i + b*ntc + c
single_preds = []
single_metrics = {}

for ch in CHANNELS:
    feats = [f"U_{ch}", NTC_COL]
    X_train_s = train[feats].to_numpy()
    X_test_s  = test[feats].to_numpy()

    beta_s = fit_least_squares(X_train_s, y_train)
    y_pred_s = predict_least_squares(beta_s, X_test_s)

    single_preds.append(y_pred_s)

    single_metrics[ch] = {
        "beta": beta_s.reshape(-1),
        "rmse": rmse(y_test, y_pred_s),
        "mae": mae(y_test, y_pred_s),
    }

# Mean of single-channel predictions (A–D)
y_test_pred_single_mean = np.mean(np.column_stack(single_preds), axis=1)
rmse_single_mean = rmse(y_test, y_test_pred_single_mean)
mae_single_mean  = mae(y_test, y_test_pred_single_mean)

print("\n--- Test performance (same split) ---")
print(f"Mean(single A-D)  RMSE={rmse_single_mean:.2f}  MAE={mae_single_mean:.2f}")
print(f"Quad model        RMSE={rmse_quad:.2f}  MAE={mae_quad:.2f}")

print("\nPer-channel (single) on test:")
for ch in CHANNELS:
    m = single_metrics[ch]
    print(f"  {ch}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  beta={m['beta']}")

# Save predictions (optional)
pred_out = test[["file", "surface", TARGET]].copy()
pred_out["T_pred_single_mean"] = y_test_pred_single_mean
pred_out["T_pred_quad"] = y_test_pred_quad
pred_csv = os.path.join(OUT_DIR, "pred_single_mean_vs_quad.csv")
pred_out.to_csv(pred_csv, index=False)
print("\nSaved predictions:", pred_csv)

# -------------------------
# PLOT 1: Pred vs True — mean(single)
# -------------------------
fig, ax = plt.subplots(figsize=(8.0, 6.6), dpi=DPI)

surfaces = sorted([s for s in test["surface"].unique() if isinstance(s, str)])
for s in surfaces:
    m = (test["surface"] == s).to_numpy()
    ax.scatter(y_test[m], y_test_pred_single_mean[m], s=POINT_SIZE, label=s, alpha=0.88)

minv = float(min(y_test.min(), y_test_pred_single_mean.min()))
maxv = float(max(y_test.max(), y_test_pred_single_mean.max()))
pad = 2.0
ax.plot([minv - pad, maxv + pad], [minv - pad, maxv + pad], linewidth=2, alpha=0.45)

ax.set_xlabel("True temperature (°C)", fontsize=FONT_AX)
ax.set_ylabel("Predicted temperature (°C)", fontsize=FONT_AX)
ax.set_title("Mean single-channel model: Predicted vs True (test set)", fontsize=FONT_TITLE)

ax.tick_params(axis="both", labelsize=FONT_TICK)
ax.grid(True, alpha=0.25)

txt = f"Test RMSE = {rmse_single_mean:.2f} °C\nTest MAE  = {mae_single_mean:.2f} °C"
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=FONT_LEGEND,
        bbox=dict(boxstyle="round", alpha=0.12))

ax.legend(title="Surface", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
          loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

fig.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig_pred_vs_true_single_mean.png")
fig.savefig(fig1_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", fig1_path)

# -------------------------
# PLOT 2: Pred vs True — quad
# -------------------------
fig, ax = plt.subplots(figsize=(8.0, 6.6), dpi=DPI)

for s in surfaces:
    m = (test["surface"] == s).to_numpy()
    ax.scatter(y_test[m], y_test_pred_quad[m], s=POINT_SIZE, label=s, alpha=0.88)

minv = float(min(y_test.min(), y_test_pred_quad.min()))
maxv = float(max(y_test.max(), y_test_pred_quad.max()))
pad = 2.0
ax.plot([minv - pad, maxv + pad], [minv - pad, maxv + pad], linewidth=2, alpha=0.45)

ax.set_xlabel("True temperature (°C)", fontsize=FONT_AX)
ax.set_ylabel("Predicted temperature (°C)", fontsize=FONT_AX)
ax.set_title("Quad-channel model: Predicted vs True (test set)", fontsize=FONT_TITLE)

ax.tick_params(axis="both", labelsize=FONT_TICK)
ax.grid(True, alpha=0.25)

txt = f"Test RMSE = {rmse_quad:.2f} °C\nTest MAE  = {mae_quad:.2f} °C"
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=FONT_LEGEND,
        bbox=dict(boxstyle="round", alpha=0.12))

ax.legend(title="Surface", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
          loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

fig.tight_layout()
fig2_path = os.path.join(OUT_DIR, "fig_pred_vs_true_quad.png")
fig.savefig(fig2_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", fig2_path)

# -------------------------
# PLOT 3: Error bar chart — mean(single) vs quad
# -------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=DPI)

labels = ["Mean single\n(A–D + NTC)", "Quad-channel\n(ABCD + NTC)"]
rmse_vals = [rmse_single_mean, rmse_quad]
mae_vals  = [mae_single_mean,  mae_quad]

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

ymax = max(rmse_vals + mae_vals)
ax.set_ylim(0, ymax * 1.18)

def label_bars(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03 * ymax, f"{h:.2f}",
                ha="center", va="bottom", fontsize=FONT_TICK)

label_bars(b1)
label_bars(b2)

fig.tight_layout(rect=[0, 0, 0.82, 1.0])
fig3_path = os.path.join(OUT_DIR, "fig_errors_single_mean_vs_quad.png")
fig.savefig(fig3_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", fig3_path)

print("\nDONE ✅")