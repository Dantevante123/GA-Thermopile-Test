import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# SETTINGS (ändra dessa)
# ------------------------------------------------------------
CSV_FEATURES = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"
# Om du vill köra ratio-delarna utan att räkna om U/ratios:
CSV_WITH_U_AND_RATIOS = None  # t.ex. r"...\features_with_U_and_ratios.csv"

OUT_DIR = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\report_plots_105_130"
os.makedirs(OUT_DIR, exist_ok=True)

# Använd bara denna temperatur-range
T_MIN_C = 105.0
T_MAX_C = 130.0

# Single-channel att visa (A/B/C/D)
SHOW_CHANNEL = "A"

# Ratio att visa (om du använder ratios)
SHOW_RATIO = ("A", "D")   # R_AD

# Ratio filter: ta bort datapunkter med låg net-signal
U_MIN = 0.05  # 50 mV

# ------------------------------------------------------------
# Importera din LM (förutsätter att du har LM/levenberg_marquardt.py i repot)
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Om du har LM som subfolder: ROOT/LM/levenberg_marquardt.py
import os, sys

# lägg till repo-root i sys.path så import funkar från root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# försök importera LM på båda sätten (med eller utan LM/)
try:
    from LM.levenberg_marquardt import lm, lm_func
except ModuleNotFoundError:
    from levenberg_marquardt import lm, lm_func 
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def rmse(y, yhat):
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat):
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)
    return float(np.mean(np.abs(y - yhat)))

def extract_surface(fname: str):
    m = re.match(r"^(J[1-6])", str(os.path.basename(fname)), flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("Saved:", path)

# ------------------------------------------------------------
# Load + filter
# ------------------------------------------------------------
df = pd.read_csv(CSV_FEATURES)

need = ["file", "y_tc_mean", "ntc_mean", "A_rms", "B_rms", "C_rms", "D_rms"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"features.csv saknar kolumner: {missing}\nFinns: {list(df.columns)}")

df["surface"] = df["file"].apply(extract_surface)

# keep only 100–130
df = df.dropna(subset=["y_tc_mean", "ntc_mean"]).copy()
df = df[(df["y_tc_mean"] >= T_MIN_C) & (df["y_tc_mean"] <= T_MAX_C)].copy()

print("Rows after 105–130°C filter:", len(df))

# ------------------------------------------------------------
# 1) Data characteristics
# ------------------------------------------------------------
# 1a Raw voltage vs temp
fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, ch in zip(axs.ravel(), ["A", "B", "C", "D"]):
    ax.scatter(df["y_tc_mean"], df[f"{ch}_rms"], s=18, alpha=0.85)
    ax.set_xlabel("Target temperature (°C)")
    ax.set_ylabel(f"{ch}_rms (V)")
    ax.set_title(f"Raw voltage vs temperature ({ch})")
    ax.grid(True, alpha=0.25)
savefig(fig, "1_raw_voltage_vs_temp_105_130.png")

# 1b Histogram of raw signal levels (RMS)
fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, ch in zip(axs.ravel(), ["A", "B", "C", "D"]):
    v = df[f"{ch}_rms"].to_numpy(float)
    v = v[np.isfinite(v)]
    ax.hist(v, bins=35)
    ax.set_xlabel(f"{ch}_rms (V)")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of signal level ({ch})")
    ax.grid(True, alpha=0.25)
savefig(fig, "1_histogram_signal_levels_105_130.png")

# ------------------------------------------------------------
# 2) Single-channel LM: refit ON 100–130 and then invert to temperature
# ------------------------------------------------------------
ch = SHOW_CHANNEL.upper()
df_sc = df.dropna(subset=[f"{ch}_rms"]).copy()

T_C = df_sc["y_tc_mean"].to_numpy(float)
Tbg_C = df_sc["ntc_mean"].to_numpy(float)
V = df_sc[f"{ch}_rms"].to_numpy(float)

# Kelvin for model
T = T_C + 273.15
Tbg = Tbg_C + 273.15
x = np.column_stack([T, Tbg])

# --- start guesses (stabilt) ---
V0_0 = float(np.mean(V))
n0 = 2.0
dTn = (T**n0 - Tbg**n0)
dTn_scale = float(np.median(np.abs(dTn))) if np.median(np.abs(dTn)) > 0 else 1.0
K0 = float((np.std(V) + 1e-6) / dTn_scale)

p0 = np.array([[V0_0], [K0], [n0]])

# --- fit LM ---
p_fit, redX2, *_ = lm(p0, x, V)
V0 = float(p_fit[0, 0])
K  = float(p_fit[1, 0])
n  = float(p_fit[2, 0])

print("\n--- Single-channel LM (refit on 105–130) ---")
print("Channel:", ch)
print("V0 =", V0)
print("K  =", K)
print("n  =", n)
print("redX2 =", float(redX2))

# Fit curve
V_fit = lm_func(x, p_fit)
res_V = V - V_fit

# 2a Fit curve vs data
fig, ax = plt.subplots(figsize=(8.6, 6.2))
ax.scatter(T_C, V, s=20, alpha=0.85, label="Measured")
ax.scatter(T_C, V_fit, s=20, alpha=0.85, label="LM fit")
ax.set_xlabel("Target temperature (°C)")
ax.set_ylabel(f"{ch}_rms (V)")
ax.set_title(f"Single-channel LM fit (Channel {ch}, 105–130°C)")
ax.grid(True, alpha=0.25)
ax.legend()
savefig(fig, f"2_single_LM_fit_curve_{ch}_105_130.png")

# 2b Residuals vs temperature (voltage)
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.scatter(T_C, res_V, s=20, alpha=0.85)
ax.axhline(0, linewidth=1)
ax.set_xlabel("Target temperature (°C)")
ax.set_ylabel("Voltage residual (V)")
ax.set_title(f"Voltage residuals vs temperature (Channel {ch})")
ax.grid(True, alpha=0.25)
savefig(fig, f"2_single_voltage_residuals_{ch}_105_130.png")

# --- Invert to temperature (robust) ---
# T_pred_K = ((V - V0)/K + Tbg^n)^(1/n)
# Guard: if K ~ 0 or inside <= 0 -> NaN
with np.errstate(divide="ignore", invalid="ignore"):
    if abs(K) < 1e-18:
        inside = np.full_like(V, np.nan, dtype=float)
    else:
        inside = (V - V0) / K + (Tbg ** n)

inside = np.where(inside > 0, inside, np.nan)
T_pred_K = inside ** (1.0 / n)
T_pred_C = T_pred_K - 273.15

m_ok = np.isfinite(T_pred_C) & np.isfinite(T_C)
sc_rmse = rmse(T_C[m_ok], T_pred_C[m_ok])
sc_mae  = mae(T_C[m_ok], T_pred_C[m_ok])

err_C = T_pred_C - T_C

# 2c Pred vs true with 45° line
fig, ax = plt.subplots(figsize=(7.8, 6.6))
ax.scatter(T_C[m_ok], T_pred_C[m_ok], s=22, alpha=0.85)
mn = float(min(T_C[m_ok].min(), T_pred_C[m_ok].min()))
mx = float(max(T_C[m_ok].max(), T_pred_C[m_ok].max()))
pad = 1.5
ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], linewidth=2, alpha=0.45)
ax.set_xlabel("True temperature (°C)")
ax.set_ylabel("Predicted temperature (°C)")
ax.set_title(f"Single-channel prediction (Channel {ch}, 105–130°C)")
ax.grid(True, alpha=0.25)
ax.text(0.02, 0.98, f"RMSE = {sc_rmse:.2f} °C\nMAE  = {sc_mae:.2f} °C\nN = {int(m_ok.sum())}",
        transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", alpha=0.12))
savefig(fig, f"2_single_pred_vs_true_{ch}_105_130.png")

# 2d Residuals in temperature domain
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.scatter(T_C[m_ok], err_C[m_ok], s=22, alpha=0.85)
ax.axhline(0, linewidth=1)
ax.set_xlabel("True temperature (°C)")
ax.set_ylabel("Temp error (°C)  (pred - true)")
ax.set_title(f"Temperature error vs temperature (Channel {ch})")
ax.grid(True, alpha=0.25)
savefig(fig, f"2_single_temp_error_vs_temp_{ch}_105_130.png")

# 2e Error histogram
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.hist(err_C[m_ok], bins=30)
ax.set_xlabel("Temp error (°C)  (pred - true)")
ax.set_ylabel("Count")
ax.set_title(f"Error distribution (Single-channel {ch})")
ax.grid(True, alpha=0.25)
savefig(fig, f"2_single_error_hist_{ch}_105_130.png")

# ------------------------------------------------------------
# 3) Ratio-based analysis (optional)
# ------------------------------------------------------------
if CSV_WITH_U_AND_RATIOS is not None:
    dfr = pd.read_csv(CSV_WITH_U_AND_RATIOS)
    dfr = dfr.dropna(subset=["file", "y_tc_mean"]).copy()
    dfr = dfr[(dfr["y_tc_mean"] >= T_MIN_C) & (dfr["y_tc_mean"] <= T_MAX_C)].copy()

    num, den = SHOW_RATIO[0].upper(), SHOW_RATIO[1].upper()
    rname = f"R_{num}{den}"
    if rname not in dfr.columns:
        # din csv brukar ha R_AD med underscore-format: R_AD
        alt = f"R_{num}{den}"
    ratio_col = f"R_{num}{den}"
    if ratio_col not in dfr.columns:
        # fallback to underscore version: R_AD
        ratio_col = f"R_{num}{den}"
    if f"R_{num}{den}" not in dfr.columns and f"R_{num}{den}" not in dfr.columns:
        # easiest: assume R_AD naming with underscore: R_AD
        ratio_col = f"R_{num}{den}"
    # In your repo it is R_AD etc with underscore: R_AD
    ratio_col = f"R_{num}{den}".replace("R_", "R_")  # no-op

    # correct naming: R_AD
    ratio_col = f"R_{num}{den}"
    if ratio_col not in dfr.columns:
        ratio_col = f"R_{num}{den}"  # still
    # final: use underscore style
    ratio_col = f"R_{num}{den}"
    if ratio_col not in dfr.columns:
        ratio_col = f"R_{num}{den}"  # ok

    # Actually your columns are like R_AD, so:
    ratio_col = f"R_{num}{den}"  # e.g. R_AD
    ratio_col = ratio_col[:2] + "_" + ratio_col[2:]  # "R_" + "AD" -> "R_AD"

    needr = [ratio_col, f"U_{num}", f"U_{den}", "y_tc_mean"]
    missr = [c for c in needr if c not in dfr.columns]
    if missr:
        print("\n[Ratio] Skipping ratio plots, missing columns:", missr)
    else:
        R = dfr[ratio_col].to_numpy(float)
        Tt = dfr["y_tc_mean"].to_numpy(float)
        U_num = dfr[f"U_{num}"].to_numpy(float)
        U_den = dfr[f"U_{den}"].to_numpy(float)

        # 3a unfiltered
        fig, ax = plt.subplots(figsize=(9.0, 6.2))
        ax.scatter(Tt, R, s=18, alpha=0.75)
        ax.set_xlabel("Target temperature (°C)")
        ax.set_ylabel(ratio_col)
        ax.set_title(f"Ratio vs temperature (unfiltered): {ratio_col}")
        ax.grid(True, alpha=0.25)
        savefig(fig, f"3_ratio_unfiltered_{ratio_col}_105_130.png")

        # 3b filtered
        m = np.isfinite(R) & np.isfinite(U_num) & np.isfinite(U_den) & (np.abs(U_num) > U_MIN) & (np.abs(U_den) > U_MIN)
        fig, ax = plt.subplots(figsize=(9.0, 6.2))
        ax.scatter(Tt[m], R[m], s=18, alpha=0.80)
        ax.set_xlabel("Target temperature (°C)")
        ax.set_ylabel(f"{ratio_col} (filtered)")
        ax.set_title(f"Ratio vs temperature (|U| > {U_MIN:.2f} V): {ratio_col}")
        ax.grid(True, alpha=0.25)
        savefig(fig, f"3_ratio_filtered_{ratio_col}_105_130.png")

        # 3c instability: |ratio| vs |den|
        fig, ax = plt.subplots(figsize=(9.0, 6.2))
        ax.scatter(np.abs(U_den), np.abs(R), s=18, alpha=0.75)
        ax.set_xlabel(f"|U_{den}| (V)")
        ax.set_ylabel(f"|{ratio_col}|")
        ax.set_title(f"Low-signal instability: |{ratio_col}| vs |U_{den}|")
        ax.grid(True, alpha=0.25)
        savefig(fig, f"3_ratio_instability_{ratio_col}_105_130.png")

# ------------------------------------------------------------
# 4) Multi-channel regression (re-fit on 100–130)
# ------------------------------------------------------------
# Create U from single-channel V0 (this is consistent inside this script)
# NOTE: This is just one reasonable choice; your best model may use per-channel V0 from per-channel LM fits.
df_reg = df.dropna(subset=["A_rms","B_rms","C_rms","D_rms","ntc_mean","y_tc_mean"]).copy()

# Quick per-channel V0 estimation from mean in the window
V0_guess = {c: float(df_reg[f"{c}_rms"].mean()) for c in ["A","B","C","D"]}
for c in ["A","B","C","D"]:
    df_reg[f"U_{c}"] = df_reg[f"{c}_rms"].astype(float) - V0_guess[c]

FEATURES = ["U_A","U_B","U_C","U_D","ntc_mean"]
X = df_reg[FEATURES].to_numpy(float)
y = df_reg["y_tc_mean"].to_numpy(float)

# Fit least squares
X_i = np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(X_i, y, rcond=None)
yhat = X_i @ beta

reg_rmse = rmse(y, yhat)
reg_mae = mae(y, yhat)

# Pred vs true
fig, ax = plt.subplots(figsize=(7.8, 6.6))
ax.scatter(y, yhat, s=22, alpha=0.85)
mn = float(min(y.min(), yhat.min()))
mx = float(max(y.max(), yhat.max()))
pad = 1.5
ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], linewidth=2, alpha=0.45)
ax.set_xlabel("True temperature (°C)")
ax.set_ylabel("Predicted temperature (°C)")
ax.set_title("Quad-channel regression (fit on 105–130°C)")
ax.grid(True, alpha=0.25)
ax.text(0.02, 0.98, f"RMSE = {reg_rmse:.2f} °C\nMAE  = {reg_mae:.2f} °C\nN = {len(y)}",
        transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", alpha=0.12))
savefig(fig, "4_multi_pred_vs_true_105_130.png")

# Error distribution
err = yhat - y
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.hist(err, bins=30)
ax.set_xlabel("Temp error (°C)  (pred - true)")
ax.set_ylabel("Count")
ax.set_title("Quad-channel regression: Error distribution (105–130°C)")
ax.grid(True, alpha=0.25)
savefig(fig, "4_multi_error_hist_105_130.png")

# Coeff bar
names = ["intercept"] + FEATURES
fig, ax = plt.subplots(figsize=(9.0, 4.8))
xpos = np.arange(len(beta))
ax.bar(xpos, beta)
ax.set_xticks(xpos)
ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_ylabel("Coefficient value")
ax.set_title("Regression coefficients (105–130°C)")
ax.grid(True, axis="y", alpha=0.25)
savefig(fig, "4_multi_coefficients_105_130.png")

# ------------------------------------------------------------
# 5) Comparison (MAE/RMSE)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.8))
labels = [f"Single-channel\nLM ({ch})", "Quad-channel\nRegression"]
rmse_vals = [sc_rmse, reg_rmse]
mae_vals = [sc_mae, reg_mae]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, rmse_vals, width=w, label="RMSE (°C)")
ax.bar(x + w/2, mae_vals,  width=w, label="MAE (°C)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Error (°C)")
ax.set_title("Model comparison (105–130°C)")
ax.grid(True, axis="y", alpha=0.25)
ax.legend()
savefig(fig, "5_comparison_mae_rmse_105_130.png")

# Save metrics summary
summary = pd.DataFrame([
    {"model": f"Single-channel LM ({ch})", "RMSE_C": sc_rmse, "MAE_C": sc_mae, "N": int(m_ok.sum())},
    {"model": "Quad-channel regression", "RMSE_C": reg_rmse, "MAE_C": reg_mae, "N": len(y)},
])
summary_path = os.path.join(OUT_DIR, "metrics_summary_105_130.csv")
summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)

print("\nDONE ✅ All plots saved to:", OUT_DIR)