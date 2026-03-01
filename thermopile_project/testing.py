import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- repo-root så att "LM/" kan importeras ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from LM.levenberg_marquardt import lm, lm_func

CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"

# -------- config --------
TEMP_MIN_C = 100
TEMP_MAX_C = 135
CHANNEL = "A"   # ändra till "B", "C", "D" för andra

# ------------------------
data = pd.read_csv(CSV_PATH)

# rensa
data = data.dropna(subset=["y_tc_mean", "ntc_mean", f"{CHANNEL}_rms"]).copy()
data = data[(data["y_tc_mean"] >= TEMP_MIN_C) & (data["y_tc_mean"] <= TEMP_MAX_C)].copy()

# arrays
T_C = data["y_tc_mean"].to_numpy(float)
Tbg_C = data["ntc_mean"].to_numpy(float)
V = data[f"{CHANNEL}_rms"].to_numpy(float)

T = T_C + 273.15
Tbg = Tbg_C + 273.15
x = np.column_stack([T, Tbg])

# -------- fit (samma modell som innan) --------
V0_0 = float(np.mean(V))
n0 = 2.0
dTn = (T**n0 - Tbg**n0)
dTn_scale = float(np.median(np.abs(dTn))) if np.median(np.abs(dTn)) > 0 else 1.0
K0 = float((np.std(V) + 1e-6) / dTn_scale)

p0 = np.array([[V0_0], [K0], [n0]])

p_fit, redX2, *_rest, cvg_hst = lm(p0, x, V)

V_fit = lm_func(x, p_fit)

# -------- analysis --------
V0 = float(p_fit[0, 0])
U = V - V0

rmse = float(np.sqrt(np.mean((V - V_fit) ** 2)))

print("\n--- Testing net signal scale ---")
print("Channel:", CHANNEL)
print("V0 =", V0)
print("RMSE (V) =", rmse)
print("U min/max (V):", float(U.min()), float(U.max()))
print("U peak-to-peak (V):", float(U.max() - U.min()))
print("U std (mV):", float(np.std(U) * 1000))

# -------- plots --------
plt.figure()
plt.scatter(T_C, U, s=22)
plt.xlabel("Target temperature (°C)")
plt.ylabel("U = V - V0 (V)")
plt.title(f"Channel {CHANNEL}: net signal")
plt.grid(True, alpha=0.25)
plt.show()

plt.figure()
plt.scatter(T_C, V - V_fit, s=22)
plt.xlabel("Target temperature (°C)")
plt.ylabel("Residual (V)")
plt.title(f"Channel {CHANNEL}: residuals")
plt.grid(True, alpha=0.25)
plt.show()
