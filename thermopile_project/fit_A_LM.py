import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from LM.levenberg_marquardt import lm, lm_func

CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"

data = pd.read_csv(CSV_PATH)

need_cols = ["y_tc_mean", "ntc_mean", "A_rms"]
data = data.dropna(subset=need_cols).copy()
data = data[(data["y_tc_mean"] >= 100) & (data["y_tc_mean"] <= 135)].copy()

T_C = data["y_tc_mean"].to_numpy(dtype=float)
Tbg_C = data["ntc_mean"].to_numpy(dtype=float)
A = data["A_rms"].to_numpy(dtype=float)

T = T_C + 273.15
Tbg = Tbg_C + 273.15
x = np.column_stack([T, Tbg])

print("x shape:", x.shape)
print("A shape:", A.shape)
print("A min/max:", float(A.min()), float(A.max()))

# ---------- startgissning ----------
V0_0 = float(np.mean(A))     # offset nära 2.5 V
n0 = 2.0

# grov K-gissning så att amplituden hamnar i rätt storleksordning
dTn = (T**n0 - Tbg**n0)
dTn_scale = float(np.median(np.abs(dTn))) if np.median(np.abs(dTn)) > 0 else 1.0
K0 = float((np.std(A) + 1e-6) / dTn_scale)

p0 = np.array([[V0_0],
               [K0],
               [n0]])

p_fit, redX2, sigma_p, sigma_y, corr_p, R_sq, cvg_hst = lm(p0, x, A)

A_fit = lm_func(x, p_fit)

print("\n--- Fit result (Channel A) ---")
print("p_fit =\n", p_fit)
print("V0 =", float(p_fit[0, 0]))
print("K  =", float(p_fit[1, 0]))
print("n  =", float(p_fit[2, 0]))
print("redX2 =", redX2)

# sanity: RMSE
rmse = float(np.sqrt(np.mean((A - A_fit) ** 2)))
print("RMSE (V) =", rmse)

plt.figure()
plt.scatter(T_C, A, s=22, alpha=0.9, label="Data (A_rms)")
plt.scatter(T_C, A_fit, s=22, alpha=0.9, label="Fit: V0 + K*(T^n - Tbg^n)")
plt.xlabel("Target temperature (°C)")
plt.ylabel("A_rms (V)")
plt.title("Channel A LM fit with offset")
plt.grid(True, alpha=0.25)
plt.legend()
plt.show()
