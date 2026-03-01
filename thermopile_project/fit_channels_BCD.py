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

# ------------------- config -------------------
TEMP_MIN_C = 105
TEMP_MAX_C = 135
CHANNELS = ["A", "B", "C", "D"]  # <-- de du vill göra nu
# (vill du göra alla: ["A","B","C","D"])

# ------------------- load + clean -------------------
data = pd.read_csv(CSV_PATH)

# krävs för alla kanaler (temperaturer)
data = data.dropna(subset=["y_tc_mean", "ntc_mean"]).copy()
data = data[(data["y_tc_mean"] >= TEMP_MIN_C) & (data["y_tc_mean"] <= TEMP_MAX_C)].copy()

def fit_one_channel(df: pd.DataFrame, ch: str, do_plot: bool = True):
    col = f"{ch}_rms"
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in CSV. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[col]).copy()

    # arrays
    T_C = df["y_tc_mean"].to_numpy(dtype=float)
    Tbg_C = df["ntc_mean"].to_numpy(dtype=float)
    V = df[col].to_numpy(dtype=float)

    # Kelvin
    T = T_C + 273.15
    Tbg = Tbg_C + 273.15
    x = np.column_stack([T, Tbg])

    # ----- start guesses -----
    V0_0 = float(np.mean(V))
    n0 = 2.0
    dTn = (T**n0 - Tbg**n0)
    dTn_scale = float(np.median(np.abs(dTn))) if np.median(np.abs(dTn)) > 0 else 1.0
    K0 = float((np.std(V) + 1e-6) / dTn_scale)

    p0 = np.array([[V0_0],
                   [K0],
                   [n0]])

    # fit
    p_fit, redX2, *_rest, cvg_hst = lm(p0, x, V)
    V_fit = lm_func(x, p_fit)

    # metrics
    rmse = float(np.sqrt(np.mean((V - V_fit) ** 2)))
    ss_res = float(np.sum((V - V_fit) ** 2))
    ss_tot = float(np.sum((V - np.mean(V)) ** 2))
    r2 = float(1 - ss_res / (ss_tot + 1e-12))

    V0 = float(p_fit[0, 0])
    K = float(p_fit[1, 0])
    n = float(p_fit[2, 0])

    print(f"\n--- Channel {ch} ---")
    print("N =", len(df))
    print("V0 =", V0)
    print("K  =", K)
    print("n  =", n)
    print("RMSE (V) =", rmse)
    print("R2 =", r2)
    print("redX2 =", float(redX2))

    if do_plot:
        plt.figure()
        plt.scatter(T_C, V, s=22, alpha=0.9, label=f"Data ({col})")
        plt.scatter(T_C, V_fit, s=22, alpha=0.9, label="Fit: V0 + K*(T^n - Tbg^n)")
        plt.xlabel("Target temperature (°C)")
        plt.ylabel(f"{col} (V)")
        plt.title(f"Channel {ch} LM fit")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.show()

    return {"channel": ch, "N": len(df), "V0": V0, "K": K, "n": n, "RMSE_V": rmse, "R2": r2}

results = []
for ch in CHANNELS:
    results.append(fit_one_channel(data, ch, do_plot=True))

# sammanfattning
out = pd.DataFrame(results)
print("\n=== Summary ===")
print(out)

# spara
out_path = os.path.join(os.path.dirname(__file__), "lm_params_BCD.csv")
out.to_csv(out_path, index=False)
print("\nSaved:", out_path)
