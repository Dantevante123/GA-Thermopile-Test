import os
import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"

# ======== KLIStra IN DINA OFFSETS (V0) HÄR ========
V0 = {
    "A": 2.600319294289444,
    "B": 2.56819319269647,          # <- byt gärna till full precision om du har den
    "C": 2.597231935273712,
    "D": 2.624025086313931,
}
# ================================================

# Undvik division nära noll (ratio exploderar annars)
EPS = 1e-9

data = pd.read_csv(CSV_PATH)

# Förväntade kolumner
need_cols = ["y_tc_mean", "ntc_mean", "A_rms", "B_rms", "C_rms", "D_rms"]
missing = [c for c in need_cols if c not in data.columns]
if missing:
    raise KeyError(f"Missing columns in CSV: {missing}\nAvailable: {list(data.columns)}")

# Rensa NaN i de kolumner vi behöver
data = data.dropna(subset=need_cols).copy()

# --- Skapa U_i = V_i - V0_i ---
for ch in ["A", "B", "C", "D"]:
    data[f"U_{ch}"] = data[f"{ch}_rms"].astype(float) - float(V0[ch])

# --- Skapa ratios ---
# R_AB = U_A / U_B etc.
def safe_ratio(num, den, eps=EPS):
    den_safe = np.where(np.abs(den) < eps, np.nan, den)  # NaN där det är för nära noll
    return num / den_safe

data["R_AB"] = safe_ratio(data["U_A"].to_numpy(), data["U_B"].to_numpy())
data["R_AC"] = safe_ratio(data["U_A"].to_numpy(), data["U_C"].to_numpy())
data["R_AD"] = safe_ratio(data["U_A"].to_numpy(), data["U_D"].to_numpy())
data["R_BC"] = safe_ratio(data["U_B"].to_numpy(), data["U_C"].to_numpy())
data["R_BD"] = safe_ratio(data["U_B"].to_numpy(), data["U_D"].to_numpy())
data["R_CD"] = safe_ratio(data["U_C"].to_numpy(), data["U_D"].to_numpy())

# --- Snabba sanity prints ---
print("\n--- Sanity check ---")
for ch in ["A", "B", "C", "D"]:
    u = data[f"U_{ch}"].to_numpy()
    print(f"U_{ch}: min/max = {float(np.nanmin(u)):.4f} / {float(np.nanmax(u)):.4f}, std (mV) = {float(np.nanstd(u)*1000):.2f}")

for r in ["R_AB","R_AC","R_AD","R_BC","R_BD","R_CD"]:
    arr = data[r].to_numpy()
    finite = np.isfinite(arr)
    frac_nan = 1.0 - float(np.mean(finite))
    if np.any(finite):
        print(f"{r}: min/max = {float(np.nanmin(arr)):.4f} / {float(np.nanmax(arr)):.4f}, NaN/inf frac = {frac_nan:.3%}")
    else:
        print(f"{r}: ALL NaN/inf (check denominators near 0)")

# --- Spara ---
out_path = os.path.join(os.path.dirname(__file__), "features_with_U_and_ratios.csv")
data.to_csv(out_path, index=False)
print("\nSaved:", out_path)
