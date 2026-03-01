# poster_plots_ratios.py
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

U_MIN = 0.05   # low-signal threshold (V)
TARGET = "y_tc_mean"  # °C

# If U columns are missing, compute U = V_rms - V0
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

# Pick ONE ratio for poster
RATIOS_TO_PLOT = [
    ("A", "D"),  # R_AB
    # ("B", "D"),
]

# Poster styling
POSTER_DPI = 220
FONT_TITLE = 16
FONT_AX = 14
FONT_TICK = 12
FONT_LEGEND = 11
POINT_SIZE = 52

# Robust axis scaling (recommended for ratios)
USE_ROBUST_YLIM = True
Y_LIM_PCT = (2, 98)  # keep central 96% of points

# -------------------------
# HELPERS
# -------------------------
def extract_surface(fname: str):
    m = re.match(r"^(J[1-6])", str(fname))
    return m.group(1) if m else None

# -------------------------
# LOAD + PREP
# -------------------------
df = pd.read_csv(CSV_PATH)
df["surface"] = df["file"].apply(extract_surface)

# Ensure U columns exist
for ch in ["A", "B", "C", "D"]:
    ucol = f"U_{ch}"
    vcol = f"{ch}_rms"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing {vcol} and {ucol}.")
        df[ucol] = df[vcol] - float(V0[ch])

need = ["surface", TARGET, "U_A", "U_B", "U_C", "U_D"]
df = df.dropna(subset=need).copy()

# Keep only 105–130 °C
T_MIN_C = 105.0
T_MAX_C = 130.0
df = df[(df[TARGET] >= T_MIN_C) & (df[TARGET] <= T_MAX_C)].copy()

surfaces = sorted([s for s in df["surface"].unique() if isinstance(s, str)])


# -------------------------
# PLOTS
# -------------------------
for num, den in RATIOS_TO_PLOT:
    u_num = df[f"U_{num}"].to_numpy(float)
    u_den = df[f"U_{den}"].to_numpy(float)

    keep = (np.abs(u_num) > U_MIN) & (np.abs(u_den) > U_MIN)
    dfk = df.loc[keep].copy()

    ratio = (dfk[f"U_{num}"] / dfk[f"U_{den}"]).to_numpy(float)
    T = dfk[TARGET].to_numpy(float)

    print(f"\nR_{num}{den}: kept {len(dfk)}/{len(df)} (removed {len(df)-len(dfk)}) with |U| > {U_MIN} V")

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=POSTER_DPI)

    # Scatter by surface
    for s in surfaces:
        m = (dfk["surface"] == s).to_numpy()
        if m.sum() == 0:
            continue
        ax.scatter(T[m], ratio[m], s=POINT_SIZE, alpha=0.85, label=s)

    ax.set_xlabel("True temperature (°C)", fontsize=FONT_AX)
    ax.set_ylabel(f"R$_{{{num}{den}}}$ = U$_{num}$ / U$_{den}$", fontsize=FONT_AX)
    ax.set_title(f"Ratio feature (filtered): R$_{{{num}{den}}}$", fontsize=FONT_TITLE)

    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.25)

    # Robust y-limits so outliers don't squash the plot
    if USE_ROBUST_YLIM and len(ratio) >= 10:
        lo, hi = np.percentile(ratio, Y_LIM_PCT)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

    # --- Put info text OUTSIDE the axes (cannot cover points) ---
    info = f"Filter: |U_{num}|, |U_{den}| > {U_MIN:.2f} V   |   N kept = {len(dfk)}"
    fig.text(0.01, 0.995, info, ha="left", va="top",
             fontsize=FONT_LEGEND, bbox=dict(boxstyle="round", alpha=0.10))

    # Legend outside
    ax.legend(title="Surface", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
              loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0, 0.82, 0.95])  # leave room for legend + top text

    out = os.path.join(OUT_DIR, f"fig_ratio_R_{num}{den}_vs_T.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)

plt.show()
