import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS
# -------------------------
CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "ratio_out")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "y_tc_mean"
T_MIN = 100.0
T_MAX = 130.0

CHANNELS = ["A", "B", "C", "D"]

# Compute U = V_rms - V0 if U columns are missing
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

# Filter low-signal points (prevents ratio blow-ups). Use 0.0 to disable.
U_MIN = 0.05  # 50 mV

# Plot styling
DPI = 220
FIG_W, FIG_H = 14, 8
FONT_TITLE = 18
FONT_AX = 12
FONT_TICK = 10
FONT_LEGEND = 10
POINT_SIZE = 20

# -------------------------
# HELPERS
# -------------------------
def extract_surface(fname: str):
    m = re.match(r"^(J[1-6])", str(fname))
    return m.group(1) if m else None

def linear_fit(x, y):
    p = np.polyfit(x, y, 1)
    return p[0], p[1]  # slope, intercept

# -------------------------
# LOAD + FILTER
# -------------------------
df = pd.read_csv(CSV_PATH)
df["surface"] = df["file"].apply(extract_surface)

need = ["file", "surface", TARGET]
df = df.dropna(subset=need).copy()
df = df[(df[TARGET] >= T_MIN) & (df[TARGET] <= T_MAX)].copy()

# Ensure U columns exist by computing from RMS if missing
for ch in CHANNELS:
    ucol = f"U_{ch}"
    vcol = f"{ch}_rms"
    if ucol not in df.columns:
        if vcol not in df.columns:
            raise ValueError(f"Missing {ucol} and {vcol} in CSV.")
        if ch not in V0:
            raise ValueError(f"Missing V0[{ch}] to compute {ucol}.")
        df[ucol] = df[vcol].astype(float) - float(V0[ch])

surfaces = sorted([s for s in df["surface"].unique() if isinstance(s, str)])
if len(surfaces) == 0:
    raise ValueError("No surfaces detected. Filenames should start with J1..J6.")

# Define 6 ratios (2x3)
RATIOS = [("A","B"), ("A","C"), ("A","D"),
          ("B","C"), ("B","D"), ("C","D")]

# -------------------------
# PLOT GRID
# -------------------------
fig, axs = plt.subplots(2, 3, figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
axs = axs.ravel()

for ax, (c1, c2) in zip(axs, RATIOS):
    for s in surfaces:
        sub = df[df["surface"] == s].copy()

        T = sub[TARGET].to_numpy(float)
        U1 = sub[f"U_{c1}"].to_numpy(float)
        U2 = sub[f"U_{c2}"].to_numpy(float)

        m = np.isfinite(T) & np.isfinite(U1) & np.isfinite(U2)
        if U_MIN > 0:
            m = m & (np.abs(U1) > U_MIN) & (np.abs(U2) > U_MIN)

        if np.sum(m) < 3:
            continue

        Tm = T[m]
        R = U1[m] / U2[m]

        ax.scatter(Tm, R, s=POINT_SIZE, alpha=0.75, label=s)

        # linear fit per surface (thin line)
        try:
            slope, intercept = linear_fit(Tm, R)
            Tfit = np.linspace(float(np.min(Tm)), float(np.max(Tm)), 80)
            ax.plot(Tfit, slope*Tfit + intercept, linewidth=1.2, alpha=0.6)
        except Exception:
            pass

    ax.set_title(f"U_{c1} / U_{c2}", fontsize=FONT_AX)
    ax.set_xlabel("Temperature (°C)", fontsize=FONT_AX)
    ax.set_ylabel("Ratio", fontsize=FONT_AX)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.25)

# Shared legend (single place, no clutter)
handles, labels = axs[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, title="Surface", fontsize=FONT_LEGEND,
               title_fontsize=FONT_LEGEND, loc="center right", bbox_to_anchor=(1.02, 0.5))

fig.suptitle("Ratio analysis (2×3): surface-dependent behaviour prevents emissivity cancellation",
             fontsize=FONT_TITLE)

out_path = os.path.join(OUT_DIR, "ratio_grid_2x3.png")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", out_path)
print("DONE ✅")