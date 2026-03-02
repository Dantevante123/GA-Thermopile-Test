import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "ratio_out")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "y_tc_mean"
T_MIN = 100.0
T_MAX = 130.0

CHANNELS = ["A", "B", "C", "D"]
V0 = {
    "A": 2.5644972181941443,
    "B": 2.549021,
    "C": 2.571482,
    "D": 2.587497,
}

# Turn OFF low-signal filtering initially (so you don't get empty plots)
U_MIN = 0.0  # set to 0.05 later if needed

GRID_STEP_C = 0.5
CLEAN_RATIO = ("A", "D")

DPI = 220
FONT_TITLE = 18
FONT_AX = 12
FONT_TICK = 10
FONT_LEGEND = 10
POINT_SIZE = 18

RATIOS = [("A","B"), ("A","C"), ("A","D"),
          ("B","C"), ("B","D"), ("C","D")]

def extract_surface(fname: str):
    m = re.match(r"^(J[1-6])", str(fname))
    return m.group(1) if m else None

def ensure_U_cols(df):
    for ch in CHANNELS:
        ucol = f"U_{ch}"
        vcol = f"{ch}_rms"
        if ucol not in df.columns:
            if vcol not in df.columns:
                raise ValueError(f"Missing {ucol} and {vcol} in CSV.")
            df[ucol] = df[vcol].astype(float) - float(V0[ch])
    return df

def compress_duplicates(T, Y):
    T = np.asarray(T, float)
    Y = np.asarray(Y, float)
    m = np.isfinite(T) & np.isfinite(Y)
    T = T[m]; Y = Y[m]
    if len(T) == 0:
        return T, Y
    order = np.argsort(T)
    T = T[order]; Y = Y[order]

    Tu, Yu = [], []
    i = 0
    while i < len(T):
        t0 = T[i]
        j = i
        while j < len(T) and T[j] == t0:
            j += 1
        Tu.append(t0)
        Yu.append(float(np.mean(Y[i:j])))
        i = j
    return np.array(Tu, float), np.array(Yu, float)

def interp_on_grid(T, Y, grid):
    T, Y = compress_duplicates(T, Y)
    if len(T) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, T, Y, left=np.nan, right=np.nan)

# -------------------------
# LOAD
# -------------------------
df = pd.read_csv(CSV_PATH)
df["surface"] = df["file"].apply(extract_surface)
df = df.dropna(subset=["file", "surface", TARGET]).copy()
df = df[(df[TARGET] >= T_MIN) & (df[TARGET] <= T_MAX)].copy()
df = ensure_U_cols(df)

surfaces = sorted([s for s in df["surface"].unique() if isinstance(s, str)])
print("Surfaces:", surfaces)
print("Rows in range:", len(df))

grid = np.arange(T_MIN, T_MAX + 1e-9, GRID_STEP_C)

# -------------------------
# FIG 1: 2x3 grid (no overlap requirement)
# -------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 8), dpi=DPI, constrained_layout=True)
axs = axs.ravel()

for ax, (c1, c2) in zip(axs, RATIOS):
    for s in surfaces:
        sub = df[df["surface"] == s]

        T = sub[TARGET].to_numpy(float)
        U1 = sub[f"U_{c1}"].to_numpy(float)
        U2 = sub[f"U_{c2}"].to_numpy(float)

        m = np.isfinite(T) & np.isfinite(U1) & np.isfinite(U2)
        if U_MIN > 0:
            m = m & (np.abs(U1) > U_MIN) & (np.abs(U2) > U_MIN)

        if np.sum(m) < 2:
            continue

        R = U1[m] / U2[m]
        Rg = interp_on_grid(T[m], R, grid)

        # plot curve; NaNs will automatically create gaps
        ax.plot(grid, Rg, linewidth=2.0, alpha=0.85, label=s)

    ax.set_title(f"U_{c1} / U_{c2}", fontsize=FONT_AX)
    ax.set_xlabel("Temperature (°C)", fontsize=FONT_AX)
    ax.set_ylabel("Ratio", fontsize=FONT_AX)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.25)

handles, labels = axs[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, title="Surface",
               fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
               loc="center right", bbox_to_anchor=(1.02, 0.5))

fig.suptitle("Ratio analysis (2×3): interpolated per surface (no forced overlap)",
             fontsize=FONT_TITLE)

grid_path = os.path.join(OUT_DIR, "ratio_grid_2x3_interpolated_no_overlap.png")
fig.savefig(grid_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", grid_path)

# -------------------------
# FIG 2: clean single ratio (scatter + interp)
# -------------------------
c1, c2 = CLEAN_RATIO
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=DPI)

total_pts = 0
for s in surfaces:
    sub = df[df["surface"] == s]
    T = sub[TARGET].to_numpy(float)
    U1 = sub[f"U_{c1}"].to_numpy(float)
    U2 = sub[f"U_{c2}"].to_numpy(float)

    m = np.isfinite(T) & np.isfinite(U1) & np.isfinite(U2)
    if U_MIN > 0:
        m = m & (np.abs(U1) > U_MIN) & (np.abs(U2) > U_MIN)

    if np.sum(m) < 2:
        continue

    total_pts += int(np.sum(m))
    R = U1[m] / U2[m]

    ax.scatter(T[m], R, s=POINT_SIZE, alpha=0.60, label=s)
    Rg = interp_on_grid(T[m], R, grid)
    ax.plot(grid, Rg, linewidth=2.2, alpha=0.90)

ax.set_title(f"Clean ratio plot: U_{c1} / U_{c2}", fontsize=FONT_TITLE)
ax.set_xlabel("Temperature (°C)", fontsize=FONT_AX)
ax.set_ylabel("Ratio", fontsize=FONT_AX)
ax.tick_params(axis="both", labelsize=FONT_TICK)
ax.grid(True, alpha=0.25)
ax.legend(title="Surface", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
          loc="center left", bbox_to_anchor=(1.02, 0.5))

ax.text(0.02, 0.98, f"Points used: {total_pts}\nU_MIN={U_MIN}",
        transform=ax.transAxes, va="top",
        fontsize=FONT_LEGEND, bbox=dict(boxstyle="round", alpha=0.12))

clean_path = os.path.join(OUT_DIR, "ratio_clean_single_no_overlap.png")
fig.savefig(clean_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", clean_path)

print("DONE ✅")