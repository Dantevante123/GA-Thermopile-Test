import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_IN = os.path.join(os.path.dirname(__file__), "features_with_U_and_ratios.csv")

RATIOS_TO_PLOT = ["R_AB", "R_AD", "R_BD"]   # börja med de stabilare
TEMP_COL = "y_tc_mean"                     # target temp (°C)
U_THRESHOLD = 0.05                         # 50 mV
SURF_VALUES = ["J1", "J2", "J3", "J4", "J5", "J6"]

def extract_surface_from_filename(s: str):
    """
    Förväntar: filnamn som börjar med J1..J6, t.ex. 'J3-T115-I3...'
    Funkar även om det är en sökväg: 'C:\\...\\J3-T115-I3.unv'
    """
    if s is None:
        return None
    s = str(s)

    # plocka bort path så vi bara har filnamnet
    base = os.path.basename(s)

    # matcha i början: J1..J6
    m = re.match(r"^(J[1-6])", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # fallback: leta var som helst i strängen
    m2 = re.search(r"(J[1-6])", base, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    return None

def filtered_ratio_df(df: pd.DataFrame, ratio_col: str):
    """
    Filtrera bort punkter där U-nämnare/täljare är för små,
    samt NaN/inf i ratio.
    """
    pair = ratio_col.split("_")[1]  # "AB"
    a, b = pair[0], pair[1]

    U_num = df[f"U_{a}"].astype(float).to_numpy()
    U_den = df[f"U_{b}"].astype(float).to_numpy()
    R = df[ratio_col].astype(float).to_numpy()

    mask = (
        np.isfinite(R) &
        np.isfinite(U_num) &
        np.isfinite(U_den) &
        (np.abs(U_num) > U_THRESHOLD) &
        (np.abs(U_den) > U_THRESHOLD)
    )
    return df.loc[mask].copy()

def plot_ratio(df: pd.DataFrame, ratio_col: str):
    dff = filtered_ratio_df(df, ratio_col)

    removed = len(df) - len(dff)
    print(f"\n{ratio_col}: kept {len(dff)}/{len(df)} (removed {removed}) with |U| > {U_THRESHOLD} V")

    plt.figure()

    for surf in SURF_VALUES:
        sub = dff[dff["surface"] == surf]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub[TEMP_COL].to_numpy(float),
            sub[ratio_col].to_numpy(float),
            s=18,
            alpha=0.85,
            label=surf
        )

    plt.xlabel("Target temperature (°C)")
    plt.ylabel(f"{ratio_col} (filtered)")
    plt.title(f"{ratio_col} vs Temperature (colored by surface)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.show()

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(
            f"Hittar inte {CSV_IN}\n"
            "Kör först make_U_and_ratios.py så att features_with_U_and_ratios.csv skapas."
        )

    df = pd.read_csv(CSV_IN)

    if "file" not in df.columns:
        raise KeyError("CSV saknar kolumnen 'file' som behövs för att extrahera yta (J1..J6).")

    # skapa surface-kolumnen
    df["surface"] = df["file"].apply(extract_surface_from_filename)

    # kolla att vi faktiskt hittade ytor
    hits = df["surface"].value_counts(dropna=False)
    print("\nSurface extraction counts:")
    print(hits)

    # behåll bara giltiga ytor
    df = df[df["surface"].isin(SURF_VALUES)].copy()

    # basic clean
    need_cols = [TEMP_COL, "surface", "U_A", "U_B", "U_C", "U_D"] + RATIOS_TO_PLOT
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    df = df.dropna(subset=[TEMP_COL, "surface"]).copy()

    # plot ratios
    for r in RATIOS_TO_PLOT:
        plot_ratio(df, r)

if __name__ == "__main__":
    main()
