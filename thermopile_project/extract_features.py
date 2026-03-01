import numpy as np
import pyuff
import os

# ==============================
# ÄNDRA BARA DENNA RAD VID BEHOV
# ==============================
UFF_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\J1-T100-I1 Tp 4_Processed.unv"

# ==============================
# TIDSFÖNSTER (sekunder)
# ==============================
T0 = 10.0
T1 = 20.0

# ==============================
# SET-ID (från inspect_unv.py)
# ==============================
TC_SETS = [50, 51, 52, 53, 54, 55]   # TC_5 ... TC_0
NTC_TEMP_SET = 56                   # Time for NTC TEMP C
RMS_SETS = {
    "A_rms": 8,
    "B_rms": 7,
    "C_rms": 6,
    "D_rms": 5,
}

# ==============================
# ROBUST UFF-LÄSNING
# ==============================
def read_xy(uff, set_id):
    # pyuff-versioner skiljer sig – detta funkar för din
    if hasattr(uff, "read_sets"):
        d = uff.read_sets(set_id)
    elif hasattr(uff, "_read_set"):
        d = uff._read_set(set_id)
    else:
        raise RuntimeError("Kan inte hitta read_sets i pyuff")

    if isinstance(d, list):
        d = d[0]

    # y-data (ordinate)
    for k in ["data", "y"]:
        if k in d:
            y = np.asarray(d[k], dtype=float)
            break
    else:
        raise KeyError(f"Inget y-data i set {set_id}")

    # x-data (abscissa / tid)
    if "x" in d:
        x = np.asarray(d["x"], dtype=float)
    elif "abscissa_inc" in d and "num_pts" in d:
        n = int(d["num_pts"])
        dx = float(d["abscissa_inc"])
        x0 = float(d.get("abscissa_min", 0.0))
        x = x0 + dx * np.arange(n)
    else:
        # fallback: index-vektor
        x = np.arange(len(y), dtype=float)

    return x, y


def window_mean(x, y, t0, t1):
    x = np.asarray(x)
    y = np.asarray(y)

    # Om y är scalar (0D) -> returnera direkt
    if y.ndim == 0:
        return float(y)

    # 1) Försök använda x som tid (sekunder)
    m = (x >= t0) & (x <= t1)
    if np.any(m):
        return float(np.mean(y[m]))

    # 2) Fallback: använd indexbaserat fönster
    # Antag ca 60 s inspelning: 10–20 s ≈ 1/6–1/3 av signalen
    n = len(y)
    i0 = int(round((t0 / 60.0) * n))
    i1 = int(round((t1 / 60.0) * n))

    i0 = max(0, min(n - 1, i0))
    i1 = max(i0 + 1, min(n, i1))

    return float(np.mean(y[i0:i1]))



# ==============================
# MAIN
# ==============================
def main():
    print(">>> extract_features.py STARTAR")

    if not os.path.exists(UFF_PATH):
        raise FileNotFoundError(UFF_PATH)

    uff = pyuff.UFF(UFF_PATH)

    print(">>> Läser TC-kanaler (target)")

    tc_vals = []
    for sid in TC_SETS:
        x, y = read_xy(uff, sid)
        val = window_mean(x, y, T0, T1)
        tc_vals.append(val)
        print(f"TC set {sid}: mean = {val:.3f} °C")

    y_target = float(np.mean(tc_vals))

    print("\n>>> Läser RMS-kanaler (features)")
    features = {}

    for name, sid in RMS_SETS.items():
        x, y = read_xy(uff, sid)
        val = window_mean(x, y, T0, T1)
        features[name] = val
        print(f"{name}: {val:.6g} V")

    # Extra feature: NTC temp
    x, y = read_xy(uff, NTC_TEMP_SET)
    ntc_mean = window_mean(x, y, T0, T1)

    print("\n==============================")
    print(f"TARGET y (mean TC0..TC5): {y_target:.3f} °C")
    print(f"NTC temp mean: {ntc_mean:.3f} °C")
    print("Features:", features)
    print("==============================\n")


if __name__ == "__main__":
    main()
