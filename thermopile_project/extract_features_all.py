import os
import glob
import csv
import numpy as np
import pyuff


# =========================
# 1) INSTÄLLNINGAR
# =========================
DATA_DIR = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV"  # <-- ÄNDRA
OUT_CSV  = os.path.join(DATA_DIR, "features.csv")

T0 = 10.0
T1 = 20.0

# Set-IDn enligt din listning (för denna typ av fil)
TC_SET_IDS = [50, 51, 52, 53, 54, 55]   # TC0..TC5
NTC_TEMP_SET_ID = 56                   # "Time for NTC TEMP C LP10Hz"

# RMS scalar-feature set-id (A,B,C,D) – enligt din listning: 8,7,6,5 är RMS A..D LP10Hz
# OBS: i din output stod:
# 5=RMS D, 6=RMS C, 7=RMS B, 8=RMS A
RMS_SET_IDS = {
    "A_rms": 8,
    "B_rms": 7,
    "C_rms": 6,
    "D_rms": 5,
}


# =========================
# 2) HJÄLPFUNKTIONER
# =========================
def read_set_dict(uff: pyuff.UFF, set_id: int) -> dict:
    """
    pyuff-versioner skiljer sig:
    - vissa har read_set
    - vissa har _read_set
    - vissa har read_sets([id])
    Vi försöker i fallande ordning.
    """
    if hasattr(uff, "read_set"):
        d = uff.read_set(set_id)
    elif hasattr(uff, "_read_set"):
        d = uff._read_set(set_id)
    elif hasattr(uff, "read_sets"):
        d = uff.read_sets([set_id])
    else:
        raise AttributeError("Din pyuff.UFF saknar read_set/_read_set/read_sets")

    # Ibland kommer list tillbaka
    if isinstance(d, list):
        d = d[0]
    if not isinstance(d, dict):
        raise TypeError(f"Förväntade dict från set {set_id}, fick {type(d)}")
    return d


def read_xy(uff: pyuff.UFF, set_id: int):
    d = read_set_dict(uff, set_id)

    x = None
    y = None

    # y-kandidater
    for k in ["data", "y", "ordinate", "ordinate_data", "ordinate_values"]:
        if k in d:
            y = d[k]
            break

    # x-kandidater
    for k in ["x", "abscissa", "abscissa_data", "abscissa_values"]:
        if k in d:
            x = d[k]
            break

    # bygg x om det går
    if x is None and ("abscissa_inc" in d) and ("num_pts" in d):
        n = int(d["num_pts"])
        dx = float(d["abscissa_inc"])
        x0 = float(d.get("abscissa_min", 0.0))
        x = x0 + dx * np.arange(n)

    if y is None:
        raise KeyError(f"Kan inte hitta y-data i set {set_id}. Keys: {list(d.keys())}")

    # Om x saknas -> antingen scalar eller index-x
    if x is None:
        y_arr = np.asarray(y)
        if y_arr.ndim == 0:
            return np.asarray([0.0]), np.asarray([float(y_arr)])
        return np.arange(len(y_arr), dtype=float), y_arr.astype(float)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x, y


def window_mean(x, y, t0, t1, relative_to_start=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # scalar-feature -> returnera direkt
    if y.ndim == 0:
        return float(y)

    if x.ndim == 0:
        x = np.arange(len(y), dtype=float)

    if relative_to_start:
        x0 = float(np.nanmin(x))
        t0 = x0 + t0
        t1 = x0 + t1

    m = (x >= t0) & (x <= t1)
    if not np.any(m):
        return float("nan")
    return float(np.mean(y[m]))



def read_scalar_feature(uff: pyuff.UFF, set_id: int) -> float:
    """
    RMS-seten verkar vara scalars i din fil.
    Vi tar mean ändå (window_mean hanterar scalar).
    """
    x, y = read_xy(uff, set_id)
    # scalar -> returnerar värdet, vektor -> tar medel
    return window_mean(x, y, T0, T1)


def extract_one_file(path: str) -> dict:
    u = pyuff.UFF(path)

    # ---- Target: mean av TC0..TC5 i fönstret
    tc_vals = []
    for sid in TC_SET_IDS:
        x, y = read_xy(u, sid)
        v = window_mean(x, y, T0, T1)
        tc_vals.append(v)

    tc_vals = np.asarray(tc_vals, dtype=float)
    if np.any(np.isnan(tc_vals)):
        raise ValueError(f"TC: Inga datapunkter i fönster {T0}-{T1}s för minst en TC-kanal.")

    y_target = float(np.mean(tc_vals))

    # ---- NTC temp mean (vektor)
    x_ntc, y_ntc = read_xy(u, NTC_TEMP_SET_ID)
    ntc_mean = window_mean(x_ntc, y_ntc, T0, T1)
    if np.isnan(ntc_mean):
        raise ValueError(f"NTC: Inga datapunkter i fönster {T0}-{T1}s.")

    # ---- RMS features (scalar)
    feats = {}
    for name, sid in RMS_SET_IDS.items():
        feats[name] = float(read_scalar_feature(u, sid))

    row = {
        "file": os.path.basename(path),
        "y_tc_mean": y_target,
        "ntc_mean": float(ntc_mean),
        **feats,
    }
    return row


# =========================
# 3) MAIN
# =========================
def main():
    print(">>> extract_features_all.py STARTAR")
    print(f">>> DATA_DIR: {DATA_DIR}")
    print(f">>> Tidsfönster: {T0}-{T1} s")

    files = []
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith((".unv", ".uff")):
            files.append(os.path.join(DATA_DIR, f))

    files = sorted(files)

    if not files:
        print("!!! Hittade inga .uff/.unv i DATA_DIR. Kolla sökvägen.")
        return

    rows = []
    failed = []

    for f in files:
        try:
            row = extract_one_file(f)
            rows.append(row)
            print(f"[OK] {row['file']}  y={row['y_tc_mean']:.3f}  ntc={row['ntc_mean']:.3f}")
        except Exception as e:
            failed.append((os.path.basename(f), str(e)))
            print(f"[FAIL] {os.path.basename(f)}  -> {e}")

    if not rows:
        print("!!! Inga filer lyckades extraheras. Avbryter utan CSV.")
        return

    # skriv CSV
    fieldnames = ["file", "y_tc_mean", "ntc_mean"] + list(RMS_SET_IDS.keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n==============================")
    print(f"KLAR ✅ Sparade {len(rows)} rader till: {OUT_CSV}")
    if failed:
        print(f"VARNING: {len(failed)} filer failade:")
        for name, err in failed[:10]:
            print(f" - {name}: {err}")
        if len(failed) > 10:
            print(" - ...")
    print("==============================\n")


if __name__ == "__main__":
    main()
