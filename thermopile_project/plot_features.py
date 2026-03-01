import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# Ändra om din CSV ligger någon annanstans
CSV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\features.csv"


def add_meta_from_filename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Försöker plocka ut J, T och I från filnamn som t.ex:
    'J6-T95-I3 Tp 4_Processed.unv'
    """
    def parse_one(name: str):
        m = re.search(r"(J\d+)-T(\d+)-I(\d+)", str(name))
        if not m:
            return None, None, None
        return m.group(1), int(m.group(2)), int(m.group(3))

    J, T, I = [], [], []
    for fn in df["file"]:
        j, t, i = parse_one(fn)
        J.append(j)
        T.append(t)
        I.append(i)

    df["J"] = J
    df["T_set"] = T
    df["I_rep"] = I
    return df


def scatter_by_group(ax, df, xcol, ycol, groupcol="J", title=None):
    groups = [g for g in sorted(df[groupcol].dropna().unique())]
    for g in groups:
        dfg = df[df[groupcol] == g]
        ax.scatter(dfg[xcol], dfg[ycol], label=g, s=18, alpha=0.85)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    ax.legend(title=groupcol, fontsize=8)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Hittar inte CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "file" in df.columns:
        df = add_meta_from_filename(df)

    # Säkerställ att kolumnerna finns
    needed = ["y_tc_mean", "ntc_mean", "A_rms", "B_rms", "C_rms", "D_rms"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Saknar kolumner i CSV: {missing}\nFinns: {list(df.columns)}")

    # Extra “bra” x-variabel att titta på
    df["dT"] = df["y_tc_mean"] - df["ntc_mean"]

    # 1) RMS vs target-temp
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    scatter_by_group(axs[0, 0], df, "y_tc_mean", "A_rms", title="A_rms vs y_tc_mean")
    scatter_by_group(axs[0, 1], df, "y_tc_mean", "B_rms", title="B_rms vs y_tc_mean")
    scatter_by_group(axs[1, 0], df, "y_tc_mean", "C_rms", title="C_rms vs y_tc_mean")
    scatter_by_group(axs[1, 1], df, "y_tc_mean", "D_rms", title="D_rms vs y_tc_mean")
    plt.show()

    # 2) RMS vs dT = (target - ntc)  (ofta tydligare)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    scatter_by_group(axs[0, 0], df, "dT", "A_rms", title="A_rms vs dT")
    scatter_by_group(axs[0, 1], df, "dT", "B_rms", title="B_rms vs dT")
    scatter_by_group(axs[1, 0], df, "dT", "C_rms", title="C_rms vs dT")
    scatter_by_group(axs[1, 1], df, "dT", "D_rms", title="D_rms vs dT")
    plt.show()

    # 3) (valfritt) kolla hur många per J
    if "J" in df.columns:
        print(df["J"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
