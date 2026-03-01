import pyuff
import numpy as np

# ändra bara denna sökväg om det behövs
UNV_PATH = r"C:\Users\SFDDWI\Desktop\DanteTermoPileUNV\J1-T100-I1 Tp 4_Processed.unv"

uff = pyuff.UFF(UNV_PATH)

n_sets = uff.get_n_sets()
print("Number of sets:", n_sets)

for i in range(1, n_sets):
    d = uff.read_sets(i, header_only=True)
    if isinstance(d, list):
        d = d[0]

    print(
        f"{i:2d} | "
        f"type={d.get('type')} | "
        f"units={d.get('ordinate_axis_units_lab')} | "
        f"id1={d.get('id1')}"
    )
