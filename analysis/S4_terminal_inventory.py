"""Supplementary S4: terminal inventory with operational status.

Produces: results/supplementary/S4_terminal_inventory.csv
"""
import os

import geopandas as gpd
import pandas as pd

from mfai import paths
from mfai.config import CITY_ORDER as ORDER

TERM_CLASS = {"Rail goods shed": "rail", "IWT terminal": "water",
              "ICD / CFS": "icd"}


def main():
    rows = []
    for c in ORDER:
        for cls, kind in TERM_CLASS.items():
            p = f"{paths.terminals(kind, c)}.gpkg"
            if not os.path.exists(p):
                continue
            t = gpd.read_file(p)
            if t.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                t = t.copy()
                t["geometry"] = t.geometry.centroid
            t = t.to_crs(4326)
            for _, r in t.iterrows():
                rows.append({
                    "City": c.capitalize(), "Class": cls,
                    "Name": r.get("name", r.get("Name", "")),
                    "Lat": round(r.geometry.y, 5),
                    "Lon": round(r.geometry.x, 5),
                    "Status": r.get("status", r.get("Status", "")),
                    "Source": r.get("source", r.get("Source", "")),
                })

    s4 = pd.DataFrame(rows).sort_values(["City", "Class", "Name"])
    print(f"\n{len(s4)} terminals\n", s4.to_string(index=False))
    print("\nCounts by city and class - check against Table 1:")
    print(pd.crosstab(s4.City, s4.Class).to_string())
    out = paths.results("supplementary") / "S4_terminal_inventory.csv"
    s4.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return s4


if __name__ == "__main__":
    main()
