"""Supplementary S1: coordinate-artefact threshold sensitivity.

Produces: results/supplementary/S1_coordinate_threshold.csv

Claim under test (Sections 3.2-3.3): index means and P10 are stable to three
decimal places across artefact thresholds T = 4..12. "Excluded %" is
relative to the rule-1-only sample (T = 99, where rule 2 never fires).
"""
import pandas as pd

from mfai import paths
from mfai.config import CITY_ORDER as ORDER
from mfai.io import load_city

COL_MFAI = "mfai_land"


def main():
    rows = []
    for c in ORDER:
        n_rule1_only = len(load_city(c, div_t=99))
        for T in range(4, 13):
            g = load_city(c, div_t=T)
            rows.append({
                "City": c.capitalize(), "T": T,
                "n": len(g),
                "Excluded %": round(100 * (1 - len(g) / n_rule1_only), 1),
                "Mean": round(g[COL_MFAI].mean(), 4),
                "P10": round(g[COL_MFAI].quantile(0.10), 4),
                "Median": round(g[COL_MFAI].median(), 4),
            })
    s1 = pd.DataFrame(rows)
    print(s1.to_string(index=False))

    print("\nRange across T = 4..12, per city:")
    for c in ORDER:
        sub = s1[s1.City == c.capitalize()]
        print(f"  {c:9s} mean {sub.Mean.min():.4f}-{sub.Mean.max():.4f} "
              f"(spread {sub.Mean.max()-sub.Mean.min():.4f}) | "
              f"P10 {sub.P10.min():.4f}-{sub.P10.max():.4f} "
              f"(spread {sub.P10.max()-sub.P10.min():.4f}) | "
              f"excluded {sub['Excluded %'].min():.1f}-"
              f"{sub['Excluded %'].max():.1f}%")
    out = paths.results("supplementary") / "S1_coordinate_threshold.csv"
    s1.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return s1


if __name__ == "__main__":
    main()
