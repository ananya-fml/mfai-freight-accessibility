"""Spatial-extent comparability: truncation radii

Produces: Table 6, Figure 3 inputs (printed)

The DBOX bounding boxes are described as "generous district bounds", but
they produce samples of wildly different spatial extent:
    Patna     median  6.6 km   p90 33.7 km   max 94.3 km
    Kochi     median 15.4 km   p90 34.1 km   max 67.2 km
    Indore    median  4.6 km   p90 16.6 km   max 49.3 km
    Varanasi  median  5.0 km   p90 19.8 km   max 37.4 km
"""
import numpy as np
import pandas as pd
from scipy import stats

from mfai.io import load_city

CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}
MODES = ["road", "rail", "water", "icd"]


def extent_profile(cities=None, div_t=8,
                   rings=(5, 10, 15, 20, 25, 30, 40, 50)):
    """Share of each city's firms within successive radii."""
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        v = d["dist_centre_km"]
        r = {"City": CITY_LABEL[c], "n": len(d),
             "median": round(v.median(), 1), "p90": round(v.quantile(.9), 1),
             "max": round(v.max(), 1)}
        for km in rings:
            r[f"<{km}km"] = round(100 * (v <= km).mean(), 1)
        rows.append(r)
    out = pd.DataFrame(rows)
    print("=" * 90)
    print("SPATIAL EXTENT — cumulative % of firms within each radius")
    print("=" * 90)
    print(out.to_string(index=False))
    print("""
  Columns are cumulative shares. A city that reaches 100% by 15 km has a
  municipal sample; one still below 90% at 30 km has a district sample.
  Cross-city comparison requires either a common radius or an explicit
  argument that the differing extents are themselves the object of study.""")
    return out


def common_radius_test(cities=None, div_t=8, radius=25.0, col="mfai_land"):
    """Recompute the cross-city comparison inside a common radius."""
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        full = d[col].dropna()
        sub = d[d.dist_centre_km <= radius]
        r = {"City": CITY_LABEL[c],
             "n full": len(d), "n within": len(sub),
             "% kept": round(100 * len(sub) / len(d), 1),
             "MFAI full": round(full.mean(), 3),
             f"MFAI <{radius:.0f}km": round(sub[col].mean(), 3),
             "shift": round(sub[col].mean() - full.mean(), 3)}
        for m in MODES:
            a = f"access_{m}"
            if a in d.columns and d[a].std() > 1e-9:
                r[f"a_{m}"] = round(sub[a].mean(), 3)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(f"MFAI <{radius:.0f}km")

    print("\n" + "=" * 90)
    print(f"COMMON RADIUS TEST — {radius:.0f} km from city centre")
    print("=" * 90)
    print(out.to_string(index=False))

    full_rank = out.sort_values("MFAI full")["City"].tolist()
    trunc_rank = out.sort_values(f"MFAI <{radius:.0f}km")["City"].tolist()
    print(f"\n  ranking, full sample     : {' < '.join(full_rank)}")
    print(f"  ranking, within {radius:.0f} km    : {' < '.join(trunc_rank)}")
    if full_rank == trunc_rank:
        print("  -> ranking PRESERVED. The cross-city comparison is not an")
        print("     artefact of differing sample extents.")
    else:
        print("  -> ranking CHANGES. Report the truncated version as primary")
        print("     and explain the difference.")
    return out


def radius_sensitivity(cities=None, div_t=8, col="mfai_land",
                       radii=(10, 15, 20, 25, 30, 40, 1000)):
    """Mean MFAI by city across truncation radii."""
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        r = {"City": CITY_LABEL[c]}
        for R in radii:
            sub = d[d.dist_centre_km <= R]
            lbl = "full" if R >= 1000 else f"{R}km"
            r[lbl] = round(sub[col].mean(), 3) if len(sub) > 30 else np.nan
            r[f"n {lbl}"] = len(sub)
        rows.append(r)
    out = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("RADIUS SENSITIVITY — mean MFAI by truncation radius")
    print("=" * 90)
    cols = ["City"] + [c for c in out.columns if not c.startswith("n ")]
    print(out[cols].to_string(index=False))
    print("\n  sample sizes:")
    ncols = ["City"] + [c for c in out.columns if c.startswith("n ")]
    print(out[ncols].to_string(index=False))
    print("""
  A city whose mean rises sharply as the radius tightens had its average
  dragged down by remote firms. If Patna converges toward the others
  under truncation, its apparent disadvantage is partly a sampling-frame
  effect and the claim must be restated as being about the district
  rather than the city.""")
    return out


def within_radius_correlations(cities=None, div_t=8, radius=25.0):
    """Re-run the monocentric control inside a common radius.

    Distance-to-centre correlations are inflated by long tails: a firm
    90 km out anchors the low end of both MFAI and NTL and manufactures
    correlation. Truncation tests whether the monocentric structure is
    a property of the urban area or of the tail.
    """
    cities = cities or CITY_ORDER

    def partial(df, x, y, ctrl):
        R = df[[x, y] + list(ctrl)].rank().corr().values
        inv = np.linalg.pinv(R)
        return -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])

    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        ntl = "ntl_log_clean" if "ntl_log_clean" in d.columns else "ntl_log"
        need = ["mfai_land", ntl, "log_dist_centre", "dist_centre_km"]
        if not all(k in d.columns for k in need):
            continue
        d = d.dropna(subset=need)
        s = d[d.dist_centre_km <= radius]
        if len(s) < 100:
            continue
        rows.append({
            "City": CITY_LABEL[c], "n": len(s),
            "MFAI–dist": round(stats.spearmanr(
                s.mfai_land, s.dist_centre_km).statistic, 3),
            "NTL–dist": round(stats.spearmanr(
                s[ntl], s.dist_centre_km).statistic, 3),
            "MFAI–NTL": round(stats.spearmanr(
                s.mfai_land, s[ntl]).statistic, 3),
            "MFAI–NTL | dist": round(
                partial(s, "mfai_land", ntl, ["log_dist_centre"]), 3),
            "ICD–NTL | dist": round(
                partial(s, "access_icd", ntl, ["log_dist_centre"]), 3)
                if "access_icd" in s.columns and s.access_icd.std() > 1e-9
                else np.nan,
        })
    out = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(f"MONOCENTRIC CONTROL WITHIN {radius:.0f} km")
    print("=" * 90)
    print(out.to_string(index=False))
    print("""
  Compare against the full-sample version. Correlations that weaken
  substantially were driven by the remote tail rather than by structure
  inside the urban area.""")
    return out


def run_all(radius=25.0):
    extent_profile()
    common_radius_test(radius=radius)
    radius_sensitivity()
    within_radius_correlations(radius=radius)


if __name__ == "__main__":
    run_all(radius=25)
