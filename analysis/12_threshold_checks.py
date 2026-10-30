"""City-balanced threshold and null multiplicity checks

Produces: results/targetability/check_*.csv

Both concerns raised in review can be settled without rerunning the null
models. Neither requires HDBSCAN more than once per city.
CHECK 1 — CITY-BALANCED POOLED THRESHOLD
Indore contributes 48,884 of 123,591 enterprises (39.6%), so it has
disproportionate influence over where the enterprise-weighted pooled tenth
percentile falls. This recomputes the threshold giving each city equal
"""
import os
import numpy as np
import pandas as pd
import hdbscan
from pyproj import Transformer

from mfai import paths
from mfai.io import load_city

np.random.seed(42)

CITIES = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {c: c.title() for c in CITIES}
UTM_EPSG = {"patna": 32645, "kolkata": 32645, "varanasi": 32644,
            "indore": 32643, "kochi": 32643}

COL_LAT, COL_LON, COL_MFAI = "lat", "lon", "mfai_land"
FALLBACK_THRESHOLD, DIVISION_T = 20, 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"
MCS, MS = 20, 10
PCT = 10


def to_utm(lats, lons, epsg):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(np.asarray(lons), np.asarray(lats))
    return np.column_stack([x, y])


def build_unique_locs(df, epsg):
    g = df.groupby([COL_LAT, COL_LON])
    locs = g.size().reset_index(name="n_ent")
    locs[COL_MFAI] = g[COL_MFAI].mean().values
    return locs.reset_index(drop=True), to_utm(
        locs[COL_LAT].values, locs[COL_LON].values, epsg)


def weighted_quantile(values, q, weights):
    """Weighted quantile via the cumulative weight distribution."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))


def run_checks():
    print("Loading …")
    ent, locs, utm = {}, {}, {}
    for c in CITIES:
        df = load_city(c)
        ent[c] = df[COL_MFAI].to_numpy(float)
        locs[c], utm[c] = build_unique_locs(df, UTM_EPSG[c])
        print(f"  {c}: {len(df):,} ent, {len(locs[c]):,} locs")

    # ── CHECK 1 ───────────────────────────────────────────────
    pooled = np.concatenate([ent[c] for c in CITIES])
    thr_ew = float(np.percentile(pooled, PCT))

    # city-balanced: every city gets equal total weight
    w = np.concatenate([np.full(len(ent[c]), 1.0 / len(ent[c])) for c in CITIES])
    thr_cb = weighted_quantile(pooled, PCT / 100.0, w)

    print("\n" + "=" * 76)
    print("CHECK 1 — POOLED THRESHOLD: enterprise-weighted vs city-balanced")
    print("=" * 76)
    print(f"  sample shares: " + ", ".join(
        f"{CITY_LABEL[c]} {100*len(ent[c])/len(pooled):.1f}%" for c in CITIES))
    print(f"\n  enterprise-weighted P{PCT}: {thr_ew:.4f}")
    print(f"  city-balanced      P{PCT}: {thr_cb:.4f}")
    print(f"  difference            : {thr_cb - thr_ew:+.4f}")

    rows = []
    for c in CITIES:
        a = ent[c]
        rows.append({
            "City": CITY_LABEL[c],
            "n": len(a),
            "flagged % (ent-wtd)": round(100 * (a < thr_ew).mean(), 1),
            "flagged % (city-bal)": round(100 * (a < thr_cb).mean(), 1),
            "change (pp)": round(100 * ((a < thr_cb).mean()
                                        - (a < thr_ew).mean()), 1),
        })
    t1 = pd.DataFrame(rows)
    print()
    print(t1.to_string(index=False))
    same_order = (t1["flagged % (ent-wtd)"].rank().tolist()
                  == t1["flagged % (city-bal)"].rank().tolist())
    print(f"\n  ordering of flagged shares preserved: {same_order}")
    print("  → If the threshold shift is small and the ordering holds, the")
    print("    enterprise-weighted pooled rule needs no rerun; report this")
    print("    table as the sensitivity.")

    # ── CHECK 2 ───────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("CHECK 2 — MULTIPLICITY: enterprises per location, and the")
    print("          enterprise- vs location-weighted in-zone statistic")
    print("=" * 76)
    rows = []
    for c in CITIES:
        L, U = locs[c], utm[c]
        m = L[COL_MFAI].to_numpy(float) < thr_ew
        fl, fu = L[m].reset_index(drop=True), U[m]
        mult_fl = fl["n_ent"].mean() if len(fl) else np.nan
        mult_all = L["n_ent"].mean()

        if len(fu) >= MCS:
            lab = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                                  metric="euclidean").fit(fu).labels_
            e = fl["n_ent"].to_numpy(float)
            izp_ent = e[lab >= 0].sum() / e.sum()
            izp_loc = (lab >= 0).mean()
        else:
            izp_ent = izp_loc = np.nan
        rows.append({
            "City": CITY_LABEL[c],
            "ent/loc flagged": round(float(mult_fl), 2),
            "ent/loc all": round(float(mult_all), 2),
            "ratio": round(float(mult_fl / mult_all), 2),
            "in-zone % (ent-wtd)": round(100 * izp_ent, 1),
            "in-zone % (loc-wtd)": round(100 * izp_loc, 1),
            "gap (pp)": round(100 * (izp_ent - izp_loc), 1),
        })
    t2 = pd.DataFrame(rows)
    print()
    print(t2.to_string(index=False))
    print("""
  'ratio' near 1 means flagged locations carry about as many enterprises
  each as the city average, so drawing random locations does not import a
  different multiplicity regime. A small 'gap' means the reported
  enterprise-weighted statistic is close to the pure location-level one,
  so spatial concentration rather than coordinate multiplicity is driving
  it. Large values in either column would mean the null needs rerunning at
  the location level.""")

    out = str(paths.results("targetability"))
    os.makedirs(out, exist_ok=True)
    t1.to_csv(f"{out}/check_city_balanced_threshold.csv", index=False)
    t2.to_csv(f"{out}/check_null_multiplicity.csv", index=False)
    print(f"\nSaved to {out}/")
    return {"threshold": t1, "multiplicity": t2,
            "thr_ew": thr_ew, "thr_cb": thr_cb}


if __name__ == "__main__":
    diag = run_checks()
