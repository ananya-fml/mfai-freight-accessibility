"""Supplementary S2: HDBSCAN parameter sweep with the Null B outcome.

Produces: results/supplementary/S2_hdbscan_sweep.csv

Does the targetability conclusion survive the parameter choice? Reports
DBCV and the distance-band-matched null (Null B) for every
(min_cluster_size, min_samples) pair. Slow: 15 settings x 999 draws per city.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import hdbscan

from mfai import paths
from mfai.config import CENTRES, CITY_ORDER as ORDER
from mfai.io import load_city_gdf

COL_MFAI = "mfai_land"
POOLED_THR = 0.7065
MCS_RANGE  = [10, 20, 30, 40, 50]
MS_RANGE   = [5, 10, 15]
N_NULL     = 999          # reduce to 199 for a first pass
BAND_KM    = 5


def flagged_locations(gdf, city):
    """Unique coordinates of enterprises below the pooled threshold,
       with enterprise counts and distance from centre."""
    lat, lon = CENTRES[city]
    cen = (gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326)
           .to_crs(gdf.crs).iloc[0])
    g = gdf.copy()
    g["x"] = g.geometry.x.round(1); g["y"] = g.geometry.y.round(1)
    flag = g[g[COL_MFAI] < POOLED_THR]
    locs = (flag.groupby(["x", "y"])
                .size().rename("n_ent").reset_index())
    allv = (g.groupby(["x", "y"]).size().rename("n_ent").reset_index())
    for d in (locs, allv):
        d["dist_km"] = np.hypot(d.x - cen.x, d.y - cen.y) / 1000.0
    return locs, allv


def in_zone_share(coords, weights, mcs, ms):
    if len(coords) < mcs:
        return np.nan, np.nan, 0
    cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms),
                         gen_min_span_tree=True).fit(coords)
    lab = cl.labels_
    share = 100 * weights[lab >= 0].sum() / weights.sum()
    try:
        dbcv = float(cl.relative_validity_)
    except Exception:
        dbcv = np.nan
    return share, dbcv, int(lab.max() + 1)


def main():
    city_data = {c: load_city_gdf(c) for c in ORDER}
    rows = []
    for c in ORDER:
        locs, allv = flagged_locations(city_data[c], c)
        coords = locs[["x", "y"]].values
        w = locs["n_ent"].values
        bands_f = (locs["dist_km"] // BAND_KM).astype(int)
        bands_a = (allv["dist_km"] // BAND_KM).astype(int)
        prof = bands_f.value_counts()
        rng = np.random.default_rng(0)

        for mcs in MCS_RANGE:
            for ms in MS_RANGE:
                obs, dbcv, nz = in_zone_share(coords, w, mcs, ms)
                if np.isnan(obs):
                    rows.append({"City": c.capitalize(), "mcs": mcs, "ms": ms,
                                 "Zones": 0, "In-zone %": np.nan,
                                 "DBCV": np.nan, "Null B mean": np.nan,
                                 "Null B p_lo": np.nan})
                    continue
                # Null B: distance-band matched draws from all locations
                null = np.empty(N_NULL)
                for k in range(N_NULL):
                    picks = []
                    for b, cnt in prof.items():
                        pool = allv.index[bands_a == b].values
                        if len(pool) == 0: continue
                        picks.append(rng.choice(pool, size=min(cnt, len(pool)),
                                                replace=False))
                    idx = np.concatenate(picks) if picks else np.array([], int)
                    sub = allv.loc[idx]
                    s, _, _ = in_zone_share(sub[["x", "y"]].values,
                                            sub["n_ent"].values, mcs, ms)
                    null[k] = s
                p_lo = (1 + np.nansum(null <= obs)) / (N_NULL + 1)
                rows.append({"City": c.capitalize(), "mcs": mcs, "ms": ms,
                             "Zones": nz, "In-zone %": round(obs, 1),
                             "DBCV": round(dbcv, 3) if dbcv == dbcv else np.nan,
                             "Null B mean": round(float(np.nanmean(null)), 1),
                             "Null B p_lo": round(p_lo, 4)})
        print(f"  {c} done")

    s2 = pd.DataFrame(rows)
    print(s2.to_string(index=False))

    print("\nConclusion stability — does the Null B reading flip across settings?")
    for c in ORDER:
        sub = s2[(s2.City == c.capitalize()) & s2["Null B p_lo"].notna()]
        sig = (sub["Null B p_lo"] < 0.05).sum()
        print(f"  {c:9s} {sig}/{len(sub)} settings significant "
              f"(p_lo range {sub['Null B p_lo'].min():.4f}-{sub['Null B p_lo'].max():.4f})")
    out = paths.results("supplementary") / "S2_hdbscan_sweep.csv"
    s2.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return s2


if __name__ == "__main__":
    main()
