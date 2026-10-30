"""Null B at the pooled P5 anchor (Kochi, Varanasi)

Produces: results/targetability/nullB_P5_kochi_varanasi.csv

The frontier run showed both cities' *most-severe* tail (pooled P5) is more
concentrated than the UNCONDITIONAL null (Kochi p_lo path: obs 0.840 vs
0.603; Varanasi obs 0.732 vs 0.143). But that could again be peripherality:
the deep tail may all sit at one far end. This runs the DISTANCE-MATCHED
null (Null B) at P5 for these two cities to decide whether the severe core
is genuinely clustered or just peripheral — the same test that dissolved
"""
import os, time, warnings
import numpy as np
import pandas as pd
from pyproj import Transformer
import hdbscan

from mfai import paths
from mfai.io import load_city

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)                      # same seed as the main runs

# ═══════════════════════════════════════════════════════════════
#  CONFIG  (matches targetability_v2.py — T=8)
# ═══════════════════════════════════════════════════════════════
ALL_CITIES = ["patna", "varanasi", "indore", "kochi", "kolkata"]   # for pooled P5
TARGETS    = ["kochi", "varanasi"]                                 # run the null here

UTM_EPSG = {"varanasi": 32644, "kochi": 32643}

COL_LAT, COL_LON, COL_MFAI, COL_DIST_CTR = "lat", "lon", "mfai_land", "dist_centre_km"

FALLBACK_THRESHOLD = 20
DIVISION_T         = 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

MCS, MS = 20, 10
N_NULL  = 9_999
POOLED_PCT = 5                          # the anchor to test

# Null B distance bands — MUST match targetability_v2.py exactly
DIST_BAND_EDGES = [0, 5, 10, 15, 20, 30, 50, np.inf]


# ═══════════════════════════════════════════════════════════════


def to_utm(lats, lons, epsg):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(np.asarray(lons), np.asarray(lats))
    return np.column_stack([x, y])


def build_unique_locs(df, epsg):
    grp = df.groupby([COL_LAT, COL_LON])
    locs = grp.size().reset_index(name="n_enterprises")
    locs[COL_MFAI] = grp[COL_MFAI].mean().values
    locs[COL_DIST_CTR] = grp[COL_DIST_CTR].mean().values
    utm = to_utm(locs[COL_LAT].values, locs[COL_LON].values, epsg)
    return locs.reset_index(drop=True), utm


def run_hdbscan(coords):
    return hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                           metric="euclidean").fit(coords).labels_


def in_zone_pct(labels, ent):
    tot = ent.sum()
    return ent[labels >= 0].sum() / tot if tot > 0 else 0.0


def empirical_pvalues(obs, null):
    B = len(null)
    pu = (1 + np.sum(null >= obs)) / (B + 1)
    pl = (1 + np.sum(null <= obs)) / (B + 1)
    return {"p_upper": pu, "p_lower": pl, "p_two": min(2 * min(pu, pl), 1.0),
            "null_mean": null.mean(),
            "null_05": np.percentile(null, 5),
            "null_95": np.percentile(null, 95)}


def null_a(all_utm, all_ent, n_sample, n_draws=N_NULL):
    pool = np.arange(len(all_utm))
    out = np.empty(n_draws)
    for i in range(n_draws):
        idx = pool if n_sample >= len(pool) else np.random.choice(
            pool, n_sample, replace=False)
        out[i] = in_zone_pct(run_hdbscan(all_utm[idx]), all_ent[idx])
    return out


def null_b(all_utm, all_ent, all_dist, flagged_dist, n_draws=N_NULL):
    """Distance-stratified null — identical logic to targetability_v2.py."""
    all_bands = np.digitize(all_dist, DIST_BAND_EDGES) - 1
    flagged_bands = np.digitize(flagged_dist, DIST_BAND_EDGES) - 1
    n_bands = len(DIST_BAND_EDGES) - 1
    band_targets = np.bincount(flagged_bands, minlength=n_bands)
    band_pools = [np.where(all_bands == b)[0] for b in range(n_bands)]
    out = np.empty(n_draws)
    for i in range(n_draws):
        parts = []
        for b in range(n_bands):
            tgt = band_targets[b]
            if tgt == 0:
                continue
            pool = band_pools[b]
            if len(pool) == 0:
                continue
            parts.append(np.random.choice(pool, size=tgt,
                                           replace=tgt > len(pool)))
        if not parts:
            out[i] = 0.0
            continue
        idx = np.concatenate(parts)
        out[i] = in_zone_pct(run_hdbscan(all_utm[idx]), all_ent[idx])
    return out


def main():

    # ── Pooled P5 across all five cities (exact, not hardcoded) ──
    print("Computing pooled P5 across all five cities …")
    all_mfai = []
    target_data = {}
    for city in ALL_CITIES:
        df = load_city(city)
        all_mfai.append(df[COL_MFAI].to_numpy(float))
        if city in TARGETS:
            locs, utm = build_unique_locs(df, UTM_EPSG[city])
            target_data[city] = (locs, utm)
            print(f"  {city}: {len(locs)} locs, {int(locs['n_enterprises'].sum())} ent")
    pooled_p5 = np.percentile(np.concatenate(all_mfai), POOLED_PCT)
    print(f"Pooled P{POOLED_PCT} = {pooled_p5:.4f}  "
          f"(frontier reported 0.6577)\n")

    rows = []
    for city in TARGETS:
        print("─" * 50)
        print(city.upper())
        locs, utm = target_data[city]
        ent_all = locs["n_enterprises"].values
        mask = locs[COL_MFAI].values < pooled_p5
        n_locs = int(mask.sum())
        n_ent = int(ent_all[mask].sum())
        print(f"  flagged at P5: {n_ent} ent, {n_locs} locs")

        if n_locs < MCS:
            print(f"  too few locations ({n_locs} < {MCS}) — skipped")
            continue

        obs = in_zone_pct(run_hdbscan(utm[mask]), ent_all[mask])
        print(f"  observed in-zone: {obs:.3f}")

        # Null A (cross-check against frontier's P5 number)
        t0 = time.time()
        na = null_a(utm, ent_all, n_locs)
        pva = empirical_pvalues(obs, na)
        print(f"  Null A: mean={pva['null_mean']:.3f} "
              f"({pva['null_05']:.3f}–{pva['null_95']:.3f}) "
              f"p_lo={pva['p_lower']:.4f} p_hi={pva['p_upper']:.4f} "
              f"[{time.time()-t0:.0f}s]  ← should match frontier")

        # Null B (the decisive test)
        t0 = time.time()
        nb = null_b(utm, ent_all, locs[COL_DIST_CTR].values,
                    locs[COL_DIST_CTR].values[mask])
        pvb = empirical_pvalues(obs, nb)
        print(f"  Null B: mean={pvb['null_mean']:.3f} "
              f"({pvb['null_05']:.3f}–{pvb['null_95']:.3f}) "
              f"p_lo={pvb['p_lower']:.4f} p_hi={pvb['p_upper']:.4f} "
              f"[{time.time()-t0:.0f}s]  ← decisive")

        verdict = ("severe core survives distance-matching (genuine cluster)"
                   if pvb["p_upper"] < 0.05 else
                   "severe-core concentration is peripheral (dissolves under Null B)")
        print(f"  → {verdict}\n")

        rows.append({"city": city, "threshold_P5": round(pooled_p5, 4),
                     "n_flag_ent": n_ent, "n_flag_locs": n_locs,
                     "obs_in_zone": round(obs, 4),
                     "nullA_mean": round(pva["null_mean"], 4),
                     "nullA_p_upper": round(pva["p_upper"], 4),
                     "nullA_p_lower": round(pva["p_lower"], 4),
                     "nullB_mean": round(pvb["null_mean"], 4),
                     "nullB_p_upper": round(pvb["p_upper"], 4),
                     "nullB_p_lower": round(pvb["p_lower"], 4),
                     "verdict": verdict})

    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "nullB_P5_kochi_varanasi.csv"), index=False)
    print(f"Saved to {out_dir}/nullB_P5_kochi_varanasi.csv")
    return rows


if __name__ == "__main__":
    rows = main()
