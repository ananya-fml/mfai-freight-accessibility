"""Reference-distance sweep, 625 combinations

Produces: Supplementary S3 (S2a-d working names) -> results/targetability/

Reviewer Major comment 3: the binding-component finding and the city
ranking both depend on the externally chosen reference distances T_m, and
that dependence must be reported in the MAIN text rather than asserted
away.
This sweeps all four anchors over a full factorial grid and asks, for
every combination:
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mfai import paths
from mfai.io import load_city

# ═══════════════════════════════════════════════════════════════
#  CONFIG — matches the pipeline (T = 8 geocoder filter)
# ═══════════════════════════════════════════════════════════════
CITIES   = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}

# Distance columns, in METRES
DIST_COL = {
    "road":  "nearest_major_road_dist_m",
    "rail":  "nearest_railway_dist_m",
    "water": "nearest_waterway_dist_m",
    "icd":   "nearest_ICD_dist_m",
}
LAND_MODES = ["road", "rail", "icd"]      # the cross-city index
ALL_MODES  = ["road", "rail", "water", "icd"]

# Indore has no locally available IWT terminal; water is n/a there and is
# excluded from its binding-component determination (Section 4.6).
NO_LOCAL_WATER = {"indore"}

FALLBACK_THRESHOLD, DIVISION_T = 20, 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

# ── Baseline (published specification) ────────────────────────
T_BASE = {"road": 2_000, "rail": 50_000, "water": 50_000, "icd": 50_000}

# ── Sweep grid, in metres ─────────────────────────────────────
T_GRID = {
    "road":  [1_000, 1_500, 2_000, 2_500, 3_000],
    "rail":  [25_000, 35_000, 50_000, 75_000, 100_000],
    "water": [25_000, 35_000, 50_000, 75_000, 100_000],
    "icd":   [25_000, 35_000, 50_000, 75_000, 100_000],
}

# Published values, for the reproduction check
PUBLISHED_MFAI = {"patna": 0.728, "indore": 0.809, "varanasi": 0.814,
                  "kochi": 0.830, "kolkata": 0.893}
PUBLISHED_BINDING = {"patna": "icd", "indore": "icd", "varanasi": "road",
                     "kochi": "icd", "kolkata": "road"}


# ═══════════════════════════════════════════════════════════════


def load_distances():
    """Return {city: {mode: distance array in metres}}."""
    out = {}
    for c in CITIES:
        df = load_city(c)
        d = {}
        for m in ALL_MODES:
            col = DIST_COL[m]
            d[m] = df[col].to_numpy(float) if col in df.columns else None
            if d[m] is None:
                print(f"  ! {c}: column {col} missing")
        out[c] = d
        print(f"  {c}: {len(df):,} enterprises")
    return out


def evaluate(dists, T):
    """One anchor combination → per-city mean access, MFAI-land, binding.

    Access is computed PER ENTERPRISE and then averaged, and MFAI-land is
    the mean of per-enterprise geometric means — not the geometric mean of
    the means, which differs by Jensen's inequality.
    """
    rows = {}
    for c in CITIES:
        d = dists[c]
        acc_mean, acc_arr = {}, {}
        for m in ALL_MODES:
            if d[m] is None:
                continue
            if c in NO_LOCAL_WATER and m == "water":
                continue                      # n/a, not a local measure
            a = T[m] / (T[m] + d[m])
            a = a[np.isfinite(a)]
            if len(a) == 0:
                continue
            acc_arr[m] = a
            acc_mean[m] = float(a.mean())

        # MFAI-land: per-enterprise geometric mean over road/rail/icd
        arrs = [T[m] / (T[m] + d[m]) for m in LAND_MODES]
        n = min(len(x) for x in arrs)
        stack = np.vstack([x[:n] for x in arrs])
        ok = np.isfinite(stack).all(axis=0)
        mfai = float(np.exp(np.log(np.clip(stack[:, ok], 1e-12, None))
                            .mean(axis=0)).mean())

        # Binding component among LOCALLY AVAILABLE modes
        order = sorted(acc_mean.items(), key=lambda kv: kv[1])
        binding, low = order[0]
        margin = order[1][1] - low if len(order) > 1 else np.nan

        rows[c] = {"mfai_land": mfai, "binding": binding,
                   "margin": margin, **{f"a_{m}": v for m, v in acc_mean.items()}}
    return rows


# ═══════════════════════════════════════════════════════════════
def run_sensitivity(verbose=True):
    print("Loading distances …")
    dists = load_distances()

    # ── Reproduction check at the published anchors ───────────
    print("\n" + "=" * 78)
    print("REPRODUCTION CHECK at baseline T (road 2 km, rail/water/ICD 50 km)")
    print("=" * 78)
    base = evaluate(dists, T_BASE)
    ok_all = True
    for c in CITIES:
        got_m, exp_m = base[c]["mfai_land"], PUBLISHED_MFAI[c]
        got_b, exp_b = base[c]["binding"], PUBLISHED_BINDING[c]
        m_ok = abs(got_m - exp_m) < 0.002
        b_ok = got_b == exp_b
        ok_all &= (m_ok and b_ok)
        print(f"  {CITY_LABEL[c]:<9} MFAI {got_m:.3f} (pub {exp_m:.3f}) "
              f"{'ok' if m_ok else 'MISMATCH':<9} "
              f"binding {got_b:<5} (pub {exp_b:<5}) "
              f"{'ok' if b_ok else 'MISMATCH'}")
    if not ok_all:
        print("\n  *** Baseline does not reproduce the published values. ***")
        print("  Check DIST_COL names and units (metres) before trusting the")
        print("  sweep — a sensitivity analysis anchored to the wrong baseline")
        print("  is worse than none.")
        return None
    print("\n  Baseline reproduces. Proceeding.\n")

    # ── Full factorial sweep ──────────────────────────────────
    keys = ["road", "rail", "water", "icd"]
    combos = list(itertools.product(*[T_GRID[k] for k in keys]))
    print("=" * 78)
    print(f"SWEEP — {len(combos)} anchor combinations")
    print("=" * 78)

    recs = []
    base_rank = [c for c, _ in sorted(
        [(c, base[c]["mfai_land"]) for c in CITIES], key=lambda kv: kv[1])]
    base_order = {c: i for i, c in enumerate(base_rank)}

    for combo in combos:
        T = dict(zip(keys, combo))
        r = evaluate(dists, T)
        means = {c: r[c]["mfai_land"] for c in CITIES}
        rank = [c for c, _ in sorted(means.items(), key=lambda kv: kv[1])]
        rho = spearmanr([base_order[c] for c in CITIES],
                        [rank.index(c) for c in CITIES]).correlation
        for c in CITIES:
            recs.append({"T_road": T["road"], "T_rail": T["rail"],
                         "T_water": T["water"], "T_icd": T["icd"],
                         "city": c, "mfai_land": r[c]["mfai_land"],
                         "binding": r[c]["binding"], "margin": r[c]["margin"],
                         "rank": rank.index(c) + 1,
                         "rank_rho_vs_base": rho,
                         "order_preserved": rank == base_rank})
    S = pd.DataFrame(recs)

    # ── 1. Binding-component stability ────────────────────────
    # crosstab rather than groupby(level=0).apply(): the latter duplicates
    # the 'city' index level on newer pandas and fails on reset_index.
    piv = (pd.crosstab(S["city"], S["binding"], normalize="index")
             .mul(100).reindex(CITIES).fillna(0).round(1))
    for m in ALL_MODES:                       # ensure every mode column exists
        if m not in piv.columns:
            piv[m] = 0.0
    piv = piv[[m for m in ALL_MODES if m in piv.columns]]
    piv.index = [CITY_LABEL[c] for c in piv.index]

    # ── 2. Margin distribution ────────────────────────────────
    marg = (S.groupby("city")["margin"]
              .agg(["min", "median", "max"]).reindex(CITIES).round(3))
    marg.index = [CITY_LABEL[c] for c in marg.index]

    # ── 3. City-mean intervals ────────────────────────────────
    span = (S.groupby("city")["mfai_land"]
              .agg(["min", "median", "max"]).reindex(CITIES).round(3))
    span["baseline"] = [round(base[c]["mfai_land"], 3) for c in CITIES]
    span["range"] = (span["max"] - span["min"]).round(3)
    span.index = [CITY_LABEL[c] for c in span.index]

    # ── 4. Ranking stability ──────────────────────────────────
    per_combo = S.drop_duplicates(
        subset=["T_road", "T_rail", "T_water", "T_icd"])
    pct_order = 100 * per_combo["order_preserved"].mean()
    rho_desc = per_combo["rank_rho_vs_base"].describe()[["min", "25%", "50%"]]
    rank_range = (S.groupby("city")["rank"].agg(["min", "max"])
                   .reindex(CITIES))
    rank_range.index = [CITY_LABEL[c] for c in rank_range.index]

    if verbose:
        print("\n1. BINDING COMPONENT — % of the anchor space in which each")
        print("   mode is the lowest-scoring component")
        print("-" * 78)
        print(piv.to_string())
        print("\n2. MARGIN to the next-lowest component, across the space")
        print("-" * 78)
        print(marg.to_string())
        print("\n3. CITY MEAN MFAI-land across the space")
        print("-" * 78)
        print(span.to_string())
        print("\n4. RANKING STABILITY")
        print("-" * 78)
        print(f"   baseline order (worst→best): "
              f"{' < '.join(CITY_LABEL[c] for c in base_rank)}")
        print(f"   exact order preserved in {pct_order:.1f}% of combinations")
        print(f"   Spearman vs baseline order — min {rho_desc['min']:.3f}, "
              f"p25 {rho_desc['25%']:.3f}, median {rho_desc['50%']:.3f}")
        print("\n   rank range per city (1 = worst served):")
        print(rank_range.to_string())

        print("\n" + "=" * 78)
        print("READING")
        print("=" * 78)
        for c in CITIES:
            row = piv.loc[CITY_LABEL[c]]
            top = row.idxmax()
            print(f"  {CITY_LABEL[c]:<9} lowest component is {top.upper():<5} "
                  f"in {row.max():.1f}% of the space "
                  f"(median margin {marg.loc[CITY_LABEL[c], 'median']:.3f})")

    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)
    S.to_csv(os.path.join(out_dir, "S2_anchor_sweep_full.csv"), index=False)
    piv.to_csv(os.path.join(out_dir, "S2a_binding_stability.csv"))
    marg.to_csv(os.path.join(out_dir, "S2b_margins.csv"))
    span.to_csv(os.path.join(out_dir, "S2c_city_mean_intervals.csv"))
    rank_range.to_csv(os.path.join(out_dir, "S2d_rank_range.csv"))
    print(f"\nSaved sweep + four summary tables to {out_dir}/")

    return {"full": S, "binding": piv, "margins": marg, "spans": span,
            "rank_range": rank_range, "pct_order_preserved": pct_order,
            "baseline": base}


if __name__ == "__main__":
    res = run_sensitivity()
