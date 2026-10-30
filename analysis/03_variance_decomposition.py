"""Variance decomposition of MFAI-land: nested sums of squares

Produces: Table 10 check -> results/targetability/variance_decomposition_check.csv

The manuscript reports three components of enterprise-level MFAI-land
variance: between cities 24.3%, between industrial groups within city
1.0%, within group within city 74.7%. Reviewer minor comment 11 asks for
the estimator behind them.
This recomputes the decomposition as a NESTED SUMS-OF-SQUARES partition,
which is the estimator that yields three shares summing exactly to 100%:
"""
import os
import numpy as np
import pandas as pd

from mfai import paths
from mfai.io import load_city

# ═══════════════════════════════════════════════════════════════
CITIES   = ["patna", "varanasi", "indore", "kochi", "kolkata"]

COL_MFAI  = "mfai_land"
COL_GROUP = "mfg_group"          # 8 manufacturing groups, G1–G8

FALLBACK_THRESHOLD = 20
DIVISION_T         = 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

# Manuscript values to check against
REPORTED = {"between_cities": 24.3, "between_groups": 1.0, "within": 74.7}
REPORTED_RATIO_RANGE = (0.976, 1.033)


def main():

    frames = []
    for city in CITIES:
        df = load_city(city)
        if COL_GROUP not in df.columns:
            print(f"*** {city}: column '{COL_GROUP}' not found. "
                  f"Available group-like columns: "
                  f"{[c for c in df.columns if 'group' in c or 'sector' in c or 'isic' in c]}")
            return
        sub = df[[COL_MFAI, COL_GROUP]].copy()
        sub["city"] = city
        frames.append(sub)
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d[COL_MFAI])]
    print(f"n = {len(d):,} enterprises, "
          f"{d['city'].nunique()} cities, "
          f"{d[COL_GROUP].nunique()} groups, "
          f"{d.groupby(['city', COL_GROUP]).ngroups} city–group cells\n")

    y = d[COL_MFAI].to_numpy(float)
    grand = y.mean()

    # ── SST ───────────────────────────────────────────────────
    SST = ((y - grand) ** 2).sum()

    # ── SS between cities ─────────────────────────────────────
    city_stats = d.groupby("city")[COL_MFAI].agg(["mean", "size"])
    SS_city = (city_stats["size"] * (city_stats["mean"] - grand) ** 2).sum()

    # ── SS between groups within city ─────────────────────────
    cg = d.groupby(["city", COL_GROUP])[COL_MFAI].agg(["mean", "size"])
    cg = cg.join(city_stats["mean"].rename("city_mean"), on="city")
    SS_group = (cg["size"] * (cg["mean"] - cg["city_mean"]) ** 2).sum()

    # ── SS within ─────────────────────────────────────────────
    d = d.join(cg["mean"].rename("cell_mean"), on=["city", COL_GROUP])
    SS_within = ((d[COL_MFAI] - d["cell_mean"]) ** 2).sum()

    # ── Identity check ────────────────────────────────────────
    total = SS_city + SS_group + SS_within
    print(f"SST                 = {SST:.6f}")
    print(f"SS_city + SS_group|city + SS_within = {total:.6f}")
    print(f"identity holds: {np.isclose(SST, total, rtol=1e-9)}\n")

    shares = {
        "between_cities": 100 * SS_city / SST,
        "between_groups": 100 * SS_group / SST,
        "within":         100 * SS_within / SST,
    }

    print("Nested sums-of-squares decomposition")
    print("─" * 52)
    for k, label in [("between_cities", "Between cities"),
                     ("between_groups", "Between groups, within city"),
                     ("within",         "Within group, within city")]:
        got, exp = shares[k], REPORTED[k]
        flag = "MATCH" if abs(got - exp) < 0.15 else "*** MISMATCH ***"
        print(f"  {label:<32} {got:6.2f}%   (reported {exp:.1f}%)  {flag}")
    print(f"  {'sum':<32} {sum(shares.values()):6.2f}%")

    if all(abs(shares[k] - REPORTED[k]) < 0.15 for k in REPORTED):
        print("\n→ Estimator CONFIRMED as nested sums-of-squares. "
              "State it in the methods as written.")
    else:
        print("\n→ MISMATCH. The reported figures came from a different "
              "estimator (likely REML variance components from a nested "
              "random-effects model). Identify which before writing the "
              "methods sentence — do not report this partition as if it "
              "were the source of the published numbers.")

    # ── Independent check: group-mean ratios ──────────────────
    cg2 = cg.reset_index()
    cg2["ratio"] = cg2["mean"] / cg2["city_mean"]
    lo, hi = cg2["ratio"].min(), cg2["ratio"].max()
    print(f"\nGroup-mean / city-mean ratio across {len(cg2)} cells: "
          f"{lo:.3f}–{hi:.3f}  (manuscript: "
          f"{REPORTED_RATIO_RANGE[0]}–{REPORTED_RATIO_RANGE[1]})")

    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([shares]).to_csv(
        os.path.join(out_dir, "variance_decomposition_check.csv"), index=False)
    cg2.to_csv(os.path.join(out_dir, "group_city_means.csv"), index=False)
    print(f"\nSaved to {out_dir}/")
    return shares, cg2


if __name__ == "__main__":
    shares, cg2 = main()
