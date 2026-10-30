"""Targetability: HDBSCAN low-proximity zones with Null A and Null B

Produces: Table 13 (null models), zone tables -> results/targetability/

Revised HDBSCAN with:
  • Primary threshold: pooled bottom decile across all five cities
  • Sensitivity threshold: city-specific bottom decile (demoted)
  • 9,999 null simulations with proper empirical tail probabilities
  • Null A  — unconditional random subset
  • Null B  — distance-from-centre-stratified random subset
"""
import os, time, warnings
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import ConvexHull
import hdbscan

from mfai import paths
from mfai.io import load_city

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION — VERIFY AGAINST YOUR CSVs
# ═══════════════════════════════════════════════════════════════

CITIES = ["patna", "varanasi", "indore", "kochi", "kolkata"]

UTM_EPSG = {
    "patna": 32645, "kolkata": 32645,
    "varanasi": 32644,
    "indore": 32643, "kochi": 32643,
}

# ── Column names (adjust if yours differ) ─────────────────────
COL_LAT        = "lat"
COL_LON        = "lon"
COL_MFAI       = "mfai_land"
COL_DIST_CTR   = "dist_centre_km"
# Modal access scores — used for zone-binding-mode analysis
COL_ACCESS     = {
    "road":  "access_road",
    "rail":  "access_rail",
    "water": "access_water",
    "icd":   "access_icd",
}
# ── Geocoder-artefact exclusion ───────────────────────────────
# Consumes the flags written by flag_geocoder_fallbacks() (Stage 0.5).
# Does NOT recompute — matches the pipeline exactly by construction.
# Exclusion rule:  fallback_rule1 | (coord_n >= THRESHOLD & coord_n_div >= T)
FALLBACK_THRESHOLD = 20      # coord_n cutoff (must match pipeline)
DIVISION_T         = 8       # *** SET TO YOUR CANONICAL T (the one giving
                             #     Patna 12,807 etc.) — sweep {4,6,8,10,12}
                             #     for the reviewer's threshold sensitivity ***
COL_RULE1          = "fallback_rule1"
COL_COORD_N        = "coord_n"
COL_COORD_NDIV     = "coord_n_div"

# ── HDBSCAN parameters (fixed for cross-city comparability) ───
MCS = 20   # min_cluster_size
MS  = 10   # min_samples

# ── Null model ────────────────────────────────────────────────
N_NULL = 9_999

# ── Null B distance-stratification bands (km) ─────────────────
DIST_BAND_EDGES = [0, 5, 10, 15, 20, 30, 50, np.inf]

# ── Zone-area sensitivity thresholds (km²) ────────────────────
AREA_THRESHOLDS = [5, 10, 25, 50, 100]

# ── Minimum flagged unique locations to attempt clustering ────
MIN_FLAGGED_LOCS = 30  # need > MCS to have any chance of a zone


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def verify_columns():
    """Print columns from the first city CSV so you can check names."""
    city = CITIES[0]
    df = pd.read_csv(paths.nearest_csv(city), nrows=3)
    print(f"Columns in {city} CSV ({len(df.columns)}):")
    for c in sorted(df.columns):
        print(f"  {c}")
    return df.columns.tolist()


def to_utm(lats, lons, epsg):
    """Convert lat/lon arrays to UTM easting/northing."""
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(np.asarray(lons), np.asarray(lats))
    return np.column_stack([x, y])


def build_unique_locs(df, epsg):
    """
    Build a unique-locations DataFrame with enterprise counts and
    mean access scores per coordinate.  Returns (df_locs, utm_coords).
    """
    grp = df.groupby([COL_LAT, COL_LON])
    locs = grp.size().reset_index(name="n_enterprises")

    # Mean modal access at each coordinate (for zone-binding analysis)
    for mode, col in COL_ACCESS.items():
        if col in df.columns:
            locs[f"a_{mode}"] = grp[col].mean().values

    # Mean MFAI and dist-from-centre
    locs[COL_MFAI] = grp[COL_MFAI].mean().values
    if COL_DIST_CTR in df.columns:
        locs[COL_DIST_CTR] = grp[COL_DIST_CTR].mean().values

    utm = to_utm(locs[COL_LAT].values, locs[COL_LON].values, epsg)
    return locs.reset_index(drop=True), utm


# ═══════════════════════════════════════════════════════════════
#  HDBSCAN + ZONE METRICS
# ═══════════════════════════════════════════════════════════════

def run_hdbscan(coords_utm, mcs=MCS, ms=MS):
    """Run HDBSCAN, return integer labels (-1 = noise)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
    )
    clusterer.fit(coords_utm)
    return clusterer.labels_


def in_zone_pct(labels, enterprise_counts):
    """Enterprise-weighted share of points assigned to any cluster."""
    in_zone = enterprise_counts[labels >= 0].sum()
    total = enterprise_counts.sum()
    return in_zone / total if total > 0 else 0.0


def zone_areas_km2(coords_utm, labels):
    """
    Convex-hull area (km²) per cluster.
    Clusters with < 3 unique points get area = 0.
    """
    areas = {}
    for z in sorted(set(labels) - {-1}):
        pts = coords_utm[labels == z]
        if len(pts) < 3:
            areas[z] = 0.0
        else:
            try:
                hull = ConvexHull(pts)
                areas[z] = hull.volume / 1e6  # m² → km²
            except Exception:
                areas[z] = 0.0
    return areas


def zone_binding_modes(df_locs_flagged, labels):
    """
    For each zone, identify the mode with the lowest mean access score
    among enterprises in that zone.  Returns {zone_id: mode_name}.
    Land modes only (road/rail/icd); water excluded for cross-city use.
    """
    land_modes = ["road", "rail", "icd"]
    cols = [f"a_{m}" for m in land_modes if f"a_{m}" in df_locs_flagged.columns]
    if not cols:
        return {}

    binding = {}
    zone_firms = {}  # {mode: total enterprises}
    mode_totals = {m: 0 for m in land_modes if f"a_{m}" in df_locs_flagged.columns}

    for z in sorted(set(labels) - {-1}):
        mask = labels == z
        sub = df_locs_flagged[mask]
        n_ent = sub["n_enterprises"].values

        means = {}
        for m in land_modes:
            c = f"a_{m}"
            if c in sub.columns:
                means[m] = np.average(sub[c].values, weights=n_ent)
        if means:
            binding[z] = min(means, key=means.get)
            mode_totals[binding[z]] = mode_totals.get(binding[z], 0) + n_ent.sum()

    return binding, mode_totals


def area_sensitivity(coords_utm, labels, enterprise_counts,
                     thresholds=AREA_THRESHOLDS):
    """
    For each area ceiling, report the share of flagged enterprises
    in zones whose convex-hull area is ≤ the ceiling.
    """
    areas = zone_areas_km2(coords_utm, labels)
    total_ent = enterprise_counts.sum()
    if total_ent == 0:
        return {t: 0.0 for t in thresholds}

    # Enterprise count per zone
    zone_ent = {}
    for z in areas:
        zone_ent[z] = enterprise_counts[labels == z].sum()

    result = {}
    for t in thresholds:
        ent_in = sum(zone_ent[z] for z, a in areas.items() if a <= t)
        result[t] = ent_in / total_ent
    return result


# ═══════════════════════════════════════════════════════════════
#  NULL MODELS
# ═══════════════════════════════════════════════════════════════

def _single_null_draw(all_utm, all_ent_counts, n_sample, idx_pool=None):
    """
    One null draw: sample n_sample locations from idx_pool (or all),
    run HDBSCAN, return enterprise-weighted in-zone share.
    """
    if idx_pool is None:
        idx_pool = np.arange(len(all_utm))
    if n_sample >= len(idx_pool):
        # Flagged group ≥ pool → sample entire pool
        idx = idx_pool.copy()
    else:
        idx = np.random.choice(idx_pool, size=n_sample, replace=False)
    coords = all_utm[idx]
    ent = all_ent_counts[idx]
    labels = run_hdbscan(coords)
    return in_zone_pct(labels, ent)


def null_model_a(all_utm, all_ent_counts, n_sample, n_draws=N_NULL):
    """
    Unconditional null: random subset of n_sample unique locations
    drawn from all city locations.
    Returns array of in-zone percentages.
    """
    results = np.empty(n_draws)
    for i in range(n_draws):
        results[i] = _single_null_draw(all_utm, all_ent_counts, n_sample)
    return results


def null_model_b(all_utm, all_ent_counts, all_dist_ctr,
                 flagged_dist_ctr, n_sample, n_draws=N_NULL):
    """
    Distance-stratified null: match the distance-from-centre profile
    of the flagged group.

    Stratifies all_utm by DIST_BAND_EDGES.  For each band, samples
    the same count as the flagged group has in that band.
    Falls back to sampling with replacement if a band is exhausted.
    """
    # Assign bands
    all_bands = np.digitize(all_dist_ctr, DIST_BAND_EDGES) - 1
    flagged_bands = np.digitize(flagged_dist_ctr, DIST_BAND_EDGES) - 1

    # Per-band target counts
    n_bands = len(DIST_BAND_EDGES) - 1
    band_targets = np.bincount(flagged_bands, minlength=n_bands)

    # Pre-compute per-band index pools
    band_pools = [np.where(all_bands == b)[0] for b in range(n_bands)]

    results = np.empty(n_draws)
    for i in range(n_draws):
        idx_parts = []
        for b in range(n_bands):
            target = band_targets[b]
            if target == 0:
                continue
            pool = band_pools[b]
            if len(pool) == 0:
                continue
            replace = target > len(pool)
            chosen = np.random.choice(pool, size=target, replace=replace)
            idx_parts.append(chosen)
        if not idx_parts:
            results[i] = 0.0
            continue
        idx = np.concatenate(idx_parts)
        coords = all_utm[idx]
        ent = all_ent_counts[idx]
        labels = run_hdbscan(coords)
        results[i] = in_zone_pct(labels, ent)

    return results


def empirical_pvalues(observed, null_dist):
    """
    Empirical p-values with standard correction (Phipson & Smyth 2010).
      p_upper = (1 + #{null >= obs}) / (B + 1)
      p_lower = (1 + #{null <= obs}) / (B + 1)
    """
    B = len(null_dist)
    p_upper = (1 + np.sum(null_dist >= observed)) / (B + 1)
    p_lower = (1 + np.sum(null_dist <= observed)) / (B + 1)
    p_two   = 2 * min(p_upper, p_lower)
    return {
        "p_upper": p_upper,
        "p_lower": p_lower,
        "p_two":   min(p_two, 1.0),
        "null_mean": np.mean(null_dist),
        "null_05":   np.percentile(null_dist, 5),
        "null_95":   np.percentile(null_dist, 95),
    }


# ═══════════════════════════════════════════════════════════════
#  SINGLE-CITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyse_city(city, df_all_locs, utm_all, threshold,
                 threshold_label="pooled_p10"):
    """
    Full targetability analysis for one city at one threshold.
    Returns a results dict.
    """
    epsg = UTM_EPSG[city]
    n_all_locs = len(df_all_locs)
    n_all_ent  = df_all_locs["n_enterprises"].sum()

    # ── Flag locations below threshold ────────────────────────
    mask = df_all_locs[COL_MFAI].values < threshold
    df_flag = df_all_locs[mask].reset_index(drop=True)
    utm_flag = utm_all[mask]
    n_flag_locs = len(df_flag)
    n_flag_ent  = df_flag["n_enterprises"].sum()
    pct_flagged = n_flag_ent / n_all_ent * 100 if n_all_ent > 0 else 0

    print(f"\n  [{threshold_label}] threshold={threshold:.4f}")
    print(f"  Flagged: {n_flag_ent} enterprises at {n_flag_locs} "
          f"unique locations ({pct_flagged:.1f}% of city)")

    res = {
        "city": city,
        "threshold_label": threshold_label,
        "threshold": threshold,
        "n_all_ent": n_all_ent,
        "n_all_locs": n_all_locs,
        "n_flag_ent": n_flag_ent,
        "n_flag_locs": n_flag_locs,
        "pct_flagged": pct_flagged,
    }

    # ── Insufficient flagged locations? ───────────────────────
    if n_flag_locs < MIN_FLAGGED_LOCS:
        print(f"  → Insufficient flagged locations ({n_flag_locs} < "
              f"{MIN_FLAGGED_LOCS}) — skipping clustering.")
        res.update({
            "n_zones": 0, "in_zone_pct": 0, "skipped": True,
        })
        return res

    # ── HDBSCAN ───────────────────────────────────────────────
    labels = run_hdbscan(utm_flag)
    n_zones = len(set(labels) - {-1})
    obs_izp = in_zone_pct(labels, df_flag["n_enterprises"].values)
    areas   = zone_areas_km2(utm_flag, labels)

    binding_info, mode_totals = zone_binding_modes(df_flag, labels)

    # ── Area sensitivity ──────────────────────────────────────
    area_sens = area_sensitivity(
        utm_flag, labels, df_flag["n_enterprises"].values
    )

    print(f"  Zones: {n_zones}, in-zone: {obs_izp:.1%}")
    for t, s in area_sens.items():
        print(f"    ≤{t} km²: {s:.1%}")

    # ── Null A (unconditional) ────────────────────────────────
    t0 = time.time()
    null_a = null_model_a(
        utm_all, df_all_locs["n_enterprises"].values, n_flag_locs
    )
    pv_a = empirical_pvalues(obs_izp, null_a)
    dt_a = time.time() - t0
    print(f"  Null A: mean={pv_a['null_mean']:.3f} "
          f"(5–95: {pv_a['null_05']:.3f}–{pv_a['null_95']:.3f}), "
          f"p_upper={pv_a['p_upper']:.4f}, "
          f"p_lower={pv_a['p_lower']:.4f}  [{dt_a:.0f}s]")

    # ── Null B (distance-stratified) ──────────────────────────
    if COL_DIST_CTR in df_all_locs.columns:
        t0 = time.time()
        null_b = null_model_b(
            utm_all,
            df_all_locs["n_enterprises"].values,
            df_all_locs[COL_DIST_CTR].values,
            df_flag[COL_DIST_CTR].values,
            n_flag_locs,
        )
        pv_b = empirical_pvalues(obs_izp, null_b)
        dt_b = time.time() - t0
        print(f"  Null B: mean={pv_b['null_mean']:.3f} "
              f"(5–95: {pv_b['null_05']:.3f}–{pv_b['null_95']:.3f}), "
              f"p_upper={pv_b['p_upper']:.4f}, "
              f"p_lower={pv_b['p_lower']:.4f}  [{dt_b:.0f}s]")
    else:
        pv_b = None
        print(f"  Null B: skipped ({COL_DIST_CTR} not in data)")

    # ── Pack results ──────────────────────────────────────────
    res.update({
        "skipped":      False,
        "n_zones":      n_zones,
        "in_zone_pct":  obs_izp,
        "zone_areas":   areas,
        "zone_binding": binding_info,
        "mode_totals":  mode_totals,
        "area_sens":    area_sens,
        # Null A
        "null_a_mean":    pv_a["null_mean"],
        "null_a_05":      pv_a["null_05"],
        "null_a_95":      pv_a["null_95"],
        "null_a_p_upper": pv_a["p_upper"],
        "null_a_p_lower": pv_a["p_lower"],
        "null_a_p_two":   pv_a["p_two"],
        # Null B
        "null_b_mean":    pv_b["null_mean"] if pv_b else None,
        "null_b_05":      pv_b["null_05"]   if pv_b else None,
        "null_b_95":      pv_b["null_95"]   if pv_b else None,
        "null_b_p_upper": pv_b["p_upper"]   if pv_b else None,
        "null_b_p_lower": pv_b["p_lower"]   if pv_b else None,
        "null_b_p_two":   pv_b["p_two"]     if pv_b else None,
    })
    return res


# ═══════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_null_table(results, null_key="a"):
    """Format Table 15-equivalent for one null variant."""
    k = f"null_{null_key}"
    rows = []
    for r in results:
        if r.get("skipped"):
            rows.append({
                "City": r["city"].title(),
                "Flagged ent.": r["n_flag_ent"],
                "Flagged locs": r["n_flag_locs"],
                "% of city": f"{r['pct_flagged']:.1f}",
                "Zones": "—",
                "In-zone %": "—",
                f"Null {null_key.upper()} mean (5–95)": "—",
                "p (upper)": "—",
                "p (lower)": "—",
                "p (two)": "—",
            })
        else:
            rows.append({
                "City": r["city"].title(),
                "Flagged ent.": r["n_flag_ent"],
                "Flagged locs": r["n_flag_locs"],
                "% of city": f"{r['pct_flagged']:.1f}",
                "Zones": r["n_zones"],
                "In-zone %": f"{r['in_zone_pct']:.1%}",
                f"Null {null_key.upper()} mean (5–95)":
                    f"{r[f'{k}_mean']:.3f} ({r[f'{k}_05']:.3f}–{r[f'{k}_95']:.3f})"
                    if r.get(f"{k}_mean") is not None else "—",
                "p (upper)":
                    f"{r[f'{k}_p_upper']:.4f}"
                    if r.get(f"{k}_p_upper") is not None else "—",
                "p (lower)":
                    f"{r[f'{k}_p_lower']:.4f}"
                    if r.get(f"{k}_p_lower") is not None else "—",
                "p (two)":
                    f"{r[f'{k}_p_two']:.4f}"
                    if r.get(f"{k}_p_two") is not None else "—",
            })
    return pd.DataFrame(rows)


def format_zone_table(results):
    """Format Table 16-equivalent with area sensitivity columns."""
    rows = []
    for r in results:
        if r.get("skipped"):
            row = {"City": r["city"].title(), "Zones": "—"}
            row.update({f"≤{t} km²": "—" for t in AREA_THRESHOLDS})
            rows.append(row)
        else:
            areas = r["zone_areas"]
            row = {
                "City":     r["city"].title(),
                "Zones":    r["n_zones"],
                "In-zone %": f"{r['in_zone_pct']:.1%}",
                "Largest km²": f"{max(areas.values()):.1f}" if areas else "—",
                "Median km²":  f"{np.median(list(areas.values())):.2f}" if areas else "—",
            }
            for t in AREA_THRESHOLDS:
                row[f"≤{t} km²"] = f"{r['area_sens'].get(t, 0):.1%}"
            rows.append(row)
    return pd.DataFrame(rows)


def format_binding_table(results):
    """Format Table 17-equivalent: enterprises by zone binding mode."""
    rows = []
    for r in results:
        if r.get("skipped") or not r.get("mode_totals"):
            continue
        row = {"City": r["city"].title()}
        mt = r["mode_totals"]
        total = sum(mt.values()) if mt else 0
        for m in ["icd", "rail", "road"]:
            n = mt.get(m, 0)
            pct = n / total * 100 if total > 0 else 0
            row[m.upper()] = f"{n} ({pct:.0f}%)" if n > 0 else "0"
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():

    # ── Step 0: Verify columns ────────────────────────────────
    print("=" * 60)
    print("STEP 0 — Column verification")
    print("=" * 60)
    cols = verify_columns()
    # Sanity checks
    for needed in [COL_LAT, COL_LON, COL_MFAI]:
        if needed not in cols:
            print(f"\n*** MISSING COLUMN: {needed} ***")
            print("Adjust the configuration at the top of this script.")
            return
    print("\nRequired columns present. Proceeding.\n")

    # ── Step 1: Load all cities ───────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading cities")
    print("=" * 60)
    city_data = {}   # {city: (df_locs, utm_coords)}
    all_mfai = []

    for city in CITIES:
        print(f"\n{city.upper()}")
        df = load_city(city)
        df_locs, utm = build_unique_locs(df, UTM_EPSG[city])
        city_data[city] = (df_locs, utm)
        all_mfai.append(df[COL_MFAI].values)
        print(f"  Unique locations: {len(df_locs)}, "
              f"enterprises: {df_locs['n_enterprises'].sum()}")

    # ── Step 2: Compute thresholds ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Thresholds")
    print("=" * 60)
    pooled = np.concatenate(all_mfai)
    pooled_p10 = np.percentile(pooled, 10)
    print(f"\nPooled P10 (n={len(pooled)}): {pooled_p10:.4f}")

    city_p10 = {}
    for city in CITIES:
        df_locs, _ = city_data[city]
        # Enterprise-level percentile: weight by n_enterprises
        vals = np.repeat(df_locs[COL_MFAI].values,
                         df_locs["n_enterprises"].values.astype(int))
        city_p10[city] = np.percentile(vals, 10)
        print(f"  {city.title()} P10: {city_p10[city]:.4f}")

    # ── Step 3: Primary analysis — pooled threshold ───────────
    print("\n" + "=" * 60)
    print("STEP 3 — PRIMARY: Pooled-P10 targetability")
    print("=" * 60)
    results_pooled = []
    for city in CITIES:
        print(f"\n{'─'*40}\n{city.upper()}")
        df_locs, utm = city_data[city]
        r = analyse_city(city, df_locs, utm, pooled_p10,
                         threshold_label="pooled_p10")
        results_pooled.append(r)

    # ── Step 4: Sensitivity — city-specific P10 ───────────────
    print("\n" + "=" * 60)
    print("STEP 4 — SENSITIVITY: City-specific P10")
    print("=" * 60)
    results_city = []
    for city in CITIES:
        print(f"\n{'─'*40}\n{city.upper()}")
        df_locs, utm = city_data[city]
        r = analyse_city(city, df_locs, utm, city_p10[city],
                         threshold_label="city_p10")
        results_city.append(r)

    # ── Step 5: Format and display ────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — Pooled threshold")
    print("=" * 60)

    print("\n── Null A (unconditional) ──")
    tbl_a = format_null_table(results_pooled, "a")
    print(tbl_a.to_string(index=False))

    print("\n── Null B (distance-stratified) ──")
    tbl_b = format_null_table(results_pooled, "b")
    print(tbl_b.to_string(index=False))

    print("\n── Zone structure & area sensitivity ──")
    tbl_z = format_zone_table(results_pooled)
    print(tbl_z.to_string(index=False))

    print("\n── Zone binding mode ──")
    tbl_bm = format_binding_table(results_pooled)
    print(tbl_bm.to_string(index=False))

    print("\n" + "=" * 60)
    print("RESULTS — City-specific P10 (sensitivity)")
    print("=" * 60)

    print("\n── Null A ──")
    tbl_a2 = format_null_table(results_city, "a")
    print(tbl_a2.to_string(index=False))

    print("\n── Zone structure & area sensitivity ──")
    tbl_z2 = format_zone_table(results_city)
    print(tbl_z2.to_string(index=False))

    # ── Save to CSV ───────────────────────────────────────────
    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)

    tbl_a.to_csv(os.path.join(out_dir, "null_a_pooled.csv"), index=False)
    tbl_b.to_csv(os.path.join(out_dir, "null_b_pooled.csv"), index=False)
    tbl_z.to_csv(os.path.join(out_dir, "zones_pooled.csv"), index=False)
    tbl_bm.to_csv(os.path.join(out_dir, "binding_pooled.csv"), index=False)
    tbl_a2.to_csv(os.path.join(out_dir, "null_a_city.csv"), index=False)
    tbl_z2.to_csv(os.path.join(out_dir, "zones_city.csv"), index=False)
    print(f"\nTables saved to {out_dir}/")

    return results_pooled, results_city


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results_pooled, results_city = main()
