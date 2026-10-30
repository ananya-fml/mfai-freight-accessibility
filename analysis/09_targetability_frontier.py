"""Absolute targetability frontier

Produces: Figure 8, Table 16 -> results/targetability/

Sweeps the flagging threshold across ABSOLUTE MFAI-land values and, at each
threshold, measures how much of the flagged population falls into compact
zones. Turns the single bottom-decile HDBSCAN result into a curve:
    x-axis: absolute MFAI-land threshold (a firm is flagged if mfai_land < t)
    y-axis: share of flagged enterprises inside compact zones (per area ceiling)
Answers the reviewer's request for a threshold-response curve (Major 8) and
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial import ConvexHull
import hdbscan

from mfai import paths
from mfai.io import load_city

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  CONFIG  (matches targetability_v2.py — T=8 confirmed)
# ═══════════════════════════════════════════════════════════════
CITIES   = ["patna", "varanasi", "indore", "kochi", "kolkata"]
UTM_EPSG = {"patna": 32645, "kolkata": 32645, "varanasi": 32644,
            "indore": 32643, "kochi": 32643}

COL_LAT, COL_LON, COL_MFAI = "lat", "lon", "mfai_land"

FALLBACK_THRESHOLD = 20
DIVISION_T         = 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

MCS, MS = 20, 10                      # HDBSCAN params (fixed)
MIN_FLAGGED_LOCS = 30                 # below this, skip clustering

# ── Threshold sweep (absolute MFAI-land) ──────────────────────
THRESH_GRID = np.round(np.arange(0.55, 0.915, 0.01), 3)

# ── Area ceilings (km²) ───────────────────────────────────────
AREA_CEILINGS = [5, 10, 25, 50, 100]
HEADLINE_CEILING = 25                 # curve drawn for this ceiling

# ── Null validation ───────────────────────────────────────────
RUN_NULL_ANCHORS = True
NULL_ANCHOR_PCTS = [5, 20]        # pooled percentiles; [5,20] to skip P10
N_NULL = 9_999

CITY_COL = {"patna": "#E76F51", "varanasi": "#F4A261", "indore": "#2A9D8F",
            "kochi": "#264653", "kolkata": "#B5179E"}


# ═══════════════════════════════════════════════════════════════
#  LOADING  (identical filter to targetability_v2)
# ═══════════════════════════════════════════════════════════════


def to_utm(lats, lons, epsg):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(np.asarray(lons), np.asarray(lats))
    return np.column_stack([x, y])


def build_unique_locs(df, epsg):
    grp = df.groupby([COL_LAT, COL_LON])
    locs = grp.size().reset_index(name="n_enterprises")
    locs[COL_MFAI] = grp[COL_MFAI].mean().values
    utm = to_utm(locs[COL_LAT].values, locs[COL_LON].values, epsg)
    return locs.reset_index(drop=True), utm


# ═══════════════════════════════════════════════════════════════
#  CLUSTER MECHANICS
# ═══════════════════════════════════════════════════════════════
def run_hdbscan(coords):
    return hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                           metric="euclidean").fit(coords).labels_


def in_zone_pct(labels, ent):
    tot = ent.sum()
    return ent[labels >= 0].sum() / tot if tot > 0 else 0.0


def zone_areas_km2(coords, labels):
    areas = {}
    for z in sorted(set(labels) - {-1}):
        pts = coords[labels == z]
        if len(pts) < 3:
            areas[z] = 0.0
        else:
            try:
                areas[z] = ConvexHull(pts).volume / 1e6
            except Exception:
                areas[z] = 0.0
    return areas


def compact_shares(coords, labels, ent, ceilings=AREA_CEILINGS):
    """Share of flagged enterprises in zones ≤ each area ceiling."""
    areas = zone_areas_km2(coords, labels)
    tot = ent.sum()
    if tot == 0:
        return {c: 0.0 for c in ceilings}, {}
    zone_ent = {z: ent[labels == z].sum() for z in areas}
    out = {}
    for c in ceilings:
        out[c] = sum(zone_ent[z] for z, a in areas.items() if a <= c) / tot
    return out, areas


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


# ═══════════════════════════════════════════════════════════════
#  FRONTIER SWEEP
# ═══════════════════════════════════════════════════════════════
def sweep_city(city, df_locs, utm):
    rows = []
    ent_all = df_locs["n_enterprises"].values
    mfai = df_locs[COL_MFAI].values
    for t in THRESH_GRID:
        mask = mfai < t
        n_locs = int(mask.sum())
        n_ent = int(ent_all[mask].sum())
        row = {"city": city, "threshold": t,
               "n_flagged_locs": n_locs, "n_flagged_ent": n_ent}
        if n_locs < MIN_FLAGGED_LOCS:
            row["in_zone_pct"] = np.nan
            row["n_clusters"] = 0
            row["largest_km2"] = np.nan
            for c in AREA_CEILINGS:
                row[f"share_le_{c}"] = np.nan
            rows.append(row)
            continue
        labels = run_hdbscan(utm[mask])
        ent = ent_all[mask]
        row["in_zone_pct"] = in_zone_pct(labels, ent)
        shares, areas = compact_shares(utm[mask], labels, ent)
        row["n_clusters"] = len(areas)
        row["largest_km2"] = max(areas.values()) if areas else np.nan
        for c in AREA_CEILINGS:
            row[f"share_le_{c}"] = shares[c]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Loading cities …")
    data, all_mfai = {}, []
    for city in CITIES:
        df = load_city(city)
        locs, utm = build_unique_locs(df, UTM_EPSG[city])
        data[city] = (locs, utm)
        all_mfai.append(df[COL_MFAI].values)
        print(f"  {city}: {len(locs)} locs, {int(locs['n_enterprises'].sum())} ent")

    pooled = np.concatenate(all_mfai)
    anchors = {p: np.percentile(pooled, p) for p in NULL_ANCHOR_PCTS}
    print("\nPooled anchor thresholds:")
    for p, v in anchors.items():
        print(f"  P{p}: {v:.4f}")

    # ── Descriptive frontier (fast) ──────────────────────────
    print("\n" + "=" * 60)
    print("FRONTIER SWEEP (descriptive, no null)")
    print("=" * 60)
    frames = []
    for city in CITIES:
        locs, utm = data[city]
        t0 = time.time()
        fr = sweep_city(city, locs, utm)
        frames.append(fr)
        print(f"  {city}: {len(THRESH_GRID)} thresholds [{time.time()-t0:.0f}s]")
    frontier = pd.concat(frames, ignore_index=True)

    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)
    frontier.to_csv(os.path.join(out_dir, "frontier_sweep.csv"), index=False)

    # ── Frontier figure (headline ceiling) ───────────────────
    fig, ax = plt.subplots(figsize=(5.51, 3.6))
    for city in CITIES:
        sub = frontier[frontier.city == city]
        # Only plot where flagged sample is large enough to be meaningful
        ok = sub["n_flagged_locs"] >= MIN_FLAGGED_LOCS
        ax.plot(sub.loc[ok, "threshold"],
                sub.loc[ok, f"share_le_{HEADLINE_CEILING}"],
                color=CITY_COL[city], lw=1.6, label=city.title())
    for p, v in anchors.items():
        ax.axvline(v, color="#CCCCCC", lw=0.6, ls="--", zorder=0)
        ax.text(v, 1.01, f"P{p}", fontsize=6, ha="center", color="#888888")
    ax.set_xlabel("Absolute MFAI-land threshold (flag if below)", fontsize=8)
    ax.set_ylabel(f"Share of flagged enterprises in zones ≤{HEADLINE_CEILING} km²",
                  fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_targetability_frontier.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_targetability_frontier.png"),
                dpi=400, bbox_inches="tight")
    print(f"\nFrontier figure + sweep saved to {out_dir}/")

    # ── Null validation at anchors only ──────────────────────
    if RUN_NULL_ANCHORS:
        print("\n" + "=" * 60)
        print(f"NULL VALIDATION at anchors {NULL_ANCHOR_PCTS} ({N_NULL} draws)")
        print("=" * 60)
        null_rows = []
        for city in CITIES:
            locs, utm = data[city]
            ent_all = locs["n_enterprises"].values
            for p, thr in anchors.items():
                mask = locs[COL_MFAI].values < thr
                n_locs = int(mask.sum())
                if n_locs < MIN_FLAGGED_LOCS:
                    print(f"  {city} P{p}: {n_locs} locs — skipped")
                    null_rows.append({"city": city, "anchor_pct": p,
                                      "threshold": thr, "n_flagged_locs": n_locs,
                                      "in_zone_pct": np.nan, "p_upper": np.nan,
                                      "p_lower": np.nan, "p_two": np.nan})
                    continue
                obs = in_zone_pct(run_hdbscan(utm[mask]), ent_all[mask])
                t0 = time.time()
                nd = null_a(utm, ent_all, n_locs)
                pv = empirical_pvalues(obs, nd)
                print(f"  {city} P{p}: obs={obs:.3f}, "
                      f"null={pv['null_mean']:.3f} "
                      f"({pv['null_05']:.3f}–{pv['null_95']:.3f}), "
                      f"p_lo={pv['p_lower']:.4f} p_hi={pv['p_upper']:.4f} "
                      f"[{time.time()-t0:.0f}s]")
                null_rows.append({"city": city, "anchor_pct": p,
                                  "threshold": thr, "n_flagged_locs": n_locs,
                                  "in_zone_pct": obs, **pv})
        pd.DataFrame(null_rows).to_csv(
            os.path.join(out_dir, "frontier_null_anchors.csv"), index=False)
        print(f"Anchor null table saved to {out_dir}/")

    plt.show()
    return frontier


if __name__ == "__main__":
    frontier = main()
