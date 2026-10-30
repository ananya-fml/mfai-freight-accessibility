"""Commodity extension: do manufacturing groups sort spatially, and does
sorting line up with freight access?

Produces: Table 11 and the sorting/LQ statistics (printed); supplementary
sorting statistics by city.

Sections, in order:
  1. Label-permutation sorting test (entropy)
  2. Permuted Duncan's dissimilarity index
  3. Location quotients - 2 km grid (1 km Kolkata), EB shrinkage, sweep
  4. Permutation null for concentration shares
  5. Concentration vs distance-matched access null
  6. Validity checks - division-level ratios, mode spread, raw distances
  7. Sorting statistics by city (supplementary)
  8. Table 11 - concentrations and freight access, consistent denominator

Run as a script; sections execute top to bottom and share state.
Needs the GHSL rasters for the built-up clip (skipped with a message if
absent). The retracted characteristic-scale test is not included.
"""
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from shapely.geometry import box

from mfai import paths
from mfai.config import CITIES, GROUP_LABELS
from mfai.io import load_city_gdf

city_data = {c: load_city_gdf(c) for c in CITIES}
# ══════════════════════════════════════════════════════════════════════
# Label-permutation sorting test (entropy)
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# SPATIAL SORTING TESTS — self-contained
# Requires: city_data, CITIES, GROUP_LABELS
# ============================================================


def build_grid(gdf, cell_size=1000):
    """Regular square grid over the extent of gdf, in gdf's CRS (metres)."""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin = np.floor(xmin / cell_size) * cell_size
    ymin = np.floor(ymin / cell_size) * cell_size
    xmax = np.ceil(xmax / cell_size) * cell_size
    ymax = np.ceil(ymax / cell_size) * cell_size
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    return gpd.GeoDataFrame({"cell_id": range(len(cells))},
                            geometry=cells, crs=gdf.crs)


def sorting_test(gdf, cell_size=1000, min_ent=10, n_perm=999, seed=0):
    """
    Label-permutation test for spatial sorting of commodity groups.

    Shuffles group labels across enterprises while holding cell locations
    and cell occupancy fixed, so the null preserves both citywide group
    base rates and the firms-per-cell distribution. Observed median cell
    entropy below the null band => groups are more spatially concentrated
    than chance.
    """
    rng = np.random.default_rng(seed)
    grid = build_grid(gdf, cell_size)
    j = gpd.sjoin(gdf[["group", "geometry"]], grid, how="inner", predicate="within")

    if len(j) == 0:
        return None

    groups = list(GROUP_LABELS.keys())
    gidx = {g: i for i, g in enumerate(groups)}
    lab_i = np.array([gidx[g] for g in j["group"].values])

    uniq, inv = np.unique(j["cell_id"].values, return_inverse=True)
    n_cells, n_groups = len(uniq), len(groups)

    def median_entropy(lab_arr):
        M = np.zeros((n_cells, n_groups))
        np.add.at(M, (inv, lab_arr), 1)
        tot = M.sum(1)
        keep = tot >= min_ent
        if keep.sum() == 0:
            return np.nan
        P = M[keep] / tot[keep][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(P * np.where(P > 0, np.log(P), 0)).sum(1) / np.log(n_groups)
        return np.median(H)

    obs = median_entropy(lab_i)
    null = np.array([median_entropy(rng.permutation(lab_i)) for _ in range(n_perm)])

    p_lo = (1 + (null <= obs).sum()) / (n_perm + 1)   # observed MORE concentrated
    p_hi = (1 + (null >= obs).sum()) / (n_perm + 1)   # observed MORE mixed

    return {
        "cells_retained": int((np.bincount(inv, minlength=n_cells) >= min_ent).sum()),
        "obs": round(float(obs), 4),
        "null_mean": round(float(null.mean()), 4),
        "null_p5": round(float(np.percentile(null, 5)), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "p_more_concentrated": round(float(p_lo), 4),
        "p_more_mixed": round(float(p_hi), 4),
    }


def pairwise_dissimilarity(gdf, cell_size=1000):
    """
    Duncan index of dissimilarity between every pair of commodity groups.
    0 = identical spatial distributions; 1 = completely segregated.
    Interpreted as the share of one group that would have to relocate to
    match the other's distribution across cells.
    """
    grid = build_grid(gdf, cell_size)
    j = gpd.sjoin(gdf[["group", "geometry"]], grid, how="inner", predicate="within")
    cnt = j.groupby(["cell_id", "group"]).size().unstack(fill_value=0)
    for g in GROUP_LABELS:
        if g not in cnt.columns:
            cnt[g] = 0
    cnt = cnt[list(GROUP_LABELS.keys())]

    share = cnt / cnt.sum(0)          # each column sums to 1
    gs = list(GROUP_LABELS.keys())
    D = pd.DataFrame(0.0, index=gs, columns=gs)
    for a in gs:
        for b in gs:
            if a != b:
                D.loc[a, b] = 0.5 * (share[a] - share[b]).abs().sum()
    return D.round(3)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
print("=== LABEL-PERMUTATION SORTING TEST (1 km cells, 999 permutations) ===")
sort_rows = []
for c in CITIES:
    r = sorting_test(city_data[c], cell_size=1000, min_ent=10, n_perm=999)
    if r is None:
        print(f"{c}: no cells")
        continue
    r["city"] = c
    sort_rows.append(r)
    verdict = ("SORTED (more concentrated than chance)"
               if r["obs"] < r["null_p5"] else
               "MORE MIXED than chance" if r["obs"] > r["null_p95"] else
               "no evidence of sorting")
    print(f"{c:9s} obs={r['obs']:.4f}  null={r['null_mean']:.4f} "
          f"[{r['null_p5']:.4f}-{r['null_p95']:.4f}]  "
          f"p_conc={r['p_more_concentrated']:.4f}  -> {verdict}")

sort_df = pd.DataFrame(sort_rows)[
    ["city", "cells_retained", "obs", "null_mean", "null_p5", "null_p95",
     "p_more_concentrated", "p_more_mixed"]
]
print("\n", sort_df.to_string(index=False))

print("\n\n=== PAIRWISE GROUP DISSIMILARITY (1 km cells) ===")
diss = {}
for c in CITIES:
    D = pairwise_dissimilarity(city_data[c], cell_size=1000)
    diss[c] = D
    off = D.values[~np.eye(len(D), dtype=bool)]
    top = D.stack().idxmax()
    print(f"\n{c}: mean D = {off.mean():.3f} | min = {off.min():.3f} | max = {off.max():.3f}")
    print(f"  most separated pair: {top[0]}–{top[1]} = {D.stack().max():.3f} "
          f"({GROUP_LABELS[top[0]]} vs {GROUP_LABELS[top[1]]})")
    print(D.to_string())


# ══════════════════════════════════════════════════════════════════════
# Permuted Duncan's dissimilarity index
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# PERMUTED DUNCAN'S D — tests whether observed pairwise
# dissimilarity exceeds what group sizes alone produce.
# Requires: city_data, CITIES, GROUP_LABELS
# ============================================================


def permuted_D(gdf, cell_size=1000, n_perm=999, seed=0):
    """
    Observed pairwise Duncan D vs a null that shuffles group labels
    across enterprises, holding cell locations and each group's total
    size fixed. Isolates genuine spatial separation from the upward
    small-N bias in D.
    """
    rng = np.random.default_rng(seed)
    grid = build_grid(gdf, cell_size)
    j = gpd.sjoin(gdf[["group", "geometry"]], grid, how="inner", predicate="within")
    if len(j) == 0:
        return None

    groups = list(GROUP_LABELS.keys())
    gidx = {g: i for i, g in enumerate(groups)}
    lab = np.array([gidx[g] for g in j["group"].values])
    _, inv = np.unique(j["cell_id"].values, return_inverse=True)
    n_cells, n_groups = inv.max() + 1, len(groups)

    def D_matrix(lab_arr):
        M = np.zeros((n_cells, n_groups))
        np.add.at(M, (inv, lab_arr), 1)
        colsum = M.sum(0)
        colsum[colsum == 0] = 1
        S = M / colsum                       # each group's share across cells
        # D[a,b] = 0.5 * sum_i |S[i,a] - S[i,b]|
        return 0.5 * np.abs(S[:, :, None] - S[:, None, :]).sum(0)

    obs = D_matrix(lab)
    null = np.empty((n_perm, n_groups, n_groups))
    for k in range(n_perm):
        null[k] = D_matrix(rng.permutation(lab))

    null_mean = null.mean(0)
    null_p95  = np.percentile(null, 95, axis=0)
    # upper-tail p: observed MORE separated than chance (Phipson-Smyth)
    p_hi = (1 + (null >= obs).sum(0)) / (n_perm + 1)
    excess = obs - null_mean

    idx = pd.Index(groups)
    return {
        "obs":       pd.DataFrame(obs,       index=idx, columns=idx).round(3),
        "null_mean": pd.DataFrame(null_mean, index=idx, columns=idx).round(3),
        "null_p95":  pd.DataFrame(null_p95,  index=idx, columns=idx).round(3),
        "excess":    pd.DataFrame(excess,    index=idx, columns=idx).round(3),
        "p_hi":      pd.DataFrame(p_hi,      index=idx, columns=idx).round(4),
    }


CELL = 2000          # match the size used for the D matrices you already have
perm_D = {}
rows = []
for c in CITIES:
    r = permuted_D(city_data[c], cell_size=CELL, n_perm=999)
    perm_D[c] = r
    if r is None:
        print(f"{c}: no cells"); continue

    gs = list(GROUP_LABELS.keys())
    off = ~np.eye(len(gs), dtype=bool)
    sig = (r["p_hi"].values < 0.05) & off

    print(f"\n=== {c.upper()} (cells {CELL}m) ===")
    print(f"mean observed D {r['obs'].values[off].mean():.3f} | "
          f"mean null D {r['null_mean'].values[off].mean():.3f} | "
          f"mean excess {r['excess'].values[off].mean():+.3f}")
    print(f"pairs significantly more separated than chance: "
          f"{sig.sum()//2} of {len(gs)*(len(gs)-1)//2}")
    print("\nexcess (observed - null mean):")
    print(r["excess"].to_string())

    # biggest genuine separations, ranked by excess not raw D
    ex = r["excess"].where(off)
    top = ex.stack().sort_values(ascending=False).head(3)
    for (a, b), v in top.items():
        print(f"  {a}-{b}: obs {r['obs'].loc[a,b]:.3f}  null {r['null_mean'].loc[a,b]:.3f} "
              f"excess {v:+.3f}  p={r['p_hi'].loc[a,b]:.4f}")

    rows.append({"city": c,
                 "mean_obs_D": round(r['obs'].values[off].mean(), 3),
                 "mean_null_D": round(r['null_mean'].values[off].mean(), 3),
                 "mean_excess": round(r['excess'].values[off].mean(), 3),
                 "sig_pairs": int(sig.sum()//2)})

print("\n\n=== SUMMARY ===")
print(pd.DataFrame(rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# Location quotients — 2 km grid, EB shrinkage, threshold sweep
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# LOCATION QUOTIENTS — 2 km primary, 20-firm floor,
# empirical-Bayes shrinkage, GHSL clipping, threshold sweep
# ============================================================

GHSL_MIN    = 0.0      # built-up threshold; 0 keeps any built-up cell
CELL_PRIMARY = 2000
MIN_FIRMS    = 20
EB_K         = 20      # shrinkage prior weight

def clip_grid_to_builtup(grid, city, ghsl_min=GHSL_MIN):
    """Keep cells whose mean GHSL built-up surface exceeds ghsl_min.
       Returns grid unchanged if the raster is unavailable."""
    try:
        import rasterio
        from rasterstats import zonal_stats
    except ImportError:
        print("  (rasterstats/rasterio unavailable - skipping GHSL clip)")
        return grid, False
    path = paths.ghsl(city)
    if not os.path.exists(path):
        print(f"  (no GHSL raster at {path} - skipping clip)")
        return grid, False
    with rasterio.open(path) as src:
        g = grid.to_crs(src.crs)
        stats = zonal_stats(g.geometry, path, stats=["mean"], nodata=src.nodata)
    grid = grid.copy()
    grid["builtup"] = [s["mean"] if s["mean"] is not None else 0.0 for s in stats]
    return grid[grid["builtup"] > ghsl_min].reset_index(drop=True), True


def clip_builtup(grid, city):
    """Same GHSL clip as clip_grid_to_builtup at the default threshold."""
    return clip_grid_to_builtup(grid, city)[0]


def compute_lq(gdf, city, cell_size=CELL_PRIMARY, min_firms=MIN_FIRMS,
               eb_k=EB_K, denom="city", pooled_shares=None, clip_ghsl=True):
    """
    Per-cell Location Quotient for each commodity group.
      LQ_raw    = (n_ig/n_i) / p_g
      LQ_shrunk = ((n_ig + k*p_g)/(n_i + k)) / p_g
    denom: "city" (default) or "pooled" (pass pooled_shares).
    Returns (cells_gdf_with_LQ_columns, sparse_cells_gdf).
    """
    grid = build_grid(gdf, cell_size)
    if clip_ghsl:
        grid, clipped = clip_grid_to_builtup(grid, city)
    else:
        clipped = False

    j = gpd.sjoin(gdf[["group", "geometry"]], grid, how="inner", predicate="within")
    cnt = j.groupby(["cell_id", "group"]).size().unstack(fill_value=0)
    for g in GROUP_LABELS:
        if g not in cnt.columns:
            cnt[g] = 0
    cnt = cnt[list(GROUP_LABELS.keys())]
    tot = cnt.sum(axis=1)

    p = (pooled_shares if denom == "pooled"
         else cnt.sum(0) / cnt.sum(0).sum())          # reference shares

    sparse_ids = tot[(tot > 0) & (tot < min_firms)].index
    keep = tot >= min_firms
    cnt_k, tot_k = cnt[keep], tot[keep]
    if len(cnt_k) == 0:
        return None, None, p, clipped

    out = pd.DataFrame({"cell_id": tot_k.index, "n_firms": tot_k.values})
    for g in GROUP_LABELS:
        raw = (cnt_k[g] / tot_k) / p[g]
        shr = ((cnt_k[g] + eb_k * p[g]) / (tot_k + eb_k)) / p[g]
        out[f"LQ_raw_{g}"] = raw.values
        out[f"LQ_{g}"]     = shr.values
        out[f"n_{g}"]      = cnt_k[g].values

    cells  = grid.merge(out, on="cell_id")
    sparse = grid[grid.cell_id.isin(sparse_ids)]
    return cells, sparse, p, clipped


# ---------- pooled reference shares (for the sensitivity run) ----------
pooled_cnt = pd.concat([city_data[c]["group"] for c in CITIES]).value_counts()
pooled_shares = (pooled_cnt / pooled_cnt.sum()).reindex(list(GROUP_LABELS.keys()))

# ---------- run ----------
lq = {}
for c in CITIES:
    print(f"\n{c}:")
    cells, sparse, p, clipped = compute_lq(city_data[c], c)
    if cells is None:
        print("  no cells clear the firm floor"); continue
    lq[c] = {"cells": cells, "sparse": sparse, "shares": p}
    print(f"  {len(cells)} cells (n>={MIN_FIRMS}), {len(sparse)} sparse, "
          f"GHSL clip {'applied' if clipped else 'NOT applied'}")
    print(f"  median firms/cell {cells['n_firms'].median():.0f}")


# ---------- hotspot threshold sweep ----------
print("\n\n=== SHARE OF EACH GROUP'S FIRMS IN HOTSPOT CELLS (shrunk LQ) ===")
sweep_rows = []
for c in CITIES:
    if c not in lq: continue
    cells = lq[c]["cells"]
    for g in GROUP_LABELS:
        tot_g = cells[f"n_{g}"].sum()
        if tot_g == 0: continue
        row = {"city": c, "group": g, "n": int(tot_g)}
        for thr in (1.25, 1.5, 2.0, 3.0):
            hot = cells[cells[f"LQ_{g}"] >= thr]
            row[f"LQ>={thr}"] = round(100 * hot[f"n_{g}"].sum() / tot_g, 1)
        sweep_rows.append(row)
sweep = pd.DataFrame(sweep_rows)
print(sweep.to_string(index=False))


# ---------- pooled-denominator sensitivity ----------
print("\n\n=== DENOMINATOR SENSITIVITY (share at LQ>=2.0) ===")
sens = []
for c in CITIES:
    cc, _, _, _ = compute_lq(city_data[c], c, denom="city")
    cp, _, _, _ = compute_lq(city_data[c], c, denom="pooled", pooled_shares=pooled_shares)
    if cc is None or cp is None: continue
    for g in GROUP_LABELS:
        t = cc[f"n_{g}"].sum()
        if t == 0: continue
        sens.append({"city": c, "group": g,
                     "citywide": round(100*cc[cc[f"LQ_{g}"]>=2.0][f"n_{g}"].sum()/t, 1),
                     "pooled":   round(100*cp[cp[f"LQ_{g}"]>=2.0][f"n_{g}"].sum()/t, 1)})
sens = pd.DataFrame(sens)
sens["diff"] = (sens["pooled"] - sens["citywide"]).round(1)
print(sens.to_string(index=False))


# ---------- maps: one panel per group, hotspots only ----------
def plot_lq_maps(city, thr=2.0, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    if city not in lq: return
    cells, sparse = lq[city]["cells"], lq[city]["sparse"]
    gs = list(GROUP_LABELS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, g in zip(axes.ravel(), gs):
        if sparse is not None and len(sparse):
            sparse.plot(ax=ax, color="#efefef", edgecolor="none")
        cells.plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=0.05)
        hot = cells[cells[f"LQ_{g}"] >= thr]
        if len(hot):
            hot.plot(ax=ax, column=f"LQ_{g}", cmap="YlOrRd",
                     vmin=thr, vmax=max(thr*2, cells[f"LQ_{g}"].max()),
                     edgecolor="white", linewidth=0.05, legend=True,
                     legend_kwds={"shrink": 0.6})
        share = (100*hot[f"n_{g}"].sum()/cells[f"n_{g}"].sum()
                 if cells[f"n_{g}"].sum() else 0)
        ax.set_title(f"{g}: {GROUP_LABELS[g]}\n{len(hot)} cells, "
                     f"{share:.0f}% of group's firms", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{city.capitalize()}: commodity hotspots "
                 f"(shrunk LQ ≥ {thr}, {CELL_PRIMARY}m cells)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{city}_LQ_hotspots.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

for c in CITIES:
    plot_lq_maps(c)


# ══════════════════════════════════════════════════════════════════════
# Permutation null for concentration shares
# ══════════════════════════════════════════════════════════════════════
def lq_hotspot_null(gdf, city, cell_size=2000, min_firms=20, eb_k=20,
                    thr=2.0, n_perm=999, seed=0):
    rng = np.random.default_rng(seed)
    grid = build_grid(gdf, cell_size)
    grid, _ = clip_grid_to_builtup(grid, city)
    j = gpd.sjoin(gdf[["group","geometry"]], grid, how="inner", predicate="within")
    groups = list(GROUP_LABELS.keys()); gi = {g:i for i,g in enumerate(groups)}
    lab = np.array([gi[g] for g in j["group"].values])
    _, inv = np.unique(j["cell_id"].values, return_inverse=True)
    nc, ng = inv.max()+1, len(groups)
    p = np.bincount(lab, minlength=ng) / len(lab)

    def shares(l):
        M = np.zeros((nc, ng)); np.add.at(M, (inv, l), 1)
        tot = M.sum(1); keep = tot >= min_firms
        if keep.sum() == 0: return np.full(ng, np.nan)
        Mk, tk = M[keep], tot[keep][:,None]
        LQ = ((Mk + eb_k*p) / (tk + eb_k)) / p
        hot = LQ >= thr
        return np.where(Mk.sum(0) > 0, (Mk*hot).sum(0) / np.maximum(Mk.sum(0),1), np.nan)

    obs = shares(lab)
    null = np.array([shares(rng.permutation(lab)) for _ in range(n_perm)])
    return pd.DataFrame({
        "group": groups,
        "obs_%": (100*obs).round(1),
        "null_%": (100*np.nanmean(null,0)).round(1),
        "null_p95_%": (100*np.nanpercentile(null,95,axis=0)).round(1),
        "p_hi": ((1 + (null >= obs).sum(0)) / (n_perm+1)).round(4),
    })

for c in CITIES:
    print(f"\n=== {c.upper()} ===")
    print(lq_hotspot_null(city_data[c], c,
                          cell_size=1000 if c=="kolkata" else 2000).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# Concentration vs distance-matched access null
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# HOTSPOT vs NON-HOTSPOT FREIGHT ACCESS, by group and city
# Requires: city_data, CITIES, GROUP_LABELS, lq (from LQ block),
#           build_grid, clip_grid_to_builtup
# ============================================================

# ---- SET THESE to match your CSV ----
COL_MFAI   = "mfai_land"
COL_ACCESS = {"road": "access_road", "rail": "access_rail", "icd": "access_icd"}
# -------------------------------------

CENTRES = {  # lat, lon — as in the manuscript
    "patna": (25.6127, 85.1416), "indore": (22.7177, 75.8545),
    "varanasi": (25.3109, 83.0104), "kochi": (9.9816, 76.2830),
    "kolkata": (22.5697, 88.3497),
}
CELL_FOR = {c: (1000 if c == "kolkata" else 2000) for c in CITIES}
THR, MIN_FIRMS, EB_K = 2.0, 20, 20


def _check_cols(gdf, city):
    need = [COL_MFAI] + list(COL_ACCESS.values())
    missing = [c for c in need if c not in gdf.columns]
    if missing:
        raise KeyError(f"[{city}] missing {missing}\nAvailable: {list(gdf.columns)}")


def centre_point(city, crs):
    lat, lon = CENTRES[city]
    return (gpd.GeoSeries([gpd.points_from_xy([lon], [lat])[0]], crs="EPSG:4326")
            .to_crs(crs).iloc[0])


def build_cell_table(gdf, city):
    """Assign enterprises to cells; compute per-cell LQ, access, distance."""
    _check_cols(gdf, city)
    cell_size = CELL_FOR[city]
    grid = build_grid(gdf, cell_size)
    grid, _ = clip_grid_to_builtup(grid, city)

    j = gpd.sjoin(gdf[["group", COL_MFAI, *COL_ACCESS.values(), "geometry"]],
                  grid, how="inner", predicate="within")
    cnt = j.groupby(["cell_id", "group"]).size().unstack(fill_value=0)
    for g in GROUP_LABELS:
        if g not in cnt.columns: cnt[g] = 0
    cnt = cnt[list(GROUP_LABELS.keys())]
    tot = cnt.sum(1)
    p = cnt.sum(0) / cnt.sum(0).sum()

    keep = tot >= MIN_FIRMS
    cnt, tot = cnt[keep], tot[keep]
    if len(cnt) == 0: return None, None, None

    lqs = {g: ((cnt[g] + EB_K*p[g]) / (tot + EB_K)) / p[g] for g in GROUP_LABELS}
    lq_df = pd.DataFrame(lqs); lq_df.index = cnt.index

    cen = centre_point(city, gdf.crs)
    cent = grid.set_index("cell_id").geometry.centroid
    dist_km = (cent.distance(cen) / 1000.0).reindex(cnt.index)

    j = j[j["cell_id"].isin(cnt.index)].copy()
    j["is_hotspot_own_group"] = [
        lq_df.loc[cid, g] >= THR for cid, g in zip(j["cell_id"], j["group"])
    ]
    j["dist_km"] = j["cell_id"].map(dist_km)
    return j, lq_df, dist_km


def access_gap(city, gdf, n_perm=999, match_distance=True, band_km=5, seed=0):
    rng = np.random.default_rng(seed)
    j, lq_df, dist_km = build_cell_table(gdf, city)
    if j is None: return None

    measures = [COL_MFAI] + list(COL_ACCESS.values())
    out = []
    bands = (dist_km // band_km).astype(int)

    for g in GROUP_LABELS.keys():
        sub = j[j["group"] == g]
        if len(sub) == 0: continue
        hot_cells = lq_df.index[lq_df[g] >= THR]
        n_hot = len(hot_cells)
        in_hot = sub[sub["cell_id"].isin(hot_cells)]
        if n_hot == 0 or len(in_hot) == 0:
            out.append({"city": city, "group": g, "n_hot_cells": 0,
                        "status": "no hotspot cells at this threshold"})
            continue

        eligible = lq_df.index
        row = {"city": city, "group": g, "n_hot_cells": n_hot,
               "n_firms_hot": len(in_hot), "n_firms_rest": len(sub) - len(in_hot),
               "dist_hot_km": round(in_hot["dist_km"].median(), 1),
               "dist_rest_km": round(sub[~sub["cell_id"].isin(hot_cells)]["dist_km"].median(), 1),
               "status": "ok"}

        # distance-band profile of the hotspot cells, for matched draws
        prof = bands.reindex(hot_cells).value_counts()

        for m in measures:
            city_sd = gdf[m].std()
            obs = in_hot[m].mean()
            rest = sub[~sub["cell_id"].isin(hot_cells)][m].mean()

            null = np.empty(n_perm)
            for k in range(n_perm):
                if match_distance:
                    picks = []
                    for b, cnt_b in prof.items():
                        pool = eligible[bands.reindex(eligible) == b]
                        if len(pool) == 0: continue
                        picks.append(rng.choice(pool, size=min(cnt_b, len(pool)),
                                                replace=False))
                    draw = np.concatenate(picks) if picks else np.array([])
                else:
                    draw = rng.choice(eligible, size=n_hot, replace=False)
                vals = sub[sub["cell_id"].isin(draw)][m]
                null[k] = vals.mean() if len(vals) else np.nan

            nm = np.nanmean(null)
            row[f"{m}_hot"]    = round(obs, 4)
            row[f"{m}_rest"]   = round(rest, 4)
            row[f"{m}_null"]   = round(nm, 4)
            row[f"{m}_d_SD"]   = round((obs - nm) / city_sd, 3) if city_sd else np.nan
            row[f"{m}_p_lo"]   = round((1 + np.nansum(null <= obs)) / (n_perm + 1), 4)
            row[f"{m}_p_hi"]   = round((1 + np.nansum(null >= obs)) / (n_perm + 1), 4)
        out.append(row)
    return pd.DataFrame(out)


# ---------- run ----------
gap = pd.concat([access_gap(c, city_data[c]) for c in CITIES], ignore_index=True)

print("=== MFAI-land: hotspot cells vs distance-matched null ===")
cols = ["city","group","n_hot_cells","n_firms_hot","dist_hot_km","dist_rest_km",
        f"{COL_MFAI}_hot", f"{COL_MFAI}_null", f"{COL_MFAI}_d_SD",
        f"{COL_MFAI}_p_lo", f"{COL_MFAI}_p_hi"]
print(gap[gap.status=="ok"][cols].to_string(index=False))

for mode, col in COL_ACCESS.items():
    print(f"\n=== {mode.upper()} access ===")
    print(gap[gap.status=="ok"][["city","group",f"{col}_hot",f"{col}_null",
                                 f"{col}_d_SD",f"{col}_p_lo",f"{col}_p_hi"]]
          .to_string(index=False))

# ---------- group mean / city mean ratios, all firms (extends the 1% result) ----------
print("\n\n=== GROUP MEAN / CITY MEAN, by mode (all firms) ===")
rows = []
for c in CITIES:
    gdf = city_data[c]; _check_cols(gdf, c)
    for g, sub in gdf.groupby("group"):
        r = {"city": c, "group": g, "n": len(sub)}
        for m in [COL_MFAI] + list(COL_ACCESS.values()):
            r[m] = round(sub[m].mean() / gdf[m].mean(), 4)
        rows.append(r)
ratios = pd.DataFrame(rows)
print(ratios.to_string(index=False))
print("\nratio ranges:")
for m in [COL_MFAI] + list(COL_ACCESS.values()):
    print(f"  {m}: {ratios[m].min():.4f} – {ratios[m].max():.4f}")


# ══════════════════════════════════════════════════════════════════════
# Validity checks — division-level ratios, mode spread, raw distances
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# THREE CHECKS ON THE "NO DIFFERENTIATION" NULL
#   1. Division-level ratios  (is G1-G8 aggregation hiding sorting?)
#   2. Per-mode spread by group (which nulls are informative vs range-restricted?)
#   3. Raw distances in km + KS tests (does the impedance transform compress it?)
# Requires: city_data, CITIES, GROUP_LABELS
# ============================================================

# ---- SET THESE ----
COL_MFAI  = "mfai_land"
COL_ACC   = {"road": "access_road", "rail": "access_rail", "icd": "access_icd"}
COL_DIST  = {"road": "dist_road_km", "rail": "dist_rail_km", "icd": "dist_icd_km"}
COL_DIV   = "isic_division"
# -------------------

def _check(gdf, cols, city):
    miss = [c for c in cols if c not in gdf.columns]
    if miss:
        raise KeyError(f"[{city}] missing {miss}\nAvailable: {list(gdf.columns)}")

MEASURES = [COL_MFAI] + list(COL_ACC.values())


# ============================================================
# CHECK 1 — division-level ratios
# ============================================================
print("="*70); print("CHECK 1: DIVISION-LEVEL GROUP/CITY RATIOS"); print("="*70)

MIN_DIV_N = 30          # divisions below this are too thin to interpret
rows = []
for c in CITIES:
    gdf = city_data[c]; _check(gdf, MEASURES + [COL_DIV], c)
    div = pd.to_numeric(gdf[COL_DIV], errors="coerce")
    for d, sub in gdf.groupby(div):
        if len(sub) < MIN_DIV_N: continue
        r = {"city": c, "division": int(d), "n": len(sub)}
        for m in MEASURES:
            r[m] = round(sub[m].mean() / gdf[m].mean(), 4)
        rows.append(r)
div_ratios = pd.DataFrame(rows)

print(f"\n{len(div_ratios)} city-division cells with n>={MIN_DIV_N}")
print("\nRatio ranges — DIVISION level (compare to group level below):")
for m in MEASURES:
    lo, hi = div_ratios[m].min(), div_ratios[m].max()
    print(f"  {m:14s} {lo:.4f} - {hi:.4f}   (spread {hi-lo:.4f})")

# group-level comparison on the same firms
rows_g = []
for c in CITIES:
    gdf = city_data[c]
    for g, sub in gdf.groupby("group"):
        r = {"city": c, "group": g, "n": len(sub)}
        for m in MEASURES:
            r[m] = round(sub[m].mean() / gdf[m].mean(), 4)
        rows_g.append(r)
grp_ratios = pd.DataFrame(rows_g)
print("\nRatio ranges — GROUP level:")
for m in MEASURES:
    lo, hi = grp_ratios[m].min(), grp_ratios[m].max()
    print(f"  {m:14s} {lo:.4f} - {hi:.4f}   (spread {hi-lo:.4f})")

print("\nMost extreme division cells (by |mfai ratio - 1|):")
tmp = div_ratios.assign(dev=(div_ratios[COL_MFAI]-1).abs())
print(tmp.nlargest(12, "dev").drop(columns="dev").to_string(index=False))

# variance decomposition: does division explain more than group?
print("\nShare of within-city variance explained (eta^2):")
for m in MEASURES:
    out = {}
    for label, key in [("group", "group"), ("division", "_div")]:
        ss_b = ss_t = 0.0
        for c in CITIES:
            gdf = city_data[c].copy()
            gdf["_div"] = pd.to_numeric(gdf[COL_DIV], errors="coerce")
            mu = gdf[m].mean()
            ss_t += ((gdf[m] - mu)**2).sum()
            for _, sub in gdf.groupby(key):
                ss_b += len(sub) * (sub[m].mean() - mu)**2
        out[label] = 100 * ss_b / ss_t
    print(f"  {m:14s} group {out['group']:.2f}%   division {out['division']:.2f}%")


# ============================================================
# CHECK 2 — spread, not just means
# ============================================================
print("\n\n" + "="*70); print("CHECK 2: PER-MODE SPREAD BY GROUP"); print("="*70)

print("\nCity-level spread (is there room to differentiate?):")
spread = []
for c in CITIES:
    gdf = city_data[c]
    r = {"city": c}
    for name, col in COL_ACC.items():
        r[f"{name}_mean"] = round(gdf[col].mean(), 3)
        r[f"{name}_SD"]   = round(gdf[col].std(), 4)
        q1, q3 = gdf[col].quantile([.25, .75])
        r[f"{name}_relIQR%"] = round(100*(q3-q1)/gdf[col].median(), 1)
    spread.append(r)
print(pd.DataFrame(spread).to_string(index=False))

print("\nGroup SD / city SD (1.0 = group as dispersed as city):")
rows = []
for c in CITIES:
    gdf = city_data[c]
    for g, sub in gdf.groupby("group"):
        r = {"city": c, "group": g, "n": len(sub)}
        for name, col in COL_ACC.items():
            r[name] = round(sub[col].std() / gdf[col].std(), 3)
        rows.append(r)
sd_ratio = pd.DataFrame(rows)
print(sd_ratio.to_string(index=False))
print("\nSD-ratio ranges:")
for name in COL_ACC:
    print(f"  {name:5s} {sd_ratio[name].min():.3f} - {sd_ratio[name].max():.3f}")


# ============================================================
# CHECK 3 — raw distances (km) and distributional tests
# ============================================================
# ---- distance columns: convert metres to km ----
COL_DIST_M = {"road": "nearest_major_road_dist_m",
              "rail": "nearest_railway_dist_m",
              "icd":  "nearest_ICD_dist_m"}

for c in CITIES:
    for name, col in COL_DIST_M.items():
        city_data[c][f"dist_{name}_km"] = city_data[c][col] / 1000.0

COL_DIST = {n: f"dist_{n}_km" for n in COL_DIST_M}

print("\n\n" + "="*70); print("CHECK 3: RAW DISTANCES AND KS TESTS"); print("="*70)

for c in CITIES:
    _check(city_data[c], list(COL_DIST.values()), c)

print("\nMedian distance (km) by group, with city median and gap:")
for c in CITIES:
    gdf = city_data[c]
    print(f"\n--- {c.upper()} ---")
    rows = []
    for g, sub in gdf.groupby("group"):
        r = {"group": g, "n": len(sub)}
        for name, col in COL_DIST.items():
            med_g, med_c = sub[col].median(), gdf[col].median()
            r[f"{name}_km"]  = round(med_g, 2)
            r[f"{name}_gap"] = round(med_g - med_c, 2)
        rows.append(r)
    t = pd.DataFrame(rows)
    for name in COL_DIST:
        t[f"{name}_km"] = t[f"{name}_km"].map(lambda v: f"{v:.2f}")
    print(t.to_string(index=False))
    print("  city medians: " + ", ".join(
        f"{n} {gdf[col].median():.2f} km" for n, col in COL_DIST.items()))

print("\n\nLargest distance gaps across all city-group cells (km):")
rows = []
for c in CITIES:
    gdf = city_data[c]
    for g, sub in gdf.groupby("group"):
        for name, col in COL_DIST.items():
            gap = sub[col].median() - gdf[col].median()
            acc_col = COL_ACC[name]
            rows.append({"city": c, "group": g, "mode": name, "n": len(sub),
                         "grp_km": round(sub[col].median(), 2),
                         "city_km": round(gdf[col].median(), 2),
                         "gap_km": round(gap, 2),
                         "acc_ratio": round(sub[acc_col].mean()/gdf[acc_col].mean(), 4)})
dist_gaps = pd.DataFrame(rows)
print(dist_gaps.assign(a=dist_gaps.gap_km.abs()).nlargest(20, "a")
      .drop(columns="a").to_string(index=False))

print("\n\nKS tests: group access distribution vs rest-of-city (Bonferroni over 120 tests)")
alpha = 0.05 / (len(CITIES) * len(GROUP_LABELS) * len(COL_ACC))
print(f"  adjusted alpha = {alpha:.5f}")
rows = []
for c in CITIES:
    gdf = city_data[c]
    for g in GROUP_LABELS:
        sub  = gdf[gdf["group"] == g]
        rest = gdf[gdf["group"] != g]
        if len(sub) < 30: continue
        for name, col in COL_ACC.items():
            st, p = ks_2samp(sub[col], rest[col])
            rows.append({"city": c, "group": g, "mode": name, "n": len(sub),
                         "KS_D": round(st, 4), "p": p,
                         "sig_bonf": p < alpha})
ks = pd.DataFrame(rows)
print(f"\n  {ks.sig_bonf.sum()} of {len(ks)} significant after Bonferroni")
print("\n  Largest KS statistics:")
print(ks.nlargest(15, "KS_D").assign(p=lambda d: d.p.map(lambda v: f"{v:.2e}"))
      .to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# Sorting statistics by city (supplementary)
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# TABLE 10a — spatial sorting, at the same resolution as 10b
#   entropy permutation test + permuted Duncan's D, per city
# Requires: city_data, GROUP_LABELS, build_grid, clip_builtup
# ============================================================

ORDER    = ["patna", "varanasi", "indore", "kochi", "kolkata"]
CELL_FOR = {c: (1000 if c == "kolkata" else 2000) for c in ORDER}
MIN_ENT, N_PERM = 10, 999


def sorting_stats(gdf, city, min_ent=MIN_ENT, n_perm=N_PERM, seed=0):
    rng  = np.random.default_rng(seed)
    grid = clip_builtup(build_grid(gdf, CELL_FOR[city]), city)
    j    = gpd.sjoin(gdf[["group", "geometry"]], grid, how="inner", predicate="within")
    if len(j) == 0:
        return None

    groups = list(GROUP_LABELS.keys())
    gi     = {g: i for i, g in enumerate(groups)}
    lab    = np.array([gi[g] for g in j["group"].values])
    _, inv = np.unique(j["cell_id"].values, return_inverse=True)
    nc, ng = inv.max() + 1, len(groups)

    def counts(l):
        M = np.zeros((nc, ng)); np.add.at(M, (inv, l), 1)
        return M

    def med_entropy(l):
        M   = counts(l); tot = M.sum(1); keep = tot >= min_ent
        if keep.sum() == 0: return np.nan
        P = M[keep] / tot[keep][:, None]
        H = -(P * np.where(P > 0, np.log(P), 0)).sum(1) / np.log(ng)
        return np.median(H)

    def D_matrix(l):
        M = counts(l); cs = M.sum(0); cs[cs == 0] = 1
        S = M / cs
        return 0.5 * np.abs(S[:, :, None] - S[:, None, :]).sum(0)

    obs_H, obs_D = med_entropy(lab), D_matrix(lab)
    null_H = np.empty(n_perm)
    null_D = np.empty((n_perm, ng, ng))
    for k in range(n_perm):
        pl = rng.permutation(lab)
        null_H[k] = med_entropy(pl)
        null_D[k] = D_matrix(pl)

    off   = ~np.eye(ng, dtype=bool)
    p_hiD = (1 + (null_D >= obs_D).sum(0)) / (n_perm + 1)
    n_sig = int(((p_hiD < 0.05) & off).sum() // 2)

    return {
        "City": city.capitalize(),
        "Cell (m)": CELL_FOR[city],
        "Obs. entropy": round(float(obs_H), 3),
        "Null entropy": round(float(np.nanmean(null_H)), 3),
        "Null 95% band": f"{np.nanpercentile(null_H,2.5):.3f}–{np.nanpercentile(null_H,97.5):.3f}",
        "p (entropy)": round((1 + np.nansum(null_H <= obs_H)) / (n_perm + 1), 3),
        "Obs. D": round(float(obs_D[off].mean()), 3),
        "Null D": round(float(null_D.mean(0)[off].mean()), 3),
        "Excess D": round(float(obs_D[off].mean() - null_D.mean(0)[off].mean()), 3),
        "Sig. pairs": f"{n_sig}/28",
    }


rows = [sorting_stats(city_data[c], c) for c in ORDER]
tab10a = pd.DataFrame([r for r in rows if r])
print(tab10a.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# Table 11 — concentrations and freight access, consistent denominator
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# TABLE 10b — recomputed on a single consistent denominator
#   share  = firms of group g in LQ>=2.0 cells / firms of g in all retained cells
#   access = same LQ>=2.0 cells vs distance-matched null
# Requires: city_data, GROUP_LABELS, build_grid, clip_builtup
# ============================================================

ORDER    = ["patna", "varanasi", "indore", "kochi", "kolkata"]
CELL_FOR = {c: (1000 if c == "kolkata" else 2000) for c in ORDER}
GROUPS_IN_TABLE = {
    "patna":    ["G2", "G5", "G6"],
    "varanasi": ["G2", "G5", "G6"],
    "indore":   ["G1", "G5", "G6"],
    "kochi":    ["G2", "G5", "G6", "G7"],
    "kolkata":  ["G2", "G5", "G6"],
}
CENTRES = {"patna": (25.6127, 85.1416), "indore": (22.7177, 75.8545),
           "varanasi": (25.3109, 83.0104), "kochi": (9.9816, 76.2830),
           "kolkata": (22.5697, 88.3497)}

COL_MFAI = "mfai_land"
COL_ROAD = "access_road"
MIN_FIRMS, EB_K, THR = 20, 20, 2.0
N_PERM, BAND_KM = 999, 5


def cell_table(gdf, city):
    """Retained cells with LQ per group, firm counts per group, and centroid distance."""
    grid = clip_builtup(build_grid(gdf, CELL_FOR[city]), city)
    j = gpd.sjoin(gdf[["group", COL_MFAI, COL_ROAD, "geometry"]], grid,
                  how="inner", predicate="within")
    cnt = j.groupby(["cell_id", "group"]).size().unstack(fill_value=0)
    for g in GROUP_LABELS:
        if g not in cnt.columns: cnt[g] = 0
    cnt = cnt[list(GROUP_LABELS.keys())]
    tot = cnt.sum(1); p = cnt.sum(0) / cnt.sum(0).sum()

    keep = tot >= MIN_FIRMS
    cnt, tot = cnt[keep], tot[keep]

    out = pd.DataFrame({"cell_id": tot.index, "n_firms": tot.values})
    for g in GROUP_LABELS:
        out[f"LQ_{g}"] = (((cnt[g] + EB_K*p[g]) / (tot + EB_K)) / p[g]).values
        out[f"n_{g}"]  = cnt[g].values
    cells = grid.merge(out, on="cell_id")

    lat, lon = CENTRES[city]
    cen = (gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
           .to_crs(cells.crs).iloc[0])
    cells["dist_km"] = cells.geometry.centroid.distance(cen) / 1000.0

    j = j[j["cell_id"].isin(cells["cell_id"])].copy()
    return cells, j


def access_null(cells, j, g, measure, n_perm=N_PERM, seed=0):
    """Observed mean in LQ>=2 cells vs cells matched on distance band."""
    rng = np.random.default_rng(seed)
    hot_ids = cells.loc[cells[f"LQ_{g}"] >= THR, "cell_id"]
    sub = j[j["group"] == g]
    in_hot = sub[sub["cell_id"].isin(hot_ids)]
    if len(hot_ids) == 0 or len(in_hot) == 0:
        return None

    obs = in_hot[measure].mean()
    sd  = j[measure].std()
    bands = (cells.set_index("cell_id")["dist_km"] // BAND_KM).astype(int)
    prof  = bands.reindex(hot_ids).value_counts()
    pool_by_band = {b: cells.loc[bands.reindex(cells.cell_id).values == b, "cell_id"].values
                    for b in prof.index}

    null = np.empty(n_perm)
    for k in range(n_perm):
        picks = [rng.choice(pool_by_band[b], size=min(n, len(pool_by_band[b])),
                            replace=False)
                 for b, n in prof.items() if len(pool_by_band[b])]
        draw = np.concatenate(picks) if picks else np.array([])
        vals = sub[sub["cell_id"].isin(draw)][measure]
        null[k] = vals.mean() if len(vals) else np.nan

    nm = np.nanmean(null)
    d  = (obs - nm) / sd if sd else np.nan
    p_lo = (1 + np.nansum(null <= obs)) / (n_perm + 1)
    p_hi = (1 + np.nansum(null >= obs)) / (n_perm + 1)
    return dict(obs=obs, null=nm, d=d, p=(p_lo if d < 0 else p_hi))


rows = []
for c in ORDER:
    cells, j = cell_table(city_data[c], c)
    for g in GROUPS_IN_TABLE[c]:
        hot = cells[cells[f"LQ_{g}"] >= THR]
        n_conc = int(hot[f"n_{g}"].sum())
        n_tot  = int(cells[f"n_{g}"].sum())
        r = {"City": c.capitalize(), "Group": g,
             "Cells": len(hot), "n_conc": n_conc, "n_total": n_tot,
             "Share %": round(100*n_conc/n_tot, 1) if n_tot else np.nan}
        a = access_null(cells, j, g, COL_MFAI)
        if a:
            r.update({"MFAI conc.": round(a["obs"], 3),
                      "Matched null": round(a["null"], 3),
                      "d (SD)": round(a["d"], 2),
                      "p (access)": round(a["p"], 3)})
        if c == "indore" and g == "G1":          # footnote value
            ar = access_null(cells, j, g, COL_ROAD)
            if ar:
                print(f"\nFOOTNOTE  Indore G1 road: obs {ar['obs']:.3f}  "
                      f"null {ar['null']:.3f}  d {ar['d']:+.2f} SD  p {ar['p']:.3f}")
        rows.append(r)

tab = pd.DataFrame(rows)
print("\n" + tab.to_string(index=False))
print("\nreconciliation check — share should equal n_conc/n_total:")
print((100*tab.n_conc/tab.n_total).round(1).equals(tab["Share %"]))
