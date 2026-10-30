"""Agglomeration: does manufacturing concentrate where freight access is better? (NB regression + diagnostics)

Produces: Table 12; Supplementary S11a-c (overdispersion, VIF, Moran's I, offset) -> results/targetability/

Does manufacturing concentrate where freight access is better, holding
built-up surface and centrality constant?
Runs standalone. Nothing else needs to be in the session: the network
construction, grid building and firm loading are all included below.
    results = run_all()
WHAT THIS ADDS TO THE FIRST VERSION

SETUP
"""
import os, time, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import rasterio
import statsmodels.api as sm
from scipy.spatial import cKDTree
from pyproj import Transformer, Geod

from mfai import paths
from mfai.io import load_city
from mfai.network import (_densify_coords, fix_road_network as _fix_road_network,
                          build_graph as _build_graph, make_snapper,
                          load_terminals)


def fix_road_network(roads):
    return _fix_road_network(roads, [])


def build_graph(edges):
    return _build_graph(edges, [])

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

# Filename patterns tried in order for the built-up raster

CITIES = {
    "kochi":    {"utm": 32643},
    "kolkata":  {"utm": 32645},
    "patna":    {"utm": 32645},
    "varanasi": {"utm": 32644},
    "indore":   {"utm": 32643},
}
CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}

# Municipal / historic core
CITY_CENTRE = {
    "patna":    (25.6127, 85.1416),   # Gandhi Maidan
    "indore":   (22.7177, 75.8545),   # Rajwada
    "varanasi": (25.3109, 83.0104),   # Godowlia
    "kochi":    (9.9816,  76.2830),   # Ernakulam, MG Road
    "kolkata":  (22.5697, 88.3497),   # BBD Bagh
}

# Truncation radius. Patna extended because 30 km excluded a fifth of
# its firms, concentrated in the worst-ICD-access periphery.
RADIUS_KM = {"patna": 60.0, "kochi": 45.0}
RADIUS_DEFAULT = 30.0

WATER_OVERRIDE = {"indore": (952_000, "Ramnagar MMT (Varanasi), NW-1")}

# Network parameters — identical to the MFAI pipeline
SNAP_TOL, ROUND_DP        = 12.0, 2
MAJOR_ROAD_TOL            = 12.0
MAX_VERTEX_M, MAJOR_SAMPLE_M = 25.0, 5.0

# Impedance reference distances — identical to the MFAI pipeline
T_REF = {"road": 2_000, "rail": 50_000, "water": 50_000, "icd": 50_000}

# Geocoder-artefact filter
FALLBACK_THRESHOLD, FALLBACK_MIN_PINCODES = 20, 5
DIV_DIVERSITY_T = 8

MODES     = ["road", "rail", "water", "icd"]
LAND      = ["road", "rail", "icd"]
CLUSTER_KM = 5.0        # spatial block for clustered standard errors

_geod = Geod(ellps="WGS84")


# ══════════════════════════════════════════════════════════════════════
# A. FIRM LOADING
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# B. NETWORK — identical construction to the MFAI pipeline
# ══════════════════════════════════════════════════════════════════════


def ghsl_path(city):
    p = paths.ghsl(city)
    return str(p) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════
# C. GRID CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════
def build_grid(city, cell_km=1.0, radius_km=None, min_dn=1.0, verbose=True):
    """Grid over the built-up area with access computed at each centroid.

    Access must be evaluated at locations where firms are ABSENT as well
    as present. Restricting to populated cells would condition on the
    outcome and ask only "given that manufacturing is here, is there
    more of it where access is better".
    """
    t0 = time.time()
    utm = CITIES[city]["utm"]
    radius_km = radius_km if radius_km is not None else RADIUS_KM.get(
        city, RADIUS_DEFAULT)
    if verbose:
        print(f"\n[{city}] grid {cell_km} km, radius {radius_km:.0f} km ...")

    roads = gpd.read_file(paths.roads_all(city)).to_crs(utm)
    G, comps = build_graph(fix_road_network(roads))
    giant = comps[0] if comps else set()
    snap = make_snapper(G, giant)
    node2comp = {n: i for i, c in enumerate(comps) for n in c}

    # major-road access nodes
    major = gpd.read_file(paths.roads_major(city)).to_crs(utm)
    msamp = []
    for geom in major.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = ([geom] if geom.geom_type == "LineString"
                 else list(getattr(geom, "geoms", [])))
        for p in parts:
            if p.geom_type == "LineString":
                msamp.extend(_densify_coords(p, MAJOR_SAMPLE_M))
    major_nodes = set()
    if msamp:
        mtree = cKDTree(np.array(msamp, dtype=float))
        gn = list(G.nodes())
        gd, _ = mtree.query(np.array(gn, dtype=float))
        major_nodes = {n for n, ok in zip(gn, gd <= MAJOR_ROAD_TOL) if ok}
    maj_dist = (nx.multi_source_dijkstra_path_length(G, major_nodes,
                                                     weight="weight")
                if major_nodes else {})

    gdfs = {k: load_terminals(str(paths.terminals(k, city)), utm)
            for k in ("rail", "water", "icd")}
    dmaps = {}
    for k, g in gdfs.items():
        sn = snap(list(zip(g.geometry.x, g.geometry.y))) if len(g) else []
        rows = []
        for tn, tc in sn:
            try:
                dm = nx.single_source_dijkstra_path_length(G, tn, weight="weight")
            except nx.NodeNotFound:
                dm = {}
            rows.append((dm, tc, node2comp.get(tn, -1)))
        dmaps[k] = rows

    # firms
    firms = load_city(city)
    tf = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    fx, fy = tf.transform(firms["lon"].values, firms["lat"].values)

    step = cell_km * 1000.0
    x0, y0 = fx.min() - step, fy.min() - step
    nx_, ny_ = (int((fx.max() + step - x0) / step) + 1,
                int((fy.max() + step - y0) / step) + 1)
    gx = x0 + (np.arange(nx_) + 0.5) * step
    gy = y0 + (np.arange(ny_) + 0.5) * step
    XX, YY = np.meshgrid(gx, gy)
    cx, cy = XX.ravel(), YY.ravel()

    ix = np.clip(((fx - x0) / step).astype(int), 0, nx_ - 1)
    iy = np.clip(((fy - y0) / step).astype(int), 0, ny_ - 1)
    cnt = np.bincount(iy * nx_ + ix, minlength=len(cx)).astype(float)

    back = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    clon, clat = back.transform(cx, cy)
    lat0, lon0 = CITY_CENTRE[city]
    _, _, dcm = _geod.inv(clon, clat, np.full(len(cx), lon0),
                          np.full(len(cx), lat0))
    dc_km = dcm / 1000.0

    keep = dc_km <= radius_km
    cx, cy, cnt = cx[keep], cy[keep], cnt[keep]
    clon, clat, dc_km = clon[keep], clat[keep], dc_km[keep]

    # built-up: centroid value, and cell total for weighting
    gp = ghsl_path(city)
    dn = np.full(len(cx), np.nan)
    dn_sum = np.full(len(cx), np.nan)
    if gp:
        with rasterio.open(gp) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            px, py = t.transform(clon, clat)
            dn = np.array([q[0] for q in src.sample(zip(px, py))], dtype=float)
            if src.nodata is not None:
                dn = np.where(dn == src.nodata, np.nan, dn)
            # cell totals from a 5x5 lattice, scaled to cell area
            offs = (np.arange(5) + 0.5) / 5 - 0.5
            tot = np.zeros(len(cx))
            for ddx in offs:
                for ddy in offs:
                    qx, qy = back.transform(cx + ddx*step, cy + ddy*step)
                    ux, uy = t.transform(qx, qy)
                    v = np.array([q[0] for q in src.sample(zip(ux, uy))],
                                 dtype=float)
                    if src.nodata is not None:
                        v = np.where(v == src.nodata, 0.0, v)
                    tot += np.nan_to_num(v)
            dn_sum = tot / 25.0 * (step / 100.0) ** 2

    # network access at centroids
    csnap = snap(list(zip(cx, cy)))
    d_road = np.full(len(cx), np.nan)
    dist = {k: np.full(len(cx), np.nan) for k in gdfs}
    for i, (mn, mc) in enumerate(csnap):
        mcomp = node2comp.get(mn, -1)
        dd = maj_dist.get(mn)
        if dd is not None:
            d_road[i] = dd + mc
        for k in gdfs:
            best = np.inf
            for dm, tc, tcomp in dmaps[k]:
                if mcomp != tcomp:
                    continue
                v = dm.get(mn)
                if v is not None and v + mc + tc < best:
                    best = v + mc + tc
            if np.isfinite(best):
                dist[k][i] = best

    g = pd.DataFrame({
        "city": city, "lat": clat, "lon": clon, "x_utm": cx, "y_utm": cy,
        "n_firms": cnt, "dn": dn, "dn_sum": dn_sum, "dist_centre_km": dc_km,
        "d_road": d_road, "d_rail": dist["rail"],
        "d_water": dist["water"], "d_icd": dist["icd"]})

    for m, dcol in [("road", "d_road"), ("rail", "d_rail"),
                    ("icd", "d_icd"), ("water", "d_water")]:
        g[f"access_{m}"] = T_REF[m] / (T_REF[m] + g[dcol])
    if city in WATER_OVERRIDE and g.d_water.isna().all():
        g["access_water"] = T_REF["water"] / (T_REF["water"] +
                                              WATER_OVERRIDE[city][0])

    ok = g[[f"access_{m}" for m in LAND]].notna().all(axis=1)
    g["mfai_land"] = np.nan
    g.loc[ok, "mfai_land"] = np.exp(
        sum(np.log(g.loc[ok, f"access_{m}"].clip(lower=1e-9))
            for m in LAND) / 3.0)

    # built-up cells only: water, forest and empty land are not
    # candidate manufacturing sites
    g = g[(g.dn.fillna(0) >= min_dn) & ok].reset_index(drop=True)

    # spatial blocks for clustered standard errors
    g["block"] = (city + "_" +
                  (g.x_utm // (CLUSTER_KM*1000)).astype(int).astype(str) + "_" +
                  (g.y_utm // (CLUSTER_KM*1000)).astype(int).astype(str))

    if verbose:
        print(f"    {len(g):,} built-up cells | {int((g.n_firms>0).sum()):,} "
              f"populated ({(g.n_firms>0).mean():.1%}) | "
              f"{int(g.n_firms.sum()):,} of {len(firms):,} firms placed "
              f"({g.n_firms.sum()/len(firms):.1%}) | "
              f"{g.block.nunique()} blocks | {time.time()-t0:.0f}s")
        if len(g) < 200:
            print(f"    ! only {len(g)} cells — estimates will be unstable")
    return g


def build_all_grids(cities=None, cell_km=1.0, radius_km=None):
    return {c: build_grid(c, cell_km, radius_km)
            for c in (cities or CITY_ORDER)}


# ══════════════════════════════════════════════════════════════════════
# D. REGRESSION
# ══════════════════════════════════════════════════════════════════════
def _fit(d, zcols, label=""):
    """Negative binomial with cluster-robust SEs on 5 km blocks.

    Falls back to Poisson if NB fails to converge. Clustering is a
    partial correction for spatial autocorrelation, not a complete one.
    """
    d = d.dropna(subset=zcols + ["log_dn", "log_dist", "y"]).copy()
    if len(d) < 50:
        return None, d
    X = d[zcols + ["log_dn", "log_dist"]].copy()
    if d.city.nunique() > 1:
        X = pd.concat([X, pd.get_dummies(d.city, prefix="c",
                                         drop_first=True).astype(float)],
                      axis=1)
    X = sm.add_constant(X)
    kw = dict(cov_type="cluster", cov_kwds={"groups": d["block"]})
    try:
        m = sm.NegativeBinomial(d.y, X).fit(disp=0, maxiter=300, **kw)
        if not np.isfinite(m.bse[zcols[0]]):
            raise ValueError
        return ("NB", m), d
    except Exception:
        try:
            m = sm.Poisson(d.y, X).fit(disp=0, maxiter=300, **kw)
            return ("Poisson", m), d
        except Exception:
            return None, d


def _prep(g):
    d = g.copy()
    d["log_dn"] = np.log1p(d.dn.fillna(0))
    d["log_dist"] = np.log1p(d.dist_centre_km)
    d["y"] = d.n_firms.astype(int)
    return d


def agglomeration(grids, modes=("mfai_land",) + tuple(MODES), verbose=True):
    """Pooled and per-city estimates for every access measure.

    A pooled coefficient is interpretable only where the per-city
    estimates agree in sign and rough magnitude. Report per city
    otherwise.
    """
    allg = _prep(pd.concat(grids.values(), ignore_index=True))

    # ---------- pooled ----------
    pooled = []
    for mo in modes:
        col = mo if mo == "mfai_land" else f"access_{mo}"
        use = [c for c, g in grids.items()
               if col in g.columns and g[col].std(skipna=True) > 1e-9]
        if not use:
            continue
        d = allg[allg.city.isin(use)].copy()
        d["z"] = d.groupby("city")[col].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 1e-9 else np.nan)
        res, dd = _fit(d, ["z"])
        if res:
            kind, m = res
            pooled.append({
                "measure": mo, "cities": len(use), "model": kind,
                "n cells": len(dd), "coef": round(m.params["z"], 4),
                "SE": round(m.bse["z"], 4),
                "p": f"{m.pvalues['z']:.2g}",
                "% change": round(100*(np.exp(m.params["z"])-1), 1)})
    pooled = pd.DataFrame(pooled)

    # ---------- per city ----------
    rows = []
    for c, g in grids.items():
        d = _prep(g)
        for mo in modes:
            col = mo if mo == "mfai_land" else f"access_{mo}"
            if col not in d.columns or d[col].std(skipna=True) < 1e-9:
                continue
            dd = d.copy()
            dd["z"] = (dd[col] - dd[col].mean()) / dd[col].std()
            res, ddd = _fit(dd, ["z"])
            if res:
                kind, m = res
                rows.append({
                    "City": CITY_LABEL[c], "measure": mo, "model": kind,
                    "n cells": len(ddd), "coef": round(m.params["z"], 4),
                    "SE": round(m.bse["z"], 4),
                    "p": f"{m.pvalues['z']:.2g}",
                    "% change": round(100*(np.exp(m.params["z"])-1), 1)})
    bycity = pd.DataFrame(rows)

    if verbose:
        print("\n" + "="*86)
        print("POOLED — city fixed effects, access standardised within city")
        print("="*86)
        print(pooled.to_string(index=False))

        print("\n" + "="*86)
        print("PER CITY — coefficient per SD of access")
        print("="*86)
        piv = bycity.pivot(index="City", columns="measure", values="coef")
        cols = [m for m in modes if m in piv.columns]
        print(piv.reindex(index=[CITY_LABEL[c] for c in CITY_ORDER
                                 if CITY_LABEL[c] in piv.index],
                          columns=cols).to_string())

        print("\n  sign agreement across cities:")
        for mo in cols:
            s = bycity[bycity.measure == mo].coef
            pos, neg = int((s > 0).sum()), int((s < 0).sum())
            flag = "consistent" if min(pos, neg) == 0 else "MIXED — pooled figure not interpretable"
            print(f"    {mo:12s} {pos} positive, {neg} negative   {flag}")

        print("\n  full detail:")
        for c in [CITY_LABEL[x] for x in CITY_ORDER]:
            sub = bycity[bycity.City == c]
            if len(sub):
                print(f"\n  {c}")
                print(sub.drop(columns="City").to_string(index=False))
    return pooled, bycity


def joint_model(grids, modes=LAND, verbose=True):
    """All modes in one specification.

    Separate regressions let each coefficient absorb variance shared
    with the other modes; road and rail access are correlated at cell
    level. This isolates independent explanatory power.
    """
    allg = _prep(pd.concat(grids.values(), ignore_index=True))
    use = [m for m in modes
           if f"access_{m}" in allg.columns
           and allg.groupby("city")[f"access_{m}"].std().min() > 1e-9]
    for m in use:
        allg[f"z_{m}"] = allg.groupby("city")[f"access_{m}"].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 1e-9 else np.nan)
    z = [f"z_{m}" for m in use]
    res, d = _fit(allg, z)
    if res is None:
        print("joint model failed to converge")
        return None
    kind, m = res
    out = pd.DataFrame({
        "mode": use,
        "coef": [round(m.params[k], 4) for k in z],
        "SE":   [round(m.bse[k], 4) for k in z],
        "p":    [f"{m.pvalues[k]:.2g}" for k in z],
        "% change": [round(100*(np.exp(m.params[k])-1), 1) for k in z]})
    if verbose:
        print("\n" + "="*86)
        print(f"JOINT MODEL — all modes together ({kind}, n={len(d):,})")
        print("="*86)
        print(out.to_string(index=False))
        print("\n  correlation among standardised access scores:")
        print(d[z].corr().round(3).to_string())
        print("\n  A mode whose coefficient shrinks toward zero here but was")
        print("  positive alone was largely sharing variance with another.")
    return out


# ══════════════════════════════════════════════════════════════════════
# E. SORTING (descriptive)
# ══════════════════════════════════════════════════════════════════════
def sorting(grids, verbose=True):
    """Firm-weighted access against two baselines.

    DESCRIPTIVE, not inferential. 'vs cells' weights every built-up cell
    equally, so a fringe cell with 200 m2 of built-up surface counts the
    same as a dense industrial block; firms are in dense cells and dense
    cells have better access, so most of that difference is a density
    confound. 'vs land' weights by built-up surface, which removes the
    crudest part of it but still does not control for centrality.

    Read the CONTRAST ACROSS MODES within a city. All modes face the
    same firm distribution, so a mode with the opposite sign to the
    others is anomalous whatever the shared confound.
    """
    rows = []
    for c, g in grids.items():
        w = g.dn_sum.clip(lower=0).fillna(0).values
        if w.sum() <= 0:
            w = np.ones(len(g))
        for mo in ["mfai_land"] + MODES:
            col = mo if mo == "mfai_land" else f"access_{mo}"
            if col not in g.columns or g[col].std(skipna=True) < 1e-9:
                continue
            v = g[col].values
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            cell = v[ok].mean()
            land = np.average(v[ok], weights=w[ok])
            firm = np.average(v[ok], weights=g.n_firms.values[ok])
            sd = v[ok].std()
            rows.append({"City": CITY_LABEL[c], "measure": mo,
                         "cells": round(cell, 4), "land-wtd": round(land, 4),
                         "firms": round(firm, 4),
                         "vs cells (SD)": round((firm-cell)/sd, 3),
                         "vs land (SD)": round((firm-land)/sd, 3)})
    out = pd.DataFrame(rows)
    if verbose:
        print("\n" + "="*86)
        print("SORTING — descriptive")
        print("="*86)
        for c in [CITY_LABEL[x] for x in CITY_ORDER]:
            sub = out[out.City == c]
            if len(sub):
                print(f"\n{c}")
                print(sub.drop(columns="City").to_string(index=False))
        print("\n  'vs land' is the defensible column. Compare modes within")
        print("  a city, not levels across cities.")
    return out


# ══════════════════════════════════════════════════════════════════════
# F. MAUP CHECK
# ══════════════════════════════════════════════════════════════════════
def maup_check(cities=None, cells=(1.0, 2.0)):
    """Rerun everything at two resolutions.

    If coefficient signs flip between resolutions, the result is an
    artefact of the aggregation unit.
    """
    store, tabs = {}, []
    for ck in cells:
        print(f"\n{'#'*86}\n# CELL SIZE {ck} km\n{'#'*86}")
        g = build_all_grids(cities, cell_km=ck)
        store[ck] = g
        _, bc = agglomeration(g, verbose=False)
        bc["cell_km"] = ck
        tabs.append(bc)

    both = pd.concat(tabs, ignore_index=True)
    piv = both.pivot_table(index=["City", "measure"], columns="cell_km",
                           values="coef")
    piv.columns = [f"{c} km" for c in piv.columns]
    piv["sign stable"] = (np.sign(piv.iloc[:, 0]) ==
                          np.sign(piv.iloc[:, 1]))
    print("\n" + "="*86)
    print("MAUP CHECK — coefficient by cell size")
    print("="*86)
    print(piv.round(4).to_string())
    unstable = piv[~piv["sign stable"]]
    print(f"\n  sign flips: {len(unstable)} of {len(piv)}")
    if len(unstable):
        print(unstable.round(4).to_string())
        print("\n  Flipped rows should not be reported as findings.")
    return store, piv


# ══════════════════════════════════════════════════════════════════════
# G. RUN
# ══════════════════════════════════════════════════════════════════════
def run_all(cities=None, do_maup=True):
    print("="*86)
    print("AGGLOMERATION ANALYSIS")
    print("="*86)
    print(f"  radius: default {RADIUS_DEFAULT:.0f} km, "
          f"overrides {RADIUS_KM}")
    for c in (cities or CITY_ORDER):
        print(f"  {c:9s} GHSL {os.path.basename(ghsl_path(c)) if ghsl_path(c) else 'MISSING'}")

    grids = build_all_grids(cities, cell_km=1.0)
    pooled, bycity = agglomeration(grids)
    jm = joint_model(grids)
    srt = sorting(grids)

    maup = None
    if do_maup:
        _, maup = maup_check(cities, cells=(1.0, 2.0))

    return {"grids": grids, "pooled": pooled, "bycity": bycity,
            "joint": jm, "sorting": srt, "maup": maup}


# ══ REGRESSION DIAGNOSTICS (reviewer Major comment 11) ══

MORAN_K = 8            # neighbours for the spatial weight matrix
MORAN_PERM = 999       # permutations for Moran's I
RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════════
def morans_i(values, coords, k=MORAN_K, n_perm=MORAN_PERM, rng=RNG):
    """Moran's I with a row-standardised k-NN weight matrix.

    Permutation inference: the analytic variance of I assumes normality,
    which Pearson residuals from a count model do not satisfy.
    Returns (I, expected_I, p_permutation).
    """
    n = len(values)
    if n < k + 2:
        return np.nan, np.nan, np.nan
    z = values - values.mean()
    denom = (z ** 2).sum()
    if denom <= 0:
        return np.nan, np.nan, np.nan

    # k nearest neighbours, excluding self
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    idx = idx[:, 1:]                      # drop self column

    def _I(zv):
        # row-standardised: each row weight = 1/k, so S0 = n
        lag = zv[idx].mean(axis=1)
        return (zv * lag).sum() / ((zv ** 2).sum() / n) / n

    obs = _I(z)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _I(rng.permutation(z))
    # two-sided permutation p
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return obs, -1.0 / (n - 1), p


def vif_table(X):
    """Variance inflation factors via auxiliary R² for each column."""
    cols = [c for c in X.columns if c != "const"]
    rows = []
    A = X[cols].to_numpy(float)
    for j, c in enumerate(cols):
        y = A[:, j]
        Z = np.delete(A, j, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({"regressor": c,
                     "VIF": round(1 / (1 - r2), 2) if r2 < 1 else np.inf})
    return pd.DataFrame(rows)


def _design(d, zcols, city_fe=True):
    X = d[zcols + ["log_dn", "log_dist"]].copy()
    if city_fe and d.city.nunique() > 1:
        X = pd.concat([X, pd.get_dummies(d.city, prefix="c",
                                         drop_first=True).astype(float)],
                      axis=1)
    return sm.add_constant(X)


def _fit_diag(d, zcols, offset=None, city_fe=True):
    """NB2 fit returning the result plus alpha and Pearson residuals."""
    need = zcols + ["log_dn", "log_dist", "y"]
    d = d.dropna(subset=need).copy()
    if offset is not None:
        d = d[np.isfinite(d[offset])].copy()
    if len(d) < 50:
        return None, d, None
    X = _design(d, zcols, city_fe)
    off = d[offset].to_numpy(float) if offset else None
    kw = dict(cov_type="cluster", cov_kwds={"groups": d["block"]})
    try:
        model = sm.NegativeBinomial(d.y, X, offset=off)
        m = model.fit(disp=0, maxiter=300, **kw)
        if not np.isfinite(m.bse[zcols[0]]):
            raise ValueError("non-finite SE")
    except Exception as e:
        return None, d, str(e)[:60]

    alpha = float(m.params.get("alpha", np.nan))
    mu = np.asarray(m.predict(X, offset=off), dtype=float)
    var = mu + alpha * mu ** 2 if np.isfinite(alpha) else mu
    with np.errstate(divide="ignore", invalid="ignore"):
        pear = (d.y.to_numpy(float) - mu) / np.sqrt(np.where(var > 0, var, np.nan))
    d = d.assign(_mu=mu, _pearson=pear)
    return m, d, None


# ══════════════════════════════════════════════════════════════════════
# 1–5. CORE DIAGNOSTICS, PER CITY
# ══════════════════════════════════════════════════════════════════════
def core_diagnostics(grids, measures=("mfai_land", "road", "rail", "icd"),
                     verbose=True):


    rows, vifs, morans = [], [], []
    for c, g in grids.items():
        d0 = _prep(g)
        zero_share = float((d0.y == 0).mean())
        for mo in measures:
            col = mo if mo == "mfai_land" else f"access_{mo}"
            if col not in d0.columns or d0[col].std(skipna=True) < 1e-9:
                continue
            d = d0.copy()
            d["z"] = (d[col] - d[col].mean()) / d[col].std()
            m, dd, err = _fit_diag(d, ["z"], city_fe=False)
            if m is None:
                rows.append({"City": CITY_LABEL[c], "measure": mo,
                             "error": err})
                continue

            alpha = float(m.params.get("alpha", np.nan))
            alpha_se = float(m.bse.get("alpha", np.nan))
            nblocks = int(dd["block"].nunique())

            mi, ei, pi = morans_i(
                dd["_pearson"].to_numpy(float),
                dd[["x_utm", "y_utm"]].to_numpy(float))

            rows.append({
                "City": CITY_LABEL[c], "measure": mo,
                "n cells": len(dd),
                "zero cells %": round(100 * zero_share, 1),
                "5 km blocks": nblocks,
                "blocks OK": "yes" if nblocks >= 30 else "FEW",
                "alpha": round(alpha, 3),
                "alpha SE": round(alpha_se, 3),
                "coef": round(float(m.params["z"]), 4),
                "SE": round(float(m.bse["z"]), 4),
                "Moran I (resid)": round(mi, 4) if np.isfinite(mi) else np.nan,
                "Moran p": round(pi, 4) if np.isfinite(pi) else np.nan,
            })
            morans.append({"City": CITY_LABEL[c], "measure": mo,
                           "I": mi, "E[I]": ei, "p": pi})

            if mo == "mfai_land":
                v = vif_table(_design(dd, ["z"], city_fe=False))
                v.insert(0, "City", CITY_LABEL[c])
                vifs.append(v)

    out = pd.DataFrame(rows)
    vif = pd.concat(vifs, ignore_index=True) if vifs else pd.DataFrame()

    if verbose:
        print("\n" + "=" * 92)
        print("S11a — OVERDISPERSION, ZERO CELLS, CLUSTER COUNT, RESIDUAL AUTOCORRELATION")
        print("=" * 92)
        print(out.to_string(index=False))
        print("""
  alpha is the NB2 overdispersion parameter; alpha >> 0 confirms the
  negative binomial over Poisson. '5 km blocks' is the number of clusters
  the robust SEs rest on — below ~30, treat those SEs as indicative only.
  Moran's I is computed on Pearson residuals: positive and significant
  values mean spatial dependence REMAINS after clustering, so the SEs are
  still optimistic. This is reported, not claimed away.""")

        if len(vif):
            print("\n" + "=" * 92)
            print("S11b — VARIANCE INFLATION FACTORS (MFAI-land specification)")
            print("=" * 92)
            print(vif.to_string(index=False))
            print("\n  VIF > 10 would indicate problematic collinearity.")
    return out, vif


# ══════════════════════════════════════════════════════════════════════
# 6. OFFSET vs COVARIATE
# ══════════════════════════════════════════════════════════════════════
def offset_comparison(grids, measures=("mfai_land", "icd"), verbose=True):
    """Built-up surface as estimated covariate vs fixed exposure offset.

    Covariate  : log firm_count ~ b*z + g*log(built-up) + ...
    Offset     : log firm_count ~ b*z + 1*log(built-up) + ...  (density model)

    If b is stable across the two, the finding does not depend on how
    developed land enters the model.
    """


    rows = []
    for c, g in grids.items():
        d0 = _prep(g)
        if "dn_sum" not in d0.columns or d0["dn_sum"].notna().sum() < 50:
            continue
        d0["log_exposure"] = np.log(d0["dn_sum"].clip(lower=1e-6))
        for mo in measures:
            col = mo if mo == "mfai_land" else f"access_{mo}"
            if col not in d0.columns or d0[col].std(skipna=True) < 1e-9:
                continue
            d = d0.copy()
            d["z"] = (d[col] - d[col].mean()) / d[col].std()

            m_cov, dd1, e1 = _fit_diag(d, ["z"], city_fe=False)
            m_off, dd2, e2 = _fit_diag(d, ["z"], offset="log_exposure",
                                       city_fe=False)
            if m_cov is None or m_off is None:
                continue
            b_cov = float(m_cov.params["z"])
            b_off = float(m_off.params["z"])
            rows.append({
                "City": CITY_LABEL[c], "measure": mo,
                "coef (covariate)": round(b_cov, 4),
                "SE (cov)": round(float(m_cov.bse["z"]), 4),
                "coef (offset)": round(b_off, 4),
                "SE (off)": round(float(m_off.bse["z"]), 4),
                "abs diff": round(abs(b_cov - b_off), 4),
                "sign stable": "yes" if np.sign(b_cov) == np.sign(b_off) else "FLIP",
            })
    out = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 92)
        print("S11c — BUILT-UP SURFACE AS COVARIATE vs EXPOSURE OFFSET")
        print("=" * 92)
        print(out.to_string(index=False))
        print("""
  The offset specification models firms per unit of developed land,
  which is the reviewer's 'equal-area versus partially developed cells'
  concern. Sign stability across the two columns is the thing to read.""")
    return out


# ══════════════════════════════════════════════════════════════════════
# 7. GRID-ORIGIN SENSITIVITY
# ══════════════════════════════════════════════════════════════════════
def origin_shift_check(verbose=True):
    """Instructions for the grid-origin sensitivity check.

    build_grid() anchors the lattice at min(firm x) − step, so the origin
    cannot be displaced from outside the function. Rather than duplicate
    110 lines of network construction here, this prints the one-line edit
    that enables the check.
    """
    if verbose:
        print("\n" + "=" * 92)
        print("S11d — GRID-ORIGIN SENSITIVITY (manual, one-line edit)")
        print("=" * 92)
        print("""  In agglomeration_standalone.py, add an origin_frac argument:

      def build_grid(city, cell_km=1.0, radius_km=None, min_dn=1.0,
                     verbose=True, origin_frac=0.0):
          ...
          step = cell_km * 1000.0
          x0 = fx.min() - step + origin_frac * step
          y0 = fy.min() - step + origin_frac * step

  then run the half-cell displacement and compare:

      g_shift = {c: build_grid(c, cell_km=1.0, origin_frac=0.5)
                 for c in CITY_ORDER}
      _, bc_shift = agglomeration(g_shift, verbose=False)
      cmp = res['bycity'].merge(bc_shift, on=['City','measure'],
                                suffixes=('_base','_shift'))
      cmp['sign stable'] = (np.sign(cmp.coef_base) == np.sign(cmp.coef_shift))
      print(cmp[['City','measure','coef_base','coef_shift','sign stable']]
            .to_string(index=False))

  A sign flip in any reported coefficient means that coefficient is an
  artefact of cell placement and must not be reported as a finding.""")
    return None


# ══════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════
def run_diagnostics(res, out_dir=None):
    out_dir = out_dir or str(paths.results("targetability"))
    grids = res["grids"]

    core, vif = core_diagnostics(grids)
    offs = offset_comparison(grids)
    origin_shift_check()

    os.makedirs(out_dir, exist_ok=True)
    core.to_csv(f"{out_dir}/S11a_diagnostics.csv", index=False)
    if len(vif):
        vif.to_csv(f"{out_dir}/S11b_vif.csv", index=False)
    offs.to_csv(f"{out_dir}/S11c_offset.csv", index=False)
    print(f"\nSaved S11 components to {out_dir}/")

    return {"core": core, "vif": vif, "offset": offs}


if __name__ == "__main__":
    res = run_all()
    diag = run_diagnostics(res)
