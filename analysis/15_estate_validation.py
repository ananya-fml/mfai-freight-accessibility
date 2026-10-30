"""Industrial-estate validation (reported as unavailable in the paper)

Produces: Supplementary S9a-b (estate coverage, terminal distance to estates)

Do manufacturing MSMEs inside designated industrial land have better
freight access than those outside?
Industrial estates are sited partly for freight access, so an index of
freight accessibility should separate them. This is convergent
validity: evidence that the measure detects a deliberate siting
decision made by someone other than the analyst.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import cKDTree

from mfai import paths
from mfai.io import load_city

# ══════════════════════════════════════════════════════════════════════
# CONFIG — EDIT
# ══════════════════════════════════════════════════════════════════════

ESTATE_POLY = {c: paths.estates(c)          # OSM landuse=industrial
               for c in ("patna", "indore", "varanasi", "kolkata")}
# kochi: not available
BOUNDARY = {            # optional; skipped if the file is absent
    "indore": paths.INPUTS / "boundaries" / "indore_boundary.gpkg",
}

UTM = {"kochi": 32643, "kolkata": 32645, "patna": 32645,
       "varanasi": 32644, "indore": 32643}
CITY_CENTRE = {"patna": (25.6127, 85.1416), "indore": (22.7177, 75.8545),
               "varanasi": (25.3109, 83.0104), "kochi": (9.9816, 76.2830),
               "kolkata": (22.5697, 88.3497)}
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}
CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]

MODES = ["road", "rail", "water", "icd"]
FALLBACK_THRESHOLD, FALLBACK_MIN_PINCODES, DIV_DIVERSITY_T = 20, 5, 8
MIN_INSIDE = 100          # below this, comparisons are not reported
BUFFER_GRID = (0, 100, 250)   # metres, sensitivity to boundary precision


# ══════════════════════════════════════════════════════════════════════
# FIRM LOADING (standalone)
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# ESTATES
# ══════════════════════════════════════════════════════════════════════
def prepare_estates(city, poly_path, boundary_path=None, buffer_m=0,
                    verbose=True):
    """Load, optionally clip and buffer, then dissolve.

    Dissolving prevents a firm inside overlapping polygons being counted
    more than once.
    """
    utm = UTM[city]
    if not os.path.exists(poly_path):
        if verbose:
            print(f"[{city}] estate file not found: {poly_path}")
        return None
    ind = gpd.read_file(poly_path).to_crs(utm)
    ind = ind[ind.geometry.notna() & ~ind.geometry.is_empty]
    ind = ind[ind.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if len(ind) == 0:
        if verbose:
            print(f"[{city}] no polygon features")
        return None
    n_all, area_all = len(ind), ind.area.sum() / 1e6

    clipped = False
    if boundary_path and os.path.exists(boundary_path):
        b = gpd.read_file(boundary_path).to_crs(utm)
        bg = b.geometry.union_all() if hasattr(b.geometry, "union_all") \
             else b.geometry.unary_union
        ind = gpd.clip(ind, bg)
        ind = ind[ind.geometry.notna() & ~ind.geometry.is_empty]
        clipped = True
        if len(ind) == 0:
            if verbose:
                print(f"[{city}] no estate land inside the boundary")
            return None

    if buffer_m:
        ind = ind.copy()
        ind["geometry"] = ind.geometry.buffer(buffer_m)

    merged = ind.geometry.union_all() if hasattr(ind.geometry, "union_all") \
             else ind.geometry.unary_union
    est = gpd.GeoDataFrame(geometry=[merged], crs=utm)
    est = est.explode(index_parts=False).reset_index(drop=True)
    est["area_km2"] = est.area / 1e6

    if verbose:
        msg = (f"[{city}] {n_all} source features, {area_all:,.1f} km2")
        if clipped:
            msg += f" -> {est.area_km2.sum():,.1f} km2 inside boundary"
        if buffer_m:
            msg += f" | +{buffer_m} m buffer"
        print(msg + f" | {len(est)} parts, largest "
              f"{est.area_km2.max():.2f} km2")
    return est


def tag_firms(city, est, div_t=DIV_DIVERSITY_T):
    d = load_city(city, div_t)
    pts = gpd.GeoDataFrame(d.copy(),
                           geometry=gpd.points_from_xy(d.lon, d.lat),
                           crs=4326).to_crs(UTM[city])
    j = gpd.sjoin(pts, est[["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    d["in_estate"] = j["index_right"].notna().values
    return d


# ══════════════════════════════════════════════════════════════════════
# COMPARISONS
# ══════════════════════════════════════════════════════════════════════
def coverage_table(estates, firms):
    rows = []
    for c in CITY_ORDER:
        if c not in firms:
            rows.append({"City": CITY_LABEL[c], "estate data": "none",
                         "parts": np.nan, "area km2": np.nan,
                         "firms": np.nan, "inside": np.nan, "% inside": np.nan})
            continue
        d, e = firms[c], estates[c]
        rows.append({"City": CITY_LABEL[c], "estate data": "OSM",
                     "parts": len(e),
                     "area km2": round(e.area_km2.sum(), 1),
                     "firms": len(d), "inside": int(d.in_estate.sum()),
                     "% inside": round(100 * d.in_estate.mean(), 2)})
    out = pd.DataFrame(rows)
    print("=" * 80)
    print("COVERAGE")
    print("=" * 80)
    print(out.to_string(index=False))
    print(f"\n  Cities with fewer than {MIN_INSIDE} firms inside are not")
    print("  compared: OSM industrial coverage captures planned estates,")
    print("  and most MSMEs are not in one.")
    return out


def raw_comparison(firms, col="mfai_land"):
    """Unadjusted. Confounded by where estates are; reported so the size
    of the confound is visible against the adjusted estimates."""
    rows = []
    for c, d in firms.items():
        if d.in_estate.sum() < MIN_INSIDE:
            continue
        a = d.loc[d.in_estate, col].dropna()
        b = d.loc[~d.in_estate, col].dropna()
        rows.append({
            "City": CITY_LABEL[c], "n in": len(a), "n out": len(b),
            "inside": round(a.mean(), 4), "outside": round(b.mean(), 4),
            "diff": round(a.mean()-b.mean(), 4),
            "SD units": round((a.mean()-b.mean())/b.std(), 3),
            "p": f"{stats.mannwhitneyu(a, b).pvalue:.2g}",
            "dist in (km)": round(d.loc[d.in_estate, "dist_centre_km"].median(), 1),
            "dist out (km)": round(d.loc[~d.in_estate, "dist_centre_km"].median(), 1)})
    out = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(f"RAW — {col}   (confounded; compare against matched below)")
    print("=" * 80)
    print(out.to_string(index=False))
    print("\n  The last two columns show the confound directly: if estate")
    print("  firms sit at a different median distance from the centre, the")
    print("  raw difference is substantially a centrality effect.")
    return out


def matched_comparison(firms, col="mfai_land", caliper=0.25, verbose=True):
    """Nearest-neighbour matching on distance to centre and log built-up.

    Each inside firm is paired with the nearest outside firm in
    standardised covariate space, with replacement, subject to a caliper
    in pooled standard deviations. This controls for centrality and
    density without imposing a functional form.
    """
    rows, pairs = [], {}
    for c, d in firms.items():
        if d.in_estate.sum() < MIN_INSIDE:
            continue
        need = [col, "dist_centre_km"]
        dn = "DN" if "DN" in d.columns and d["DN"].notna().any() else None
        s = d.dropna(subset=need + ([dn] if dn else [])).copy()
        cov = ["dist_centre_km"] + ([dn] if dn else [])
        Z = s[cov].copy()
        if dn:
            Z[dn] = np.log1p(Z[dn].clip(lower=0))
        Z = (Z - Z.mean()) / Z.std()

        ins = s.index[s.in_estate]
        out_ = s.index[~s.in_estate]
        if len(ins) < MIN_INSIDE or len(out_) < MIN_INSIDE:
            continue
        tree = cKDTree(Z.loc[out_].values)
        dist, idx = tree.query(Z.loc[ins].values)
        ok = dist <= caliper
        if ok.sum() < MIN_INSIDE:
            if verbose:
                print(f"  [{c}] only {int(ok.sum())} matched within caliper")
            continue
        a = s.loc[ins[ok], col].values
        b = s.loc[out_[idx[ok]], col].values
        diff = a - b
        w = stats.wilcoxon(diff) if len(diff) > 20 else None
        rows.append({
            "City": CITY_LABEL[c], "matched": int(ok.sum()),
            "% matched": round(100*ok.mean(), 1),
            "inside": round(a.mean(), 4), "matched control": round(b.mean(), 4),
            "diff": round(diff.mean(), 4),
            "SD units": round(diff.mean()/s[col].std(), 3),
            "p": f"{w.pvalue:.2g}" if w is not None else "—"})
        pairs[c] = pd.DataFrame({"inside": a, "control": b, "diff": diff})

    out = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 80)
        print(f"MATCHED — {col}   (primary specification)")
        print("=" * 80)
        print(out.to_string(index=False) if len(out) else "  no city qualified")
        print(f"\n  Nearest neighbour on distance to centre and log built-up,")
        print(f"  with replacement, caliper {caliper} SD. Wilcoxon signed-rank")
        print("  on paired differences.")
    return out, pairs


def matched_by_mode(firms, caliper=0.25):
    """Matched difference for each mode.

    Which mode drives any estate difference is informative. A difference
    concentrated in road access is weak — roads are near-universal. A
    difference in ICD or rail access is stronger, since those reflect
    facility siting.
    """
    rows = []
    for c, d in firms.items():
        if d.in_estate.sum() < MIN_INSIDE:
            continue
        r = {"City": CITY_LABEL[c]}
        for m in ["mfai_land"] + MODES:
            col = m if m == "mfai_land" else f"access_{m}"
            if col not in d.columns or d[col].std(skipna=True) < 1e-9:
                r[m] = np.nan
                continue
            t, _ = matched_comparison({c: d}, col, caliper, verbose=False)
            r[m] = t["SD units"].iloc[0] if len(t) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("MATCHED DIFFERENCE BY MODE (SD units, inside minus control)")
    print("=" * 80)
    print(out.to_string(index=False))
    print("\n  A difference concentrated in road access is weak evidence:")
    print("  road access is near-universal. Differences in rail or ICD")
    print("  access reflect facility siting and are more informative.")
    return out


def stratified_comparison(firms, col="mfai_land", band_km=5.0, min_n=30):
    """Inside versus outside within distance bands."""
    allrows = []
    for c, d in firms.items():
        if d.in_estate.sum() < MIN_INSIDE:
            continue
        s = d.dropna(subset=[col, "dist_centre_km"]).copy()
        s["band"] = (s.dist_centre_km // band_km).astype(int) * band_km
        rows = []
        for b, g in s.groupby("band"):
            a, o = g.loc[g.in_estate, col], g.loc[~g.in_estate, col]
            if len(a) < min_n or len(o) < min_n:
                continue
            rows.append({"band": f"{b:.0f}–{b+band_km:.0f}",
                         "n in": len(a), "n out": len(o),
                         "diff": round(a.mean()-o.mean(), 4),
                         "SD units": round((a.mean()-o.mean())/o.std(), 3)})
        if not rows:
            continue
        t = pd.DataFrame(rows)
        print(f"\n{CITY_LABEL[c]}")
        print(t.to_string(index=False))
        print(f"  bands positive: {int((t['diff']>0).sum())} of {len(t)} | "
              f"weighted mean diff "
              f"{np.average(t['diff'], weights=t['n in']):+.4f}")
        t["City"] = CITY_LABEL[c]
        allrows.append(t)
    print("\n  Consistent sign across bands is the evidence; a single band")
    print("  could reflect the position of one estate.")
    return pd.concat(allrows, ignore_index=True) if allrows else pd.DataFrame()


def pooled_regression(firms, col="mfai_land"):
    """Estate dummy with centrality and built-up controls, city fixed
    effects, standard errors clustered by city."""
    parts = []
    for c, d in firms.items():
        if d.in_estate.sum() < MIN_INSIDE:
            continue
        s = d.copy(); s["city"] = c
        parts.append(s)
    if not parts:
        print("\n  no city qualified for pooled regression")
        return None
    a = pd.concat(parts, ignore_index=True)
    dn = "DN" if "DN" in a.columns and a["DN"].notna().any() else None
    a = a.dropna(subset=[col, "dist_centre_km"] + ([dn] if dn else []))
    # within-city standardisation of the outcome, so the coefficient is
    # in SD units and comparable across cities
    a["z"] = a.groupby("city")[col].transform(
        lambda s: (s - s.mean()) / s.std())
    X = pd.DataFrame({"in_estate": a.in_estate.astype(float),
                      "log_dist": np.log1p(a.dist_centre_km)}, index=a.index)
    if dn:
        X["log_dn"] = np.log1p(a[dn].clip(lower=0))
    X = pd.concat([X, pd.get_dummies(a.city, prefix="c",
                                     drop_first=True).astype(float)], axis=1)
    m = sm.OLS(a.z, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": a.city})
    print("\n" + "=" * 80)
    print(f"POOLED REGRESSION — z({col}) ~ in_estate + controls + city FE")
    print("=" * 80)
    print(m.summary().tables[1])
    print(f"\n  n = {len(a):,} | cities = {a.city.nunique()} | "
          f"inside = {int(a.in_estate.sum()):,}")
    print("  Coefficient is in within-city SD of the outcome.")
    print("  Cross-sectional and observational: estates were sited for land,")
    print("  power and water as well as freight, so this is an association,")
    print("  not an effect of estate designation.")
    return m


def buffer_sensitivity(estate_paths, boundary=None, col="mfai_land",
                       buffers=BUFFER_GRID):
    """Does the result depend on treating estate boundaries as exact?

    Geocoding error on Indian addresses is 20-50 m, so firms just outside
    an estate boundary may belong to it.
    """
    rows = []
    for bm in buffers:
        est, firms = {}, {}
        for c, p in estate_paths.items():
            e = prepare_estates(c, p, (boundary or {}).get(c), bm,
                                verbose=False)
            if e is None:
                continue
            est[c] = e
            firms[c] = tag_firms(c, e)
        t, _ = matched_comparison(firms, col, verbose=False)
        for _, r in t.iterrows():
            rows.append({"buffer m": bm, "City": r["City"],
                         "matched": r["matched"],
                         "SD units": r["SD units"], "p": r["p"]})
    out = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("BUFFER SENSITIVITY")
    print("=" * 80)
    if len(out):
        print(out.pivot(index="City", columns="buffer m",
                        values="SD units").to_string())
    return out


def terminal_geography(city, est):
    """Distance from each terminal to the nearest estate polygon.

    Bears on whether capacity was placed to serve planned industrial
    land or the dispersed MSME base.
    """
    utm = UTM[city]
    eg = est.geometry.union_all() if hasattr(est.geometry, "union_all") \
         else est.geometry.unary_union
    out = []
    for label in ("rail", "water", "icd"):
        stem = paths.terminals(label, city)
        g = None
        for ext in (".gpkg", ".csv"):
            p = f"{stem}{ext}"
            if os.path.exists(p):
                g = (gpd.read_file(p).to_crs(utm) if ext == ".gpkg" else None)
                if g is None:
                    df = pd.read_csv(p)
                    df.columns = [x.strip() for x in df.columns]
                    la = [x for x in df.columns if x.lower() in ("latitude","lat")][0]
                    lo = [x for x in df.columns if x.lower() in ("longitude","lon","lng")][0]
                    g = gpd.GeoDataFrame(df,
                        geometry=gpd.points_from_xy(df[lo], df[la]),
                        crs=4326).to_crs(utm)
                break
        if g is None or len(g) == 0:
            continue
        d = [p.distance(eg)/1000 for p in g.geometry.representative_point()]
        out.append({"City": CITY_LABEL[city], "mode": label, "n": len(g),
                    "median km to estate": round(float(np.median(d)), 2),
                    "min km": round(float(np.min(d)), 2),
                    "within 2 km": int(sum(x <= 2 for x in d))})
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════════════
def run_all(estate_paths=None, boundary=None, buffer_m=0,
            col="mfai_land", caliper=0.25, do_buffer=True):
    estate_paths = estate_paths or ESTATE_POLY
    boundary = boundary if boundary is not None else BOUNDARY

    est, firms = {}, {}
    print("=" * 80)
    print("INDUSTRIAL ESTATE VALIDATION")
    print("=" * 80)
    for c, p in estate_paths.items():
        e = prepare_estates(c, p, boundary.get(c), buffer_m)
        if e is None:
            continue
        est[c] = e
        firms[c] = tag_firms(c, e)
        print(f"    firms inside: {int(firms[c].in_estate.sum()):,} of "
              f"{len(firms[c]):,} ({firms[c].in_estate.mean():.2%})")

    cov = coverage_table(est, firms)
    usable = [c for c in firms if firms[c].in_estate.sum() >= MIN_INSIDE]
    print(f"\n  usable for comparison: "
          f"{', '.join(CITY_LABEL[c] for c in usable) or 'none'}")
    if not usable:
        return {"coverage": cov, "estates": est, "firms": firms}

    raw = raw_comparison(firms, col)
    matched, pairs = matched_comparison(firms, col, caliper)
    bymode = matched_by_mode(firms, caliper)
    print("\n" + "=" * 80)
    print(f"STRATIFIED BY DISTANCE TO CENTRE — {col}")
    print("=" * 80)
    strat = stratified_comparison(firms, col)
    reg = pooled_regression(firms, col)

    tg = pd.concat([terminal_geography(c, est[c]) for c in est],
                   ignore_index=True)
    print("\n" + "=" * 80)
    print("TERMINAL DISTANCE TO NEAREST ESTATE")
    print("=" * 80)
    print(tg.to_string(index=False))

    buf = buffer_sensitivity(estate_paths, boundary, col) if do_buffer else None

    return {"coverage": cov, "estates": est, "firms": firms, "raw": raw,
            "matched": matched, "pairs": pairs, "by_mode": bymode,
            "stratified": strat, "regression": reg, "terminals": tg,
            "buffer": buf}


if __name__ == "__main__":
    res = run_all()
