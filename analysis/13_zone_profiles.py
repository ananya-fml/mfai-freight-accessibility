"""Zone profiles: locality, extent, binding mode

Produces: Tables 15, 17 -> results/targetability/zone_profiles.csv

Fills the ⟦TODO⟧ placeholders in Section 5.8 ("Where the zones are").
Reproduces the pooled-threshold clustering exactly as in the main analysis
(same T=8 filter, same pooled P10, same HDBSCAN parameters), then emits one
row per zone with everything needed to write a geographic description —
except the place names, which you read off a map using the coordinates this
prints.
"""
import os
import numpy as np
import pandas as pd
import hdbscan
from pyproj import Transformer
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from mfai import paths
from mfai.io import load_city

np.random.seed(42)
# ═══════════════════════════════════════════════════════════════
CITIES = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {c: c.title() for c in CITIES}
UTM_EPSG = {"patna": 32645, "kolkata": 32645, "varanasi": 32644,
            "indore": 32643, "kochi": 32643}

COL_LAT, COL_LON, COL_MFAI = "lat", "lon", "mfai_land"
ACCESS = {"road": "access_road", "rail": "access_rail", "icd": "access_icd"}

FALLBACK_THRESHOLD, DIVISION_T = 20, 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

MCS, MS = 20, 10
POOLED_PCT = 10

TOP_N = 3            # always name this many per city
SHARE_FLOOR = 0.10   # plus any zone above this share of city flagged mass
MEDOID_CAP = 3000    # subsample above this many points for the medoid


def scale_class(a):
    if not np.isfinite(a):
        return "—"
    if a <= 5:
        return "site"
    if a <= 25:
        return "neighbourhood"
    if a <= 100:
        return "district"
    return "regional"


def build_unique_locs(df, epsg):
    g = df.groupby([COL_LAT, COL_LON])
    locs = g.size().reset_index(name="n_ent")
    locs[COL_MFAI] = g[COL_MFAI].mean().values
    for m, col in ACCESS.items():
        if col in df.columns:
            locs[f"a_{m}"] = g[col].mean().values
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(locs[COL_LON].values, locs[COL_LAT].values)
    return locs.reset_index(drop=True), np.column_stack([x, y])


def shape_descriptors(pts_utm):
    """Principal-axis extents in km, elongation ratio, and compass bearing."""
    if len(pts_utm) < 3:
        return np.nan, np.nan, np.nan, np.nan
    c = pts_utm - pts_utm.mean(axis=0)
    cov = np.cov(c.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = c @ vecs                       # coords in principal-axis frame
    major = (proj[:, 0].max() - proj[:, 0].min()) / 1000.0
    minor = (proj[:, 1].max() - proj[:, 1].min()) / 1000.0
    ratio = major / minor if minor > 1e-9 else np.inf
    # bearing of the major axis, compass degrees, folded to 0–180
    vx, vy = vecs[0, 0], vecs[1, 0]
    bearing = (np.degrees(np.arctan2(vx, vy))) % 180.0
    return major, minor, ratio, bearing


def medoid_index(pts_utm):
    """Index of the point minimising total distance to all others."""
    n = len(pts_utm)
    if n == 1:
        return 0
    if n > MEDOID_CAP:
        sub = np.random.choice(n, MEDOID_CAP, replace=False)
        d = cdist(pts_utm[sub], pts_utm[sub]).sum(axis=1)
        return int(sub[int(np.argmin(d))])
    d = cdist(pts_utm, pts_utm).sum(axis=1)
    return int(np.argmin(d))


def run_profiles(verbose=True):

    print("Loading …")
    data, all_mfai = {}, []
    for c in CITIES:
        df = load_city(c)
        locs, utm = build_unique_locs(df, UTM_EPSG[c])
        data[c] = (locs, utm)
        all_mfai.append(df[COL_MFAI].to_numpy(float))
        print(f"  {c}: {len(locs):,} locations")

    thr = float(np.percentile(np.concatenate(all_mfai), POOLED_PCT))
    print(f"\nPooled P{POOLED_PCT} = {thr:.4f}  (main analysis: 0.7065)\n")

    rows = []
    for c in CITIES:
        locs, utm = data[c]
        m = locs[COL_MFAI].to_numpy(float) < thr
        fl, fu = locs[m].reset_index(drop=True), utm[m]
        city_ent = float(fl["n_ent"].sum())
        if len(fu) < MCS:
            print(f"  {CITY_LABEL[c]}: {len(fu)} flagged locations — no clustering")
            continue

        lab = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                              metric="euclidean").fit(fu).labels_
        back = Transformer.from_crs(f"EPSG:{UTM_EPSG[c]}", "EPSG:4326",
                                    always_xy=True)

        for z in sorted(set(lab) - {-1}):
            k = lab == z
            pts, sub = fu[k], fl[k].reset_index(drop=True)
            n_ent = float(sub["n_ent"].sum())

            # area
            if len(pts) >= 3:
                try:
                    area = ConvexHull(pts).volume / 1e6
                except Exception:
                    area = np.nan
            else:
                area = np.nan

            # centroid and medoid, back to lat/lon
            cx, cy = pts.mean(axis=0)
            clon, clat = back.transform(cx, cy)
            mi = medoid_index(pts)
            mlon, mlat = back.transform(pts[mi, 0], pts[mi, 1])

            major, minor, ratio, bearing = shape_descriptors(pts)

            # binding mode within the zone (enterprise-weighted)
            w = sub["n_ent"].to_numpy(float)
            means = {}
            for mo in ACCESS:
                col = f"a_{mo}"
                if col in sub.columns:
                    means[mo] = float(np.average(sub[col].to_numpy(float),
                                                 weights=w))
            binding = min(means, key=means.get) if means else "—"

            rows.append({
                "city": CITY_LABEL[c], "zone": int(z),
                "n_ent": int(n_ent), "n_locs": int(k.sum()),
                "share_city_flagged_%": round(100 * n_ent / city_ent, 1),
                "area_km2": round(area, 1) if np.isfinite(area) else np.nan,
                "scale_class": scale_class(area),
                "centroid_lat": round(clat, 5), "centroid_lon": round(clon, 5),
                "medoid_lat": round(mlat, 5), "medoid_lon": round(mlon, 5),
                "major_km": round(major, 1) if np.isfinite(major) else np.nan,
                "minor_km": round(minor, 1) if np.isfinite(minor) else np.nan,
                "elongation": round(ratio, 1) if np.isfinite(ratio) else np.nan,
                "bearing_deg": round(bearing) if np.isfinite(bearing) else np.nan,
                "binding_mode": binding.upper(),
            })

    P = pd.DataFrame(rows)
    if P.empty:
        print("no zones found")
        return P

    # ── selection: top N by enterprise count, plus share floor ──
    P["rank_in_city"] = (P.groupby("city")["n_ent"]
                          .rank(ascending=False, method="first").astype(int))
    P["selected"] = ((P["rank_in_city"] <= TOP_N)
                     | (P["share_city_flagged_%"] >= 100 * SHARE_FLOOR))
    P = P.sort_values(["city", "rank_in_city"]).reset_index(drop=True)

    if verbose:
        print("=" * 100)
        print("ALL ZONES")
        print("=" * 100)
        cols = ["city", "rank_in_city", "n_ent", "n_locs",
                "share_city_flagged_%", "area_km2", "scale_class",
                "elongation", "bearing_deg", "binding_mode", "selected"]
        print(P[cols].to_string(index=False))

        print("\n" + "=" * 100)
        print("ZONES TO NAME — look these up")
        print("=" * 100)
        print("""Use the MEDOID for naming: it is a real enterprise location, so it always
falls on developed land. The centroid is the zone's average position and can
sit in open country when a zone is curved or ribbon-shaped.

Read 'elongation' with 'bearing': a ratio above about 3 means the zone is a
corridor rather than a pocket, and the bearing tells you which axis to follow
(0/180 = N–S, 090 = E–W, 045 = NE–SW, 135 = NW–SE).
""")
        for c in P["city"].unique():
            sel = P[(P.city == c) & P.selected]
            print(f"\n{'─'*72}\n{c}  ({len(sel)} zones to name)\n{'─'*72}")
            for _, r in sel.iterrows():
                shape = ("corridor" if r.elongation >= 3
                         else "elongated pocket" if r.elongation >= 2
                         else "compact pocket")
                print(f"  Zone {r.zone}  (rank {r.rank_in_city})  "
                      f"{r.n_ent:,} enterprises, {r['share_city_flagged_%']}% of city flagged")
                print(f"    area {r.area_km2} km² ({r.scale_class}), "
                      f"{r.major_km}×{r.minor_km} km, {shape}, "
                      f"bearing {r.bearing_deg:.0f}°, binding {r.binding_mode}")
                print(f"    MEDOID   {r.medoid_lat}, {r.medoid_lon}")
                print(f"    centroid {r.centroid_lat}, {r.centroid_lon}")
                print(f"    https://www.google.com/maps/search/?api=1&query="
                      f"{r.medoid_lat},{r.medoid_lon}")
                print(f"    name: ______________________________________")

        print("\n" + "=" * 100)
        print("COORDINATE LIST (paste into a map, one per line)")
        print("=" * 100)
        for _, r in P[P.selected].iterrows():
            print(f"{r.medoid_lat}, {r.medoid_lon}   # {r.city} zone {r.zone}, "
                  f"{r.n_ent:,} ent, {r.area_km2} km², {r.binding_mode}")

    out = str(paths.results("targetability"))
    os.makedirs(out, exist_ok=True)
    P.to_csv(f"{out}/zone_profiles.csv", index=False)
    P[P.selected].to_csv(f"{out}/zone_profiles_to_name.csv", index=False)
    print(f"\nSaved zone_profiles.csv and zone_profiles_to_name.csv to {out}/")
    return P


if __name__ == "__main__":
    prof = run_profiles()
