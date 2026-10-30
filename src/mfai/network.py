"""Stage 1: road graph construction and network distances to the nearest
major road, rail goods shed, inland-waterway terminal and ICD.

The graph is built over all (densified) vertices rather than linemerge
endpoints, which cut the median snap connector to ~12 m and removed a
density-correlated bias in road distances (see build_graph).
"""
import math
import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
from scipy.spatial import cKDTree

from . import paths
from .config import (WATER_OVERRIDE, SNAP_TOL, ROUND_DP,
                     MAJOR_ROAD_TOL, MAX_VERTEX_M, MAJOR_SAMPLE_M)

CARRY = ["sector_code", "isic_division", "isic_section", "pincode_clean",
         "mfg_group", "coord_n", "coord_n_pin", "coord_n_div",
         "fallback_rule1"]


def check_paths(city):
    print(f"\n[{city}] input check (data = {paths.DATA})")
    req = [paths.enterprises(city), paths.roads_all(city),
           paths.roads_major(city)]
    ok = True
    for p in req:
        e = os.path.exists(p)
        print(f"  {'OK     ' if e else 'MISSING'}  {p}")
        ok &= e
    print("  terminals:")
    for kind in ("rail", "water", "icd"):
        stem = paths.terminals(kind, city)
        found = [e for e in (".csv", ".gpkg")
                 if os.path.exists(f"{stem}{e}")]
        print(f"    {stem.name:40s} {found if found else 'ABSENT'}")
    if not any(os.path.exists(f"{paths.terminals('water', city)}{e}")
               for e in (".csv", ".gpkg")):
        if city in WATER_OVERRIDE:
            d, nm = WATER_OVERRIDE[city]
            print(f"    -> override: {nm} at {d/1000:.0f} km")
        else:
            print(f"    -> !! no waterway terminals and NO OVERRIDE")
    return ok


def fix_road_network(roads_gdf, log):
    geoms = []
    for g in roads_gdf.geometry:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "MultiLineString":
            geoms.extend(list(g.geoms))
        elif g.geom_type == "LineString":
            geoms.append(g)
    log.append(f"  exploded {len(roads_gdf)} features -> {len(geoms)} LineStrings")

    merged = linemerge(unary_union(geoms))
    noded = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    log.append(f"  noded -> {len(noded)} segments")

    endpts = []
    for ln in noded:
        c = list(ln.coords)
        endpts.append(c[0]); endpts.append(c[-1])
    endpts = np.array(endpts)
    pairs = cKDTree(endpts).query_pairs(SNAP_TOL)
    parent = list(range(len(endpts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    clusters = {}
    for i in range(len(endpts)):
        clusters.setdefault(find(i), []).append(i)
    snap_map = {}
    for members in clusters.values():
        rep = tuple(endpts[members].mean(axis=0).round(ROUND_DP))
        for m in members:
            snap_map[m] = rep
    log.append(f"  snapped endpoints into {len(clusters)} nodes (tol={SNAP_TOL} m)")

    out, idx = [], 0
    for ln in noded:
        c = list(ln.coords)
        c[0]  = snap_map[idx]; idx += 1
        c[-1] = snap_map[idx]; idx += 1
        clean = [c[0]] + [tuple(np.round(x, ROUND_DP)) for x in c[1:-1]] + [c[-1]]
        ded = [clean[0]]
        for x in clean[1:]:
            if x != ded[-1]:
                ded.append(x)
        if len(ded) >= 2:
            ls = LineString(ded)
            if ls.length > 0.01:
                out.append(ls)
    log.append(f"  rebuilt {len(out)} edges")
    return out


def _densify_coords(ls, step_m):
    coords = list(ls.coords)
    if step_m is None:
        return coords
    out = []
    for a, b in zip(coords[:-1], coords[1:]):
        out.append(a)
        seg = math.hypot(b[0]-a[0], b[1]-a[1])
        if seg > step_m:
            for k in range(1, int(seg // step_m) + 1):
                t = (k * step_m) / seg
                if t >= 1.0:
                    break
                out.append((a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])))
    out.append(coords[-1])
    return out


def build_graph(edges, log):
    """Graph over ALL vertices, not only linemerge endpoints.

    Endpoint-only nodes gave connectors correlated with density
    (Spearman vs DN = -0.220) and road distances inflated 1.67x against
    straight-line. Vertex-level cut the connector to ~12 m median and
    inflation to 1.33x, with rail and ICD medians moving <0.3%.
    """
    G = nx.Graph()
    for ls in edges:
        coords = [tuple(np.round(c, ROUND_DP))
                  for c in _densify_coords(ls, MAX_VERTEX_M)]
        for a, b in zip(coords[:-1], coords[1:]):
            if a == b:
                continue
            w = math.hypot(a[0]-b[0], a[1]-b[1])
            if G.has_edge(a, b):
                if w < G[a][b]["weight"]:
                    G[a][b]["weight"] = w
            else:
                G.add_edge(a, b, weight=w)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    log.append(f"  graph: {G.number_of_nodes():,} nodes, "
               f"{G.number_of_edges():,} edges, {len(comps)} components")
    if comps:
        log.append(f"  giant component {100*len(comps[0])/G.number_of_nodes():.1f}% "
                   f"| edge/node {G.number_of_edges()/G.number_of_nodes():.3f}")
    return G, comps


def make_snapper(G, giant=None):
    nodes = [n for n in G.nodes() if giant is None or n in giant]
    tree = cKDTree(np.array(nodes, dtype=float))
    def snap(points_xy):
        if len(points_xy) == 0:
            return []
        d, idx = tree.query(np.asarray(points_xy, dtype=float))
        return [(nodes[i], float(dd)) for dd, i in zip(d, idx)]
    return snap


def load_terminals(base, utm_epsg):
    empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs=utm_epsg)
    for ext in (".gpkg", ".csv"):
        p = base + ext
        if not os.path.exists(p):
            continue
        if ext == ".gpkg":
            g = gpd.read_file(p).to_crs(utm_epsg)
            if len(g) == 0:
                return empty
            nc = next((c for c in g.columns if c.lower() == "name"), None)
            g = g.copy()
            g["name"] = g[nc] if nc else [f"t{i}" for i in range(len(g))]
            g["geometry"] = g.geometry.representative_point()
            return g[["name", "geometry"]]
        df = pd.read_csv(p)
        if len(df) == 0:
            return empty
        df.columns = [c.strip() for c in df.columns]
        latc = [c for c in df.columns if c.lower() in ("latitude","lat")][0]
        lonc = [c for c in df.columns if c.lower() in ("longitude","lon","lng")][0]
        nc   = [c for c in df.columns if c.lower() == "name"][0]
        df = df.rename(columns={latc:"lat", lonc:"lon", nc:"name"})
        df = df.dropna(subset=["lat","lon"])
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                                crs=4326).to_crs(utm_epsg)
    return empty


def process_city(city, utm_epsg):
    t0 = time.time()
    log = [f"=== {city.upper()} network & freight accessibility ===\n"]
    outdir = paths.results(f"pipeline/{city}")

    roads = gpd.read_file(paths.roads_all(city)).to_crs(utm_epsg)
    log.append(f"Loaded {len(roads)} road features (EPSG:{utm_epsg})")
    edges = fix_road_network(roads, log)
    G, comps = build_graph(edges, log)
    giant = comps[0] if comps else set()
    snap = make_snapper(G, giant)
    node2comp = {n: i for i, c in enumerate(comps) for n in c}

    gpd.GeoDataFrame(geometry=edges, crs=utm_epsg).to_file(
        f"{outdir}/{city}_roads_fixed.gpkg", driver="GPKG")

    # ---- major-road node tagging ----
    major = gpd.read_file(paths.roads_major(city)).to_crs(utm_epsg)
    name_col = next((c for c in major.columns if c.lower() == "name"), None)
    hw_col   = next((c for c in major.columns if c.lower() == "highway"), None)

    def seg_label(i):
        nm = major[name_col].iloc[i] if name_col else None
        if isinstance(nm, str) and nm.strip():
            return nm.strip()
        hw = major[hw_col].iloc[i] if hw_col else None
        return (f"{str(hw).title()} (unnamed)" if isinstance(hw, str)
                else "Major (unnamed)")

    msamp, mseg = [], []
    for si, geom in enumerate(major.geometry):
        if geom is None or geom.is_empty:
            continue
        parts = ([geom] if geom.geom_type == "LineString"
                 else list(getattr(geom, "geoms", [])))
        for p in parts:
            if p.geom_type != "LineString":
                continue
            for c in _densify_coords(p, MAJOR_SAMPLE_M):
                msamp.append(c); mseg.append(si)

    major_nodes, major_name = set(), {}
    if msamp:
        mtree = cKDTree(np.array(msamp, dtype=float))
        gnodes = list(G.nodes())
        gd, gidx = mtree.query(np.array(gnodes, dtype=float))
        hit = gd <= MAJOR_ROAD_TOL
        for n, ok, si in zip(gnodes, hit, gidx):
            if ok:
                major_nodes.add(n); major_name[n] = seg_label(mseg[si])
        reached = len(set(mseg[i] for i in gidx[hit]))
        log.append(f"Major roads: {len(major)} segments, "
                   f"{major.length.sum()/1000:,.0f} km | "
                   f"{len(major_nodes):,} nodes tagged | "
                   f"{reached}/{len(major)} segments reached "
                   f"({100*reached/len(major):.1f}%)")

    maj_dist, maj_path = (nx.multi_source_dijkstra(G, major_nodes, weight="weight")
                          if major_nodes else ({}, {}))

    # ---- terminals ----
    rail = load_terminals(str(paths.terminals("rail", city)), utm_epsg)
    iwai = load_terminals(str(paths.terminals("water", city)), utm_epsg)
    icd  = load_terminals(str(paths.terminals("icd", city)), utm_epsg)
    log.append(f"\nTerminals: rail {len(rail)} | waterway {len(iwai)} | ICD {len(icd)}")
    if len(rail) + len(icd) == 0:
        raise RuntimeError(f"[{city}] no rail or ICD terminals — check paths")
    if len(iwai) == 0 and city not in WATER_OVERRIDE:
        raise RuntimeError(f"[{city}] no waterway terminals and no override")

    # ---- MSMEs ----
    msme = gpd.read_file(paths.cleaned(city)).to_crs(utm_epsg)
    msme = msme[msme.geometry.notna() & ~msme.geometry.is_empty].reset_index(drop=True)
    msme["msme_id"] = msme.index
    msme["_x"] = msme.geometry.x.round(1)
    msme["_y"] = msme.geometry.y.round(1)
    uniq = msme.drop_duplicates(subset=["_x","_y"]).reset_index(drop=True)
    uniq["loc_id"] = uniq.index
    log.append(f"MSMEs (manufacturing, unfiltered): {len(msme):,} | "
               f"unique locations: {len(uniq):,}")

    msme_snap = snap(list(zip(uniq.geometry.x, uniq.geometry.y)))
    snaps = {k: (snap(list(zip(g.geometry.x, g.geometry.y))) if len(g) else [])
             for k, g in (("railway", rail), ("waterway", iwai), ("ICD", icd))}

    def all_dist(term_snap):
        rows = {}
        for ti, (tn, tc) in enumerate(term_snap):
            try:
                dm = nx.single_source_dijkstra_path_length(G, tn, weight="weight")
            except nx.NodeNotFound:
                dm = {}
            rows[ti] = (dm, tc, node2comp.get(tn, -1))
        return rows

    dmaps = {k: all_dist(v) for k, v in snaps.items()}
    gdfs  = {"railway": rail, "waterway": iwai, "ICD": icd}

    near = {k: {} for k in gdfs}
    for li, (mn, mc) in enumerate(msme_snap):
        mcomp = node2comp.get(mn, -1)
        for kind in gdfs:
            best, bname = np.inf, None
            for ti, (dm, tc, tcomp) in dmaps[kind].items():
                d = dm.get(mn)
                if d is None or mcomp != tcomp:
                    continue
                tot = d + mc + tc
                if tot < best:
                    best, bname = tot, gdfs[kind]["name"].iloc[ti]
            near[kind][li] = (best if np.isfinite(best) else np.nan, bname)

    maj_dm, maj_nm = {}, {}
    for li, (mn, mc) in enumerate(msme_snap):
        d = maj_dist.get(mn)
        maj_dm[li] = (d + mc) if d is not None else np.nan
        p = maj_path.get(mn)
        maj_nm[li] = major_name.get(p[0]) if p else None

    # ---- assemble ----
    ml = msme.merge(uniq[["loc_id","_x","_y"]], on=["_x","_y"], how="left")
    full = ml[["msme_id","loc_id"]].copy()
    full["lat"] = msme.to_crs(4326).geometry.y.values
    full["lon"] = msme.to_crs(4326).geometry.x.values
    if "DN" in msme.columns:
        full["DN"] = msme["DN"].values
    for c in CARRY:
        if c in msme.columns:
            full[c] = msme[c].values

    full["nearest_major_road_name"]   = full["loc_id"].map(maj_nm).values
    full["nearest_major_road_dist_m"] = full["loc_id"].map(maj_dm).values
    full["snap_connector_m"] = full["loc_id"].map(
        {li: round(mc,1) for li,(mn,mc) in enumerate(msme_snap)}).values
    for kind in gdfs:
        full[f"nearest_{kind}_dist_m"] = full["loc_id"].map(
            {li: v[0] for li, v in near[kind].items()}).values
        full[f"nearest_{kind}_name"] = full["loc_id"].map(
            {li: v[1] for li, v in near[kind].items()}).values

    full["water_source"] = "network"
    if full["nearest_waterway_dist_m"].isna().all():
        d, nm = WATER_OVERRIDE[city]
        full["nearest_waterway_dist_m"] = float(d)
        full["nearest_waterway_name"]   = nm
        full["water_source"] = "nearest_operational_terminal"
        log.append(f"  waterway: no in-district terminal; using {nm} "
                   f"at {d/1000:.0f} km")

    # ---- write, preserving externally added covariates ----
    out = full.drop(columns="loc_id")
    csvp = paths.nearest_csv(city)

    if os.path.exists(csvp):
        prev = pd.read_csv(csvp)
        if "msme_id" in prev.columns and len(prev) == len(out):
            # any column the pipeline does not itself produce was added
            # downstream (NTL, GHSL epochs, distance to centre, enterprise
            # name) and would otherwise be silently lost on every rerun
            extra = [c for c in prev.columns if c not in out.columns]
            if extra:
                out = out.merge(prev[["msme_id"] + extra],
                                on="msme_id", how="left")
                log.append(f"  preserved {len(extra)} external column(s): "
                           f"{', '.join(extra)}")
        else:
            log.append("  ! existing CSV has a different row count or no "
                       "msme_id — external columns NOT preserved")

    out.to_csv(csvp, index=False)
    log.append(f"Wrote {len(out):,} rows, {len(out.columns)} columns "
               f"-> {city}_msme_nearest.csv")

    for kind in ["major_road"] + list(gdfs):
        v = full[f"nearest_{kind}_dist_m"].dropna()
        if len(v):
            log.append(f"  {kind:12s}: {len(v):,}/{len(full):,} | "
                       f"median {v.median()/1000:.2f} km | "
                       f"p95 {v.quantile(.95)/1000:.2f} km")
    sc = full["snap_connector_m"].dropna()
    log.append(f"  connector: median {sc.median():.1f} m | "
               f"p95 {sc.quantile(.95):.1f} m")
    log.append(f"\nDone in {time.time()-t0:.1f} s")

    with open(outdir / f"{city}_report.txt", "w") as f:
        f.write("\n".join(log))
    print("\n".join(log)); print()
