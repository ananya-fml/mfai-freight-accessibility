"""Figure 7: low-proximity zones at the pooled threshold

Produces: Figure 7 -> results/targetability/

Five panels, one per city, showing WHERE low-proximity enterprises sit and
whether they gather into anything a site-specific intervention could
address. Figure 8 (the frontier) already carries the compact-versus-diffuse
argument quantitatively; this figure's job is morphology — Patna's citywide
scatter, Kochi's coastal string, Indore's fragments, and Kolkata's near-empty
panel.
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer
from scipy.spatial import ConvexHull
import hdbscan

from mfai import paths
from mfai.io import load_city

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# CONFIG — matches targetability_v2.py (T = 8)
# ══════════════════════════════════════════════════════════════════════
CITIES   = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}
UTM_EPSG = {"patna": 32645, "kolkata": 32645, "varanasi": 32644,
            "indore": 32643, "kochi": 32643}

COL_LAT, COL_LON, COL_MFAI = "lat", "lon", "mfai_land"
FALLBACK_THRESHOLD, DIVISION_T = 20, 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

MCS, MS = 20, 10            # identical to the main analysis
POOLED_PCT = 10             # primary threshold

# Terminal styling — matches the documented QGIS convention
TERM_STYLE = {
    "rail":  dict(marker="s", color="#E76F51", label="Rail freight terminal"),
    "water": dict(marker="^", color="#2A9D8F", label="IWT terminal"),
    "icd":   dict(marker="o", color="#B5179E", label="ICD / CFS"),
}

NOISE_COLOR = "#BBBBBB"
FIG_W, FIG_H = 7.48, 5.4     # Elsevier double-column width


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════


def to_utm(lats, lons, epsg):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(np.asarray(lons), np.asarray(lats))
    return np.column_stack([x, y])


def build_unique_locs(df, epsg):
    grp = df.groupby([COL_LAT, COL_LON])
    locs = grp.size().reset_index(name="n_enterprises")
    locs[COL_MFAI] = grp[COL_MFAI].mean().values
    return locs.reset_index(drop=True), to_utm(
        locs[COL_LAT].values, locs[COL_LON].values, epsg)


def load_terminals(city, kind, epsg):
    """Returns Nx2 array of UTM coords, or empty array."""
    base = str(paths.terminals(kind, city))
    for ext in (".gpkg", ".csv"):
        p = base + ext
        if not os.path.exists(p):
            continue
        try:
            if ext == ".gpkg":
                g = gpd.read_file(p)
                if len(g) == 0:
                    return np.empty((0, 2))
                g = g.to_crs(f"EPSG:{epsg}")
                pts = g.geometry.representative_point()
                return np.column_stack([pts.x.values, pts.y.values])
            df = pd.read_csv(p)
            if len(df) == 0:
                return np.empty((0, 2))
            df.columns = [c.strip() for c in df.columns]
            latc = [c for c in df.columns if c.lower() in ("latitude", "lat")][0]
            lonc = [c for c in df.columns
                    if c.lower() in ("longitude", "lon", "lng")][0]
            return to_utm(df[latc].values, df[lonc].values, epsg)
        except Exception as e:
            print(f"    ! {city}/{kind}: {str(e)[:50]}")
            return np.empty((0, 2))
    return np.empty((0, 2))


# ══════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════
def nice_scale_length(span_m):
    """Pick a round scale-bar length ≈ 1/4 of the panel width."""
    target = span_m / 4.0
    for v in [100, 200, 500, 1_000, 2_000, 5_000, 10_000,
              20_000, 25_000, 50_000, 100_000]:
        if v >= target:
            return v
    return 100_000


def draw_scale_bar(ax, x0, x1, y0, y1):
    L = nice_scale_length(x1 - x0)
    pad_x = (x1 - x0) * 0.06
    pad_y = (y1 - y0) * 0.07
    bx, by = x0 + pad_x, y0 + pad_y
    ax.plot([bx, bx + L], [by, by], color="#333333", lw=1.4,
            solid_capstyle="butt", zorder=6)
    for xx in (bx, bx + L):                       # end ticks
        ax.plot([xx, xx], [by, by + (y1 - y0) * 0.014],
                color="#333333", lw=1.4, zorder=6)
    lab = f"{L/1000:g} km" if L >= 1000 else f"{L:g} m"
    ax.text(bx + L / 2, by + (y1 - y0) * 0.022, lab, ha="center",
            va="bottom", fontsize=5.6, color="#333333", zorder=6)


def draw_panel(ax, city, locs, utm, labels, terminals):
    flagged_n = len(utm)
    zones = sorted(set(labels) - {-1})

    # ── extent: flagged points plus a margin; terminals only if close ──
    if flagged_n:
        x0, x1 = utm[:, 0].min(), utm[:, 0].max()
        y0, y1 = utm[:, 1].min(), utm[:, 1].max()
    else:
        x0 = x1 = y0 = y1 = 0.0
    span = max(x1 - x0, y1 - y0, 5_000)
    cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
    half = span * 0.62                       # square panel, 24% margin
    x0, x1 = cxm - half, cxm + half
    y0, y1 = cym - half, cym + half

    # ── zone hulls, drawn under the points ──
    if zones:
        cmap = plt.get_cmap("turbo")
        cols = [cmap(v) for v in np.linspace(0.08, 0.92, len(zones))]
    else:
        cols = []
    for z, col in zip(zones, cols):
        pts = utm[labels == z]
        if len(pts) >= 3:
            try:
                h = ConvexHull(pts)
                ax.add_patch(MplPolygon(pts[h.vertices], closed=True,
                                        facecolor=col, alpha=0.16,
                                        edgecolor=col, linewidth=0.5,
                                        zorder=2))
            except Exception:
                pass

    # ── points ──
    noise = labels == -1
    if noise.any():
        ax.scatter(utm[noise, 0], utm[noise, 1], s=1.6, c=NOISE_COLOR,
                   linewidths=0, zorder=3, rasterized=True)
    for z, col in zip(zones, cols):
        m = labels == z
        ax.scatter(utm[m, 0], utm[m, 1], s=2.6, color=col,
                   linewidths=0, zorder=4, rasterized=True)

    # ── terminals (only those inside the panel) ──
    for kind, style in TERM_STYLE.items():
        T = terminals.get(kind, np.empty((0, 2)))
        if len(T) == 0:
            continue
        keep = ((T[:, 0] >= x0) & (T[:, 0] <= x1) &
                (T[:, 1] >= y0) & (T[:, 1] <= y1))
        if keep.any():
            ax.scatter(T[keep, 0], T[keep, 1], marker=style["marker"],
                       c=style["color"], s=26, edgecolor="white",
                       linewidth=0.5, zorder=5)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#999999")

    n_ent = int(locs["n_enterprises"].sum()) if flagged_n else 0
    ax.set_title(f"{CITY_LABEL[city]}", fontsize=8, pad=3)
    ax.text(0.5, -0.045,
            f"{n_ent:,} enterprises · {flagged_n:,} locations · "
            f"{len(zones)} zone{'s' if len(zones) != 1 else ''}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color="#555555")
    draw_scale_bar(ax, x0, x1, y0, y1)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():

    print("Loading cities …")
    data, all_mfai = {}, []
    for c in CITIES:
        df = load_city(c)
        locs, utm = build_unique_locs(df, UTM_EPSG[c])
        data[c] = (locs, utm)
        all_mfai.append(df[COL_MFAI].to_numpy(float))
        print(f"  {c}: {len(locs)} locs")

    thr = np.percentile(np.concatenate(all_mfai), POOLED_PCT)
    print(f"\nPooled P{POOLED_PCT} = {thr:.4f}  (main analysis: 0.7065)\n")

    panels = {}
    for c in CITIES:
        locs, utm = data[c]
        mask = locs[COL_MFAI].to_numpy(float) < thr
        fl, fu = locs[mask].reset_index(drop=True), utm[mask]
        if len(fu) >= MCS:
            labels = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                                     metric="euclidean").fit(fu).labels_
        else:
            labels = np.full(len(fu), -1)
        terms = {k: load_terminals(c, k, UTM_EPSG[c]) for k in TERM_STYLE}
        panels[c] = (fl, fu, labels, terms)
        nz = len(set(labels) - {-1})
        print(f"  {c}: {int(fl['n_enterprises'].sum()):,} ent, "
              f"{len(fu):,} locs, {nz} zones "
              f"(terminals: {', '.join(f'{k}={len(v)}' for k, v in terms.items())})")

    # ── figure: 2 x 3, legend in the sixth cell ──
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H))
    axes = axes.ravel()
    for ax, c in zip(axes, CITIES):
        draw_panel(ax, c, *panels[c])

    lax = axes[5]
    lax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444",
               markersize=3.4, linestyle="", label="Flagged location, in zone"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NOISE_COLOR,
               markersize=3.4, linestyle="", label="Flagged location, unclustered"),
        MplPolygon([[0, 0]], facecolor="#888888", alpha=0.25,
                   edgecolor="#888888", label="Zone extent (convex hull)"),
    ] + [
        Line2D([0], [0], marker=s["marker"], color="w",
               markerfacecolor=s["color"], markeredgecolor="white",
               markersize=5, linestyle="", label=s["label"])
        for s in TERM_STYLE.values()
    ]
    lax.legend(handles=handles, loc="center", fontsize=6.2, frameon=False,
               labelspacing=0.9, handletextpad=0.6)
    lax.text(0.5, 0.06,
             "Zone colours distinguish zones within a panel only.\n"
             "Each panel has its own scale — read the scale bars.",
             transform=lax.transAxes, ha="center", va="bottom",
             fontsize=5.6, color="#666666", style="italic")

    plt.tight_layout(w_pad=1.0, h_pad=2.0)
    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 400})]:
        fig.savefig(os.path.join(out_dir, f"fig7_zones_pooled.{ext}"),
                    bbox_inches="tight", **kw)
    print(f"\nSaved fig7_zones_pooled.pdf / .png to {out_dir}/")

    # ── export for QGIS, in case cartographic finishing is wanted ──
    rows = []
    for c, (fl, fu, labels, _) in panels.items():
        rows.append(pd.DataFrame({
            "city": c, "lat": fl[COL_LAT].values, "lon": fl[COL_LON].values,
            "x_utm": fu[:, 0], "y_utm": fu[:, 1],
            "n_enterprises": fl["n_enterprises"].values,
            "mfai_land": fl[COL_MFAI].values, "zone": labels}))
    ex = pd.concat(rows, ignore_index=True)
    ex.to_csv(os.path.join(out_dir, "fig7_flagged_points.csv"), index=False)
    print(f"Exported {len(ex):,} flagged points with zone labels "
          f"→ fig7_flagged_points.csv")

    plt.show()
    return panels, ex


if __name__ == "__main__":
    panels, ex = main()
