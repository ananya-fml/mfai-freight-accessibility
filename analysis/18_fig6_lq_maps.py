"""Figure 6: commodity location-quotient maps

Produces: Figure 6 -> results/figures/LQ_matrix_final.{pdf,png}

LQ MATRIX — Pinheiro-style, with district boundaries
  tile basemap · graticule · per-panel north arrow + scale bar
  · shared colourbar · district outline per city row
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx

import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from shapely.geometry import box
from pyproj import Transformer

from mfai import paths
from mfai.config import GROUP_LABELS
from mfai.io import load_city_gdf

OUT_DIR     = str(paths.results("figures"))

ORDER    = ["patna", "indore", "varanasi", "kochi", "kolkata"]
GROUPS   = ["G2", "G5", "G6"]
CELL_FOR = {c: (1000 if c == "kolkata" else 2000) for c in ORDER}
SPAN_KM  = {"patna": 55, "indore": 46, "varanasi": 44, "kochi": 54, "kolkata": 22}
# ---- district boundary files. Point these at whatever you have. ----
# Any polygon file works (.gpkg/.shp/.geojson). Missing entries are skipped
# and that city simply gets no outline. Set DISTRICT_DIR if they live together.
DISTRICT_FILE = {c: [paths.district(c)] for c in ORDER}
OUTLINE_STYLE = dict(color="#444444", lw=0.7, linestyle="--")

MIN_FIRMS, EB_K, HOTSPOT_THR, DRAW_MIN = 20, 20, 2.0, 1.2
PANEL_IN, DPI = 2.55, 400
SHOW_TERMINALS, GRATICULE = True, True
BASEMAP = ctx.providers.Esri.WorldGrayCanvas

BOUNDS = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
COLORS = ["#fdd49e", "#fc8d59", "#e34a33", "#b30000", "#67000d"]
CMAP   = LinearSegmentedColormap.from_list("lq", COLORS, N=len(COLORS))
NORM   = BoundaryNorm(BOUNDS, CMAP.N)

TERM_STYLE = {
    "rail": dict(kind="rail", marker="s",
                 color="#E76F51", label="Rail goods shed"),
    "iwt":  dict(kind="water", marker="^",
                 color="#2A9D8F", label="IWT terminal"),
    "icd":  dict(kind="icd", marker="o",
                 color="#B5179E", label="ICD / CFS"),
}

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TO_LL = Transformer.from_crs(3857, 4326, always_xy=True)
TO_WM = Transformer.from_crs(4326, 3857, always_xy=True)


def build_grid(gdf, cs):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin, ymin = np.floor(xmin/cs)*cs, np.floor(ymin/cs)*cs
    xmax, ymax = np.ceil(xmax/cs)*cs,  np.ceil(ymax/cs)*cs
    cells = [box(x, y, x+cs, y+cs) for x in np.arange(xmin, xmax, cs)
             for y in np.arange(ymin, ymax, cs)]
    return gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=gdf.crs)


def clip_builtup(grid, city):
    try:
        import rasterio
        from rasterstats import zonal_stats
        p = paths.ghsl(city)
        if not os.path.exists(p): return grid
        with rasterio.open(p) as src:
            st = zonal_stats(grid.to_crs(src.crs).geometry, p,
                             stats=["mean"], nodata=src.nodata)
        grid = grid.copy(); grid["bu"] = [s["mean"] or 0.0 for s in st]
        return grid[grid.bu > 0].reset_index(drop=True)
    except (ImportError, OSError):
        print(f"  {city}: GHSL clip unavailable"); return grid


def lq_cells(gdf, city):
    grid = clip_builtup(build_grid(gdf, CELL_FOR[city]), city)
    j = gpd.sjoin(gdf[["group", "geometry"]], grid, how="inner", predicate="within")
    cnt = j.groupby(["cell_id", "group"]).size().unstack(fill_value=0)
    for g in GROUP_LABELS:
        if g not in cnt.columns: cnt[g] = 0
    cnt = cnt[list(GROUP_LABELS.keys())]
    tot = cnt.sum(1); p = cnt.sum(0)/cnt.sum(0).sum()
    keep = tot >= MIN_FIRMS
    cnt, tot = cnt[keep], tot[keep]
    out = pd.DataFrame({"cell_id": tot.index})
    for g in GROUP_LABELS:
        out[f"LQ_{g}"] = (((cnt[g] + EB_K*p[g])/(tot + EB_K))/p[g]).values
    return grid.merge(out, on="cell_id"), p


def load_points(candidates, crs=3857):
    for p in candidates:
        if os.path.exists(p):
            try:
                g = gpd.read_file(p)
                if g.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                    g = g.copy(); g["geometry"] = g.geometry.centroid
                return g.to_crs(crs)
            except Exception as e:
                print(f"    {os.path.basename(p)}: {type(e).__name__}")
    return None


def load_district(city, city_gdf):
    """Load a district polygon and keep only parts near the study area."""
    for p in DISTRICT_FILE.get(city, []):
        if not os.path.exists(p): continue
        try:
            d = gpd.read_file(p).to_crs(city_gdf.crs)
            # keep polygons intersecting the enterprise extent (drops other
            # districts if the file covers a whole state)
            hull = city_gdf.geometry.union_all().convex_hull
            d = d[d.intersects(hull)]
            if len(d) == 0:
                print(f"  {city}: district file has no polygon over the data")
                return None
            print(f"  {city}: district boundary from {os.path.basename(p)}")
            return d.to_crs(3857)
        except Exception as e:
            print(f"  {city}: district read failed ({type(e).__name__})")
    print(f"  {city}: no district file found — outline omitted")
    return None


def nice_ticks(lo, hi, n=3):
    step = (hi - lo)/n
    mag  = 10**np.floor(np.log10(step))
    for m in (1, 2, 2.5, 5, 10):
        if step <= m*mag: step = m*mag; break
    return np.arange(np.ceil(lo/step)*step, hi, step)


city_data = {c: load_city_gdf(c) for c in ORDER}
CELLS, SHARE, TERMS, WIN, MSC, OUTLINE = {}, {}, {}, {}, {}, {}


def prepare():
    print("preparing layers ...")
    for c in ORDER:
        cells, SHARE[c] = lq_cells(city_data[c], c)
        CELLS[c] = cells.to_crs(3857)
        OUTLINE[c] = load_district(c, city_data[c])
        TERMS[c] = {}
        if SHOW_TERMINALS:
            for k, s in TERM_STYLE.items():
                t = load_points([f"{paths.terminals(s['kind'], c)}.gpkg"])
                if t is not None: TERMS[c][k] = t
        lat = city_data[c].to_crs(4326).geometry.y.mean()
        MSC[c] = 1/np.cos(np.radians(lat))
        cen = CELLS[c].geometry.union_all().centroid
        h = SPAN_KM[c]*500*MSC[c]
        WIN[c] = (cen.x-h, cen.x+h, cen.y-h, cen.y+h)
        print(f"    {len(CELLS[c])} cells @ {CELL_FOR[c]}m, terminals {sorted(TERMS[c])}")


def draw(name="LQ_matrix_final"):
    nrow, ncol = len(ORDER), len(GROUPS)
    L, R, T, B, GX, GY = 0.80, 1.05, 0.44, 0.78, 0.30, 0.34
    W = L + ncol*PANEL_IN + (ncol-1)*GX + R
    H = T + nrow*PANEL_IN + (nrow-1)*GY + B
    fig = plt.figure(figsize=(W, H))

    y = H - T
    for i, c in enumerate(ORDER):
        y -= PANEL_IN
        lon0, lat0 = TO_LL.transform(WIN[c][0], WIN[c][2])
        lon1, lat1 = TO_LL.transform(WIN[c][1], WIN[c][3])
        xs, ys = nice_ticks(lon0, lon1, 3), nice_ticks(lat0, lat1, 3)

        for k, g in enumerate(GROUPS):
            ax = fig.add_axes([(L + k*(PANEL_IN+GX))/W, y/H,
                               PANEL_IN/W, PANEL_IN/H])
            ax.set_xlim(WIN[c][0], WIN[c][1]); ax.set_ylim(WIN[c][2], WIN[c][3])

            if OUTLINE[c] is not None:
                OUTLINE[c].boundary.plot(ax=ax, zorder=2, **OUTLINE_STYLE)

            drawn = CELLS[c][CELLS[c][f"LQ_{g}"] >= DRAW_MIN]
            if len(drawn):
                drawn.plot(ax=ax, column=f"LQ_{g}", cmap=CMAP, norm=NORM,
                           alpha=0.85, ec="white", lw=0.15, zorder=3)
                hot = drawn[drawn[f"LQ_{g}"] >= HOTSPOT_THR]
                if len(hot):
                    hot.dissolve().boundary.plot(ax=ax, color="black",
                                                 lw=0.7, zorder=4)
            for kind, s in TERM_STYLE.items():
                if kind in TERMS[c]:
                    ms = 9 if c == "kochi" else 12
                    TERMS[c][kind].plot(ax=ax, marker=s["marker"], color=s["color"],
                                        markersize=ms, ec="white", lw=0.4,
                                        alpha=0.92, zorder=5)
            try:
                ctx.add_basemap(ax, source=BASEMAP, crs="EPSG:3857",
                                attribution=False, zorder=1)
            except Exception as e:
                print(f"    {c}/{g} basemap: {type(e).__name__}")

            if GRATICULE:
                ax.set_xticks([TO_WM.transform(v, ys[0])[0] for v in xs])
                ax.set_yticks([TO_WM.transform(xs[0], v)[1] for v in ys])
                ax.set_xticklabels([f"{v:.2f}°E" for v in xs], fontsize=4.6)
                ax.set_yticklabels([f"{v:.2f}°N" for v in ys], fontsize=4.6,
                                   rotation=90, va="center")
                if k != 0: ax.set_yticklabels([])
                ax.tick_params(length=2, width=0.4, top=True, right=True, pad=1.5)
            else:
                ax.set_xticks([]); ax.set_yticks([])

            ax.annotate("N", xy=(0.955, 0.935), xytext=(0.955, 0.845),
                        xycoords="axes fraction", ha="center", fontsize=6, zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7))

            span = WIN[c][1]-WIN[c][0]
            sb_km = 10 if SPAN_KM[c] >= 30 else 5
            sb = sb_km*1000*MSC[c]
            x0 = WIN[c][0] + 0.06*span
            y0 = WIN[c][2] + (0.13 if c == "kolkata" else 0.09)*(WIN[c][3]-WIN[c][2])
            ax.add_patch(mpatches.Rectangle((x0-0.025*span, y0-0.028*span),
                         sb+0.10*span, 0.088*span, fc="white", ec="none",
                         alpha=0.78, zorder=6))
            ax.add_patch(mpatches.Rectangle((x0, y0-0.006*span), sb, 0.012*span,
                         fc="none", ec="black", lw=0.4, zorder=7))
            ax.add_patch(mpatches.Rectangle((x0, y0-0.006*span), sb/2, 0.012*span,
                         fc="black", ec="black", lw=0.4, zorder=7))
            ax.text(x0+sb, y0+0.020*span, f"{sb_km} km", ha="center",
                    va="bottom", fontsize=5.2, zorder=7)

            if i == 0:
                ax.set_title(f"{g}  {GROUP_LABELS[g]}", fontsize=8, pad=5)
            if k == 0:
                ax.text(-0.20, 0.5, f"{c.capitalize()}\nn = {len(city_data[c]):,}",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=8, linespacing=1.5)
            ax.text(0.035, 0.955, f"{100*SHARE[c][g]:.0f}% of city",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.0,
                    color="#222222", zorder=8,
                    bbox=dict(fc="white", ec="#999999", lw=0.3, alpha=0.85, pad=1.5))
            for s_ in ax.spines.values(): s_.set_edgecolor("#444444")
        y -= GY

    cax = fig.add_axes([(W-R+0.34)/W, 0.32, 0.16/W, 0.34])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                      ticks=BOUNDS, spacing="uniform", extend="max")
    cb.ax.set_yticklabels(["1.2", "1.5", "2.0", "2.5", "3.0", "4.0"])
    cb.set_label("Location quotient", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2)
    cb.outline.set_linewidth(0.5)

    kinds = [k for k in TERM_STYLE if any(k in TERMS[c] for c in ORDER)]
    handles = [Line2D([], [], marker=TERM_STYLE[k]["marker"], color="none",
                      markerfacecolor=TERM_STYLE[k]["color"],
                      markeredgecolor="white", markersize=6.5) for k in kinds]
    handles += [Line2D([], [], color="black", lw=0.9),
                Line2D([], [], color=OUTLINE_STYLE["color"], lw=0.9,
                       linestyle="--")]
    labels = [TERM_STYLE[k]["label"] for k in kinds] + \
             ["LQ ≥ 2.0 boundary", "District boundary"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=7, handlelength=1.5,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.032))
    fig.text(0.5, 0.008,
             "Basemap: Esri World Gray Canvas.  Analysis in city-specific UTM "
             "zones; panels displayed in Web Mercator.  Cells below LQ 1.2 not "
             "shown.  Map lines delineate study areas and do not necessarily "
             "depict accepted national boundaries.",
             ha="center", fontsize=5.4, color="#666666")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{name}.{ext}", dpi=DPI, facecolor="white",
                    bbox_inches="tight")
    print(f"\nsaved {OUT_DIR}/{name}.pdf  ({W:.1f} × {H:.1f} in)")
    plt.show()


if __name__ == "__main__":
    prepare()
    draw()
