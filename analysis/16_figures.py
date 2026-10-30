"""Publication figures, one function per figure

Produces: Figures 1-6 and S1-S2 -> results/figures/ (Figure 7 comes from 11_fig7_zones.py)

Each function produces a single standalone figure sized to Elsevier
column widths, so figures can be placed independently in the manuscript.
    make_charts()                       # all nine charts
    make_maps(res)                      # map fallbacks, if not using QGIS
    fig2_modes()                        # or call individually
    fig7_zones_city("kolkata", res)
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from scipy import stats
from pyproj import Transformer

from mfai import paths
from mfai.io import load_city

# ══════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════
W1, W15, W2 = 3.54, 5.51, 7.48        # inches: single, 1.5, double column

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.4,
    "lines.linewidth": 1.3, "legend.frameon": False,
    "legend.handlelength": 1.8, "legend.columnspacing": 1.2,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUTDIR = str(paths.results("figures"))

CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}
UTM = {"kochi": 32643, "kolkata": 32645, "patna": 32645,
       "varanasi": 32644, "indore": 32643}
CITY_CENTRE = {"patna": (25.6127, 85.1416), "indore": (22.7177, 75.8545),
               "varanasi": (25.3109, 83.0104), "kochi": (9.9816, 76.2830),
               "kolkata": (22.5697, 88.3497)}

CITY_STYLE = {
    "patna":    {"color": "#1b3a6b", "ls": "-"},
    "indore":   {"color": "#c1121f", "ls": (0, (5, 1.5))},
    "varanasi": {"color": "#d98c00", "ls": (0, (4, 1.2, 1, 1.2))},
    "kochi":    {"color": "#2a9d8f", "ls": (0, (1.4, 1.2))},
    "kolkata":  {"color": "#6a4c93", "ls": (0, (3, 1, 1, 1, 1, 1))},
}
MODE_STYLE = {
    "road":  {"color": "#264653", "ls": "-",                 "label": "Major road"},
    "rail":  {"color": "#e76f51", "ls": (0, (5, 1.5)),       "label": "Rail goods shed"},
    "water": {"color": "#2a9d8f", "ls": (0, (4, 1.2, 1, 1.2)), "label": "IWT terminal"},
    "icd":   {"color": "#b5179e", "ls": (0, (1.4, 1.2)),     "label": "ICD"},
}
MODES = ["road", "rail", "water", "icd"]
COL = "mfai_land"

FALLBACK_THRESHOLD, FALLBACK_MIN_PINCODES, DIV_DIVERSITY_T = 20, 5, 8


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════


def _terminals(city, kind):
    stem = paths.terminals(kind, city)
    for ext in (".gpkg", ".csv"):
        p = f"{stem}{ext}"
        if not os.path.exists(p):
            continue
        if ext == ".gpkg":
            return gpd.read_file(p).to_crs(UTM[city])
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        la = [c for c in df.columns if c.lower() in ("latitude", "lat")][0]
        lo = [c for c in df.columns if c.lower() in ("longitude", "lon", "lng")][0]
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lo], df[la]),
                                crs=4326).to_crs(UTM[city])
    return None


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/{name}.{ext}")
    print(f"  {OUTDIR}/{name}.png / .pdf")


def _nice_scale(extent_m):
    """Round scale-bar length, roughly a fifth of the map width."""
    target = extent_m / 5
    for v in (1, 2, 5, 10, 20, 25, 50):
        if target <= v * 1000 * 1.5:
            return v
    return 50


def _scalebar(ax, km=None, frac=(0.05, 0.05), fs=6):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    km = km or _nice_scale(x1 - x0)
    L = km * 1000
    sx, sy = x0 + frac[0]*(x1-x0), y0 + frac[1]*(y1-y0)
    ax.add_patch(Rectangle((sx, sy), L, 0.010*(y1-y0), fc="black",
                           ec="black", lw=0.4, zorder=20))
    ax.text(sx + L/2, sy + 0.022*(y1-y0), f"{km} km", ha="center",
            va="bottom", fontsize=fs, zorder=20)


def _north(ax, frac=(0.95, 0.86), fs=7):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    X, Y = x0 + frac[0]*(x1-x0), y0 + frac[1]*(y1-y0)
    ax.annotate("N", xy=(X, Y), xytext=(X, Y - 0.09*(y1-y0)),
                ha="center", fontsize=fs, fontweight="bold", zorder=20,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9))


def _mapaxes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — context map, ONE CITY PER FIGURE
# ══════════════════════════════════════════════════════════════════════
def fig1_map_city(city, gridsize=40, width=W1, save=True):
    """Manufacturing density and terminal geography for one city.

    Cities are drawn separately because extents range from 13.5 km
    (Kolkata) to 94 km (Patna): a shared panel row at true aspect either
    misaligns the panels or reduces the compact cities to a dot. Each
    figure carries its own scale bar.
    """
    d = load_city(city)
    tf = Transformer.from_crs("EPSG:4326", UTM[city], always_xy=True)
    x, y = tf.transform(d.lon.values, d.lat.values)
    span = max(x.max()-x.min(), y.max()-y.min())
    h = width * (y.max()-y.min()) / max(x.max()-x.min(), 1) + 0.55

    fig, ax = plt.subplots(figsize=(width, min(max(h, 2.0), 5.0)))
    try:
        gpd.read_file(paths.roads_major(city)).to_crs(
            UTM[city]).plot(ax=ax, color="0.82", lw=0.3, zorder=1)
    except Exception:
        pass

    hb = ax.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=1,
                   cmap="Blues", linewidths=0, zorder=2, alpha=0.92)

    for k, mk, ms in [("rail", "s", 20), ("water", "^", 22), ("icd", "o", 22)]:
        t = _terminals(city, k)
        if t is None or len(t) == 0:
            continue
        p = t.geometry.representative_point()
        ax.scatter(p.x, p.y, marker=mk, s=ms, zorder=6,
                   facecolor=MODE_STYLE[k]["color"], edgecolor="white",
                   linewidth=0.5)

    _mapaxes(ax)
    _scalebar(ax); _north(ax)
    ax.set_title(f"{CITY_LABEL[city]}   n = {len(d):,}", fontsize=8.5)

    cb = fig.colorbar(hb, ax=ax, fraction=0.032, pad=0.015)
    cb.set_label("Enterprises per cell (log)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)

    if save:
        _save(fig, f"fig1_map_{city}")
    return fig


def fig1_legend(save=True):
    """Shared legend for the city maps, placed once in the caption row."""
    fig, ax = plt.subplots(figsize=(W2, 0.34))
    ax.axis("off")
    h = [Line2D([], [], marker=m, ls="", mfc=MODE_STYLE[k]["color"],
                mec="white", mew=0.5, ms=5, label=MODE_STYLE[k]["label"])
         for k, m in [("rail", "s"), ("water", "^"), ("icd", "o")]]
    h.append(Line2D([], [], color="0.82", lw=1.2, label="Major roads"))
    ax.legend(handles=h, loc="center", ncol=4, frameon=False, fontsize=7.5)
    if save:
        _save(fig, "fig1_legend")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — mode-specific access  [HEADLINE]
# ══════════════════════════════════════════════════════════════════════
def fig2_modes(cities=None, width=W2, save=True):
    """Binding-constraint result.

    The x-axis starts at 0 so Indore's constant IWT score (0.0499) is
    visible; the previous version clipped at 0.15 and omitted the very
    line its caption described.
    """
    cities = cities or CITY_ORDER
    fig, axes = plt.subplots(1, len(cities), figsize=(width, 2.15),
                             sharex=True, sharey=True)
    grid = np.linspace(0.0, 1.0, 600)

    for ax, c in zip(axes, cities):
        d = load_city(c)
        means = {}
        for m in MODES:
            col = f"access_{m}"
            if col not in d.columns:
                continue
            v = d[col].dropna().values
            if v.std() < 1e-9:
                ax.axvline(v[0], color=MODE_STYLE[m]["color"], lw=1.2,
                           ls=(0, (1.2, 1.2)), zorder=5)
                ax.annotate(f"{v[0]:.2f}", xy=(v[0], 0.92),
                            xycoords=("data", "axes fraction"),
                            fontsize=6, color=MODE_STYLE[m]["color"],
                            ha="left", rotation=90, va="top")
                means[m] = v[0]
                continue
            ax.plot(grid, stats.gaussian_kde(v)(grid),
                    color=MODE_STYLE[m]["color"], ls=MODE_STYLE[m]["ls"],
                    lw=1.25)
            means[m] = v.mean()

        land = {m: means[m] for m in ["road", "rail", "icd"] if m in means}
        b = min(land, key=land.get)
        s = sorted(land.values())
        ax.set_title(f"{CITY_LABEL[c]}\n{MODE_STYLE[b]['label']}, "
                     f"margin {s[1]-s[0]:.3f}", fontsize=7.5, linespacing=1.35)
        ax.set_yticks([])
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlabel("$a = T/(T+d)$")

    axes[0].set_ylabel("Density")
    h = [Line2D([], [], color=MODE_STYLE[m]["color"], ls=MODE_STYLE[m]["ls"],
                label=MODE_STYLE[m]["label"]) for m in MODES]
    fig.legend(handles=h, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.16))
    if save:
        _save(fig, "fig2_modes")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — truncation radius  [TWO MECHANISMS]
# ══════════════════════════════════════════════════════════════════════
def fig3_truncation(cities=None, radii=(5,10,15,20,25,30,40,60,200),
                    width=W15, save=True):
    cities = cities or CITY_ORDER
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(width, 2.5),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    for c in cities:
        d = load_city(c)
        xs, ys, ns = [], [], []
        for R in radii:
            s = d[d.dist_centre_km <= R]
            if len(s) < 50:
                continue
            xs.append(R); ys.append(s[COL].mean()); ns.append(100*len(s)/len(d))
        a1.plot(xs, ys, marker="o", ms=2.8, label=CITY_LABEL[c], **CITY_STYLE[c])
        a2.plot(xs, ns, marker="o", ms=2.8, **CITY_STYLE[c])

    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks([5, 10, 20, 30, 60, 200])
        a.set_xticklabels(["5", "10", "20", "30", "60", "full"])
        a.set_xlabel("Truncation radius (km)")
    a1.set_ylabel("Mean MFAI (road, rail, ICD)")
    a2.set_ylabel("Enterprises retained (%)")
    a1.set_title("Accessibility by spatial extent", fontsize=8)
    a2.set_title("Settlement extent", fontsize=8)
    a1.legend(loc="center left", bbox_to_anchor=(0.02, 0.32), fontsize=6.5)
    if save:
        _save(fig, "fig3_truncation")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — ECDF  [CONTINUITY]
# ══════════════════════════════════════════════════════════════════════
def fig4_ecdf(cities=None, width=W15, save=True):
    """Bandwidth-free. A shoulder here is a real distributional feature."""
    cities = cities or CITY_ORDER
    fig, ax = plt.subplots(figsize=(width, 2.7))
    for c in cities:
        v = np.sort(load_city(c)[COL].dropna().values)
        ax.plot(v, np.arange(1, len(v)+1)/len(v), label=CITY_LABEL[c],
                **CITY_STYLE[c])
    ax.axhline(0.10, color="0.3", lw=0.6, ls=(0, (3, 2)))
    ax.text(0.215, 0.125, "bottom decile", fontsize=6.5, color="0.35")
    ax.set_xlabel("MFAI (road, rail, ICD)")
    ax.set_ylabel("Cumulative share of enterprises")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=7)
    if save:
        _save(fig, "fig4_ecdf")
    return fig


def fig4b_dip(cities=None, width=W1, save=True):
    """Dip statistic with p-values annotated.

    Separate figure: the previous version created the axis inside a try
    block and left it empty when diptest was unavailable.
    """
    try:
        import diptest
    except ImportError:
        print("  fig4b skipped — pip install diptest")
        return None
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        v = load_city(c)[COL].dropna().values
        dp, p = diptest.diptest(v)
        rows.append({"city": c, "dip": dp, "p": p, "n": len(v)})
    t = pd.DataFrame(rows).sort_values("dip")

    fig, ax = plt.subplots(figsize=(width, 2.1))
    for i, r in enumerate(t.itertuples()):
        ax.barh(i, r.dip, color=CITY_STYLE[r.city]["color"], height=0.6,
                alpha=0.9)
        ax.text(r.dip + 0.00035, i, f"p = {r.p:.3g}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([CITY_LABEL[c] for c in t.city])
    ax.set_xlabel("Hartigan's dip statistic")
    ax.set_xlim(0, t.dip.max()*1.45)
    ax.grid(axis="y", visible=False)
    if save:
        _save(fig, "fig4b_dip")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — variance decomposition
# ══════════════════════════════════════════════════════════════════════
def fig5_variance_modes(cities=None, width=W15, save=True):
    cities = cities or CITY_ORDER
    shares = {}
    for c in cities:
        d = load_city(c)
        contrib = {}
        for m in MODES:
            col = f"access_{m}"
            contrib[m] = ((0.25**2) * np.log(d[col].clip(lower=1e-9)).var()
                          if col in d.columns and d[col].std() > 1e-9 else 0.0)
        tot = sum(contrib.values())
        shares[c] = {m: 100*contrib[m]/tot for m in MODES}

    order = sorted(cities, key=lambda c: shares[c]["road"])
    fig, ax = plt.subplots(figsize=(width, 2.3))
    bottom = np.zeros(len(order))
    for m in MODES:
        v = np.array([shares[c][m] for c in order])
        ax.barh([CITY_LABEL[c] for c in order], v, left=bottom,
                color=MODE_STYLE[m]["color"], label=MODE_STYLE[m]["label"],
                height=0.6, edgecolor="white", linewidth=0.7)
        for i, (a, b) in enumerate(zip(v, bottom)):
            if a > 6:
                ax.text(b + a/2, i, f"{a:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
        bottom += v
    ax.set_xlabel("Share of within-city MFAI variance (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42),
              fontsize=7)
    if save:
        _save(fig, "fig5_variance_modes")
    return fig


def fig5b_variance_components(cities=None, width=W1, save=True):
    cities = cities or CITY_ORDER
    frames = []
    for c in cities:
        d = load_city(c)
        frames.append(pd.DataFrame({COL: d[COL], "city": c,
                                    "grp": d.get("mfg_group")}))
    df = pd.concat(frames, ignore_index=True).dropna(subset=[COL])
    y = df[COL]; g = y.mean()
    sst = ((y-g)**2).sum()
    cm = df.groupby("city")[COL].transform("mean")
    ssc = ((cm-g)**2).sum()
    if df.grp.notna().any():
        gm = df.groupby(["city", "grp"])[COL].transform("mean")
        ssg = ((gm-cm)**2).sum(); ssr = ((y-gm)**2).sum()
    else:
        ssg, ssr = 0.0, ((y-cm)**2).sum()

    parts = [100*ssc/sst, 100*ssg/sst, 100*ssr/sst]
    lbl = ["Between cities", "Between sectors,\nwithin city",
           "Within sector,\nwithin city"]
    cols = ["#4a5759", "#b0c4b1", "#e8b4bc"]

    fig, ax = plt.subplots(figsize=(width, 2.5))
    b = 0
    for p, l, cc in zip(parts, lbl, cols):
        ax.bar(0, p, bottom=b, color=cc, width=0.5, edgecolor="white", lw=0.8)
        if p > 4:
            ax.text(0, b + p/2, f"{l}\n{p:.1f}%", ha="center", va="center",
                    fontsize=7, linespacing=1.3)
        else:
            ax.annotate(f"{l}  {p:.1f}%", xy=(0.26, b + p/2),
                        xytext=(0.62, b + p/2), fontsize=6.8, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
        b += p
    ax.set_xticks([]); ax.set_xlim(-0.45, 1.25); ax.set_ylim(0, 100)
    ax.set_ylabel("Share of total variance (%)")
    ax.grid(axis="x", visible=False)
    if save:
        _save(fig, "fig5b_variance_components")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — agglomeration coefficients
# ══════════════════════════════════════════════════════════════════════
AGG = pd.DataFrame([
    ("patna","road",0.8735,0.0872), ("patna","rail",-0.2980,0.1348),
    ("patna","water",0.5887,0.2244), ("patna","icd",-0.2838,0.1129),
    ("indore","road",0.6329,0.1113), ("indore","rail",0.7625,0.1818),
    ("indore","icd",0.3589,0.1341),
    ("varanasi","road",0.3187,0.0842), ("varanasi","rail",0.4216,0.0808),
    ("varanasi","water",-0.4132,0.1283), ("varanasi","icd",-0.1086,0.1526),
    ("kochi","road",0.3287,0.0472), ("kochi","rail",0.2404,0.0687),
    ("kochi","water",0.1021,0.1044), ("kochi","icd",0.3751,0.1097),
], columns=["city", "mode", "coef", "se"])


def fig6_agglomeration(tab=None, width=W2, save=True):
    tab = AGG if tab is None else tab
    cities = [c for c in CITY_ORDER if c in set(tab.city)]
    fig, axes = plt.subplots(1, 4, figsize=(width, 2.2), sharex=True,
                             sharey=True)
    for ax, m in zip(axes, MODES):
        sub = tab[tab["mode"] == m]
        for i, c in enumerate(cities):
            r = sub[sub.city == c]
            if not len(r):
                ax.plot(0, i, marker="x", ms=4, color="0.7")
                continue
            b, se = r.coef.iloc[0], r.se.iloc[0]
            ax.errorbar(b, i, xerr=1.96*se, fmt="o", ms=3.4,
                        color=CITY_STYLE[c]["color"], capsize=1.8, lw=1.0,
                        elinewidth=1.0)
        ax.axvline(0, color="0.3", lw=0.6, ls=(0, (3, 2)))
        ax.set_yticks(range(len(cities)))
        ax.set_yticklabels([CITY_LABEL[c] for c in cities])
        ax.set_title(MODE_STYLE[m]["label"], fontsize=7.5)
        ax.set_xlabel("coefficient per SD")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[0].annotate("Kolkata excluded:\n169 cells at 1 km",
                     xy=(0.02, -0.42), xycoords="axes fraction",
                     fontsize=6.3, color="0.35", va="top")
    if save:
        _save(fig, "fig6_agglomeration")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — deprivation zones, ONE CITY PER FIGURE
# ══════════════════════════════════════════════════════════════════════
def fig7_zones_city(city, res, width=W1, save=True):
    """Zone structure for one city.

    Drawn separately: cities differ in extent by a factor of seven, and
    a shared row at true aspect misaligns. Zone colours are nominal, so
    the legend describes categories rather than a specific colour.
    """
    r = res["cities"].get(city)
    if r is None:
        return None
    loc, z = r["loc"], r["zones"]
    d = load_city(city)
    tf = Transformer.from_crs("EPSG:4326", UTM[city], always_xy=True)
    bx, by = tf.transform(d.lon.values, d.lat.values)
    h = width * (by.max()-by.min()) / max(bx.max()-bx.min(), 1) + 0.75

    fig, ax = plt.subplots(figsize=(width, min(max(h, 2.2), 5.0)))
    ax.scatter(bx, by, s=0.25, c="0.90", lw=0, zorder=1)

    noise = loc.zone == -1
    ax.scatter(loc.loc[noise, "_x"], loc.loc[noise, "_y"], s=2.6,
               c="0.55", lw=0, zorder=2)

    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(z), 1)) % 1.0) \
        if len(z) <= 20 else plt.cm.turbo(np.linspace(0.05, 0.95, len(z)))
    from scipy.spatial import ConvexHull
    for i, zid in enumerate(z.zone.values):
        m = loc.zone == zid
        col = cmap[i]
        ax.scatter(loc.loc[m, "_x"], loc.loc[m, "_y"], s=4.2, color=col,
                   lw=0, zorder=3)
        pts = loc.loc[m, ["_x", "_y"]].values
        if len(pts) >= 3:
            try:
                hl = ConvexHull(pts)
                poly = pts[np.append(hl.vertices, hl.vertices[0])]
                ax.plot(poly[:, 0], poly[:, 1], color=col, lw=0.8,
                        alpha=0.9, zorder=4)
            except Exception:
                pass

    _mapaxes(ax); _scalebar(ax); _north(ax)
    cc = r["concentration"]
    small = z[z.area_km2 <= 25].firms.sum() if len(z) else 0
    ax.set_title(f"{CITY_LABEL[city]}\n"
                 f"{cc['n_zones']} zones · {cc['firms_in_zones_pct']:.0f}% in zones · "
                 f"{100*small/cc['firms']:.0f}% in zones < 25 km²\n"
                 f"largest zone {z.area_km2.max():.0f} km²",
                 fontsize=7.3, linespacing=1.4)
    if save:
        _save(fig, f"fig7_zones_{city}")
    return fig


def fig7_legend(save=True):
    fig, ax = plt.subplots(figsize=(W2, 0.34))
    ax.axis("off")
    h = [Line2D([], [], marker="o", ls="", mfc="0.90", mec="none", ms=3.5,
                label="All manufacturing MSMEs"),
         Line2D([], [], marker="o", ls="", mfc="0.55", mec="none", ms=4,
                label="Lowest decile, not in a zone"),
         Line2D([], [], marker="o", ls="", mfc="#1f77b4", mec="none", ms=4,
                label="Lowest decile, in a zone (colour denotes zone)")]
    ax.legend(handles=h, loc="center", ncol=3, frameon=False, fontsize=7)
    if save:
        _save(fig, "fig7_legend")
    return fig


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY
# ══════════════════════════════════════════════════════════════════════
def figS1_bandwidth(cities=None, bws=(0.15, 0.3, 0.5, 0.8), width=W2,
                    save=True):
    cities = cities or CITY_ORDER
    fig, axes = plt.subplots(1, len(cities), figsize=(width, 1.9))
    greys = np.linspace(0.75, 0.05, len(bws))
    for ax, c in zip(axes, cities):
        v = load_city(c)[COL].dropna().values
        g = np.linspace(v.min(), v.max(), 512)
        for bw, sh in zip(bws, greys):
            ax.plot(g, stats.gaussian_kde(v, bw_method=bw)(g), color=str(sh),
                    lw=1.0, label=f"{bw}")
        ax.set_title(CITY_LABEL[c], fontsize=7.5)
        ax.set_yticks([])
        ax.set_xlim(0.3, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlabel("MFAI")
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=6, title="bandwidth", title_fontsize=6,
                    loc="upper left")
    if save:
        _save(fig, "figS1_bandwidth")
    return fig


def figS2_monocentric(cities=None, min_n=30, width=W2, save=True):
    """Bins holding fewer than min_n enterprises are dropped: Patna's
    tail previously oscillated on a handful of observations."""
    cities = cities or CITY_ORDER
    fig, axes = plt.subplots(1, len(cities), figsize=(width, 2.0),
                             sharey=True)
    for ax, c in zip(axes, cities):
        d = load_city(c)
        hi = d.dist_centre_km.quantile(0.97)
        bins = np.arange(0, hi + 3, 3)
        cut = pd.cut(d.dist_centre_km, bins)
        keep = d.groupby(cut, observed=True).size() >= min_n
        for m in MODES:
            col = f"access_{m}"
            if col not in d.columns or d[col].std() < 1e-9:
                continue
            g = d.groupby(cut, observed=True)[col].mean()[keep]
            ax.plot([i.mid for i in g.index], g.values,
                    color=MODE_STYLE[m]["color"], ls=MODE_STYLE[m]["ls"],
                    lw=1.1)
        ax.set_title(CITY_LABEL[c], fontsize=7.5)
        ax.set_xlabel("km from centre")
    axes[0].set_ylabel("Mean access score")
    h = [Line2D([], [], color=MODE_STYLE[m]["color"], ls=MODE_STYLE[m]["ls"],
                label=MODE_STYLE[m]["label"]) for m in MODES]
    fig.legend(handles=h, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.17))
    if save:
        _save(fig, "figS2_monocentric")
    return fig


def figS3_threshold(res, width=W15, save=True):
    fig, ax = plt.subplots(figsize=(width, 2.5))
    for c in CITY_ORDER:
        t = res["threshold"].get(c)
        if t is None:
            continue
        rel = t[t.type == "relative"]
        ax.plot([int(s[1:]) for s in rel.threshold], rel.firms_in_zones_pct,
                marker="o", ms=3, label=CITY_LABEL[c], **CITY_STYLE[c])
    ax.set_xlabel("Threshold (percentile of MFAI)")
    ax.set_ylabel("Lowest-decile enterprises inside zones (%)")
    ax.set_xticks([5, 10, 15, 20])
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    if save:
        _save(fig, "figS3_threshold")
    return fig


def figS4_null(res, width=W15, save=True):
    n = res["null"].copy()
    order = {CITY_LABEL[c]: i for i, c in enumerate(CITY_ORDER)}
    n = n.assign(_o=n.City.map(order)).sort_values("_o", ascending=False)
    fig, ax = plt.subplots(figsize=(width, 2.4))
    for i, r in enumerate(n.itertuples()):
        lo, hi = getattr(r, "_8"), getattr(r, "_9")
        ax.plot([lo, hi], [i, i], color="0.78", lw=6, solid_capstyle="butt",
                zorder=1)
        ax.plot(getattr(r, "_7"), i, "|", color="0.3", ms=10, mew=1.4, zorder=2)
        cc = next(v["color"] for k, v in CITY_STYLE.items()
                  if CITY_LABEL[k] == r.City)
        ax.plot(getattr(r, "_6"), i, "o", color=cc, ms=5.5, zorder=3)
    ax.set_yticks(range(len(n))); ax.set_yticklabels(n.City)
    ax.set_xlabel("Lowest-decile enterprises inside zones (%)")
    ax.set_xlim(-3, 105)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="0.3", ms=5, label="observed"),
        Line2D([], [], marker="|", ls="", color="0.3", mew=1.4, label="null mean"),
        Line2D([], [], color="0.78", lw=6, label="null 5–95%")],
        loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6.8)
    if save:
        _save(fig, "figS4_null")
    return fig


# ══════════════════════════════════════════════════════════════════════
def make_charts():
    print("charts")
    fig2_modes(); fig3_truncation(); fig4_ecdf(); fig4b_dip()
    fig5_variance_modes(); fig5b_variance_components()
    fig6_agglomeration(); figS1_bandwidth(); figS2_monocentric()
    plt.show()


def make_maps(res=None, cities=None):
    print("maps")
    for c in (cities or CITY_ORDER):
        fig1_map_city(c)
    fig1_legend()
    if res is not None:
        for c in (cities or CITY_ORDER):
            fig7_zones_city(c, res)
        fig7_legend()
        figS3_threshold(res); figS4_null(res)
    plt.show()


def export_for_qgis(res=None, outdir=None):
    """Write firm points and flagged locations for QGIS map production.
    Output contains enterprise coordinates: keep it out of version control."""
    outdir = outdir or str(paths.results("qgis"))
    for c in CITY_ORDER:
        d = load_city(c)
        keep = [k for k in ["msme_id", COL, "mfg_group", "isic_division",
                            "access_road", "access_rail", "access_icd"]
                if k in d.columns]
        gpd.GeoDataFrame(d[keep],
                         geometry=gpd.points_from_xy(d.lon, d.lat),
                         crs=4326).to_file(f"{outdir}/{c}_firms.gpkg",
                                           driver="GPKG")
        if res is not None and res["cities"].get(c) is not None:
            loc = res["cities"][c]["loc"]
            gpd.GeoDataFrame(loc[["n_firms", "zone", COL]],
                             geometry=gpd.points_from_xy(loc.lon, loc.lat),
                             crs=4326).to_file(f"{outdir}/{c}_flagged.gpkg",
                                               driver="GPKG")
        print(f"  {outdir}/{c}_*.gpkg")


if __name__ == "__main__":
    make_charts()
    make_maps()
