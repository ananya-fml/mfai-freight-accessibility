"""Mode severity x spatial variability; composite masking

Produces: Tables 8-9, Figure 5 -> results/targetability/

Replaces the per-mode variance table (Table 9) with a two-axis diagnostic:
    Severity      D_cm = -log(mean access_cm)     [a LEVEL statistic]
    Variability   V_cm = Var[log(access_icm)]     [a SPREAD statistic]
computed per city c and land mode m (road, rail, ICD) directly from the
firm-level access scores, so the figure is self-consistent.
Water is EXCLUDED: log is ill-behaved at Indore's 0.05 override and the
"""
import os
import numpy as np
import pandas as pd

from mfai import paths
from mfai.io import load_city

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ═══════════════════════════════════════════════════════════════
#  CONFIG  (must match your pipeline — T=8 confirmed)
# ═══════════════════════════════════════════════════════════════
CITIES   = ["patna", "varanasi", "indore", "kochi", "kolkata"]

COL_MFAI    = "mfai_land"
ACCESS_COLS = {"road": "access_road", "rail": "access_rail", "icd": "access_icd"}
LAND_MODES  = ["road", "rail", "icd"]          # water excluded (see docstring)

# Geocoder-fallback exclusion — consume stored flags, do NOT recompute
FALLBACK_THRESHOLD = 20
DIVISION_T         = 8
COL_RULE1, COL_COORD_N, COL_COORD_NDIV = "fallback_rule1", "coord_n", "coord_n_div"

EPS = 1e-6   # clamp for log guard

# Composite-masking: bottom-decile agreement between MFAI and each mode.
# City-relative (each city's own decile) and EQUAL-SIZED, to isolate rank
# discordance from the coverage gap that absolute thresholds create for
# uniformly-poor modes. Absolute/pooled version also computed → supplement.
DECILE_FRAC = 0.10

# Elsevier single-column width
FIG_W, FIG_H = 3.54, 3.4

# Marker shape identifies city. Colour creates a restrained visual hierarchy:
# road and rail remain grey context; all ICD observations retain full colour.
# Marker size is held constant so it cannot be mistaken for a data variable.
CITY_MARK = {"patna": "o", "varanasi": "s", "indore": "^",
             "kochi": "D", "kolkata": "P"}
MODE_COL = {
    "road": "#C7C7C7",   # light-grey context
    "rail": "#858585",   # dark-grey context
    "icd":  "#B5179E",   # focal mode
}
POINT_SIZE = 42


# ═══════════════════════════════════════════════════════════════


def severity_variability(df, mode):
    """D = -log(mean a);  V = Var[log a].  Enterprise-level."""
    col = ACCESS_COLS[mode]
    a = df[col].to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    a = np.clip(a, EPS, None)
    mean_a = a.mean()
    D = -np.log(mean_a)
    V = np.var(np.log(a), ddof=1)
    return {
        "mean_access": mean_a,
        "D_severity":  D,
        "D_linear":    1 - mean_a,     # reported for reference
        "V_variability": V,
        "n": len(a),
    }


def _masking_metrics(L_mfai, L_mode):
    """Set-agreement between the bottom-MFAI and bottom-mode groups.

    overlap = share of MFAI-poor that are also mode-poor
              (are the composite-flagged actually poor on this mode?)
    miss    = share of mode-poor NOT in MFAI-poor
              (what fraction of the genuinely mode-poor does MFAI miss?)
    jaccard = symmetric overlap (supplement only)

    With equal-sized sets miss ≈ 1 − overlap; ties at decile boundaries make
    the two sets slightly unequal, so both are reported rather than assumed
    complementary.
    """
    inter  = int(np.sum(L_mfai & L_mode))
    union  = int(np.sum(L_mfai | L_mode))
    n_mfai = int(np.sum(L_mfai))
    n_mode = int(np.sum(L_mode))
    overlap = inter / n_mfai if n_mfai else np.nan
    miss    = (n_mode - inter) / n_mode if n_mode else np.nan
    jaccard = inter / union if union else np.nan
    return {"overlap": overlap, "miss": miss, "jaccard": jaccard,
            "n_mfai": n_mfai, "n_mode": n_mode}


def composite_masking(df, mfai_thr, mode_thr):
    """Masking metrics for one city given MFAI and per-mode thresholds.

    Pass city-relative deciles (main table) or pooled-absolute deciles
    (supplement). Spearman(MFAI, mode) is threshold-free and added alongside.
    """
    mfai = df[COL_MFAI].to_numpy(float)
    L_mfai = mfai < mfai_thr
    out = {}
    for m in LAND_MODES:
        a = df[ACCESS_COLS[m]].to_numpy(float)
        L_mode = a < mode_thr[m]
        met = _masking_metrics(L_mfai, L_mode)
        ok = np.isfinite(mfai) & np.isfinite(a)
        met["spearman"] = spearmanr(mfai[ok], a[ok]).correlation
        out[m] = met
    return out


def main():
    print("Loading cities …")
    rows = []
    city_dfs = {}
    for city in CITIES:
        df = load_city(city)
        city_dfs[city] = df
        for mode in LAND_MODES:
            r = severity_variability(df, mode)
            r.update({"city": city, "mode": mode})
            rows.append(r)

    res = pd.DataFrame(rows)[
        ["city", "mode", "mean_access", "D_linear",
         "D_severity", "V_variability", "n"]
    ]

    # ── Table ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Severity (D = -log mean a) and Variability (V = Var log a)")
    print("=" * 64)
    disp = res.copy()
    for c in ["mean_access", "D_linear", "D_severity", "V_variability"]:
        disp[c] = disp[c].round(4)
    print(disp.to_string(index=False))

    # ── Dividing lines: cross-cell medians ───────────────────
    D_med = res["D_severity"].median()
    V_med = res["V_variability"].median()
    print(f"\nQuadrant dividers (medians): D={D_med:.4f}, V={V_med:.4f}")

    # ── Quadrant classification ──────────────────────────────
    def quad(r):
        hi_D = r["D_severity"] >= D_med
        hi_V = r["V_variability"] >= V_med
        if hi_D and not hi_V: return "system-level (not targetable)"
        if hi_D and hi_V:     return "severe, differentiated"
        if not hi_D and hi_V: return "localised pockets"
        return "low priority"
    res["quadrant"] = res.apply(quad, axis=1)
    print("\nClassification:")
    print(res[["city", "mode", "quadrant"]].to_string(index=False))

    # ── Figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Median-split crosshairs removed from the figure: the box classification
    # misclassifies Patna ICD (moderate V, but no spatial structure — the
    # HDBSCAN null in §5.8 is the arbiter of targetability, not the median).
    # Present as a continuous D–V scatter; the console still prints the box
    # labels as an internal diagnostic only. Re-enable for exploration if wanted.
    # ax.axvline(V_med, color="#BBBBBB", lw=0.6, zorder=1)
    # ax.axhline(D_med, color="#BBBBBB", lw=0.6, zorder=1)

    for _, r in res.iterrows():
        is_icd = r["mode"] == "icd"
        is_indore_icd = is_icd and r["city"] == "indore"

        # Publication-safe emphasis: every point has the same size. ICD cells
        # remain in full colour; road/rail cells are grey contextual points.
        # Indore ICD is distinguished only by a darker outline and one label.
        ax.scatter(
            r["V_variability"],
            r["D_severity"],
            marker=CITY_MARK[r["city"]],
            color=MODE_COL[r["mode"]],
            s=POINT_SIZE,
            edgecolor="#222222" if is_indore_icd else "white",
            linewidth=0.9 if is_indore_icd else 0.4,
            alpha=1.0 if is_icd else 0.72,
            zorder=5 if is_indore_icd else (4 if is_icd else 2),
        )

        if is_indore_icd:
            ax.annotate(
                "Indore ICD",
                xy=(r["V_variability"], r["D_severity"]),
                xytext=(7, 8),
                textcoords="offset points",
                fontsize=6.5,
                fontweight="bold",
                ha="left",
                va="bottom",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#444444",
                    "linewidth": 0.6,
                },
                zorder=6,
            )

    ax.set_xlabel(r"Spatial variability  $V=\mathrm{Var}[\log a]$", fontsize=8)
    ax.set_ylabel(r"Deficit severity  $D=-\log(\overline{a})$", fontsize=8)
    ax.tick_params(labelsize=7)

    # Legends: modes (colour) + cities (marker)
    from matplotlib.lines import Line2D
    mode_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODE_COL[m], markeredgecolor="white",
               markersize=6, label=m.upper(), linestyle="")
        for m in LAND_MODES
    ]
    city_handles = [Line2D([0], [0], marker=CITY_MARK[c], color="#555555",
                    markerfacecolor="#555555", markersize=6,
                    label=c.title(), linestyle="")
                    for c in CITIES]
    leg1 = ax.legend(handles=mode_handles, title="Mode", fontsize=6,
                     title_fontsize=6, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=city_handles, title="City", fontsize=6,
              title_fontsize=6, loc="lower right", framealpha=0.9)

    # Caption note for manuscript: marker size is constant and has no
    # quantitative meaning; colour highlights ICD observations, while road and
    # rail are shown in grey as context. Indore ICD alone is labelled.
    plt.tight_layout()
    out_dir = str(paths.results("targetability"))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_severity_variability.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_severity_variability.png"),
                dpi=400, bbox_inches="tight")
    res.to_csv(os.path.join(out_dir, "severity_variability.csv"), index=False)
    print(f"\nSaved figure + table to {out_dir}/")
    plt.show()

    # ═══════════════════════════════════════════════════════════
    #  COMPOSITE MASKING
    # ═══════════════════════════════════════════════════════════
    # Does the composite ranking locate each mode's poorly-served firms?
    # MAIN table: city-relative equal-sized deciles (isolates rank
    #   discordance). SUPPLEMENT: pooled-absolute deciles (adds the coverage
    #   gap for uniformly-poor modes) + Jaccard + Spearman.

    # Pooled-absolute thresholds (supplement)
    pooled_mfai_thr = np.nanquantile(
        np.concatenate([city_dfs[c][COL_MFAI].to_numpy(float) for c in CITIES]),
        DECILE_FRAC)
    pooled_mode_thr = {
        m: np.nanquantile(
            np.concatenate([city_dfs[c][ACCESS_COLS[m]].to_numpy(float)
                            for c in CITIES]), DECILE_FRAC)
        for m in LAND_MODES}

    main_rows, supp_rows = [], []
    for city in CITIES:
        df = city_dfs[city]
        # City-relative thresholds (main)
        rel_mfai_thr = np.nanquantile(df[COL_MFAI].to_numpy(float), DECILE_FRAC)
        rel_mode_thr = {m: np.nanquantile(df[ACCESS_COLS[m]].to_numpy(float),
                                          DECILE_FRAC) for m in LAND_MODES}
        rel = composite_masking(df, rel_mfai_thr, rel_mode_thr)
        ab  = composite_masking(df, pooled_mfai_thr, pooled_mode_thr)
        for m in LAND_MODES:
            main_rows.append({
                "city": city.title(), "mode": m.upper(),
                "overlap_%": round(100 * rel[m]["overlap"], 1),
                "miss_%":    round(100 * rel[m]["miss"], 1),
                "spearman":  round(rel[m]["spearman"], 3),
            })
            supp_rows.append({
                "city": city.title(), "mode": m.upper(),
                "rel_overlap_%": round(100 * rel[m]["overlap"], 1),
                "rel_miss_%":    round(100 * rel[m]["miss"], 1),
                "rel_jaccard":   round(rel[m]["jaccard"], 3),
                "abs_overlap_%": round(100 * ab[m]["overlap"], 1),
                "abs_miss_%":    round(100 * ab[m]["miss"], 1),
                "abs_jaccard":   round(ab[m]["jaccard"], 3),
                "spearman":      round(rel[m]["spearman"], 3),
                "n_mfai_rel":    rel[m]["n_mfai"], "n_mode_rel": rel[m]["n_mode"],
                "n_mode_abs":    ab[m]["n_mode"],
            })

    mask_main = pd.DataFrame(main_rows)
    mask_supp = pd.DataFrame(supp_rows)

    print("\n" + "=" * 64)
    print("COMPOSITE MASKING — city-relative bottom deciles (MAIN)")
    print("overlap = % of MFAI-poor also mode-poor;  "
          "miss = % of mode-poor missed by MFAI")
    print("=" * 64)
    # Pivot to city × mode for readability
    piv_ov = mask_main.pivot(index="city", columns="mode", values="overlap_%")
    piv_ms = mask_main.pivot(index="city", columns="mode", values="miss_%")
    print("\nOverlap %  (MFAI-poor that are also mode-poor):")
    print(piv_ov.to_string())
    print("\nMiss %  (mode-poor that MFAI does NOT flag):")
    print(piv_ms.to_string())
    print("\nExpected signature: Indore ICD low overlap / high miss "
          "(uniformly poor → masked); Kochi ICD higher overlap / lower miss.")

    mask_main.to_csv(os.path.join(out_dir, "masking_main.csv"), index=False)
    mask_supp.to_csv(os.path.join(out_dir, "masking_supplement.csv"), index=False)
    print(f"\nSaved masking tables to {out_dir}/")

    return res, mask_main, mask_supp


if __name__ == "__main__":
    res, mask_main, mask_supp = main()
