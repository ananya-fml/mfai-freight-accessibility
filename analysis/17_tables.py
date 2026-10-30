"""Publication tables: LaTeX / CSV export

Produces: Main-text Tables 1-17 and supplementary S1, S4, S6, S9 -> results/tables/

Seventeen main-text tables and nine supplementary, each with caption and
note, exported to LaTeX (booktabs), CSV and Markdown.
    tabs = make_all(res_target=res)
    export(tabs, fmt="latex")
PROVENANCE
Tables are marked [COMPUTED], [STORED] or [MANUAL] in their docstring.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from mfai import paths
from mfai.io import load_city, load_raw

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
OUTDIR = str(paths.results("tables"))

CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}
MODES = ["road", "rail", "water", "icd"]
MODE_LABEL = {"road": "Major road", "rail": "Rail goods shed",
              "water": "IWT terminal", "icd": "ICD"}
DIST_COL = {"road": "nearest_major_road_dist_m",
            "rail": "nearest_railway_dist_m",
            "water": "nearest_waterway_dist_m",
            "icd": "nearest_ICD_dist_m"}
COL = "mfai_land"

CITY_CENTRE = {"patna": (25.6127, 85.1416), "indore": (22.7177, 75.8545),
               "varanasi": (25.3109, 83.0104), "kochi": (9.9816, 76.2830),
               "kolkata": (22.5697, 88.3497)}

GROUPS = {
    "G1 Food, beverages, tobacco":            "G1_food_bev_tobacco",
    "G2 Textiles, apparel, leather":          "G2_textile_apparel_leather",
    "G3 Wood, paper, printing":               "G3_wood_paper_printing",
    "G4 Petroleum, chemicals, pharma":        "G4_petro_chem_pharma",
    "G5 Rubber, plastics, minerals":          "G5_rubber_plastic_mineral",
    "G6 Basic and fabricated metals":         "G6_metals",
    "G7 Electronics, machinery":              "G7_electronics_machinery",
    "G8 Transport eq., furniture, other":     "G8_transport_furniture_oth",
}

FALLBACK_THRESHOLD, FALLBACK_MIN_PINCODES, DIV_DIVERSITY_T = 20, 5, 8


def _tag(df, caption, note="", source="COMPUTED"):
    df.attrs["caption"] = caption
    df.attrs["note"] = note
    df.attrs["source"] = source
    return df


def _stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ══════════════════════════════════════════════════════════════════════
# MAIN TABLES
# ══════════════════════════════════════════════════════════════════════
def t1_cities():
    """[MANUAL + COMPUTED] Study cities and terminal inventory."""
    cfg = {
        "patna":    ("Bihar", "Inland river port (NW-1), weak rail freight"),
        "varanasi": ("Uttar Pradesh", "Inland river port, recent MMLP investment"),
        "kolkata":  ("West Bengal", "Metropolitan port city, compact core"),
        "kochi":    ("Kerala", "Coastal major port, linear conurbation"),
        "indore":   ("Madhya Pradesh", "Landlocked, peripheral ICD capacity"),
    }
    counts = {"patna": (3, 6, 1), "varanasi": (4, 4, 1), "kolkata": (5, 5, 6),
              "kochi": (11, 9, 8), "indore": (5, 0, 5)}
    rows = []
    for c in CITY_ORDER:
        st, desc = cfg[c]
        r, w, i = counts[c]
        d = load_city(c)
        rows.append({"City": CITY_LABEL[c], "State": st, "Configuration": desc,
                     "Rail goods sheds": r, "IWT terminals": w, "ICD/CFS": i,
                     "Analysis sample": f"{len(d):,}"})
    t = pd.DataFrame(rows)
    return _tag(t, "Study cities, freight infrastructure and analysis sample.",
                "Terminal inventories include operational and "
                "under-development facilities; status is reported in Table S5. "
                "Indore has no inland waterway terminal within its district; "
                "see Section 3.3.")


def t2_sample():
    """[COMPUTED] Sample construction and geocoder artefact removal."""
    rows = []
    for c in CITY_ORDER:
        raw = load_raw(c)
        kept = load_city(c)
        rem = len(raw) - len(kept)
        rows.append({
            "City": CITY_LABEL[c],
            "Manufacturing MSMEs": len(raw),
            "Artefacts removed": rem,
            "% removed": round(100*rem/len(raw), 1),
            "Analysis sample": len(kept),
            "Unique locations": kept[["lat", "lon"]].drop_duplicates().shape[0],
            "Enterprises per location": round(
                len(kept)/kept[["lat", "lon"]].drop_duplicates().shape[0], 2),
        })
    t = pd.DataFrame(rows)
    tot = t.iloc[:, 1:].sum(numeric_only=True)
    t.loc[len(t)] = ["Total", int(tot["Manufacturing MSMEs"]),
                     int(tot["Artefacts removed"]),
                     round(100*tot["Artefacts removed"]/tot["Manufacturing MSMEs"], 1),
                     int(tot["Analysis sample"]), int(tot["Unique locations"]),
                     round(tot["Analysis sample"]/tot["Unique locations"], 2)]
    for c in ["Manufacturing MSMEs", "Artefacts removed", "Analysis sample",
              "Unique locations"]:
        t[c] = t[c].map("{:,}".format)
    return _tag(t, "Sample construction.",
                "Artefacts identified by two independent rules (Section 3.5): "
                "at least 20 enterprises at one coordinate spanning at least "
                "five postal codes, or at least eight ISIC divisions. "
                "Registry co-location means effective spatial sample size is "
                "roughly half the enterprise count.")


GRAPH_DIAG = pd.DataFrame([
    ("Connector distance, median (m)",              "35.7",  "12.1"),
    ("Connector distance, 95th percentile (m)",     "220.7", "59.5"),
    ("Spearman (connector, built-up density)",      "−0.220", "−0.139"),
    ("Road distance / straight-line, median",       "1.67", "1.33"),
    ("Rail median, change",                         "—",    "< 0.3%"),
    ("ICD median, change",                          "—",    "< 0.3%"),
], columns=["Diagnostic", "Endpoint-only graph", "Vertex-level graph"])


def t3_graph():
    """[STORED] Effect of graph construction on measured distances.

    Regenerate: rerun Stage 1 with build_graph reverted to endpoint-only
    and compare snap_connector_m distributions.
    """
    return _tag(GRAPH_DIAG.copy(),
                "Effect of routable graph construction on measured distances "
                "(Indore).",
                "A graph built on line-merge endpoints places nodes only at "
                "junctions, so enterprises connect at distant junctions and "
                "the resulting error correlates with urban form. The final "
                "two rows confirm that node density changed while network "
                "topology did not.", source="STORED")


def t4_reference():
    """[MANUAL] Reference distances and their anchors."""
    t = pd.DataFrame([
        ("Major road", "2 km",
         "UDPFI/URDPFI: arterial minimum intersection spacing 500 m; "
         "sub-arterial spacing 0.5 km (CBD) to 3–5 km (residential). Half "
         "the mean spacing implies a catchment edge of 1–2.5 km."),
        ("ICD", "50 km",
         "CBIC Circular No. 50/2020-Customs, para 4.1.4: no greenfield ICD "
         "within 100 km of an existing ICD. Statutory 100 km spacing implies "
         "a ~50 km catchment under nearest-facility assignment."),
        ("Rail goods shed", "50 km",
         "Weakest anchor. Intermodal first-mile drayage is conventionally "
         "50–100 km. No Indian Railways or CONCOR catchment norm identified; "
         "stated as an assumption."),
        ("IWT terminal", "50 km",
         "By analogy with ICD. No statutory anchor: the CBIC circular "
         "relaxes distance rules for waterway-linked facilities."),
    ], columns=["Mode", "$T_m$", "Basis"])
    return _tag(t, "Impedance reference distances.",
                "$T_m$ is the catchment edge: the distance at which access is "
                "judged half that of co-location. All four are swept in the "
                "sensitivity analysis (Table S2).", source="MANUAL")


def t5_distances():
    """[COMPUTED] Network distance to nearest facility."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c)
        r = {"City": CITY_LABEL[c]}
        for m in MODES:
            v = d[DIST_COL[m]].dropna() / 1000.0
            if v.std() < 1e-9:
                r[MODE_LABEL[m]] = f"{v.median():.0f}$^{{a}}$"
            else:
                r[MODE_LABEL[m]] = (f"{v.median():.2f} "
                                    f"[{v.quantile(.25):.2f}–{v.quantile(.75):.2f}]")
        rows.append(r)
    t = pd.DataFrame(rows)
    return _tag(t, "Network distance to the nearest facility, km "
                   "(median [interquartile range]).",
                "$^{a}$ Out-of-district terminal, constant for all "
                "enterprises. The median enterprise in every city lies within "
                "500 m of a major road; ICD distances vary more than fivefold "
                "across cities.")


def t6_access():
    """[COMPUTED] Mean access scores and binding modal constraint."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c)
        r = {"City": CITY_LABEL[c], "$n$": f"{len(d):,}"}
        for m in MODES:
            col = f"access_{m}"
            r[MODE_LABEL[m]] = round(d[col].mean(), 3) if col in d.columns else np.nan
        land = {m: d[f"access_{m}"].mean() for m in ["road", "rail", "icd"]}
        srt = sorted(land.items(), key=lambda kv: kv[1])
        r["MFAI"] = round(d[COL].mean(), 3)
        r["Binding mode"] = MODE_LABEL[srt[0][0]]
        r["Margin"] = round(srt[1][1] - srt[0][1], 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    return _tag(t, "Mean access scores, composite index and binding modal "
                   "constraint.",
                "Access scores $a = T/(T+d)$, range 0–1, higher is better. "
                "MFAI is the three-mode index (road, rail, ICD), comparable "
                "across cities. The binding mode is the lowest-mean land "
                "mode; the margin is the gap to the next lowest. A margin "
                "below roughly 0.05 indicates no mode binds distinctly and "
                "the ordering should not be read as a finding. Indore's IWT "
                "score is a constant (Section 3.3).")


def t7_truncation(radii=(10, 15, 20, 25, 30, 40)):
    """[COMPUTED] Mean MFAI by truncation radius."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c)
        r = {"City": CITY_LABEL[c]}
        for R in radii:
            s = d[d.dist_centre_km <= R]
            r[f"{R} km"] = round(s[COL].mean(), 3) if len(s) > 50 else np.nan
        r["Full"] = round(d[COL].mean(), 3)
        r["Swing"] = round(r[f"{radii[0]} km"] - r["Full"], 3)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("Full")
    return _tag(t, "Mean MFAI by truncation radius from the city centre.",
                "The ordering is preserved at every radius. Swing is the "
                "difference between the 10 km and full-extent values: a large "
                "swing indicates that low aggregate accessibility arises from "
                "settlement dispersion (Kochi), a small one that it reflects a "
                "shortfall of provision (Patna). See Section 6.2.")


def t8_distribution():
    """[COMPUTED] Distribution of MFAI, with dip statistics if available."""
    try:
        import diptest
        have = True
    except ImportError:
        have = False
        print("  t8: pip install diptest for dip columns")
    rows = []
    for c in CITY_ORDER:
        v = load_city(c)[COL].dropna()
        r = {"City": CITY_LABEL[c], "$n$": f"{len(v):,}",
             "Mean": round(v.mean(), 3), "SD": round(v.std(), 3),
             "P10": round(v.quantile(.10), 3),
             "Median": round(v.median(), 3),
             "P90": round(v.quantile(.90), 3),
             "IQR": round(v.quantile(.75)-v.quantile(.25), 3),
             "P90/P10": round(v.quantile(.90)/v.quantile(.10), 3),
             "CV": round(v.std()/v.mean(), 3),
             "Skew": round(v.skew(), 2)}
        if have:
            dp, p = diptest.diptest(v.values)
            r["Dip"] = round(dp, 4)
            r["Dip $p$"] = f"{p:.3g}"
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("Mean")
    return _tag(t, "Distribution of MFAI by city.",
                "Hartigan's dip test rejects unimodality in four of five "
                "cities, but at these sample sizes it rejects on slight "
                "departures and the dip statistic is the informative "
                "quantity. Patna has a higher dip statistic than Kochi with a "
                "much larger $p$-value, purely because its sample is less "
                "than half the size. Every city is unimodal at kernel "
                "bandwidths of 0.5 and above (Figure S1).")


def t9_variance_modes():
    """[COMPUTED] Contribution of each mode to within-city index variance."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c)
        contrib = {}
        for m in MODES:
            col = f"access_{m}"
            contrib[m] = ((0.25**2) * np.log(d[col].clip(lower=1e-9)).var()
                          if col in d.columns and d[col].std() > 1e-9 else 0.0)
        tot = sum(contrib.values())
        r = {"City": CITY_LABEL[c]}
        r.update({MODE_LABEL[m]: round(100*contrib[m]/tot, 1) for m in MODES})
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("Major road", ascending=False)
    return _tag(t, "Contribution of each mode to within-city MFAI variance (\\%).",
                "First-order sensitivity approximation on the log scale, since "
                "aggregation is geometric (Paruolo et al., 2013). Road "
                "dominates where terminal access is uniform, whether uniformly "
                "good (Kolkata) or uniformly poor (Indore). Where terminal "
                "access genuinely varies between enterprises, terminals "
                "contribute the majority.")


def t10_variance_components():
    """[COMPUTED] Variance components of enterprise-level MFAI."""
    frames = []
    for c in CITY_ORDER:
        d = load_city(c)
        frames.append(pd.DataFrame({COL: d[COL], "city": c,
                                    "grp": d.get("mfg_group")}))
    df = pd.concat(frames, ignore_index=True).dropna(subset=[COL])
    y = df[COL]; g = y.mean(); sst = ((y-g)**2).sum()
    cm = df.groupby("city")[COL].transform("mean")
    ssc = ((cm-g)**2).sum()
    if df.grp.notna().any():
        gm = df.groupby(["city", "grp"])[COL].transform("mean")
        ssg = ((gm-cm)**2).sum(); ssr = ((y-gm)**2).sum()
    else:
        ssg, ssr = 0.0, ((y-cm)**2).sum()
    t = pd.DataFrame([
        {"Component": "Between cities", "Sum of squares": round(ssc, 2),
         "\\% of total": round(100*ssc/sst, 1)},
        {"Component": "Between industrial groups, within city",
         "Sum of squares": round(ssg, 2), "\\% of total": round(100*ssg/sst, 1)},
        {"Component": "Within group, within city",
         "Sum of squares": round(ssr, 2), "\\% of total": round(100*ssr/sst, 1)},
    ])
    return _tag(t, "Variance components of enterprise-level MFAI.",
                "Industrial sector explains one per cent of the variation in "
                "freight accessibility. Nearly three-quarters is "
                "micro-geographic, arising below the level of both city and "
                "sector: freight accessibility is a property of where an "
                "enterprise sits, not of what it makes.")


def _partial(d, x, y, ctrl):
    R = d[[x, y] + list(ctrl)].rank().corr().values
    inv = np.linalg.pinv(R)
    return -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])


def t11_correlations():
    """[COMPUTED] Discriminant validity correlations."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c)
        ntl = ("ntl_log_clean" if "ntl_log_clean" in d.columns
               else "ntl_log" if "ntl_log" in d.columns else None)
        if ntl is None or "DN" not in d.columns:
            continue
        s = d.dropna(subset=[COL, ntl, "DN", "dist_centre_km",
                             "log_dist_centre"])
        rows.append({
            "City": CITY_LABEL[c],
            "MFAI–built-up": round(stats.spearmanr(s[COL], s.DN).statistic, 3),
            "MFAI–NTL": round(stats.spearmanr(s[COL], s[ntl]).statistic, 3),
            "Built-up–NTL": round(stats.spearmanr(s.DN, s[ntl]).statistic, 3),
            "MFAI–distance": round(stats.spearmanr(
                s[COL], s.dist_centre_km).statistic, 3),
            "NTL–distance": round(stats.spearmanr(
                s[ntl], s.dist_centre_km).statistic, 3),
            "MFAI–NTL | dist": round(
                _partial(s, COL, ntl, ["log_dist_centre"]), 3),
            "MFAI–NTL | dist, built-up": round(
                _partial(s, COL, ntl, ["log_dist_centre", "DN"]), 3),
        })
    t = pd.DataFrame(rows)
    return _tag(t, "Discriminant validity: Spearman correlations at "
                   "enterprise level.",
                "Nighttime luminosity is very largely a monocentric gradient "
                "in all five cities, so as an activity measure it is close to "
                "a restatement of centrality. Built-up density and luminosity "
                "are near-duplicates where both are high. Most of the raw "
                "MFAI–NTL association is shared centrality: partialling out "
                "distance collapses it substantially.")


def t12_mode_diagnostics():
    """[COMPUTED] Mode-level diagnostics with predictor spread."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c)
        ntl = ("ntl_log_clean" if "ntl_log_clean" in d.columns
               else "ntl_log" if "ntl_log" in d.columns else None)
        if ntl is None:
            continue
        for m in MODES:
            a = f"access_{m}"
            if a not in d.columns or d[a].std() < 1e-9:
                continue
            s = d.dropna(subset=[a, ntl, "DN", "dist_centre_km",
                                 "log_dist_centre"])
            iqr = s[a].quantile(.75) - s[a].quantile(.25)
            rows.append({
                "City": CITY_LABEL[c], "Mode": MODE_LABEL[m],
                "Relative IQR (\\%)": round(100*iqr/s[a].mean(), 1),
                "vs NTL": round(stats.spearmanr(s[a], s[ntl]).statistic, 3),
                "vs built-up": round(stats.spearmanr(s[a], s.DN).statistic, 3),
                "vs distance": round(stats.spearmanr(
                    s[a], s.dist_centre_km).statistic, 3),
                "vs NTL | dist": round(
                    _partial(s, a, ntl, ["log_dist_centre"]), 3),
            })
    t = pd.DataFrame(rows)
    return _tag(t, "Mode-level diagnostics.",
                "Relative IQR is the interquartile range as a share of the "
                "mean. A near-zero correlation on a predictor with little "
                "spread is range restriction, not independence: Kolkata's ICD "
                "access varies over 5.6\\% of its mean and its low correlation "
                "must not be read alongside Patna's. Patna's ICD access has "
                "adequate spread and is independent of all three covariates, "
                "while its waterway access — with identical spread — is not.")


def t13_redundancy(decile=0.10):
    """[COMPUTED] Overlap of the lowest decile by each measure."""
    rows = []
    for c in CITY_ORDER:
        d = load_city(c).reset_index(drop=True)
        ntl = ("ntl_log_clean" if "ntl_log_clean" in d.columns
               else "ntl_log" if "ntl_log" in d.columns else None)
        if ntl is None or "DN" not in d.columns:
            continue
        s = d.dropna(subset=[COL, ntl, "DN"]).reset_index(drop=True)
        k = max(1, int(len(s)*decile))
        f_m = set(s[COL].nsmallest(k).index)
        f_n = set(s[ntl].nsmallest(k).index)
        f_d = set(s.DN.nsmallest(k).index)
        rows.append({
            "City": CITY_LABEL[c], "Flagged": f"{k:,}",
            "Overlap with NTL (\\%)": round(100*len(f_m & f_n)/k, 1),
            "Overlap with built-up (\\%)": round(100*len(f_m & f_d)/k, 1),
            "Identified only by MFAI (\\%)": round(
                100*len(f_m - f_n - f_d)/k, 1)})
    t = pd.DataFrame(rows)
    return _tag(t, "Overlap of the lowest accessibility decile with "
                   "activity-based screens.",
                "Partial overlap between the lowest deciles of imperfectly "
                "correlated variables is arithmetically expected; this is a "
                "threshold-overlap statement rather than evidence of "
                "orthogonality. Kolkata's figure is the largest and the least "
                "meaningful, its flagged decile being a threshold artefact on "
                "a uniformly well-served city.")


# ── STORED: agglomeration ─────────────────────────────────────────────
AGG = pd.DataFrame([
    ("patna","MFAI",0.1711,0.1606,0.29,685),
    ("patna","road",0.8735,0.0872,1.3e-23,685),
    ("patna","rail",-0.2980,0.1348,0.027,685),
    ("patna","water",0.5887,0.2244,0.0087,685),
    ("patna","icd",-0.2838,0.1129,0.012,685),
    ("indore","MFAI",0.7778,0.1186,5.5e-11,707),
    ("indore","road",0.6329,0.1113,1.3e-08,707),
    ("indore","rail",0.7625,0.1818,2.7e-05,707),
    ("indore","icd",0.3589,0.1341,0.0074,707),
    ("varanasi","MFAI",0.4156,0.1148,0.0003,790),
    ("varanasi","road",0.3187,0.0842,0.00015,790),
    ("varanasi","rail",0.4216,0.0808,1.8e-07,790),
    ("varanasi","water",-0.4132,0.1283,0.0013,790),
    ("varanasi","icd",-0.1086,0.1526,0.48,790),
    ("kochi","MFAI",0.5452,0.0715,2.4e-14,1464),
    ("kochi","road",0.3287,0.0472,3.2e-12,1464),
    ("kochi","rail",0.2404,0.0687,0.00047,1464),
    ("kochi","water",0.1021,0.1044,0.33,1464),
    ("kochi","icd",0.3751,0.1097,0.00063,1464),
], columns=["city", "mode", "coef", "se", "p", "cells"])


def t14_agglomeration(tab=None):
    """[STORED] Enterprise count response to access.

    Regenerate: agglomeration_standalone.run_all(), read res["bycity"].
    """
    tab = AGG if tab is None else tab
    rows = []
    for c in CITY_ORDER:
        sub = tab[tab.city == c]
        if not len(sub):
            continue
        r = {"City": CITY_LABEL[c], "Cells": f"{int(sub.cells.iloc[0]):,}"}
        for m in ["MFAI"] + MODES:
            q = sub[sub["mode"] == m]
            key = "MFAI" if m == "MFAI" else MODE_LABEL[m]
            r[key] = (f"{q.coef.iloc[0]:.3f}{_stars(q.p.iloc[0])}"
                      if len(q) else "—")
        rows.append(r)
    t = pd.DataFrame(rows)
    return _tag(t, "Enterprise count response to access, by city "
                   "(coefficient per within-city standard deviation).",
                "Negative binomial, controlling for log built-up surface and "
                "log distance from centre, standard errors clustered on 5 km "
                "spatial blocks. $^{*}p<0.05$, $^{**}p<0.01$, "
                "$^{***}p<0.001$. Kolkata is excluded: 169 cells at 1 km and "
                "43 at 2 km, insufficient spatial variation, non-convergent "
                "at the coarser resolution. Patna is estimated at 60 km and "
                "Kochi at 45 km; Patna's ICD coefficient is $+0.033$ "
                "($p=0.62$) within 30 km. Road access is endogenous, so a "
                "positive road coefficient is expected and uninformative. "
                "Signs differ across cities for every terminal mode, so "
                "pooled estimates are not reported.", source="STORED")


# ── STORED: targetability ─────────────────────────────────────────────
NULL_MODEL = pd.DataFrame([
    ("kochi",   2777, 5,  7.4, 84.8, 62.9, 54.7, 68.9,  4.67, 0.000),
    ("varanasi", 929, 5,  1.2, 60.9, 24.7,  0.0, 53.8,  1.52, 0.030),
    ("kolkata", 2489, 14, 3.6, 92.0, 91.2, 56.9,100.0,  0.05, 0.790),
    ("patna",   1290, 8,  2.4, 67.2, 90.1, 58.0, 99.4, -1.78, 0.890),
    ("indore",  4892, 18, 2.6, 67.7, 86.9, 80.9, 92.9, -4.36, 1.000),
], columns=["city","firms","zones_obs","zones_null","obs","null_mean",
            "null_p05","null_p95","z","p"])

ZONE_STRUCT = pd.DataFrame([
    ("kolkata", 14, 92.0, 92.0, 14,   6.0,  1.60),
    ("indore",  18, 67.7, 51.1, 14, 368.9,  6.06),
    ("varanasi", 5, 60.9, 25.9,  2, 143.8, 48.95),
    ("patna",    8, 67.2, 15.8,  3, 257.8, 56.11),
    ("kochi",    5, 84.8,  0.0,  0, 661.9, 60.88),
], columns=["city","zones","in_zones","in_small","n_small","largest","median"])

ZONE_STRUCT_OPT = pd.DataFrame([
    ("kolkata", 17, 96.1, 96.1, 17,   6.0,  1.27),
    ("indore",  10, 65.9, 25.5,  4, 319.6, 34.04),
    ("varanasi", 5, 60.9, 25.9,  2, 143.8, 48.95),
    ("patna",    9, 77.9, 16.7,  3, 305.4,100.67),
    ("kochi",    6, 78.9,  9.6,  4, 610.5, 18.04),
], columns=["city","zones","in_zones","in_small","n_small","largest","median"])

ZONE_BINDING = pd.DataFrame([
    ("kochi",   2178, 0,   178),
    ("patna",    659, 0,   208),
    ("varanasi",   0, 0,   566),
    ("kolkata",    0, 0,  2291),
    ("indore",   393, 412, 2509),
], columns=["city", "icd", "rail", "road"])


def t15_null(tab=None):
    """[STORED] Observed against null spatial concentration.

    Regenerate: targetability_hdbscan.run_all(), read res["null"].
    """
    tab = NULL_MODEL if tab is None else tab
    t = tab.copy()
    t["City"] = t.city.map(CITY_LABEL)
    t["Flagged enterprises"] = t.firms.map("{:,}".format)
    t["Zones, observed"] = t.zones_obs
    t["Zones, null"] = t.zones_null
    t["In zones, observed (\\%)"] = t.obs
    t["In zones, null (\\%)"] = t.null_mean.round(1).astype(str) + " [" + \
        t.null_p05.round(1).astype(str) + "–" + t.null_p95.round(1).astype(str) + "]"
    t["$z$"] = t.z.map("{:+.2f}".format)
    t["$p$"] = t.p.map("{:.3f}".format)
    out = t[["City", "Flagged enterprises", "Zones, observed", "Zones, null",
             "In zones, observed (\\%)", "In zones, null (\\%)", "$z$", "$p$"]]
    return _tag(out, "Observed against null spatial concentration of the "
                     "lowest accessibility decile.",
                "Null is 100 random subsets of equal size drawn from the same "
                "enterprise population; the bracketed range is the 5th to "
                "95th percentile. MFAI is a smooth function of distance to "
                "fixed terminals, so low-accessibility regions are contiguous "
                "by construction and flagged enterprises cluster whatever the "
                "threshold; excess concentration over the null is the "
                "evidence. In Patna and Indore flagged enterprises are "
                "significantly \\emph{less} concentrated than an arbitrary "
                "equally sized group, because they are peripheral and the "
                "periphery is sparse. $p$ is one-sided.", source="STORED")


def t16_zones(tab=None, tab_opt=None):
    """[STORED] Zone structure under fixed and per-city parameters.

    Regenerate: targetability_hdbscan.run_all() then compact_concentration().
    """
    a = (ZONE_STRUCT if tab is None else tab).copy()
    b = (ZONE_STRUCT_OPT if tab_opt is None else tab_opt).copy()
    m = a.merge(b, on="city", suffixes=("", "_opt"))
    m["City"] = m.city.map(CITY_LABEL)
    out = pd.DataFrame({
        "City": m.City,
        "Zones": m.zones,
        "In zones (\\%)": m.in_zones,
        "In zones < 25 km$^2$ (\\%)": m.in_small,
        "Compact zones": m.n_small,
        "Largest zone (km$^2$)": m.largest,
        "Median zone (km$^2$)": m["median"],
        "In zones < 25 km$^2$, per-city optimum (\\%)": m.in_small_opt,
    })
    return _tag(out, "Zone structure of the lowest accessibility decile.",
                "HDBSCAN on unique coordinates, $\\textit{min\\_cluster\\_size}=20$, "
                "$\\textit{min\\_samples}=10$; the final column uses each "
                "city's relative-validity optimum. The in-zone share alone "
                "conflates many compact zones with one diffuse region: "
                "Kochi's 84.8\\% comprises 68.5\\% inside a single 662 km$^2$ "
                "zone of 17.7 km radius, which is not an addressable site. "
                "Indore is the only city whose compact-zone share is "
                "parameter-sensitive and should be read as a range.",
                source="STORED")


def t17_zone_binding(tab=None):
    """[STORED] Enterprises in zones by the mode binding within the zone.

    Regenerate: zone_binding_mode(res).
    """
    tab = ZONE_BINDING if tab is None else tab
    binding_city = {"patna": "ICD", "indore": "ICD", "varanasi": "Major road",
                    "kochi": "ICD", "kolkata": "Major road"}
    t = tab.copy()
    t["City"] = t.city.map(CITY_LABEL)
    tot = t[["icd", "rail", "road"]].sum(axis=1)
    for k, lab in [("icd", "ICD"), ("rail", "Rail"), ("road", "Major road")]:
        t[lab] = t[k].map("{:,}".format) + " (" + \
            (100*t[k]/tot).round(0).astype(int).astype(str) + "\\%)"
    t["City-level binding mode"] = t.city.map(binding_city)
    t["Agreement"] = np.where(
        [b.lower().startswith(t.loc[i, ["icd", "rail", "road"]].idxmax()[:3])
         for i, b in enumerate(t["City-level binding mode"])], "yes", "no")
    out = t[["City", "ICD", "Rail", "Major road",
             "City-level binding mode", "Agreement"]]
    return _tag(out, "Enterprises in zones, by the mode binding within the "
                     "zone.",
                "Four of five cities agree with the city-level binding "
                "constraint. Indore does not: its ICD access is uniformly poor "
                "and contributes 13\\% of within-city variance while road "
                "contributes 70\\%, so the composite discriminates on road and "
                "the zones it identifies are road-poor locations in a city "
                "whose binding constraint is ICD. A composite index can "
                "identify the wrong targets even where the level analysis "
                "identifies the right constraint.", source="STORED")


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY
# ══════════════════════════════════════════════════════════════════════
def s1_threshold_sensitivity(grid=(4, 6, 8, 10, 12)):
    """[COMPUTED] Geocoder threshold sensitivity."""
    rows = []
    for c in CITY_ORDER:
        raw = load_raw(c)
        for T in grid:
            d = load_city(c, div_t=T)
            rows.append({"City": CITY_LABEL[c], "$T$": T, "$n$": f"{len(d):,}",
                         "Excluded (\\%)": round(100*(1-len(d)/len(raw)), 1),
                         "Mean MFAI": round(d[COL].mean(), 4),
                         "P10": round(d[COL].quantile(.10), 4),
                         "IQR": round(d[COL].quantile(.75)-d[COL].quantile(.25), 4)})
    t = pd.DataFrame(rows)
    return _tag(t, "Sensitivity to the sectoral-diversity threshold.",
                "The excluded share moves by up to ten percentage points "
                "across the range while index means and tenth percentiles are "
                "stable to three decimal places: the outcome is insensitive to "
                "a parameter that materially changes sample size.")


def s4_groups():
    """[COMPUTED] Industrial group composition and relative accessibility."""
    counts, rel = [], []
    for c in CITY_ORDER:
        d = load_city(c)
        cm = d[COL].mean()
        rc = {"City": CITY_LABEL[c]}
        rr = {"City": CITY_LABEL[c]}
        for lab, key in GROUPS.items():
            s = d[d.mfg_group == key]
            rc[lab] = len(s)
            rr[lab] = round(s[COL].mean()/cm, 3) if len(s) else np.nan
        counts.append(rc); rel.append(rr)
    a = pd.DataFrame(counts)
    b = pd.DataFrame(rel)
    a = _tag(a, "Enterprises by industrial group and city.", "")
    b = _tag(b, "Mean MFAI by industrial group, relative to city mean.",
             "A value of 1.000 is the city average. All forty city–group "
             "combinations lie between 0.976 and 1.033, corroborating the "
             "1.0\\% between-sector variance share in Table 10.")
    return a, b


def s6_network():
    """[STORED] Network construction diagnostics.

    Regenerate: read results/pipeline/{city}/{city}_report.txt.
    """
    t = pd.DataFrame([
        ("Patna",    "481,983", "492,411", 94.8, 1569, "1,294", 94.3, 12.0, 113.4),
        ("Indore",   "579,983", "605,608", 98.4, 1824, "1,488", 95.4, 11.4,  53.2),
        ("Varanasi", "830,183", "850,895", 98.9,  830,   "651", 95.8, 12.2,  52.1),
        ("Kochi",    "792,378", "808,394", 97.5, 3369, "2,001", 94.4, 12.3,  97.9),
        ("Kolkata",  "231,873", "246,858", 98.3, 1909,   "419", 91.1, 10.6,  29.8),
    ], columns=["City", "Nodes", "Edges", "Giant component (\\%)",
                "Major road segments", "Major road km",
                "Segments reached (\\%)", "Connector median (m)",
                "Connector p95 (m)"])
    return _tag(t, "Routable network construction diagnostics.",
                "The graph is built over all vertices of the noded edge set, "
                "densified to a maximum 25 m spacing. Connector distance is "
                "from the enterprise to its snapped graph node and enters "
                "every terminal distance.", source="STORED")


ESTATE_COV = pd.DataFrame([
    ("patna",    82, 11.9,   86, 0.67),
    ("indore",   57, 13.1, 1697, 3.47),
    ("varanasi", 39,  2.5,    6, 0.06),
    ("kolkata", 282, 11.4,  587, 2.36),
], columns=["city", "parts", "area", "inside", "pct"])

ESTATE_TERM = pd.DataFrame([
    ("patna",   "Rail",  3, 3.32, 1.47, 1), ("patna",   "IWT",  6, 2.25, 0.05, 2),
    ("patna",   "ICD",   1, 1.48, 1.48, 1),
    ("indore",  "Rail",  5, 0.26, 0.10, 4), ("indore",  "ICD",  5, 3.28, 0.26, 2),
    ("varanasi","Rail",  4, 1.72, 0.25, 2), ("varanasi","IWT",  4, 1.97, 1.81, 3),
    ("varanasi","ICD",   1, 1.95, 1.95, 1),
    ("kolkata", "Rail",  5, 0.15, 0.00, 5), ("kolkata", "IWT",  5, 0.01, 0.00, 5),
    ("kolkata", "ICD",   6, 0.12, 0.00, 6),
], columns=["city", "mode", "n", "median_km", "min_km", "within_2km"])


def s9_estates():
    """[STORED] Industrial estate coverage and terminal proximity.

    Regenerate: estate_validation_all_cities.run_all().
    """
    a = ESTATE_COV.copy()
    a["City"] = a.city.map(CITY_LABEL)
    a = a[["City", "parts", "area", "inside", "pct"]]
    a.columns = ["City", "Polygon parts", "Area (km$^2$)",
                 "Enterprises inside", "\\% inside"]
    a.loc[len(a)] = ["Kochi", "—", "—", "—", "—"]
    a = _tag(a, "OpenStreetMap industrial land coverage.",
             "Coverage captures 0.06\\% to 3.47\\% of enterprises. The "
             "variable is closer to `planned industrial estate' than to "
             "`industrial land': scattered urban manufacturing is largely "
             "unmapped. Only Indore and Kolkata had sufficient enterprises "
             "inside for comparison, and the matched difference was null in "
             "both. No polygons were available for Kochi.", source="STORED")

    b = ESTATE_TERM.copy()
    b["City"] = b.city.map(CITY_LABEL)
    b = b[["City", "mode", "n", "median_km", "min_km", "within_2km"]]
    b.columns = ["City", "Mode", "$n$", "Median km to estate", "Min km",
                 "Within 2 km"]
    b = _tag(b, "Terminal distance to the nearest industrial estate polygon.",
             "Indore's rail goods sheds sit a median 0.26 km from estate land "
             "against 3.28 km for its ICDs, consistent with estates having "
             "been sited relative to the rail freight geography of their era "
             "and container terminals arriving later. Suggestive and "
             "untested.", source="STORED")
    return a, b


# ══════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════
def _latex(df, label):
    # Values are already rounded per column; print them as stored (7.4, not
    # pandas' default 7.400000) so the .tex matches the .csv.
    body = df.to_latex(index=False, escape=False, column_format=None,
                       caption=df.attrs.get("caption", ""),
                       label=f"tab:{label}", float_format=lambda x: f"{x}",
                       na_rep="—")
    body = body.replace(
        "\\begin{table}", "\\begin{table}[htbp]\n\\centering\n\\small")
    note = df.attrs.get("note", "")
    if note:
        body = body.replace(
            "\\end{table}",
            f"\\begin{{tablenotes}}\\footnotesize\n\\item {note}\n"
            f"\\end{{tablenotes}}\n\\end{{table}}")
    return body


def export(tabs, fmt="latex", outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    for name, df in tabs.items():
        if df is None:
            continue
        if fmt == "latex":
            with open(f"{outdir}/{name}.tex", "w") as f:
                f.write(_latex(df, name))
        elif fmt == "csv":
            df.to_csv(f"{outdir}/{name}.csv", index=False)
        elif fmt == "md":
            with open(f"{outdir}/{name}.md", "w") as f:
                f.write(f"**{df.attrs.get('caption','')}**\n\n")
                f.write(df.to_markdown(index=False))
                if df.attrs.get("note"):
                    f.write(f"\n\n*{df.attrs['note']}*\n")
    print(f"wrote {len(tabs)} tables to {outdir}/ as {fmt}")


def make_all(res_target=None, agg_tab=None, verbose=True):
    s4a, s4b = s4_groups()
    s9a, s9b = s9_estates()
    tabs = {
        "t1_cities": t1_cities(),
        "t2_sample": t2_sample(),
        "t3_graph": t3_graph(),
        "t4_reference": t4_reference(),
        "t5_distances": t5_distances(),
        "t6_access": t6_access(),
        "t7_truncation": t7_truncation(),
        "t8_distribution": t8_distribution(),
        "t9_variance_modes": t9_variance_modes(),
        "t10_variance_components": t10_variance_components(),
        "t11_correlations": t11_correlations(),
        "t12_mode_diagnostics": t12_mode_diagnostics(),
        "t13_redundancy": t13_redundancy(),
        "t14_agglomeration": t14_agglomeration(agg_tab),
        "t15_null": t15_null(),
        "t16_zones": t16_zones(),
        "t17_zone_binding": t17_zone_binding(),
        "s1_threshold": s1_threshold_sensitivity(),
        "s4a_group_counts": s4a,
        "s4b_group_relative": s4b,
        "s6_network": s6_network(),
        "s9a_estate_coverage": s9a,
        "s9b_estate_terminals": s9b,
    }
    if verbose:
        for name, df in tabs.items():
            src = df.attrs.get("source", "COMPUTED")
            print("\n" + "="*78)
            print(f"{name}   [{src}]")
            print(df.attrs.get("caption", ""))
            print("="*78)
            print(df.to_string(index=False))
            if df.attrs.get("note"):
                print(f"\nNote. {df.attrs['note']}")
    return tabs


if __name__ == "__main__":
    tabs = make_all()
    export(tabs, fmt="latex")
    export(tabs, fmt="csv")
