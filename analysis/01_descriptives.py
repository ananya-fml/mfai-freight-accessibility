"""Descriptive statistics: sample, distances, access scores, binding component

Produces: Tables 4 and 7 (working versions) -> results/descriptives/

Paper-ready descriptive statistics. Run after the pipeline.
    tabs = build_all_tables()
    export_tables(tabs, fmt="csv")     # or "latex"
COMPARABILITY RULES ENFORCED THROUGHOUT
  a_road, a_rail, a_water, a_icd   comparable across ALL cities
                                   (externally anchored T)
"""
import numpy as np
import pandas as pd

from mfai import paths
from mfai.io import load_city, load_raw, apply_fallback_filter

CITY_ORDER = ["patna", "varanasi", "indore", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "varanasi": "Varanasi", "indore": "Indore",
              "kochi": "Kochi", "kolkata": "Kolkata"}

GROUP_LABEL = {
    "G1_food_bev_tobacco":        "G1 Food, beverages, tobacco",
    "G2_textile_apparel_leather": "G2 Textiles, apparel, leather",
    "G3_wood_paper_printing":     "G3 Wood, paper, printing",
    "G4_petro_chem_pharma":       "G4 Petroleum, chemicals, pharma",
    "G5_rubber_plastic_mineral":  "G5 Rubber, plastics, minerals",
    "G6_metals":                  "G6 Basic and fabricated metals",
    "G7_electronics_machinery":   "G7 Electronics, machinery",
    "G8_transport_furniture_oth": "G8 Transport eq., furniture, other",
}
GROUP_ORDER = list(GROUP_LABEL)

MODES = ["road", "rail", "water", "icd"]
MODE_LABEL = {"road": "Major road", "rail": "Rail goods shed",
              "water": "IWT terminal", "icd": "ICD"}
DIST_COL = {"road": "nearest_major_road_dist_m",
            "rail": "nearest_railway_dist_m",
            "water": "nearest_waterway_dist_m",
            "icd":  "nearest_ICD_dist_m"}


def _load(cities=None, div_t=8):
    """Filtered data for all cities, concatenated."""
    cities = cities or CITY_ORDER
    frames = []
    for c in cities:
        d = load_city(c, div_t=div_t)          # from the pipeline
        d["city"] = c
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# TABLE 1 — Sample construction
# ══════════════════════════════════════════════════════════════════════
def table_sample(cities=None, div_t=8):
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        raw = load_raw(c)
        kept = apply_fallback_filter(raw, div_t)
        rows.append({
            "City": CITY_LABEL[c],
            "Manufacturing MSMEs": len(raw),
            "Geocoder artefacts removed": len(raw) - len(kept),
            "% removed": round(100 * (1 - len(kept)/len(raw)), 1),
            "Analysis sample": len(kept),
            "Unique locations": kept[["lat", "lon"]].drop_duplicates().shape[0],
            "ISIC divisions": kept["isic_division"].nunique(),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# TABLE 2 — Network distances to freight infrastructure (km)
# ══════════════════════════════════════════════════════════════════════
def table_distances(cities=None, div_t=8, stat="median"):
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        r = {"City": CITY_LABEL[c]}
        for m in MODES:
            v = d[DIST_COL[m]].dropna() / 1000.0
            if m == "water" and d["water_source"].iloc[0] != "network":
                r[MODE_LABEL[m]] = f"{v.median():.0f}*"
            elif stat == "median":
                r[MODE_LABEL[m]] = (f"{v.median():.2f} "
                                    f"[{v.quantile(.25):.2f}–{v.quantile(.75):.2f}]")
            else:
                r[MODE_LABEL[m]] = f"{v.mean():.2f}"
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["note"] = ("Median [IQR] network distance, km. "
                         "* out-of-district terminal, constant for all firms.")
    return out


# ══════════════════════════════════════════════════════════════════════
# TABLE 3 — Access scores and composite index by city
# ══════════════════════════════════════════════════════════════════════
def table_access(cities=None, div_t=8):
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        r = {"City": CITY_LABEL[c], "n": len(d)}
        for m in MODES:
            r[MODE_LABEL[m]] = round(d[f"access_{m}"].mean(), 3)
        land = d[[f"access_{m}" for m in ["road", "rail", "icd"]]].mean()
        order = land.sort_values()
        r["MFAI (3-mode)"] = round(d["mfai_land"].mean(), 3)
        r["MFAI (4-mode)"] = round(d["mfai"].mean(), 3)
        r["Binding mode"]  = MODE_LABEL[order.index[0].replace("access_", "")]
        r["Margin"]        = round(order.iloc[1] - order.iloc[0], 3)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("MFAI (3-mode)")
    out.attrs["note"] = (
        "Access scores a = T/(T+d), 0–1, higher is better. Binding mode is "
        "the lowest-mean land mode; margin is the gap to the next lowest. "
        "A margin below ~0.05 indicates no mode binds distinctly. "
        "The 4-mode index is not comparable for Indore.")
    return out


# ══════════════════════════════════════════════════════════════════════
# TABLE 4 — Within-city distribution of MFAI
# ══════════════════════════════════════════════════════════════════════
def table_distribution(cities=None, div_t=8, col="mfai_land"):
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        v = d[col].dropna()
        rows.append({
            "City": CITY_LABEL[c],
            "Mean": round(v.mean(), 3),
            "SD": round(v.std(), 3),
            "P10": round(v.quantile(.10), 3),
            "P25": round(v.quantile(.25), 3),
            "Median": round(v.median(), 3),
            "P75": round(v.quantile(.75), 3),
            "P90": round(v.quantile(.90), 3),
            "IQR": round(v.quantile(.75) - v.quantile(.25), 3),
            "P90/P10": round(v.quantile(.90) / v.quantile(.10), 3),
            "CV": round(v.std() / v.mean(), 3),
            "Skew": round(v.skew(), 2),
        })
    out = pd.DataFrame(rows).sort_values("Mean")
    out.attrs["note"] = (
        f"Distribution of {col}. P90/P10 and CV measure within-city "
        "inequality in freight accessibility; a high mean with a low CV "
        "indicates uniform good access, a low mean with a low CV indicates "
        "a city-wide constraint rather than internal inequality.")
    return out


# ══════════════════════════════════════════════════════════════════════
# TABLE 5 — Variance decomposition by city
# ══════════════════════════════════════════════════════════════════════
def table_variance(cities=None, div_t=8, weights=None):
    """Share of within-city MFAI variance attributable to each mode.

    First-order sensitivity on the log scale, since aggregation is
    geometric (Paruolo, Saisana & Saltelli 2013). Nominal weight is not
    importance.
    """
    cities = cities or CITY_ORDER
    weights = weights or {m: 0.25 for m in MODES}
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        contrib = {}
        for m in MODES:
            lv = np.log(d[f"access_{m}"].clip(lower=1e-9)).var()
            contrib[m] = (weights[m] ** 2) * lv
        tot = sum(contrib.values())
        r = {"City": CITY_LABEL[c]}
        for m in MODES:
            r[MODE_LABEL[m]] = round(100 * contrib[m] / tot, 1) if tot else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["note"] = (
        "Percentage of within-city MFAI variance contributed by each mode "
        "under equal nominal weights. Road dominates where terminal access "
        "is uniform across firms; terminals dominate where it varies.")
    return out


# ══════════════════════════════════════════════════════════════════════
# TABLE 6 — MFAI by manufacturing group and city
# ══════════════════════════════════════════════════════════════════════
def table_group_city(cities=None, div_t=8, col="mfai_land", relative=False):
    """Absolute (comparable via mfai_land) or city-relative index.

    The relative version divides each group mean by its city mean, so it
    is comparable across cities by construction and isolates sectoral
    from geographic variation.
    """
    cities = cities or CITY_ORDER
    df = _load(cities, div_t)
    piv = df.pivot_table(index="mfg_group", columns="city",
                         values=col, aggfunc="mean")
    piv = piv.reindex(GROUP_ORDER)[[c for c in cities if c in piv.columns]]
    if relative:
        city_mean = df.groupby("city")[col].mean()
        piv = piv.div(city_mean, axis=1)
    piv.columns = [CITY_LABEL[c] for c in piv.columns]
    piv.index = [GROUP_LABEL[g] for g in piv.index]
    piv["All cities"] = (df.groupby("mfg_group")[col].mean()
                           .reindex(GROUP_ORDER).values
                         if not relative else np.nan)
    if relative:
        piv = piv.drop(columns="All cities")
    out = piv.round(3)
    out.attrs["note"] = (
        ("Group mean divided by city mean; 1.000 = city average."
         if relative else
         f"Mean {col} by manufacturing group. Comparable across cities."))
    return out


def table_group_counts(cities=None, div_t=8):
    cities = cities or CITY_ORDER
    df = _load(cities, div_t)
    piv = df.pivot_table(index="mfg_group", columns="city",
                         values="msme_id", aggfunc="count")
    piv = piv.reindex(GROUP_ORDER)[[c for c in cities if c in piv.columns]]
    piv["Total"] = piv.sum(axis=1)
    piv.columns = [CITY_LABEL.get(c, c) for c in piv.columns]
    piv.index = [GROUP_LABEL[g] for g in piv.index]
    return piv.fillna(0).astype(int)


# ══════════════════════════════════════════════════════════════════════
# TABLE 7 — Access profile by manufacturing group, pooled
# ══════════════════════════════════════════════════════════════════════
def table_group_profile_fixed(cities=None, div_t=8):
    """which industries are best and worst served and on which mode
    Sector profile with city composition removed.

    Pooling raw values across cities lets city composition masquerade as
    a sector effect: G1 is 48% Indore, whose a_water is a constant 0.0499,
    so the pooled G1 IWT figure reflects sample mix, not the sector.
    """
    cities = cities or CITY_ORDER
    df = _load(cities, div_t)
    modes = ["road", "rail", "water", "icd"]
    for m in modes:
        c = f"access_{m}"
        g = df.groupby("city")[c]
        df[f"z_{m}"] = (df[c] - g.transform("mean")) / g.transform("std")
    df["z_mfai"] = ((df.mfai_land - df.groupby("city").mfai_land.transform("mean"))
                    / df.groupby("city").mfai_land.transform("std"))
    out = (df.groupby("mfg_group")[[f"z_{m}" for m in modes] + ["z_mfai"]]
             .mean().round(3))
    out["n"] = df.groupby("mfg_group").size()
    out.index = [GROUP_LABEL[g] for g in out.index]
    print(out.sort_values("z_mfai").to_string())
    print("\n  Within-city standardised. 0 = city average for that mode.")
    print("  Values within ±0.10 SD indicate no meaningful sector effect,")
    print("  consistent with the 1.0% between-sector variance share.")
    return out


# ══════════════════════════════════════════════════════════════════════
# TABLE 8 — Variance components: city vs sector
# ══════════════════════════════════════════════════════════════════════
def table_variance_components(cities=None, div_t=8, col="mfai_land"):
    """Is freight accessibility a function of WHICH CITY or WHICH SECTOR?

    One-way decomposition of total variance in firm-level accessibility
    into between-city, between-group-within-city, and residual
    (within-city-within-group) components.
    """
    cities = cities or CITY_ORDER
    df = _load(cities, div_t).dropna(subset=[col])
    y = df[col]
    grand = y.mean()
    ss_total = ((y - grand) ** 2).sum()

    city_mean = df.groupby("city")[col].transform("mean")
    ss_city = ((city_mean - grand) ** 2).sum()

    cg_mean = df.groupby(["city", "mfg_group"])[col].transform("mean")
    ss_group_within = ((cg_mean - city_mean) ** 2).sum()

    ss_resid = ((y - cg_mean) ** 2).sum()

    out = pd.DataFrame([
        {"Component": "Between cities",
         "Sum of squares": ss_city, "% of total": 100*ss_city/ss_total},
        {"Component": "Between sectors, within city",
         "Sum of squares": ss_group_within,
         "% of total": 100*ss_group_within/ss_total},
        {"Component": "Within sector, within city (residual)",
         "Sum of squares": ss_resid, "% of total": 100*ss_resid/ss_total},
    ])
    out["Sum of squares"] = out["Sum of squares"].round(2)
    out["% of total"] = out["% of total"].round(1)
    out.attrs["note"] = (
        "Decomposition of firm-level accessibility variance. A large "
        "residual share means accessibility is determined by micro-location "
        "rather than by city or sector, which is the precondition for "
        "site-specific rather than city-wide intervention.")
    return out


# ══════════════════════════════════════════════════════════════════════
# BUILD AND EXPORT
# ══════════════════════════════════════════════════════════════════════
def build_all_tables(cities=None, div_t=8):
    cities = cities or CITY_ORDER
    t = {
        "T1_sample":            table_sample(cities, div_t),
        "T2_distances":         table_distances(cities, div_t),
        "T3_access":            table_access(cities, div_t),
        "T4_distribution":      table_distribution(cities, div_t),
        "T5_variance_by_mode":  table_variance(cities, div_t),
        "T6_group_counts":      table_group_counts(cities, div_t),
        "T7_group_city_abs":    table_group_city(cities, div_t, relative=False),
        "T8_group_city_rel":    table_group_city(cities, div_t, relative=True),
        "T9_group_profile":     table_group_profile_fixed(cities, div_t),
        "T10_variance_comp":    table_variance_components(cities, div_t),
    }
    for name, tab in t.items():
        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)
        print(tab.to_string())
        if tab.attrs.get("note"):
            print(f"\nNote: {tab.attrs['note']}")
    return t


def export_tables(tabs, fmt="csv", outdir=None):
    outdir = outdir or paths.results("descriptives")
    for name, tab in tabs.items():
        if fmt == "csv":
            tab.to_csv(f"{outdir}/{name}.csv")
        elif fmt == "latex":
            with open(f"{outdir}/{name}.tex", "w") as f:
                f.write(tab.to_latex(index=True, escape=True,
                                     caption=tab.attrs.get("note", name)))
    print(f"\nWrote {len(tabs)} tables to {outdir}/ as {fmt}")


if __name__ == "__main__":
    tabs = build_all_tables()
    export_tables(tabs, fmt="csv")
