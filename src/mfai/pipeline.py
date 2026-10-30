"""Pipeline runners and summary checks.

    python scripts/run_pipeline.py            # all cities
    python scripts/run_pipeline.py patna      # one city
"""
import pandas as pd

from .config import CITIES, UTM, SHORT, LAND_MODES, DIV_DIVERSITY_T as DIV_DIVERSITY_T_ANALYSIS
from .io import load_city, load_raw, apply_fallback_filter
from .clean import clean_msme, audit_fallbacks, flag_geocoder_fallbacks
from .network import check_paths, process_city
from .index import add_mfai
from .rasters import patch_dn_column


def summary_table(cities, div_t=None):
    rows = []
    for c in cities:
        df = load_city(c, div_t)
        r = {"city": c, "n": len(df)}
        for m in SHORT:
            col = f"access_{m}"
            if col in df.columns and not df[col].isna().all():
                r[f"a_{m}"] = round(df[col].mean(), 3)
        r["water_src"] = ("net" if df["water_source"].iloc[0] == "network"
                          else "override")
        r["mfai"]      = round(df["mfai"].mean(), 3)
        r["mfai_land"] = round(df["mfai_land"].mean(), 3)
        land = df[[f"access_{m}" for m in LAND_MODES]].mean().sort_values()
        r["binding"] = land.index[0].replace("access_", "")
        r["margin"]  = round(land.iloc[1] - land.iloc[0], 3)
        rows.append(r)
    out = pd.DataFrame(rows)
    print("\n" + "="*78)
    print(f"MFAI SUMMARY (T={div_t or DIV_DIVERSITY_T_ANALYSIS})")
    print("="*78)
    print(out.to_string(index=False))
    print("\n  mfai       4-mode. Compare only within the same water_src.")
    print("  mfai_land  3-mode. Comparable across all cities.")
    print("  binding    lowest-mean LAND mode; margin is the gap to next.")
    return out


def group_sizes(cities, div_t=None):
    """Firms per manufacturing group. Check before clustering."""
    rows = []
    for c in cities:
        d = load_city(c, div_t)
        r = {"city": c, "total": len(d)}
        r.update(d["mfg_group"].value_counts().to_dict())
        rows.append(r)
    out = pd.DataFrame(rows).fillna(0)
    grp = sorted(c for c in out.columns if c.startswith("G"))
    out = out[["city", "total"] + grp]
    out[["total"] + grp] = out[["total"] + grp].astype(int)
    print("\n" + "="*78)
    print(f"FIRMS PER MANUFACTURING GROUP (T={div_t or DIV_DIVERSITY_T_ANALYSIS})")
    print("="*78)
    print(out.to_string(index=False))
    print(f"\n  smallest cell: {out[grp].values.min():,}")
    print("  Groups below ~300 firms will yield few or no clusters at")
    print("  min_cluster_size >= 10 with a 30-firm reporting floor.")
    return out


def threshold_sensitivity(cities, T_grid=(4, 6, 8, 10, 12)):
    """Sample size and MFAI distribution across thresholds. Run the same
    table on correlation and clustering OUTPUTS — that is the evidence a
    referee needs, not that the firm count is stable."""
    rows = []
    for c in cities:
        raw = load_raw(c)
        for T in T_grid:
            d = apply_fallback_filter(raw, T)
            rows.append({"city": c, "T": T, "n": len(d),
                         "pct_excl": round(1 - len(d)/len(raw), 3),
                         "mfai_mean": round(d["mfai"].mean(), 4),
                         "mfai_p10":  round(d["mfai"].quantile(.10), 4),
                         "mfai_iqr":  round(d["mfai"].quantile(.75) -
                                            d["mfai"].quantile(.25), 4)})
    out = pd.DataFrame(rows)
    print("\n" + "="*78)
    print("THRESHOLD SENSITIVITY")
    print("="*78)
    for c in cities:
        print(f"\n{c}")
        print(out[out.city == c].drop(columns="city").to_string(index=False))
    return out


def run_city(city, audit=False):
    if not check_paths(city):
        raise RuntimeError(f"[{city}] missing required inputs")
    clean_msme(city)
    if audit:
        audit_fallbacks(city)
    flag_geocoder_fallbacks(city)
    process_city(city, UTM[city])
    add_mfai(city)


def run_all(audit=False, dn_target="dn_2020_300m"):
    """Full pipeline. Run ``mfai.rasters.switch_epoch()`` once beforehand to
    sample the observed 2020 GHSL epoch; ``DN`` is then pointed at the
    300 m focal mean, the density measure used in the paper."""
    patch_dn_column(target=dn_target)
    for c in CITIES:
        run_city(c, audit=audit)
        print("-"*70)
    summary_table(list(CITIES))
    group_sizes(list(CITIES))
    return threshold_sensitivity(list(CITIES))
