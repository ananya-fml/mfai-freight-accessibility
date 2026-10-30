"""Stage 0 (cleaning, ISIC classification) and Stage 0.5 (geocoder-fallback
flagging). Nothing is dropped at Stage 0.5; the filter is applied at analysis
time by :func:`mfai.io.load_city`.
"""
import numpy as np
import pandas as pd
import geopandas as gpd

from . import paths
from .config import (DBOX, DIV_TO_GROUP, RESTRICT_DIVISIONS,
                     FALLBACK_THRESHOLD, FALLBACK_MIN_PINCODES,
                     DIV_DIVERSITY_T as DIV_DIVERSITY_T_ANALYSIS)


def clean_pincode(s):
    """Parse pincodes to Int64, rounding float representation error.
    682001.9999999999 -> 682002."""
    v = pd.to_numeric(s, errors="coerce").round()
    return v.where(v.between(100000, 999999)).astype("Int64")


def find_pincode_col(g):
    cands = [c for c in g.columns
             if "pin" in c.lower()
             and "valid" not in c.lower()
             and "clean" not in c.lower()]
    return cands[0] if cands else None


def isic_section(d):
    try:
        d = int(d)
    except (ValueError, TypeError):
        return None
    for lo, hi, sec in [(1,3,"A"),(5,9,"B"),(10,33,"C"),(35,35,"D"),
                        (36,39,"E"),(41,43,"F"),(45,47,"G"),(49,53,"H"),
                        (55,56,"I"),(58,63,"J"),(64,66,"K"),(68,68,"L"),
                        (69,75,"M"),(77,82,"N"),(84,84,"O"),(85,85,"P"),
                        (86,88,"Q"),(90,93,"R"),(94,96,"S"),(97,98,"T"),
                        (99,99,"U")]:
        if lo <= d <= hi:
            return sec
    return None


def clean_msme(city):
    log = [f"=== {city.upper()} MSME audit & cleaning ===\n"]
    g = gpd.read_file(paths.enterprises(city))
    n0 = len(g)
    log.append(f"Input records: {n0:,}")

    latc = [c for c in g.columns if c.lower() in ("lat", "latitude")][0]
    lonc = [c for c in g.columns
            if c.lower() in ("long", "lon", "lng", "longitude")][0]

    bad_geom = g.geometry.isna() | g.geometry.is_empty
    bad_num  = g[latc].isna() | g[lonc].isna() | (g[latc] == 0) | (g[lonc] == 0)
    lo1, lo2, la1, la2 = DBOX[city]
    oob = ~(g[lonc].between(lo1, lo2) & g[latc].between(la1, la2))
    drop = bad_geom | bad_num | oob
    log.append(f"  bad geometry {int(bad_geom.sum())} | "
               f"bad lat-lon {int(bad_num.sum())} | "
               f"outside bbox {int(oob.sum())}")
    g = g[~drop].copy()
    log.append(f"  -> {len(g):,} remain")

    fn = [c for c in g.columns if c.lower().replace("_","") == "filename"]
    fn_col = fn[0] if fn else None
    codes = (g[fn_col].astype(str).str.extract(r"^(\d+)")[0] if fn_col
             else pd.Series([None]*len(g), index=g.index))
    g["sector_code"] = codes

    div = codes.astype(str).str[:2]
    g["isic_division"] = div.where(div.str.match(r"^\d{2}$"), None)
    g["isic_section"]  = div.map(isic_section)
    sec_counts = g["isic_section"].value_counts(dropna=False).to_dict()
    log.append(f"  ISIC sections (full register): {sec_counts}")

    # ---- pincode parse ----
    pin_col = find_pincode_col(g)
    if pin_col:
        parsed = clean_pincode(g[pin_col])
        g["pincode_clean"] = parsed.astype("float64")
        n_bad = int(parsed.isna().sum())
        naive = (g[pin_col].astype(str)
                 .str.replace(r"\.0$", "", regex=True)
                 .str.match(r"^\d{6}$"))
        recovered = int((parsed.notna() & ~naive).sum())
        log.append(f"  pincode '{pin_col}' | unparseable {n_bad:,} "
                   f"({100*n_bad/len(g):.1f}%) | recovered by rounding "
                   f"{recovered:,}")
    else:
        g["pincode_clean"] = np.nan
        log.append("  no pincode column found")

    # ---- duplicates ----
    ent = [c for c in g.columns
           if c.lower().replace(" ","").startswith("enterprise")]
    names = (g[ent[0]].astype(str).str.strip().str.upper() if ent
             else pd.Series([""]*len(g), index=g.index))
    key = (names + "|" + g[latc].round(5).astype(str)
           + "|" + g[lonc].round(5).astype(str))
    sect = g[fn_col].astype(str) if fn_col else g["sector_code"].astype(str)
    dup = (key + "|" + sect).duplicated(keep="first")
    log.append(f"  exact duplicates dropped: {int(dup.sum()):,}")
    g = g[~dup].copy()

    # ---- ANALYSIS FRAME ----
    n_pre = len(g)
    dnum = pd.to_numeric(g["isic_division"], errors="coerce")
    g = g[dnum.isin(RESTRICT_DIVISIONS)].copy()
    dnum = pd.to_numeric(g["isic_division"], errors="coerce")
    g["mfg_group"] = dnum.map(DIV_TO_GROUP)
    log.append(f"\n  ANALYSIS FRAME: ISIC divisions 10-33 (manufacturing)")
    log.append(f"    {n_pre:,} -> {len(g):,} "
               f"({100*(n_pre-len(g))/max(n_pre,1):.1f}% removed)")
    log.append(f"    divisions present: {sorted(int(x) for x in dnum.unique())}")
    log.append(f"    groups: {g['mfg_group'].value_counts().sort_index().to_dict()}")
    log.append(f"\nFinal cleaned manufacturing records: {len(g):,}")

    out = paths.cleaned(city)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out, driver="GPKG")
    (paths.results("pipeline") / f"{city}_dataquality.txt").write_text(
        "\n".join(log))
    print("\n".join(log)); print()


# ======================================================================
# STAGE 0.5 — FLAG GEOCODER FALLBACKS (nothing removed)
# ======================================================================
def _fallback_stats(g):
    """Per-coordinate profile. Shared by the flagger and the audit."""
    g = g.copy()
    g["_lat6"] = g.geometry.y.round(6)
    g["_lon6"] = g.geometry.x.round(6)

    if "pincode_clean" in g.columns:
        g["_pin_eff"] = clean_pincode(g["pincode_clean"])
        note = "rounded parse (from clean_msme)"
    else:
        pc = find_pincode_col(g)
        if pc:
            g["_pin_eff"] = clean_pincode(g[pc])
            note = "rounded parse (recomputed)"
        else:
            g["_pin_eff"] = pd.NA
            note = "no pincode column"

    n_ok = int(g["_pin_eff"].notna().sum())
    note += f" — {n_ok:,}/{len(g):,} parseable"

    agg = {"n": ("_lat6", "size"), "n_pin": ("_pin_eff", "nunique")}
    if "isic_division" in g.columns:
        agg["n_div"] = ("isic_division", "nunique")
    st = g.groupby(["_lat6", "_lon6"]).agg(**agg).reset_index()
    return g, st, note, n_ok


def flag_geocoder_fallbacks(city):
    """Attach per-coordinate statistics; drop nothing.

    Rule 1 (pincode dispersion) is evaluated here since its parameters
    are fixed. Rule 2 (ISIC diversity) is stored as the raw
    coord_n_div count so its threshold can be varied downstream.

    Rule 1: >=20 firms at one coordinate spanning >=5 distinct pincodes.
      Catches CITY-centroid fallback.
    Rule 2: >=20 firms spanning >=T ISIC divisions, any pincode count.
      Catches PINCODE-centroid fallback, invisible to rule 1 because
      those coordinates have one pincode and look like a real estate.
    """
    path = paths.cleaned(city)
    g0 = gpd.read_file(path)
    g, st, note, n_ok = _fallback_stats(g0)

    if n_ok == 0:
        print(f"[{city}] !! no parseable pincodes — rule 1 cannot fire")

    st = st.rename(columns={"n": "coord_n", "n_pin": "coord_n_pin",
                            "n_div": "coord_n_div"})
    g = g.merge(st, on=["_lat6", "_lon6"], how="left")
    g["fallback_rule1"] = ((g.coord_n >= FALLBACK_THRESHOLD) &
                           (g.coord_n_pin >= FALLBACK_MIN_PINCODES))

    g.drop(columns=["_lat6", "_lon6", "_pin_eff"], errors="ignore").to_file(
        path, driver="GPKG")

    big = g.coord_n >= FALLBACK_THRESHOLD
    print(f"[{city}] fallback FLAGGING (nothing dropped) | {note}")
    print(f"    firms at coords with n>={FALLBACK_THRESHOLD}: {int(big.sum()):,}")
    print(f"    rule 1 flags {int(g.fallback_rule1.sum()):,} firms")
    for T in (4, 6, 8, 10, 12):
        k = g.fallback_rule1 | (big & (g.coord_n_div >= T))
        print(f"    would exclude at T={T:>2}: {int(k.sum()):>7,} ({k.mean():.1%})")


def audit_fallbacks(city, T_grid=(4, 6, 8, 10, 12, 15, 20)):
    """Threshold sweep, n_div histogram, exclusion profile."""
    g0 = gpd.read_file(paths.cleaned(city))
    n0 = len(g0)
    _, st, note, _ = _fallback_stats(g0)
    big = st.n >= FALLBACK_THRESHOLD
    r1  = big & (st.n_pin >= FALLBACK_MIN_PINCODES)

    print("=" * 74)
    print(f"{city.upper()} FALLBACK AUDIT   n = {n0:,}  | {note}")
    print("=" * 74)
    print(f"\nDIV_DIVERSITY_T SWEEP")
    print(f"  {'T':>4} {'r2 only':>9} {'coords':>8} {'firms':>10} "
          f"{'% city':>8} {'retained':>9}")
    for T in T_grid:
        r2 = big & (st.n_div >= T)
        kill = r1 | r2
        f_ = int(st.loc[kill, "n"].sum())
        print(f"  {T:>4} {int((r2 & ~r1).sum()):>9} {int(kill.sum()):>8} "
              f"{f_:>10,} {f_/n0:>7.1%} {int((big & ~kill).sum()):>9}")

    print(f"\n  n_div distribution among coords with n>={FALLBACK_THRESHOLD}:")
    for k, v in st.loc[big, "n_div"].value_counts().sort_index().items():
        print(f"    {int(k):>3} divisions : {v:>4}  {'#'*min(48, int(v))}")

    kill = r1 | (big & (st.n_div >= DIV_DIVERSITY_T_ANALYSIS))
    fb = st.loc[kill, ["_lat6","_lon6"]].assign(_excl=True)
    g2 = g0.copy()
    g2["_lat6"] = g2.geometry.y.round(6)
    g2["_lon6"] = g2.geometry.x.round(6)
    g2 = g2.merge(fb, on=["_lat6","_lon6"], how="left")
    g2["_excl"] = g2["_excl"].fillna(False)

    print(f"\nEXCLUSION PROFILE at T={DIV_DIVERSITY_T_ANALYSIS}: "
          f"{g2._excl.mean():.1%} ({int(g2._excl.sum()):,} firms)")
    comp = (g2.groupby("isic_division")
              .agg(total=("_excl","size"), excluded=("_excl","sum")))
    comp["pct"] = 100 * comp.excluded / comp.total
    comp = comp[comp.total >= 100].sort_values("pct", ascending=False)
    print(comp.round(1).to_string())
    print(f"  spread: {comp.pct.max()-comp.pct.min():.1f} percentage points")

    if "DN" in g2.columns:
        e = g2.loc[g2._excl, "DN"].dropna()
        r = g2.loc[~g2._excl, "DN"].dropna()
        if len(e) > 50 and len(r) > 50:
            rel = abs(e.median() - r.median()) / max(r.median(), 1)
            print(f"\n  DN: excluded median {e.median():.0f} | "
                  f"retained median {r.median():.0f} "
                  f"({rel:.1%} difference)")
            if e.median() < r.median() and rel > 0.05:
                print("  -> excluded firms sit in materially lower-density")
                print("     areas. Peripheral firms are the most likely to be")
                print("     freight-disadvantaged, so this biases AGAINST")
                print("     detecting deprivation — conservative for the")
                print("     targeting claim.")
            else:
                print("  -> no material density difference")
    return st, comp
