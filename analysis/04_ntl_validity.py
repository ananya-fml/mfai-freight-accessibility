"""Night-time lights: stack QC, sampling, discriminant validity

Produces: Section 4.6 statistics; writes ntl_median / ntl_log into the analysis CSVs

Three stages:
  A  qc_ntl_stack(city)      check the 48-band stack for zero-fill
  B  sample_ntl(city)        add ntl_median to the firm table
  C  orthogonality(cities)   is freight disadvantage redundant with
                             economic marginality?
THE QUESTION STAGE C ANSWERS
"""
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

from mfai import paths
from mfai.io import load_city

CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}


def check_inputs():
    for c in CITY_ORDER:
        med, stack, csv = (paths.ntl_median(c), paths.ntl_stack(c),
                           paths.nearest_csv(c))
        print(f"{c:9s} median {'OK' if os.path.exists(med) else 'MISSING':7s} "
              f"stack {'OK' if os.path.exists(stack) else 'MISSING':7s} "
              f"csv {'OK' if os.path.exists(csv) else 'MISSING'}")


# ══════════════════════════════════════════════════════════════════════
# A. QC — did zero-fill contaminate the stack?
# ══════════════════════════════════════════════════════════════════════
def qc_ntl_stack(city):
    """Per-band zero share, negatives, and coverage.

    The stack was initialised with np.zeros and reproject was called
    without dst_nodata, so any area a source tile does not cover keeps
    the value 0 — indistinguishable from genuinely dark. np.nanmedian
    then treats those as real zeros. Monsoon months over Kochi and
    Kolkata are the likely offenders.
    """
    path = paths.ntl_stack(city)
    if not os.path.exists(path):
        print(f"[{city}] stack not found: {path}")
        return None

    rows = []
    with rasterio.open(path) as src:
        desc = src.descriptions
        for i in range(1, src.count + 1):
            b = src.read(i)
            rows.append({
                "band": i,
                "month": desc[i-1] if desc and desc[i-1] else f"b{i}",
                "pct_zero": round(100 * float((b == 0).mean()), 1),
                "pct_neg":  round(100 * float((b < 0).mean()), 1),
                "median":   round(float(np.median(b[b != 0])), 3) if (b != 0).any() else np.nan,
                "p99":      round(float(np.percentile(b, 99)), 2),
            })
    df = pd.DataFrame(rows)

    print(f"\n[{city}] NTL stack QC  ({len(df)} bands)")
    print(f"  zero share : median {df.pct_zero.median():.1f}%  "
          f"min {df.pct_zero.min():.1f}%  max {df.pct_zero.max():.1f}%")
    print(f"  negatives  : median {df.pct_neg.median():.1f}%  "
          f"(small negatives are legitimate VIIRS background subtraction)")

    bad = df[df.pct_zero > df.pct_zero.median() + 10]
    if len(bad):
        print(f"\n  !! {len(bad)} band(s) with elevated zero share — likely "
              f"cloud gaps or partial tile coverage:")
        print(bad[["band", "month", "pct_zero", "median"]].to_string(index=False))
        print("  A 48-month median tolerates a few bad months, but if these")
        print("  cluster in the monsoon they bias peri-urban cells downward.")
    else:
        print("  -> no band stands out; zero-fill is not a problem here")
    return df


# ══════════════════════════════════════════════════════════════════════
# B. Sample NTL at firm coordinates
# ══════════════════════════════════════════════════════════════════════
def sample_ntl(city, write=True):
    """Add ntl_median to {city}_msme_nearest.csv.

    Firm coordinates are EPSG:4326; the raster is Mollweide, so points
    are transformed before sampling.
    """
    med  = paths.ntl_median(city)
    csvp = paths.nearest_csv(city)
    if not os.path.exists(med):
        print(f"[{city}] median composite not found: {med}")
        return None

    df = pd.read_csv(csvp)
    with rasterio.open(med) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tf.transform(df["lon"].values, df["lat"].values)
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))],
                        dtype="float64")
        nod = src.nodata
    if nod is not None:
        vals = np.where(vals == nod, np.nan, vals)

    df["ntl_median"] = np.round(vals, 4)
    # log transform for analysis: radiance is heavily right-skewed and
    # can be slightly negative, so shift before logging
    shift = max(0.0, -np.nanmin(vals)) + 0.01
    df["ntl_log"] = np.round(np.log(vals + shift), 4)

    if write:
        df.to_csv(csvp, index=False)

    v = pd.Series(vals).dropna()
    print(f"[{city}] ntl_median sampled at {len(df):,} firms | "
          f"median {v.median():.2f} | p10 {v.quantile(.10):.2f} | "
          f"p90 {v.quantile(.90):.2f} | zeros {(v == 0).mean():.1%} | "
          f"NaN {np.isnan(vals).mean():.1%}")
    return df


def sample_all(cities=None):
    cities = cities or CITY_ORDER
    for c in cities:
        qc_ntl_stack(c)
        sample_ntl(c)


# ══════════════════════════════════════════════════════════════════════
# C. Orthogonality
# ══════════════════════════════════════════════════════════════════════
def _prep(city, div_t=8):
    d = load_city(city, div_t=div_t)
    keep = ["mfai_land", "DN"] + (["ntl_log"] if "ntl_log" in d.columns else [])
    d = d.dropna(subset=[c for c in keep if c in d.columns])
    return d


def orthogonality(cities=None, div_t=8, decile=0.10):
    """Correlation, partial correlation, and the redundancy test."""
    cities = cities or CITY_ORDER

    # ---- 1. pairwise and partial correlation ----
    print("=" * 78)
    print("1. IS MFAI REDUNDANT WITH ECONOMIC-ACTIVITY MEASURES?")
    print("=" * 78)
    rows = []
    for c in cities:
        d = _prep(c, div_t)
        r = {"City": CITY_LABEL[c], "n": len(d)}
        r["MFAI–DN"] = round(stats.spearmanr(d.mfai_land, d.DN).statistic, 3)
        if "ntl_log" in d.columns:
            r["MFAI–NTL"] = round(stats.spearmanr(d.mfai_land, d.ntl_log).statistic, 3)
            r["DN–NTL"]   = round(stats.spearmanr(d.DN, d.ntl_log).statistic, 3)
            # partial: MFAI vs NTL controlling for DN, on ranks
            R = d[["mfai_land", "ntl_log", "DN"]].rank().corr().values
            inv = np.linalg.pinv(R)
            r["MFAI–NTL | DN"] = round(
                -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1]), 3)
        rows.append(r)
    corr = pd.DataFrame(rows)
    print(corr.to_string(index=False))
    print("""
  Spearman throughout. If MFAI–DN and MFAI–NTL are both weak (|rho| < 0.3),
  freight accessibility is largely independent of economic activity and
  the index is not a proxy for peripherality.
  A high DN–NTL means the two activity measures are near-duplicates;
  the partial column then shows whether NTL adds anything beyond DN.""")

    # ---- 2. the joint distribution ----
    print("\n" + "=" * 78)
    print("2. WHERE DO FREIGHT-POOR AND ECONOMICALLY-ACTIVE PLACES DIVERGE?")
    print("=" * 78)
    cells = []
    for c in cities:
        d = _prep(c, div_t)
        act = "ntl_log" if "ntl_log" in d.columns else "DN"
        lo_m = d.mfai_land <= d.mfai_land.quantile(0.25)
        hi_m = d.mfai_land >= d.mfai_land.quantile(0.75)
        lo_a = d[act] <= d[act].quantile(0.25)
        hi_a = d[act] >= d[act].quantile(0.75)
        n = len(d)
        cells.append({
            "City": CITY_LABEL[c],
            "Poor freight, active":   round(100*(lo_m & hi_a).sum()/n, 1),
            "Poor freight, inactive": round(100*(lo_m & lo_a).sum()/n, 1),
            "Good freight, active":   round(100*(hi_m & hi_a).sum()/n, 1),
            "Good freight, inactive": round(100*(hi_m & lo_a).sum()/n, 1),
        })
    joint = pd.DataFrame(cells)
    print(joint.to_string(index=False))
    print("""
  Percent of firms in each quartile combination (independence -> 6.25%).
  "Poor freight, active" is the policy-relevant cell: economically live
  manufacturing that is freight-disconnected. A share well above 6.25%
  means the two dimensions genuinely diverge.""")

    # ---- 3. the redundancy test ----
    print("\n" + "=" * 78)
    print("3. REDUNDANCY TEST — would existing data already find these firms?")
    print("=" * 78)
    rows = []
    for c in cities:
        d = _prep(c, div_t).reset_index(drop=True)
        act = "ntl_log" if "ntl_log" in d.columns else "DN"
        k = max(1, int(len(d) * decile))
        mfai_flag = set(d.mfai_land.nsmallest(k).index)
        act_flag  = set(d[act].nsmallest(k).index)
        dn_flag   = set(d.DN.nsmallest(k).index)
        rows.append({
            "City": CITY_LABEL[c],
            "n flagged": k,
            "Overlap with NTL-based": round(100*len(mfai_flag & act_flag)/k, 1),
            "Overlap with DN-based":  round(100*len(mfai_flag & dn_flag)/k, 1),
            "Unique to MFAI": round(
                100*len(mfai_flag - act_flag - dn_flag)/k, 1),
        })
    red = pd.DataFrame(rows)
    print(red.to_string(index=False))
    print(f"""
  Bottom {decile:.0%} by each measure. "Unique to MFAI" is the share of
  freight-deprived firms that neither a nighttime-lights nor a built-up
  density screen would have identified.

  This is the paper's justification stated as a number. A high value
  means the index finds firms no existing indicator does. A low value
  means a planner with census and satellite data already has them, and
  the contribution is measurement precision rather than new targets.""")

    return {"corr": corr, "joint": joint, "redundancy": red}


def mode_vs_both(cities=None, div_t=8):
    """Mode access vs NTL and vs DN, with predictor spread reported.

    A near-zero correlation means orthogonality only if the predictor
    varies. Kolkata's ICD access has a relative IQR of 5.5% against
    Patna's 12.9%, so its low correlation is partly range restriction.
    """
    from scipy import stats
    rows = []
    for c in (cities or CITY_ORDER):
        d = load_city(c, div_t=div_t)
        ntl = "ntl_log_clean" if "ntl_log_clean" in d.columns else "ntl_log"
        for m in ["road", "rail", "water", "icd"]:
            a = f"access_{m}"
            if a not in d.columns or d[a].std() < 1e-9:
                continue
            s = d.dropna(subset=[a, ntl, "DN"])
            iqr = s[a].quantile(.75) - s[a].quantile(.25)
            rows.append({
                "City": CITY_LABEL[c], "mode": m,
                "mean": round(s[a].mean(), 3),
                "IQR": round(iqr, 4),
                "rel IQR %": round(100 * iqr / s[a].mean(), 1),
                "vs NTL": round(stats.spearmanr(s[a], s[ntl]).statistic, 3),
                "vs DN": round(stats.spearmanr(s[a], s.DN).statistic, 3),
            })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\n  Read 'vs NTL' and 'vs DN' only where 'rel IQR %' is")
    print("  substantial. A near-zero correlation on a near-constant")
    print("  predictor is range restriction, not independence.")
    return out


if __name__ == "__main__":
    check_inputs()
    sample_all()                 # QC + sample all five cities
    res = orthogonality()
    mode_vs_both()
