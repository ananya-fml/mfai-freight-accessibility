"""NTL band robustness and monocentric control

Produces: Section 4.6 robustness; writes ntl_median_clean and dist_centre_km into the analysis CSVs

Two things the correlation results need before they can be reported.
A. NTL ROBUSTNESS
   The 48-band medians include months that are entirely zero-filled
   (2022_07 is 100% zero in Indore, Kochi and Kolkata; Kolkata also
   loses 2022_06). Each dead band contributes a hard 0 to every cell's
   median, pulling it downward. With 8 compromised bands Kochi's median
"""
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer, Geod
from scipy import stats

from mfai import paths
from mfai.io import load_city


CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]
CITY_LABEL = {"patna": "Patna", "indore": "Indore", "varanasi": "Varanasi",
              "kochi": "Kochi", "kolkata": "Kolkata"}

# ── City centres ───────────────────────────────────────────────────────
# Municipal / historic core. State the source in the methods; the exact
# point matters less than using a consistent, defensible definition.
#   Patna     Gandhi Maidan
#   Indore    Rajwada Palace
#   Varanasi  Godowlia / Dashashwamedh
#   Kochi     Ernakulam town centre (MG Road)
#   Kolkata   BBD Bagh (Dalhousie Square)
CITY_CENTRE = {
    "patna":    (25.6127, 85.1416),
    "indore":   (22.7177, 75.8545),
    "varanasi": (25.3109, 83.0104),
    "kochi":    (9.9816,  76.2830),
    "kolkata":  (22.5697, 88.3497),
}

_geod = Geod(ellps="WGS84")


# ══════════════════════════════════════════════════════════════════════
# A. NTL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════
def clean_ntl_median(city, max_zero_pct=60.0, write=True):
    """Recompute the NTL median from bands below a zero-share ceiling.

    Adds ntl_median_clean and ntl_log_clean to the firm table so the
    original and cleaned versions can be compared directly.
    """
    stack_p = paths.ntl_stack(city)
    csvp    = paths.nearest_csv(city)
    if not os.path.exists(stack_p):
        print(f"[{city}] stack missing"); return None

    with rasterio.open(stack_p) as src:
        desc = src.descriptions
        keep, drop = [], []
        for i in range(1, src.count + 1):
            b = src.read(i)
            z = 100 * float((b == 0).mean())
            (keep if z <= max_zero_pct else drop).append(
                (i, desc[i-1] if desc and desc[i-1] else f"b{i}", z))
        if not keep:
            print(f"[{city}] no clean bands"); return None

        arr = np.stack([src.read(i) for i, _, _ in keep]).astype("float64")
        arr[arr == 0] = np.nan          # treat fill as missing, not dark
        med = np.nanmedian(arr, axis=0)
        crs, transform = src.crs, src.transform

    print(f"[{city}] kept {len(keep)}/{len(keep)+len(drop)} bands "
          f"(zero share <= {max_zero_pct:.0f}%)")
    if drop:
        print(f"    dropped: {', '.join(m for _, m, _ in drop)}")

    df = pd.read_csv(csvp)
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tf.transform(df["lon"].values, df["lat"].values)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    H, W = med.shape
    rows = np.clip(np.asarray(rows), 0, H-1)
    cols = np.clip(np.asarray(cols), 0, W-1)
    vals = med[rows, cols]

    df["ntl_median_clean"] = np.round(vals, 4)
    shift = np.nanmax([0.0, -np.nanmin(vals)]) + 0.01
    df["ntl_log_clean"] = np.round(np.log(vals + shift), 4)
    if write:
        df.to_csv(csvp, index=False)

    if "ntl_median" in df.columns:
        both = df[["ntl_median", "ntl_median_clean"]].dropna()
        rho = stats.spearmanr(both.ntl_median, both.ntl_median_clean).statistic
        print(f"    median {df.ntl_median.median():.2f} -> "
              f"{df.ntl_median_clean.median():.2f}  "
              f"(rank corr with original {rho:.4f})")
        if rho > 0.99:
            print("    -> ranks essentially unchanged; the dead bands were")
            print("       a uniform level shift, not a spatial distortion")
        else:
            print("    -> ranks shifted; use the cleaned version")
    return df


# ══════════════════════════════════════════════════════════════════════
# B. DISTANCE TO CENTRE
# ══════════════════════════════════════════════════════════════════════
def add_dist_centre(city, centre=None, write=True):
    """Great-circle distance from each firm to the city centre, km."""
    centre = centre or CITY_CENTRE[city]
    csvp = paths.nearest_csv(city)
    df = pd.read_csv(csvp)
    clat, clon = centre
    _, _, d = _geod.inv(df["lon"].values, df["lat"].values,
                        np.full(len(df), clon), np.full(len(df), clat))
    df["dist_centre_km"] = np.round(d / 1000.0, 3)
    df["log_dist_centre"] = np.round(np.log1p(df["dist_centre_km"]), 4)
    if write:
        df.to_csv(csvp, index=False)
    v = df["dist_centre_km"]
    print(f"[{city}] dist to centre: median {v.median():.1f} km | "
          f"p90 {v.quantile(.90):.1f} km | max {v.max():.1f} km")
    return df


def prepare_all(cities=None, max_zero_pct=60.0):
    cities = cities or CITY_ORDER
    for c in cities:
        clean_ntl_median(c, max_zero_pct)
        add_dist_centre(c)
        print()


# ══════════════════════════════════════════════════════════════════════
# C. PARTIAL CORRELATION CONTROLLING FOR DISTANCE TO CENTRE
# ══════════════════════════════════════════════════════════════════════
def _partial_spearman(df, x, y, controls):
    """Spearman partial correlation via the inverse rank-correlation
    matrix."""
    cols = [x, y] + list(controls)
    R = df[cols].rank().corr().values
    inv = np.linalg.pinv(R)
    return -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])


def monocentric_test(cities=None, div_t=8, ntl_col=None):
    """Does the MFAI-activity relationship survive controlling for
    distance to centre?

    If the partial correlations collapse toward zero, MFAI, NTL and
    built-up density are all expressions of one monocentric gradient and
    the index adds nothing. If they survive, the relationship is real.
    """
    cities = cities or CITY_ORDER
    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        col = ntl_col or ("ntl_log_clean" if "ntl_log_clean" in d.columns
                          else "ntl_log")
        need = ["mfai_land", "DN", col, "log_dist_centre"]
        if not all(k in d.columns for k in need):
            print(f"[{c}] missing {[k for k in need if k not in d.columns]}")
            continue
        d = d.dropna(subset=need)
        rows.append({
            "City": CITY_LABEL[c], "n": len(d),
            "MFAI–dist": round(stats.spearmanr(
                d.mfai_land, d.dist_centre_km).statistic, 3),
            "NTL–dist": round(stats.spearmanr(
                d[col], d.dist_centre_km).statistic, 3),
            "MFAI–NTL": round(stats.spearmanr(
                d.mfai_land, d[col]).statistic, 3),
            "MFAI–NTL | dist": round(
                _partial_spearman(d, "mfai_land", col, ["log_dist_centre"]), 3),
            "MFAI–NTL | dist,DN": round(
                _partial_spearman(d, "mfai_land", col,
                                  ["log_dist_centre", "DN"]), 3),
        })
    out = pd.DataFrame(rows)
    print("=" * 78)
    print("MONOCENTRIC CONTROL")
    print("=" * 78)
    print(out.to_string(index=False))
    print("""
  If MFAI-NTL collapses toward zero once distance to centre is
  partialled out, both variables were tracking one centre-periphery
  gradient and MFAI is not measuring freight geography specifically.
  If it survives, the association is genuine.

  Note the sign of MFAI-dist: strongly negative means accessibility
  falls with distance from the core, i.e. the index IS largely
  monocentric in that city. A weak or positive value means terminals
  are sited away from the centre and accessibility does not follow
  the urban gradient.""")
    return out


def mode_monocentric(cities=None, div_t=8):
    """Which modes are monocentric and which are not?

    A mode whose access is uncorrelated with distance to centre has a
    geography set by facility siting rather than by the urban gradient.
    That is what makes it a policy variable rather than a restatement of
    peripherality.
    """
    cities = cities or CITY_ORDER
    modes = ["road", "rail", "water", "icd"]
    raw, part = [], []
    for c in cities:
        d = load_city(c, div_t=div_t)
        col = "ntl_log_clean" if "ntl_log_clean" in d.columns else "ntl_log"
        if "log_dist_centre" not in d.columns:
            continue
        d = d.dropna(subset=["log_dist_centre", col])
        r1 = {"City": CITY_LABEL[c]}
        r2 = {"City": CITY_LABEL[c]}
        for m in modes:
            a = f"access_{m}"
            if a not in d.columns or d[a].std() < 1e-9:
                r1[m] = r2[m] = np.nan
                continue
            r1[m] = round(stats.spearmanr(d[a], d.dist_centre_km).statistic, 3)
            r2[m] = round(_partial_spearman(d, a, col, ["log_dist_centre"]), 3)
        raw.append(r1); part.append(r2)

    t1, t2 = pd.DataFrame(raw), pd.DataFrame(part)
    print("\n" + "=" * 78)
    print("MODE ACCESS vs DISTANCE TO CENTRE  (Spearman)")
    print("=" * 78)
    print(t1.to_string(index=False))
    print("\n  Strongly negative = that mode's access is a monocentric")
    print("  gradient. Near zero = its geography is independent of the")
    print("  urban core, set by where the facilities were sited.")

    print("\n" + "=" * 78)
    print("MODE ACCESS vs NTL, CONTROLLING FOR DISTANCE TO CENTRE")
    print("=" * 78)
    print(t2.to_string(index=False))
    print("""
  The key row is ICD. Patna and Indore showed ICD-NTL of 0.008 and
  0.018 — statistically independent of economic activity, in exactly the
  two cities where ICD is decisively the binding constraint (margins
  0.169 and 0.150). Their ICDs sit 32 km and 22 km out, well beyond the
  lit area, whereas Varanasi and Kochi have ICDs at ~11 km, inside it.

  If those near-zero values survive the distance control, the claim is:
  where ICD access binds, no existing economic indicator can locate the
  firms it binds on. That is the justification for the index.""")
    return t1, t2


if __name__ == "__main__":
    prepare_all()          # clean NTL medians + distance to centre
    monocentric_test()
    mode_monocentric()
