"""Raster sampling at enterprise coordinates.

Replaces the modelled 2030 GHSL built-up surface with the observed 2020
epoch. GHS-BUILT-S epochs through 2020 rest on Sentinel-2/Landsat
observation; 2025 and 2030 are modelled. Enterprises are 2020-24 Udyam
registrations and the NTL composite covers 2022-25, so the observed epoch
is the defensible choice. The 300 m focal mean (``dn_2020_300m``) absorbs
20-50 m geocoding error and is the density measure used in the paper.
"""
import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from . import paths
from .config import CITY_ORDER, CITY_LABEL
from .io import load_city

NEW_COL = "dn_2020"      # observed epoch
OLD_COL = "dn_2030"      # existing modelled values, renamed

# ══════════════════════════════════════════════════════════════════════
# 1. SAMPLING
# ══════════════════════════════════════════════════════════════════════
def sample_raster(city, path, colname, write=True):
    """Point-sample a raster at firm coordinates.

    No clipping, no raster reprojection, no vectorising, no spatial
    join — the points are transformed into the raster's CRS instead.
    """
    csvp = paths.nearest_csv(city)
    if not os.path.exists(path):
        print(f"[{city}] raster missing: {path}")
        return None
    df = pd.read_csv(csvp)
    with rasterio.open(path) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tf.transform(df.lon.values, df.lat.values)
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
    df[colname] = np.round(vals, 1)
    if write:
        df.to_csv(csvp, index=False)
    v = pd.Series(vals).dropna()
    print(f"[{city}] {colname}: median {v.median():.0f} | "
          f"p10 {v.quantile(.10):.0f} | p90 {v.quantile(.90):.0f} | "
          f"zeros {(v == 0).mean():.1%} | NaN {np.isnan(vals).mean():.1%}")
    return df


def sample_raster_focal(city, path, colname, radius_cells=1,
                        write=True):
    """Mean of a (2r+1)^2 window around each firm.

    Geocoding error on Indian addresses is realistically 20-50 m. At
    100 m resolution a single-cell read is sensitive to that error;
    radius_cells=1 gives a 300 m window that absorbs it and is closer to
    the intended construct, neighbourhood built-up density.
    """
    csvp = paths.nearest_csv(city)
    if not os.path.exists(path):
        print(f"[{city}] raster missing: {path}")
        return None
    df = pd.read_csv(csvp)
    with rasterio.open(path) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tf.transform(df.lon.values, df.lat.values)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    H, W = arr.shape
    r = radius_cells
    rows = np.clip(np.asarray(rows), 0, H - 1)
    cols = np.clip(np.asarray(cols), 0, W - 1)
    out = np.full(len(df), np.nan)
    for i, (rr, cc) in enumerate(zip(rows, cols)):
        r0, r1 = max(0, rr - r), min(H, rr + r + 1)
        c0, c1 = max(0, cc - r), min(W, cc + r + 1)
        blk = arr[r0:r1, c0:c1]
        if blk.size and not np.isnan(blk).all():
            out[i] = np.nanmean(blk)
    df[colname] = np.round(out, 1)
    if write:
        df.to_csv(csvp, index=False)
    v = pd.Series(out).dropna()
    print(f"[{city}] {colname} (focal r={r}): median {v.median():.0f} | "
          f"NaN {np.isnan(out).mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. THE SWITCH
# ══════════════════════════════════════════════════════════════════════
def switch_epoch(cities=None, tile=None, focal=True):
    """Rename the existing DN to dn_2030, sample the observed epoch.

    tile=None      expects paths.ghsl(city)
    tile="path"    single multi-city raster used for all cities
    focal=True     also writes dn_2020_300m, a 3x3 focal mean
    """
    cities = cities or CITY_ORDER
    for c in cities:
        csvp = paths.nearest_csv(c)
        df = pd.read_csv(csvp)

        # preserve the modelled values under an explicit name
        if "DN" in df.columns and OLD_COL not in df.columns:
            df = df.rename(columns={"DN": OLD_COL})
            df.to_csv(csvp, index=False)
            print(f"[{c}] renamed DN -> {OLD_COL}")
        elif OLD_COL in df.columns:
            print(f"[{c}] {OLD_COL} already present")
        else:
            print(f"[{c}] no DN column found")

        path = tile or paths.ghsl(c)
        sample_raster(c, path, NEW_COL)
        if focal:
            sample_raster_focal(c, path, f"{NEW_COL}_300m", 1)
        print()


# ══════════════════════════════════════════════════════════════════════
# 3. DOES THE EPOCH MATTER?
# ══════════════════════════════════════════════════════════════════════
def compare_epochs(cities=None, div_t=8):
    """Compare the two epochs, and re-run the DN-dependent statistics
    under each.

    If MFAI-DN correlations and the partial correlations are stable, the
    epoch choice does not affect conclusions and the observed epoch is
    reported with a one-line robustness note. If they differ, the
    observed epoch is primary and the difference must be explained.
    """
    from scipy import stats
    cities = cities or CITY_ORDER

    def partial(d, x, y, ctrl):
        R = d[[x, y] + list(ctrl)].rank().corr().values
        inv = np.linalg.pinv(R)
        return -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])

    rows = []
    for c in cities:
        d = load_city(c, div_t=div_t)
        have = [k for k in (OLD_COL, NEW_COL, f"{NEW_COL}_300m")
                if k in d.columns]
        if NEW_COL not in have:
            print(f"[{c}] {NEW_COL} missing — run switch_epoch() first")
            continue
        ntl = ("ntl_log_clean" if "ntl_log_clean" in d.columns
               else "ntl_log" if "ntl_log" in d.columns else None)
        r = {"City": CITY_LABEL[c], "n": len(d)}

        if OLD_COL in have:
            a = d[[OLD_COL, NEW_COL]].dropna()
            r["median 2030"] = round(a[OLD_COL].median(), 0)
            r["median 2020"] = round(a[NEW_COL].median(), 0)
            r["rank corr"] = round(
                stats.spearmanr(a[OLD_COL], a[NEW_COL]).statistic, 4)
            r["MFAI–DN 2030"] = round(stats.spearmanr(
                *d[["mfai_land", OLD_COL]].dropna().values.T).statistic, 3)

        r["MFAI–DN 2020"] = round(stats.spearmanr(
            *d[["mfai_land", NEW_COL]].dropna().values.T).statistic, 3)
        if f"{NEW_COL}_300m" in have:
            r["MFAI–DN focal"] = round(stats.spearmanr(
                *d[["mfai_land", f"{NEW_COL}_300m"]].dropna().values.T
            ).statistic, 3)

        if ntl and "log_dist_centre" in d.columns:
            s = d.dropna(subset=["mfai_land", ntl, "log_dist_centre", NEW_COL])
            r["MFAI–NTL | d,DN2020"] = round(
                partial(s, "mfai_land", ntl, ["log_dist_centre", NEW_COL]), 3)
            if OLD_COL in have:
                s2 = d.dropna(subset=["mfai_land", ntl,
                                      "log_dist_centre", OLD_COL])
                r["MFAI–NTL | d,DN2030"] = round(
                    partial(s2, "mfai_land", ntl,
                            ["log_dist_centre", OLD_COL]), 3)
        rows.append(r)

    out = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("EPOCH COMPARISON")
    print("=" * 88)
    print(out.to_string(index=False))
    print("""
  rank corr    agreement between epochs in firm ordering. Above ~0.98 and
               the epoch choice cannot change any rank-based result.
  median       2020 below 2030 is expected: the modelled epoch adds
               built-up, mostly peri-urban.
  MFAI–DN      if the 2030 value is closer to zero, the modelled epoch was
               attenuating the correlation, consistent with it inflating
               peri-urban density where low-MFAI firms sit.
  focal        3x3 mean, more robust to 20-50 m geocoding error. If it
               differs materially from the single-cell read, prefer it.

  Then re-run, with DN pointed at dn_2020:
      orthogonality()      monocentric_test()      agglomeration test""")
    return out


def patch_dn_column(cities=None, target=NEW_COL):
    """Point the existing analysis scripts at the new epoch.

    Those scripts reference a column literally named "DN". Rather than
    edit each one, write the chosen epoch into a DN column.
    """
    cities = cities or CITY_ORDER
    for c in cities:
        csvp = paths.nearest_csv(c)
        if not csvp.exists():
            print(f"[{c}] no analysis CSV yet — run the pipeline first"); continue
        df = pd.read_csv(csvp)
        if target not in df.columns:
            print(f"[{c}] {target} missing"); continue
        df["DN"] = df[target]
        df.to_csv(csvp, index=False)
        print(f"[{c}] DN <- {target}  "
              f"(dn_2030 and {target} both retained)")
