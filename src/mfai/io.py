"""Canonical analysis-time loader.

The geocoder-artefact filter is applied here, using the per-coordinate
occupancy columns stored by the pipeline (``coord_n``, ``coord_n_pin``,
``coord_n_div``). It is never baked into the stored files, and it must not
be recomputed from rounded lon/lat: rounding merges coordinates the
pipeline treated as distinct and changes the Indore sample by two
enterprises (48,886 instead of 48,884).
"""
import numpy as np
import pandas as pd

from . import paths
from .config import (CENTRES, CITIES, DIV_DIVERSITY_T, FALLBACK_THRESHOLD,
                     UTM)


def artefact_mask(df, div_t=DIV_DIVERSITY_T, threshold=FALLBACK_THRESHOLD):
    """True for enterprises at a geocoder-fallback coordinate.

    Rule 1: >= threshold firms at one coordinate spanning >= 5 pincodes
            (city-centroid fallback; stored as ``fallback_rule1``).
    Rule 2: >= threshold firms spanning >= div_t ISIC divisions
            (pincode-centroid fallback).
    """
    return (df["fallback_rule1"].fillna(False).astype(bool)
            | ((df["coord_n"] >= threshold) & (df["coord_n_div"] >= div_t)))


def apply_fallback_filter(df, div_t=None, threshold=FALLBACK_THRESHOLD,
                          verbose=False):
    div_t = div_t or DIV_DIVERSITY_T
    excl = artefact_mask(df, div_t, threshold)
    out = df[~excl].copy()
    if verbose:
        print(f"  T={div_t}: {int(excl.sum()):,} excluded ({excl.mean():.1%})"
              f" -> {len(out):,} retained")
    return out


def load_raw(city):
    return pd.read_csv(paths.nearest_csv(city))


def load_city(city, div_t=None, verbose=False):
    """Filtered analysis sample for one city (a DataFrame)."""
    df = load_raw(city)
    if verbose:
        print(f"[{city}] {len(df):,} rows", end="  ")
    d = apply_fallback_filter(df, div_t, verbose=verbose).reset_index(drop=True)
    if "dist_centre_km" not in d.columns:
        from pyproj import Geod
        lat0, lon0 = CENTRES[city]
        _, _, m = Geod(ellps="WGS84").inv(
            d.lon.values, d.lat.values,
            np.full(len(d), lon0), np.full(len(d), lat0))
        d["dist_centre_km"] = m / 1000.0
    if "DN" not in d.columns:
        for alt in ("dn_2020_300m", "dn_2020", "dn_2030"):
            if alt in d.columns:
                d["DN"] = d[alt]
                break
    return d


def load_city_gdf(city, div_t=None):
    """Filtered sample as a GeoDataFrame in the city's UTM zone, with a
    short ``group`` code (G1..G8). Used by the commodity extension."""
    import geopandas as gpd
    d = load_city(city, div_t)
    d["group"] = d["mfg_group"].str[:2]
    d = d.dropna(subset=["group", "lon", "lat"])
    return gpd.GeoDataFrame(
        d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326"
    ).to_crs(epsg=UTM[city])


def load_all(cities=None, div_t=None):
    cities = cities or CITIES
    return {c: load_city(c, div_t) for c in cities}
