"""Reproduction checks against the published tables.

Run with the data archive available:

    MFAI_DATA=/path/to/MFAI_archive pytest -q

Tests that need the data are skipped when it is absent.
"""
import numpy as np
import pandas as pd
import pytest

from mfai import paths
from mfai.config import CITIES
from mfai.index import access_scores, weighted_geometric
from mfai.io import load_city, load_raw

HAVE_DATA = all(paths.nearest_csv(c).exists() for c in CITIES)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="MFAI_DATA not set")

# Table 2 (raw, analysis sample) and Table 6 (mean MFAI-land)
TABLE2 = {"patna": (14_316, 12_807), "varanasi": (10_973, 9_289),
          "indore": (67_270, 48_884), "kochi": (43_464, 27_759),
          "kolkata": (26_129, 24_852)}
TABLE6_MFAI = {"patna": 0.728, "varanasi": 0.814, "indore": 0.809,
               "kochi": 0.830, "kolkata": 0.893}


# ── no data needed ──────────────────────────────────────────────────────
def test_access_score_is_half_at_reference_distance():
    df = pd.DataFrame({"nearest_major_road_dist_m": [2_000.0],
                       "nearest_railway_dist_m": [50_000.0],
                       "nearest_waterway_dist_m": [50_000.0],
                       "nearest_ICD_dist_m": [50_000.0]})
    a = access_scores(df)
    assert np.allclose(a.values, 0.5)


def test_geometric_mean_renormalises_weights_over_available_modes():
    acc = pd.DataFrame({"road": [0.8], "rail": [0.5], "icd": [0.2],
                        "water": [np.nan]})
    m, w = weighted_geometric(acc, {"road": .25, "rail": .25, "icd": .25,
                                    "water": .25})
    assert set(w) == {"road", "rail", "icd"}
    assert np.isclose(sum(w.values()), 1.0)
    assert np.isclose(m.iloc[0], (0.8 * 0.5 * 0.2) ** (1 / 3))


# ── needs data ──────────────────────────────────────────────────────────
@needs_data
@pytest.mark.parametrize("city", CITIES)
def test_table2_sample_sizes(city):
    raw, kept = TABLE2[city]
    assert len(load_raw(city)) == raw
    assert len(load_city(city)) == kept


@needs_data
@pytest.mark.parametrize("city", CITIES)
def test_table6_mean_mfai_land(city):
    assert round(load_city(city).mfai_land.mean(), 3) == TABLE6_MFAI[city]


@needs_data
def test_recomputing_the_filter_from_rounded_coordinates_is_wrong():
    """Guards against reintroducing the rounded-coordinate filter."""
    df = load_raw("indore")
    key = df.lon.round(6).astype(str) + "_" + df.lat.round(6).astype(str)
    g = df.assign(_k=key).groupby("_k")
    art = (g["_k"].transform("size") >= 20) & (
        (g["pincode_clean"].transform("nunique") >= 5)
        | (g["isic_division"].transform("nunique") >= 8))
    assert int((~art).sum()) == 48_886 != len(load_city("indore"))


@needs_data
def test_table10_variance_components():
    d = pd.concat([load_city(c).assign(city=c) for c in CITIES])
    y, grand = d.mfai_land, d.mfai_land.mean()
    sst = ((y - grand) ** 2).sum()
    cm = d.groupby("city").mfai_land.transform("mean")
    gm = d.groupby(["city", "mfg_group"]).mfai_land.transform("mean")
    shares = [((cm - grand) ** 2).sum() / sst, ((gm - cm) ** 2).sum() / sst,
              ((y - gm) ** 2).sum() / sst]
    assert [round(100 * s, 1) for s in shares] == [24.3, 1.0, 74.7]
