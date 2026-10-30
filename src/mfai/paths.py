"""File locations.

Data live outside the repository. Point ``MFAI_DATA`` at the data archive
(the folder holding ``01_inputs/`` and ``02_analysis_data/``); results are
written under ``MFAI_RESULTS`` (default: ``results/`` in the repo).

    export MFAI_DATA=/path/to/MFAI_archive
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MFAI_DATA", REPO / "data"))
RESULTS = Path(os.environ.get("MFAI_RESULTS", REPO / "results"))

INPUTS = DATA / "01_inputs"
ANALYSIS = DATA / "02_analysis_data"


def results(sub=""):
    p = RESULTS / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── inputs ──────────────────────────────────────────────────────────────
def enterprises(city):
    return INPUTS / "enterprises" / f"msme_{city}_builtup.gpkg"


def roads_all(city):
    return INPUTS / "networks" / f"{city}_all_roads.gpkg"


def roads_major(city):
    return INPUTS / "networks" / f"{city}_roads_clipped.gpkg"


TERMINAL_STEM = {"rail": "railway_freight_terminals_{c}",
                 "water": "iwai_terminal_{c}",
                 "icd": "icd_{c}"}


def terminals(kind, city):
    """Path stem (no extension); loaders try .gpkg then .csv."""
    return INPUTS / "terminals" / TERMINAL_STEM[kind].format(c=city)


def ghsl(city):
    return INPUTS / "rasters" / f"GHSL_{city}_100m_2020.tif"


def ntl_median(city):
    return INPUTS / "rasters" / f"{city}_ntl_median_500m.tif"


def ntl_stack(city):
    return INPUTS / "rasters" / f"{city}_ntl_stack_500m.tif"


def district(city):
    return INPUTS / "boundaries" / f"{city}_district.gpkg"


def estates(city):
    return INPUTS / "estates" / f"landuse_industrial_{city}.gpkg"


# ── pipeline products ───────────────────────────────────────────────────
def cleaned(city):
    return ANALYSIS / "cleaned" / f"msme_{city}_clean.gpkg"


def nearest_csv(city):
    """Main analysis file: one row per enterprise, unfiltered."""
    return ANALYSIS / f"{city}_msme_nearest.csv"
