"""Run the MFAI pipeline: clean -> flag artefacts -> network distances -> index.

    python scripts/run_pipeline.py              # all five cities
    python scripts/run_pipeline.py patna kochi  # selected cities
    python scripts/run_pipeline.py --audit      # print the artefact audit

Requires MFAI_DATA to point at the data archive (see data/README.md).
Run `python -c "from mfai.rasters import switch_epoch; switch_epoch()"`
once beforehand to sample the observed 2020 GHSL built-up surface.
"""
import argparse

from mfai.config import CITIES
from mfai.pipeline import run_all, run_city, summary_table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cities", nargs="*", choices=CITIES, default=[],
                    metavar="city")
    ap.add_argument("--audit", action="store_true",
                    help="print the geocoder-fallback threshold audit")
    a = ap.parse_args()
    if not a.cities:
        run_all(audit=a.audit)
    else:
        for c in a.cities:
            run_city(c, audit=a.audit)
        summary_table(a.cities)


if __name__ == "__main__":
    main()
