# Data

The data are not stored in this repository. Point `MFAI_DATA` at a copy of
the data archive:

```bash
export MFAI_DATA=/path/to/MFAI_archive
```

## Expected layout

```
MFAI_archive/
├── 01_inputs/
│   ├── enterprises/  msme_{city}_builtup.gpkg        Udyam enterprise points
│   ├── networks/     {city}_all_roads.gpkg           full OSM road network
│   │                 {city}_roads_clipped.gpkg       major-road layer
│   ├── terminals/    railway_freight_terminals_{city}.gpkg
│   │                 iwai_terminal_{city}.gpkg       (absent for Indore)
│   │                 icd_{city}.gpkg
│   ├── rasters/      GHSL_{city}_100m_2020.tif       GHS-BUILT-S, 2020 epoch
│   │                 {city}_ntl_median_500m.tif      VIIRS median composite
│   │                 {city}_ntl_stack_500m.tif       VIIRS monthly stack
│   ├── boundaries/   {city}_district.gpkg
│   └── estates/      landuse_industrial_{city}.gpkg  (absent for Kochi)
└── 02_analysis_data/
    ├── {city}_msme_nearest.csv                        pipeline output
    └── cleaned/msme_{city}_clean.gpkg                 stage 0 output
```

`{city}` is one of `patna`, `varanasi`, `indore`, `kochi`, `kolkata`.

## Sources

| Layer | Source |
|---|---|
| Enterprises | Udyam Registration portal, Ministry of MSME, Government of India |
| Road networks, industrial land use | OpenStreetMap contributors (ODbL) |
| Rail goods sheds, IWT terminals, ICD/CFS | Compiled from PM Gati Shakti layers, India-WRIS, IWAI, Indian Railways, PIB releases and port authorities, then checked by hand; per-terminal source in `results/published/supplementary/S4_terminal_inventory.csv` |
| Built-up surface | GHS-BUILT-S R2023A, European Commission JRC |
| Night-time lights | VIIRS DNB monthly composites, Earth Observation Group |

## Availability

Enterprise identifying details and enterprise-level coordinates are not
redistributed. The analysis CSVs also carry enterprise names, so they must
never be committed. `.gitignore` excludes them, along with every `.gpkg`
and `.tif` file.

The aggregate tables reported in the paper are in `results/published/`.
Rerunning the code against the archive regenerates them.

## The artefact filter

`{city}_msme_nearest.csv` holds the **full** manufacturing register before
geocoder-artefact removal. Always load it through `mfai.io.load_city()`,
which applies the filter from the stored occupancy columns:

```python
artefact = (df.coord_n >= 20) & ((df.coord_n_pin >= 5) | (df.coord_n_div >= 8))
```

Do not recompute the coordinate key from rounded `lon`/`lat`. Rounding
merges coordinates that the pipeline treated as distinct, and Indore comes
out at 48,886 instead of 48,884.
