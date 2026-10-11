# Data

The repository contains code only. Every dataset below is public, but they have different terms. Nothing here
redistributes imagery or third-party elevation data.

## Sources

| role | dataset | access | terms |
|---|---|---|---|
| label | NOAA 2022 topobathy lidar DEM, Kahoʻolawe / Lānaʻi / Maui / Molokaʻi / Oʻahu (NOAA OCM project 10335), 0.5 m bare earth, EPSG:6634 | NOAA Digital Coast, `noaa-nos-coastal-lidar-pds` S3 bucket (VRT) | public domain (US Government) |
| prior | Copernicus GLO-30 DEM, 1° tiles N20–N21, W158–W156 | Copernicus Data Space or the AWS open-data bucket `copernicus-dem-30m` | Copernicus DEM licence; derived products must carry the notice below |
| base imagery | County of Maui aerial imagery, 2021–2023 (`Pictometry2023_MauiCounty282` tile service) | County of Maui public ArcGIS tile services | County of Maui terms; used as model input only, never redistributed |
| imagery | Maxar Vivid 2022 mosaic via the Hawaiʻi Statewide GIS Program | `geodata.hawaii.gov` ImageServer | Hawaiʻi Statewide GIS Program terms; used as model input only, never redistributed |
| imagery | USDA NAIP 2021 Hawaiʻi, RGB + NIR | NOAA Digital Coast mosaic `HI_NAIP_2021_9668` | public domain |
| coastline | County of Maui 2017 Detailed Coastline | County of Maui open data | County of Maui terms |
| coastline (map mask) | USACE 2013 topobathy lidar land polygons | NOAA Digital Coast | public domain |
| evaluation | ICESat-2 ATL03 / ATL08 through SlideRule (ATL06-SR) | `sliderule` Python client | NASA open data |
| evaluation | FABDEM v1-2 | University of Bristol | CC BY-NC-SA 4.0; scored only, never redistributed |
| evaluation | GEDTM30 | OpenGeoHub / Zenodo | CC BY 4.0 |

Required notice on anything derived from GLO-30, including the published map:

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
> under COPERNICUS by the European Union and ESA; all rights reserved.

## Training cache

The training cache is about 50 GB of memory-mapped arrays, 8,453 training and 1,107 validation chips. Its layout is
documented in [`data/dataset.py`](../src/canopy2ground/data/dataset.py). It is built from the sources above:

```bash
python -m canopy2ground.build vrt     --out data/lidar.vrt
python -m canopy2ground.build chips   --lidar-vrt data/lidar.vrt --corpus data/corpus --tile-cache data/tc
python -m canopy2ground.build cache   --corpus data/corpus --out data/cache
python -m canopy2ground.build prior   --cache data/cache --dem data/priors/copernicus
python -m canopy2ground.build layers  --cache data/cache --out data/layers --tile-cache data/tc
python -m canopy2ground.build land    --cache data/cache --coastline data/coast/mauico_detailedcoast_county.geojson
python -m canopy2ground.build regions --cache data/cache
```

- **Selecting and splitting chips.** `chips` tries every 512 m cell over Maui County. A cell is kept when lidar
  covers at least 40% of it with at least 1,024 valid pixels and County imagery exists.
- **The split.** It is fixed by an md5 hash of 20 × 20-chip blocks. Training chips within one chip of a validation
  block are then removed.
- **Reproducibility.** Rebuilding from the public sources reproduces the published cache's chip set and row order
  exactly. Masks, labels and imagery are bit-identical (one Vivid pixel differs by one grey level), and the warped prior agrees
  to 2.4e-4 m.

## Region labels

Hold-out experiments use region labels computed from the chip grid: `east_maui`, `west_maui`, `molokai_dry`,
`molokai_wet`, `lanai` and `kahoolawe`. The channels between the islands are wider than one chip, so connected
components of the chip grid are the islands. East and West Maui are split at the isthmus, and Molokaʻi at 157.00° W.
