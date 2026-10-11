#!/bin/bash
# The East Maui map, end to end: imagery store -> 6-model ensemble -> coastline -> EPSG:3857 -> web tiles.
#     bash scripts/reproduce_map.sh
# Run from the repo root with the package installed (pip install -e ".[geo,figures,tiles]").
# Steps 2 and 3 need a CUDA GPU; on a Slurm cluster run cluster/prefetch.sh with an output path
# instead, which submits cluster/predict.slurm, and continue here from step 4.
#
# Inputs, none of which are redistributed in this repo:
#   $PRIOR   Copernicus GLO-30 DEM tiles (Copernicus_DSM_COG_10_*_DEM.tif) covering N20-N21, W157-W156,
#            from the public bucket s3://copernicus-dem-30m/.
#   $COAST/mauico_detailedcoast_county.geojson    County of Maui 2017 Detailed Coastline (County GIS), lon/lat.
#   $COAST/mauico_detailedcoast_eastmaui.geojson  the same, clipped to mapping.EAST_MAUI_BBOX.
#   $COAST/usace2013_lidar_landpoly_eastmaui.geojson
#            land above mean high water (DEM > 0.27 m LMSL) of the 2013 USACE NCMP topobathy lidar
#            (NOAA OCM dataset 9337), polygonised, clipped to the same box, features with kind "land_above_mhw".
#   $RUNS/<member>/model.pt  the six ensemble members (cluster/train_ensemble.sh).
set -euo pipefail
DATA=${DATA:-data}; PRIOR=${PRIOR:-$DATA/priors/copernicus}; COAST=${COAST:-$DATA/coast}
RUNS=${RUNS:-runs}; MAPS=${MAPS:-maps}; TILES=${TILES:-tiles}; SITE=${SITE:-site}
STORE=$MAPS/store_east_maui
MAP=$MAPS/east_maui_dtm.tif
PY=${PY:-python}
mkdir -p "$MAPS" "$TILES" "$SITE"

# 1. Imagery for every 512 m tile (resumable: rerun until every row is present).
$PY -m canopy2ground.prefetch --preset east_maui --prior "$PRIOR" --out "$STORE" --layers vivid naip --workers 4

# 2. Six members x 16 one-pixel shifts, averaged at the residual (canopy2ground.predict).
$PY -m canopy2ground.predict --store "$STORE" --prior "$PRIOR" --out "$MAP" --shift-tta 4 --fused-attention \
    --coastline "$COAST/mauico_detailedcoast_county.geojson" \
    --checkpoint "$RUNS"/{b2_s0,b2_s1,b5_s0,b5_s1,b5_coast_s0,b5_coast_s1}/model.pt

# 3. Water to NaN, and Copernicus on the few land pixels the predictor skipped.
$PY -m canopy2ground.mapping.coast_mask --pred "$MAP" --out "${MAP%.tif}_coast.tif" --prior "$PRIOR" \
    --county "$COAST/mauico_detailedcoast_eastmaui.geojson" \
    --usace "$COAST/usace2013_lidar_landpoly_eastmaui.geojson"

# 4. Onto the viewer's EPSG:3857 grid.
$PY -m canopy2ground.mapping.warp "${MAP%.tif}_coast.tif" "$TILES/east_maui_dtm_coast_3857.tif"

# 5. The prior's archive (the before/after toggle) and prior_3857.tif, the surface's gap filler.
$PY -m canopy2ground.mapping.prior_tiles --prior "$PRIOR" --work "$TILES" --publish "$SITE"

# 6. NOAA 2022 lidar over Maui island at 2 m (streams ~1 GB of COG overviews).
$PY -m canopy2ground.mapping.lidar_mosaic --vrt "$DATA/lidar.vrt" --out "$TILES/maui_lidar_2m.vrt"

# 7. Lidar + model as one surface, z8-z14 everywhere and z15 over the model, WebP tiles.
$PY -m canopy2ground.mapping.surface --model "$TILES/east_maui_dtm_coast_3857.tif" \
    --lidar "$TILES/maui_lidar_2m.vrt" --prior "$TILES/prior_3857.tif" --work "$TILES" --publish "$SITE"

# 8. The outline of measured ground, drawn over the map.
$PY -m canopy2ground.mapping.footprint --mosaic "$TILES/maui_lidar_2m.vrt" \
    --clip "$COAST/mauico_detailedcoast_county.geojson" --out "$SITE/lidar_coverage.geojson"

# 9. Publication figures.
$PY -m canopy2ground.figures.panels crop --store "$STORE" --map "${MAP%.tif}_coast.tif" --prior "$PRIOR" \
    --out figs/hanawi_gorge.npz
$PY -m canopy2ground.figures.panels overview --surface "$TILES/maui_surface_3857.tif" \
    --model "$TILES/east_maui_dtm_coast_3857.tif" --footprint "$SITE/lidar_coverage.geojson" \
    --out figs/overview.npz
$PY -m canopy2ground.figures.paper --panel figs/hanawi_gorge.npz --overview figs/overview.npz \
    --results results --out figs
