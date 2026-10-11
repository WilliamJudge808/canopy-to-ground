"""Build the training cache from public data, one resumable step at a time.

    python -m canopy2ground.build vrt     --out data/lidar.vrt
    python -m canopy2ground.build chips   --lidar-vrt data/lidar.vrt --corpus data/corpus --tile-cache data/tc
    python -m canopy2ground.build cache   --corpus data/corpus --out data/cache
    python -m canopy2ground.build prior   --cache data/cache --dem data/priors/copernicus
    python -m canopy2ground.build layers  --cache data/cache --out data/layers --tile-cache data/tc
    python -m canopy2ground.build land    --cache data/cache \\
                                          --coastline data/coast/mauico_detailedcoast_county.geojson
    python -m canopy2ground.build regions --cache data/cache

``chips`` is the long step (every 512 m cell over Maui County is tried
against the lidar; about 9.8k are kept). The prior is the directory of
Copernicus GLO-30 1-degree tiles over the county (N20 and N21, W156 to W158,
those that exist) from the ``copernicus-dem-30m`` AWS open-data bucket. The
coastline is the County of Maui 2017 Detailed Coastline, as GeoJSON in EPSG:4326.
"""
