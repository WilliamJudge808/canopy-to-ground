"""Post-processing of a predicted DTM into the published web map.

Stages, each a ``python -m canopy2ground.mapping.<module>`` CLI:

1. ``coast_mask``     model on land, NaN on water (County coastline plus 2013 USACE lidar land).
2. ``warp``           onto the viewer's EPSG:3857 grid.
3. ``prior_tiles``    the Copernicus prior as its own archive, for the viewer's before/after toggle.
4. ``lidar_mosaic``   the NOAA 2022 lidar over Maui island at 2 m.
5. ``surface``        lidar and model merged into one surface, tiled as terrain-RGB PMTiles, with a z15
                      detail level over the model's footprint.
6. ``footprint``      the outline of surveyed ground, as GeoJSON for the viewer.

See ``scripts/reproduce_map.sh`` for the full sequence.
"""

EAST_MAUI_BBOX = (-156.30, 20.55, -155.94, 20.99)   # W, S, E, N: the lidar void the map exists for
MAUI_BBOX = (-156.75, 20.47, -155.92, 21.06)        # Maui island plus Kahoolawe: the extent of the tiles
