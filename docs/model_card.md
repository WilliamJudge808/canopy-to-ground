# Model card: East Maui bare-earth ensemble (v3)

| | |
|---|---|
| Task | 1 m bare-earth elevation (DTM) from 1 m imagery and a 30 m DEM |
| Architecture | `TerrainFusionNet`, see [method.md](method.md) |
| Members | 2 × MiT-B2 ([b2.yaml](../configs/b2.yaml)), 2 × MiT-B5 ([b5.yaml](../configs/b5.yaml)), 2 × MiT-B5 coast-aware ([b5_coast.yaml](../configs/b5_coast.yaml)); seeds 0 and 1 each |
| Parameters | 39.7 M (MiT-B2 member), 96.9 M (MiT-B5), 97.1 M (MiT-B5 coast-aware) |
| Inference | residual averaged over members and 16 one-pixel shifts, added to the prior |
| Training data | NOAA 2022 Maui County lidar DTM; 8,453 chips of 512 m on Maui, Molokaʻi, Lānaʻi and Kahoʻolawe |
| Inputs | County of Maui aerial RGB, Maxar Vivid RGB, USDA NAIP RGBN, Copernicus GLO-30 DEM, County 2017 coastline |
| Output | 1 m grid in EPSG:6634 (NAD83(PA11) / UTM zone 4N); heights in metres in the vertical datum of the NOAA 2022 lidar |

## Intended use

Visualising terrain where no lidar exists: drainage, landform and route context in East Maui. A starting surface
for hydrology or habitat work that will be checked against field data.

## Not for

Engineering, flood mapping, hazard zoning, navigation, or any decision that needs surveyed elevations. The map is a
prediction. Its error is typically under a metre on ground like the validation set, and several metres on cliffs,
gulch walls and under closed canopy.

## Performance

| set | median abs. error | notes |
|---|---:|---|
| 1,107 validation chips, same survey, spatially blocked; also used for model selection | 0.46 m | per-chip MAE; mean 2.66 m; Copernicus 1.27 m, FABDEM 1.25 m |
| wet east Molokaʻi, region withheld (separate model) | 2.57 m | Copernicus 6.12 m, FABDEM 4.85 m |
| ICESat-2 over East Maui, 6,732 segments | 1.08 m | per 40 m segment, debiased per overpass; includes surveyed ground; Copernicus 1.49 m |
| ICESat-2 on unsurveyed ground only, 2,643 segments | 1.26 m | added after the pre-registered check; Copernicus 2.30 m |

Details and confidence intervals are in [results.md](results.md).

## Known failure modes

- **Closed canopy.** The first version still sat about 0.9 m high under dense canopy, where the prior sits 4.9 m
  high. This was not re-measured for the published ensemble. One tested loss weight on the bias traded it for MAE
  about one for one.
- **Cliffs and gulch walls.** These dominate the error tail. The model damps relief it cannot place, and the
  amplitude loss only partly counters this.
- **Imagery gaps.** Tiles where the prior is mostly void are not predicted. Where the County photograph is
  missing, the Vivid mosaic stands in as the base photograph. Where an extra layer is missing, the model sees
  zeros plus an availability flag.
- **Off-survey shift.** Median error is 2.7× higher on ICESat-2 unsurveyed ground (1.26 m, a different metric)
  and 5.6× higher on the withheld region (2.57 m) than on the validation chips. Generalisation to other islands rests
  on one withheld region and one seed.
- **Resolution.** The output grid is 1 m. The finest wavelength that still matches lidar was 42 m at the median for
  the first version (23 m or finer on a quarter of chips).
