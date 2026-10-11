# Results

Raw outputs of the final checks behind the published numbers. The chip-level files were regenerated with this package's evaluation tools and match the published numbers
exactly; the ICESat-2 files were reproduced with the same tools. All come from
[`src/canopy2ground/evaluation`](../src/canopy2ground/evaluation).

| file | what | command |
|---|---|---|
| `external_val.json` | 6-model ensemble vs Copernicus, FABDEM and GEDTM30 on the 1,107 validation chips, at 1 m and 30 m, by slope bin | `python -m canopy2ground.evaluation.external score --checkpoint <6 members> ...` |
| `external_molokai.json` | the same on wet east Molokaʻi with the model trained without it | `... external score --split train --regions molokai_wet --checkpoint holdout_molokai_wet_s0.pt ...` |
| `ensembles.json` | the 6-model ensemble vs its 4-member predecessor on the validation chips, paired, with input-attribution decoys | `python -m canopy2ground.evaluation.ensemble --set ... --baseline ...` |
| `block_bootstrap_val.json`, `block_bootstrap_molokai.json` | the same paired comparisons resampling whole 10 km split blocks instead of chips | `python -m canopy2ground.evaluation.blocks --result results/external_val.json --cache data/cache` |
| `icesat_v3.json` | ICESat-2 over East Maui, v1 / v2 / v3 vs the prior, pre-registered | `python -m canopy2ground.evaluation.icesat2 score ...` |
| `icesat_v3_nolidar.json` | the same, restricted to segments outside the lidar footprint (added afterwards) | `... icesat2 score --outside lidar_coverage.geojson ...` |

Paired deltas are `error(reference) − error(model)`, so positive means the model is better. ICESat-2 deltas are
medians, with 95% intervals from 10,000 bootstrap draws that resample whole overpasses. Chip deltas are means, with
intervals from 4,000 draws that resample chips.
