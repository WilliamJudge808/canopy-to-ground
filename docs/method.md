# Method

## Problem

The input is a 30 m radar DEM (Copernicus GLO-30) plus 1 m imagery of the same ground. The output is a 1 m
bare-earth DTM. Three things make this harder than generic depth super-resolution:

1. **The target is under canopy.** The radar sees the top of the forest. Measured on wet east Molokai, GLO-30 sits a
   median +4.9 m high under dense canopy and +0.9 m on open ground. The photographs also show canopy, not ground.
2. **The scale jump is 30×.** Published DEM super-resolution usually works at 2–10×. At 30 m a 40 m wide gulch is
   one or two pixels.
3. **The imagery is heterogeneous.** County aerial photographs, Maxar Vivid mosaics and NAIP four-band imagery come
   from different years, sensors, sun angles and resolutions, and coverage has holes. A layer missing on a chip
   must be distinguishable from a dark image.

Labels come from the NOAA 2022 topobathy lidar, which covers most of Maui County except windward East Maui.

## Data

- 512 × 512 m chips at 1 m on a fixed UTM grid (EPSG:6634): 8,453 training and 1,107 validation chips.
- **Validation split.** It is spatial. 10 × 10 km blocks are hashed to validation, and training chips within 512 m
  of a validation block are dropped.
- **Channels:**
  - Optical (12 channels): base photograph RGB, Vivid RGB, NAIP RGBN, and one availability plane per extra layer.
  - Terrain (3 channels): computed on the fly from the prior. They are the per-chip z-scored elevation, slope
    magnitude and a 315°/45° hillshade.
  - The coast-aware members add a land/sea plane (County 2017 coastline) to both streams, and absolute elevation
    (`dem / 1000`) to the terrain stream. Absolute elevation is the quantity the per-chip z-score discards.
- **Normalisation.** It runs on the GPU from raw uint8 and bool arrays. A memory-mapped cache serves
  random crops without a dataloader worker pool ([`data/dataset.py`](../src/canopy2ground/data/dataset.py)).

## Architecture

`TerrainFusionNet` ([`models/fusion_net.py`](../src/canopy2ground/models/fusion_net.py)):

| stage | resolution (512 m chip) | operation |
|---|---|---|
| imagery encoder | 128 → 16 px | SegFormer MiT-B2 or B5, ImageNet initialised |
| terrain encoder | 256 → 16 px | ResNet-18, ImageNet initialised |
| fusion, 4 m and 8 m | 128, 64 px | `GatedFusionBlock`: `proj(img) + σ(W·topo) ⊙ conv([img, topo])` |
| fusion, 16 m and 32 m | 32, 16 px | `SpatialCrossAttention`: terrain queries, imagery keys and values, residual on imagery |
| decoder | 16 → 512 px | U-Net |
| 1 m detail | 512 px | fixed Laplacian plus two convolutions on the raw optical stack |
| head | 512 px | residual `r`, zeroed outside the valid mask (lidar ∧ prior for the original members, prior only for coast-aware ones); elevation = prior + `r` |

Design notes:

- **Residual on the prior.** The prior is accurate at long wavelengths: its median per-chip error is 1.27 m on
  validation chips, while their relief is hundreds of metres. Predicting a residual keeps that and spends capacity on
  the short wavelengths.
- **Fusion by scale.** Fine scales need local, per-pixel arbitration between "the photo shows an edge here" and
  "the prior says this is a slope". A gate conditioned on terrain does that. Coarse scales need global context:
  imagery value is concentrated in spatial arrangement above 256 m (scrambling 64 px patches costs 0.28 m more than
  scrambling 256 px blocks). Cross-attention gives the 16 m and 32 m features a receptive field of the whole chip.
- **Two decoders.** MiT emits no H/2 feature, so the original U-Net had an empty skip there. Together with
  nearest-neighbour upsampling, this imprinted a 2 m / 4 m lattice on the output, visible as sharp peaks in the
  hillshade spectrum. The `decoder_v2` variant uses bilinear upsampling, a learned stride-2 stem in the empty skip,
  and a wider 1 m end. In the coast-aware members no 2.0 m or 4.0 m peak remains without TTA.

## Objective

[`losses.py`](../src/canopy2ground/losses.py). Under uncertainty an L1-optimal model emits only r times the true
relief in each band, where r is its correlation with the truth. Early versions of this model emitted 0.43–0.57 of
the true relief at 2–16 m and filled gulches in. On lidar chips 98.5% of the variance sits at
wavelengths above 128 m, and under 0.1% in the 4–32 m band where gulches and ridges are. A per-pixel loss therefore
gives these bands almost no weight. The objective adds per-band terms:

```
L = 0.3·L1 + 0.5·(1/6)·Σ_b w_b·L1_b/σ_b + 1.0·Σ_b w_b·(1 − r_b)/Σw + 0.5·Σ_b w_b·|log(σ̂_b/σ_b)|/Σw + 0.2·L_fft [+ 0.3·L_sea]
```

- `b` runs over six Laplacian octaves (2–4 m up to 64–128 m).
- `w = (0.15, 1, 1, 1, 0.5, 0.25)`. The 2–4 m band is mostly texture and residual canopy.
- `σ_b` is the target's own spread in band `b`.
- The correlation term cannot be lowered by shrinking the output toward the mean. The amplitude term pays for
  keeping relief.
- `L_sea` pulls pixels seaward of the County coastline to 0 m with a ±0.5 m dead zone. Lidar hydro-flattens the sea
  at 0.00 m, and without the term the model invents land in coves.

## Training

[`train.py`](../src/canopy2ground/train.py), [`configs/`](../configs):
- AdamW, lr 1e-3, weight decay 1e-4, cosine schedule to 1e-5, 66 epochs, batch 8, fp16 autocast, gradient clip 1.0.
- The encoders train at 0.1× (MiT-B2) or 0.3× (MiT-B5) the head's learning rate. They are frozen for two epochs,
  then released one stage per epoch, deepest first. Frozen stages stay in eval mode, so the ResNet's BatchNorm
  statistics do not drift.
- Crop side cycles through 512, 256 and 128 px (coast-aware members: 512, 384, 256 px). Resolution stays 1 m; only
  the extent of context changes.
- Each epoch is one pass over the training chips without replacement. Gain and bias jitter is applied to the base
  photograph.
- No geometric augmentation is used. A d8 test (rotations and mirrors) cut imagery-attributable skill by a quarter.
  A plausible reason is that rotation moves the sun direction in image coordinates, and mirroring reverses the
  handedness of the shading. This was not tested separately.

## Inference

[`predict.py`](../src/canopy2ground/predict.py):
1. The region is cut into 512 m tiles with 64 m overlap.
2. Each tile is predicted by all six members at 16 one-pixel input shifts.
3. Residuals are averaged in each shift's own frame and added to the prior once.
4. Tiles are blended with a cosine feather.
5. Output is streamed row by row, so memory stays bounded for any region.

Imagery for a region is fetched beforehand ([`prefetch.py`](../src/canopy2ground/prefetch.py)), so the GPU job
needs no network access.

The published map is then masked to the coastline (County 2017 ∪ USACE 2013 lidar land), reprojected to EPSG:3857
and tiled as terrain-RGB WebP PMTiles ([`mapping/`](../src/canopy2ground/mapping)).
