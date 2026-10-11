"""Memory-mapped chip dataset and batch assembly.

A cache split (``<root>/<split>/``) holds N co-registered 512 x 512 chips at
1 m, one ``.npy`` per channel:

    rgb.npy               uint8 (N,512,512,3)  base aerial photograph
    lidar.npy             f32   (N,512,512)    NOAA 2022 lidar DTM (target)
    lmask.npy             bool  (N,512,512)    lidar ground valid
    dem_copernicus.npy    f32   (N,512,512)    Copernicus GLO-30, lanczos to 1 m
    dmask_copernicus.npy  bool  (N,512,512)    prior valid
    land.npy              uint8 (N,512,512)    County coastline (land = 1), optional
    meta.json             {"n": N, "chips": [{"i": row, "id": chip_id, ...}, ...]}

Extra imagery layers live under ``<layer_root>/<split>/`` as ``img_<name>.npy``
(uint8, N x 512 x 512 x C) and ``avail_<name>.npy`` (float32, N): the fraction
of the chip the layer covers, in [0, 1]. Region labels for hold-out experiments
are in ``<root>/island_labels.json``.

Each extra layer adds one constant availability plane to the optical input.
Training fills it with the chip's stored coverage fraction and zeroes the
layer's bands where that fraction is 0. Inference (``predict.tile_batch``)
fills it with 1.0 when the layer was fetched, which ``prefetch`` only accepts at
coverage >= 0.5, and with 0 (bands zeroed) otherwise. The published models were
trained this way.

``python -m canopy2ground.build`` builds a cache from the source rasters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..grid import CHIP_M as CHIP  # px: chips are gridded at 1 m
from .topo import topo_features

LAYER_CHANNELS = {"vivid": 3, "naip": 4}         # NAIP = R, G, B, NIR
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GRAY_MEAN, GRAY_STD = 0.449, 0.226               # for bands with no ImageNet statistic (NIR)


def extra_channels_for(layers) -> int:
    """Optical channels added by extra layers: their bands plus one availability plane each."""
    return sum(LAYER_CHANNELS[n] for n in layers) + len(layers)


@dataclass(frozen=True)
class DataConfig:
    root: str
    layer_root: str | None = None
    layers: tuple[str, ...] = ("vivid", "naip")
    coast: bool = False
    regions: tuple[str, ...] | None = None        # restrict to these region labels
    prior: str = "copernicus"


class ChipDataset:
    def __init__(self, cfg: DataConfig, split: str = "train", regions=None):
        d = Path(cfg.root) / split
        self.split = split
        self.layers = tuple(cfg.layers)
        for name in self.layers:
            if name not in LAYER_CHANNELS:
                raise ValueError(f"unknown layer {name!r}; expected one of {sorted(LAYER_CHANNELS)}")

        self.rgb = np.load(d / "rgb.npy", mmap_mode="r")
        self.lidar = np.load(d / "lidar.npy", mmap_mode="r")
        self.lmask = np.load(d / "lmask.npy", mmap_mode="r")
        self.dem = np.load(d / f"dem_{cfg.prior}.npy", mmap_mode="r")
        pm = d / f"dmask_{cfg.prior}.npy"
        self.prior_mask = np.load(pm, mmap_mode="r") if pm.exists() else None
        self.land = np.load(d / "land.npy", mmap_mode="r") if cfg.coast else None

        meta = json.loads((d / "meta.json").read_text(encoding="utf-8-sig"))
        self.chips = meta.get("chips", [])
        regions = regions if regions is not None else cfg.regions
        if regions:
            labels = json.loads((Path(cfg.root) / "island_labels.json").read_text(encoding="utf-8"))[split]
            self.rows = [i for i in range(meta["n"]) if labels.get(str(i)) in set(regions)]
            if not self.rows:
                raise ValueError(f"regions={regions} matched no {split} chips")
        else:
            self.rows = list(range(meta["n"]))

        layer_dir = Path(cfg.layer_root or cfg.root) / split
        self.extra_img = {n: np.load(layer_dir / f"img_{n}.npy", mmap_mode="r") for n in self.layers}
        self.extra_avail = {n: np.load(layer_dir / f"avail_{n}.npy", mmap_mode="r") for n in self.layers}

    def __len__(self) -> int:
        return len(self.rows)

    def crop(self, i: int, side: int = CHIP, rng: np.random.Generator | None = None) -> dict:
        """Aligned crop of chip ``i`` (dataset index). Centre crop when ``rng`` is None."""
        row = self.rows[i]
        if side >= CHIP:
            o0 = o1 = 0
            side = CHIP
        elif rng is None:
            o0 = o1 = (CHIP - side) // 2
        else:
            o0 = int(rng.integers(0, CHIP - side + 1))
            o1 = int(rng.integers(0, CHIP - side + 1))
        s = (slice(o0, o0 + side), slice(o1, o1 + side))

        prior_valid = np.ones((side, side), bool)
        if self.prior_mask is not None:
            prior_valid &= np.asarray(self.prior_mask[row][s])
        out = {
            "row": row,
            "rgb": np.asarray(self.rgb[row][s]),
            "lidar": np.asarray(self.lidar[row][s], dtype=np.float32),
            "dem": np.asarray(self.dem[row][s], dtype=np.float32),
            "mask": np.asarray(self.lmask[row][s]) & prior_valid,
            "prior_valid": prior_valid,
            "extra": {n: np.asarray(self.extra_img[n][row][s]) for n in self.layers},
        }
        if self.land is not None:
            out["land"] = np.asarray(self.land[row][s]).astype(bool)
        return out


@dataclass
class BatchOptions:
    side: int = CHIP
    augment: bool = False                # photometric gain/bias jitter on the base photograph
    decoy: str | None = None             # "scramble_rgb" | "scramble_dem": attribution controls
    decoy_src: list[int] | None = field(default=None)


def make_batch(ds: ChipDataset, indices, opts: BatchOptions | None = None,
               rng: np.random.Generator | None = None, device="cuda") -> dict | None:
    """Stack crops into a model batch; normalisation and topo features run on ``device``.

    Crops with under 10% valid lidar or no relief are dropped; returns None if
    nothing survives. ``decoy`` swaps one modality for that of the chip in
    ``decoy_src`` (pick one far away, so its imagery is uncorrelated), which is
    how imagery- and prior-attributable skill are measured.
    """
    opts = opts or BatchOptions()
    src = opts.decoy_src if opts.decoy_src is not None else list(indices)
    crops, gains, biases = [], [], []
    for idx, sidx in zip(indices, src):
        c = ds.crop(idx, side=opts.side, rng=rng)
        if c["mask"].mean() < 0.10:
            continue
        v = c["lidar"][c["mask"]]
        if v.size < 64 or float(v.max() - v.min()) < 1e-3:
            continue
        c["avail_row"] = c["row"]
        if opts.decoy == "scramble_rgb":
            d = ds.crop(sidx, side=opts.side)
            c["rgb"], c["extra"], c["avail_row"] = d["rgb"], d["extra"], d["row"]
        elif opts.decoy == "scramble_dem":
            d = ds.crop(sidx, side=opts.side)
            c["dem"] = d["dem"]
            if "land" in c:          # the coastline is prior-side information and moves with it
                c["land"], c["prior_valid"] = d["land"], d["prior_valid"]
        if opts.augment and rng is not None:
            gains.append(1.0 + rng.normal(0, 0.05))
            biases.append(rng.normal(0, 0.02))
        else:
            gains.append(1.0)
            biases.append(0.0)
        crops.append(c)
    if not crops:
        return None

    def up(key):
        return torch.from_numpy(np.stack([c[key] for c in crops])).to(device, non_blocking=True)

    b = len(crops)
    rgb = up("rgb").permute(0, 3, 1, 2).float().div_(255.0)
    if opts.augment and rng is not None:
        g = torch.as_tensor(gains, dtype=torch.float32, device=rgb.device).view(b, 1, 1, 1)
        o = torch.as_tensor(biases, dtype=torch.float32, device=rgb.device).view(b, 1, 1, 1)
        rgb = (rgb * g + o).clamp_(0.0, 1.0)
    mean = torch.as_tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    dem, lidar, mask = up("dem"), up("lidar"), up("mask").float()

    coast = "land" in crops[0]
    if coast:
        # Inputs are computed where the prior is valid (what inference sees),
        # never on the lidar footprint, which would leak the label's shape.
        land = up("land").float()
        in_mask = up("prior_valid").float()
        topo_mask = in_mask
    else:
        topo_mask = mask
    dem_z, gx, gy, slope_mag, hillshade = topo_features(dem, topo_mask)

    out = {
        "rgb": (rgb - mean) / std, "dem": dem, "lidar": lidar, "mask": mask,
        "dem_z": dem_z, "gx": gx, "gy": gy, "slope_mag": slope_mag, "hillshade": hillshade,
        "target_res": (lidar - dem) * mask,
        "chip_ids": [c["row"] for c in crops],
    }
    if coast:
        out.update(land=land * in_mask, sea=(1.0 - land) * in_mask, in_mask=in_mask,
                   dem_abs=dem * in_mask / 1000.0)
    if ds.layers:
        out.update(_extra_layers(ds, crops, device, mask.shape[-2:]))
    return out


def _extra_layers(ds: ChipDataset, crops, device, hw) -> dict:
    ch_mean, ch_std, ch_layer = [], [], []
    for li, name in enumerate(ds.layers):
        c = LAYER_CHANNELS[name]
        ch_mean += list(IMAGENET_MEAN) + [GRAY_MEAN] * (c - 3)
        ch_std += list(IMAGENET_STD) + [GRAY_STD] * (c - 3)
        ch_layer += [li] * c
    stack = np.stack([np.concatenate([c["extra"][n] for n in ds.layers], axis=-1) for c in crops])
    avail = [[float(ds.extra_avail[n][c["avail_row"]]) for n in ds.layers] for c in crops]

    x = torch.from_numpy(stack).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.as_tensor(ch_mean, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(ch_std, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    av = torch.as_tensor(avail, dtype=torch.float32, device=x.device)          # (B, n_layers)
    present = (av > 0.0).float()[:, ch_layer].view(len(crops), -1, 1, 1)
    b, n = av.shape
    return {
        # a missing layer is exactly zero, not a normalised black image
        "extra_img": ((x - m) / s) * present,
        "extra_avail": av.view(b, n, 1, 1).expand(b, n, *hw).contiguous(),
    }
