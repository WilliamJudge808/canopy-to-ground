"""Run an ensemble over a region and write a 1 m GeoTIFF.

    python -m canopy2ground.predict --checkpoint m1.pt m2.pt ... \\
        --store maps/store_east_maui --prior data/priors/copernicus \\
        --coastline data/coast/mauico_detailedcoast_county.geojson \\
        --out maps/east_maui_dtm.tif --shift-tta 4

The region is cut into overlapping 512 m tiles. Each tile is predicted by
every ensemble member at 16 one-pixel input shifts, the residuals are averaged,
added to the prior once, and the tiles are blended with a cosine feather. Each
member runs under fp16 autocast and is redone in fp32 if fp16 overflows. Rows
are streamed to disk, so memory stays at a few hundred MB for any region.

Imagery comes from a store written by ``canopy2ground.prefetch`` (one
``row_NNNN.npz`` per tile row). Fetching is kept out of this process so the
GPU job runs offline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .coast import Coastline
from .data.dataset import CHIP, GRAY_MEAN, GRAY_STD, IMAGENET_MEAN, IMAGENET_STD, LAYER_CHANNELS
from .data.topo import topo_features
from .grid import CRS as WORK_CRS
from .metrics import predict_batch
from .models import load_checkpoint
from .models.speed import use_fused_mit_attention
from .prior import PriorRaster


def checkpoint_layers(meta: dict) -> tuple[str, ...]:
    """Imagery layers a checkpoint was trained on (current or original checkpoint format)."""
    if "train_config" in meta:
        return tuple(meta["train_config"]["data"]["layers"])
    return tuple(meta.get("config", {}).get("layers", ()))


def feather(n: int, edge: int) -> np.ndarray:
    """1-D cosine ramp of width ``edge`` at both ends, 1 in the middle.

    Ramps of two tiles overlapping by ``edge`` sum to 1. The ramp excludes its
    end points, so no pixel gets zero weight (the map's outer edge stays valid).
    """
    w = np.ones(n, np.float32)
    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge + 2)[1:-1]))
        w[:edge], w[-edge:] = ramp, ramp[::-1]
    return w


def _shift(t: torch.Tensor, sy: int, sx: int, pad: int, mode: str) -> torch.Tensor:
    x = t if t.dim() == 4 else t[:, None]
    h, w = x.shape[-2:]
    x = F.pad(x.float(), (pad, pad, pad, pad), mode=mode)[..., pad + sy:pad + sy + h, pad + sx:pad + sx + w]
    return x if t.dim() == 4 else x[:, 0]


def _shifted_stack(t: torch.Tensor, shifts, pad: int, mode: str) -> torch.Tensor:
    return torch.cat([_shift(t, sy, sx, pad, mode) for sy, sx in shifts])


@torch.no_grad()
def ensemble_residual(models, b: dict, n_shift: int = 1, chunk: int = 8) -> torch.Tensor:
    """Mean residual (1, H, W) over ``models`` and over n_shift x n_shift input shifts.

    The original decoder imprints a lattice with 2 and 4 pixel periods, locked
    to the input grid. Shifting the input by k pixels moves that lattice by k
    while the terrain moves with the input, so un-shifting each prediction and
    averaging over a full period (0..3 in x and y) cancels the lattice and
    keeps the terrain. ``b`` holds rgb / dem / mask (and optional extra layers
    and coast inputs); terrain features are computed here, per shift.
    """
    p = max(1, n_shift)
    shifts = [(sy, sx) for sy in range(p) for sx in range(p)]
    h, w = b["dem"].shape[-2:]
    acc = torch.zeros((1, h, w), device=b["dem"].device)
    for i in range(0, len(shifts), chunk):
        part = shifts[i:i + chunk]

        bb = {k: _shifted_stack(b[k], part, p, "reflect")
              for k in ("rgb", "extra_img", "extra_avail", "dem") if k in b}
        for k in ("mask", "in_mask", "land"):
            if k in b:
                bb[k] = _shifted_stack(b[k], part, p, "replicate")
        if "in_mask" in bb:
            bb["dem_abs"] = bb["dem"] * bb["in_mask"] / 1000.0
        bb["dem_z"], _, _, bb["slope_mag"], bb["hillshade"] = topo_features(bb["dem"], bb["mask"])
        r = sum(predict_batch(m, bb) for m in models) / len(models)    # (n, H, W), shifted frames
        for j, (sy, sx) in enumerate(part):
            u = F.pad(r[j:j + 1, None], (p, p, p, p), mode="replicate")
            acc += u[..., p - sy:p - sy + h, p - sx:p - sx + w][:, 0]
    return acc / len(shifts)


def tile_batch(rgb: np.ndarray, dem: np.ndarray, valid: np.ndarray, extra: dict, layers,
               land: np.ndarray | None = None, device: str | torch.device = "cuda") -> dict:
    """Model inputs for one tile, normalised exactly as in training."""
    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(device)

    norm_rgb = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    b = {"rgb": t(np.moveaxis(norm_rgb, -1, 0)), "dem": t(dem), "mask": t(valid)}
    if land is not None:
        b["in_mask"] = b["mask"]
        b["land"] = t(land.astype(np.float32) * valid)
    if layers:
        imgs, avail = [], []
        for name in layers:
            arr, c = extra.get(name), LAYER_CHANNELS[name]
            if arr is None:
                imgs.append(np.zeros((c, CHIP, CHIP), np.float32))
                avail.append(np.zeros((CHIP, CHIP), np.float32))
                continue
            f = arr.astype(np.float32) / 255.0
            mean = np.r_[IMAGENET_MEAN, [GRAY_MEAN] * (c - 3)]
            std = np.r_[IMAGENET_STD, [GRAY_STD] * (c - 3)]
            imgs.append(np.moveaxis((f - mean) / std, -1, 0))
            avail.append(np.ones((CHIP, CHIP), np.float32))
        b["extra_img"] = t(np.concatenate(imgs, 0))
        b["extra_avail"] = t(np.stack(avail, 0))
    return b


class RowWriter:
    """Accumulate feathered tiles in a rolling buffer and stream finished rows to a GeoTIFF."""

    def __init__(self, path: Path, width: int, height: int, x0: float, y1: float, stride: int, crs: str):
        import rasterio
        from rasterio.transform import from_origin
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dst = rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                                 dtype="float32", crs=crs, transform=from_origin(x0, y1, 1.0, 1.0),
                                 nodata=np.nan, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
        self.w, self.h, self.buf_h = width, height, CHIP + stride
        self.acc = np.zeros((self.buf_h, width), np.float32)
        self.wgt = np.zeros((self.buf_h, width), np.float32)
        self.top = 0

    def add(self, tile: np.ndarray, weight: np.ndarray, oy: int, ox: int) -> None:
        by = oy - self.top
        hh, ww = min(CHIP, self.buf_h - by), min(CHIP, self.w - ox)
        self.acc[by:by + hh, ox:ox + ww] += tile[:hh, :ww] * weight[:hh, :ww]
        self.wgt[by:by + hh, ox:ox + ww] += weight[:hh, :ww]

    def flush(self, upto: int) -> None:
        from rasterio.windows import Window
        n = min(upto, self.h) - self.top
        if n <= 0:
            return
        out = np.where(self.wgt[:n] > 1e-6, self.acc[:n] / np.maximum(self.wgt[:n], 1e-6), np.nan)
        self.dst.write(out.astype(np.float32), 1, window=Window(0, self.top, self.w, n))
        for a in (self.acc, self.wgt):
            a[:self.buf_h - n] = a[n:]
            a[self.buf_h - n:] = 0.0
        self.top += n

    def close(self, **tags: str) -> None:
        self.flush(self.h)
        self.dst.update_tags(**tags)
        self.dst.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="ensemble members, averaged at the residual")
    ap.add_argument("--store", required=True, type=Path, help="imagery store from canopy2ground.prefetch")
    ap.add_argument("--prior", required=True, help="prior GeoTIFF or directory of tiles")
    ap.add_argument("--coastline", help="coastline GeoJSON (required if any member uses coast inputs)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shift-tta", type=int, default=4,
                    help="n: average over an n x n grid of one-pixel shifts; 4 = 16 passes")
    ap.add_argument("--fused-attention", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    if a.fused_attention:
        use_fused_mit_attention()
    models, metas = zip(*(load_checkpoint(p) for p in a.checkpoint))
    models = [m.to(a.device) for m in models]
    member_layers = {checkpoint_layers(m) for m in metas}
    if len(member_layers) != 1:
        ap.error(f"ensemble members were trained on different imagery layers: {member_layers}")
    layers = member_layers.pop()
    coast = any(m.cfg.coast_inputs for m in models)
    if coast and not a.coastline:
        ap.error("--coastline is required: a member was trained with coast inputs")
    coastline = Coastline(a.coastline, WORK_CRS) if coast else None

    grid = json.loads((a.store / "grid.json").read_text())
    if tuple(grid["layers"]) != layers:
        ap.error(f"store has layers {grid['layers']}, checkpoints need {list(layers)}")
    x0, y1, nx, ny, stride = grid["x0"], grid["y1"], grid["nx"], grid["ny"], grid["stride"]
    width, height = grid["width"], grid["height"]
    weight = np.outer(feather(CHIP, CHIP - stride), feather(CHIP, CHIP - stride))
    prior = PriorRaster(a.prior, WORK_CRS)
    out = RowWriter(a.out, width, height, x0, y1, stride, WORK_CRS)

    t0, written = time.time(), 0
    for iy in range(ny):
        with np.load(a.store / f"row_{iy:04d}.npz") as z:
            row = {k: z[k] for k in z.files}
        for ix in range(nx):
            if row["skip"][ix] or not row["rgb_ok"][ix]:
                continue
            ox, oy = ix * stride, iy * stride
            bounds = (x0 + ox, y1 - oy - CHIP, x0 + ox + CHIP, y1 - oy)
            dem, valid = prior.read(bounds, CHIP)
            if valid.mean() < 0.5:
                continue
            land = None
            if coastline is not None:
                land = coastline.land_mask(bounds, CHIP)
                if land is None:
                    continue                                  # open sea
            extra = {n: row[n][ix] if row[n + "_ok"][ix] else None for n in layers}
            b = tile_batch(row["rgb"][ix], dem, valid.astype(np.float32), extra, layers, land, a.device)
            res = ensemble_residual(models, b, a.shift_tta)
            tile = (b["dem"] + res * b["mask"])[0].cpu().numpy()
            out.add(tile, weight * valid, oy, ox)
            written += 1
        out.flush((iy + 1) * stride)
        print(f"row {iy + 1}/{ny}  {written} tiles  {(time.time() - t0) / 60:.1f} min", flush=True)
    out.close(checkpoints=",".join(Path(p).name for p in a.checkpoint), shift_tta=str(a.shift_tta),
              note="predicted bare-earth DTM; not a survey")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
