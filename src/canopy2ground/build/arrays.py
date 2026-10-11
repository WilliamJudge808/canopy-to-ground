"""Per-chip input arrays added to an assembled cache: the prior, extra imagery layers and the land mask.

Each is warped or rasterised onto the chip footprint given by its id, in
cache row order, so they align row for row with ``rgb.npy`` and ``lidar.npy``.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .. import grid
from ..coast import Coastline
from ..data.dataset import LAYER_CHANNELS
from ..prior import PriorRaster
from ..sources import LAYERS, Fetcher, SqliteCache, read_layer
from ..sources.raster import gdal_env
from .cache import CHIP, SPLITS, NpyWriter, chip_ids


def build_prior(cache: Path, dem: Path, splits=SPLITS) -> None:
    """``dem_copernicus.npy`` (float32, 0 in voids) and ``dmask_copernicus.npy`` from Copernicus GLO-30."""
    prior = PriorRaster(dem, grid.CRS)
    for split in splits:
        d = Path(cache) / split
        ids = chip_ids(d)
        with (NpyWriter(d / "dem_copernicus.npy", (len(ids), CHIP, CHIP), np.float32) as out,
              NpyWriter(d / "dmask_copernicus.npy", (len(ids), CHIP, CHIP), np.bool_) as mask):
            n_void = 0
            for cid in ids:
                elev, valid = prior.read(grid.chip_bounds(cid), CHIP)
                out.append(elev)
                mask.append(valid)
                n_void += not valid.all()
        print(f"{split}: prior for {len(ids)} chips ({n_void} with voids)", flush=True)
    prior.close()


def build_land(cache: Path, coastline: Path, splits=SPLITS) -> None:
    """``land.npy`` (uint8, 1 = land) from the County coastline, rasterised as at inference."""
    coast = Coastline(coastline, grid.CRS)
    for split in splits:
        d = Path(cache) / split
        ids = chip_ids(d)
        n_coast = 0
        with NpyWriter(d / "land.npy", (len(ids), CHIP, CHIP), np.uint8) as out:
            for cid in ids:
                land = coast.land_mask(grid.chip_bounds(cid), CHIP)
                land = np.zeros((CHIP, CHIP), np.uint8) if land is None else land
                n_coast += 0 < land.mean() < 1
                out.append(land)
        print(f"{split}: land mask for {len(ids)} chips ({n_coast} on the coast)", flush=True)


def build_layers(cache: Path, out_root: Path, layers=LAYERS, splits=SPLITS, workers: int = 8,
                 cache_root: Path | None = None) -> None:
    """``out_root/<split>/img_<layer>.npy`` (uint8) and ``avail_<layer>.npy`` (float32 coverage fraction).

    A failed or empty fetch is stored as a zero image with coverage 0, which
    the model sees through the availability plane.
    """
    fetcher = Fetcher()
    vivid_cache = SqliteCache(Path(cache_root) / "vivid_2022" / "bbox_cache.sqlite") if cache_root else None

    def fetch(name: str, cid: str) -> tuple[np.ndarray, float]:
        try:
            return read_layer(name, grid.chip_bounds(cid), grid.CRS, fetcher, vivid_cache)
        except Exception as e:
            print(f"  [warn] {cid} {name}: {e}", flush=True)
            return np.zeros((CHIP, CHIP, LAYER_CHANNELS[name]), np.uint8), 0.0

    for split in splits:
        ids = chip_ids(Path(cache) / split)
        d = Path(out_root) / split
        d.mkdir(parents=True, exist_ok=True)
        avail = {n: np.zeros(len(ids), np.float32) for n in layers}
        window: deque = deque()
        with gdal_env(), ExitStack() as stack, ThreadPoolExecutor(max_workers=workers) as pool:
            imgs = {n: stack.enter_context(NpyWriter(d / f"img_{n}.npy", (len(ids), CHIP, CHIP,
                                                                         LAYER_CHANNELS[n]), np.uint8))
                    for n in layers}
            for k, cid in enumerate(ids):
                for n in layers:
                    window.append((k, n, pool.submit(fetch, n, cid)))
                    _drain(window, imgs, avail, keep=4 * workers)
            _drain(window, imgs, avail, keep=0)
        for n in layers:
            np.save(d / f"avail_{n}.npy", avail[n])
            print(f"{split}: {n} for {len(ids)} chips ({int((avail[n] > 0).sum())} covered)", flush=True)


def _drain(window: deque, imgs: dict, avail: dict, keep: int) -> None:
    """Write finished fetches in submission order (so rows stream in order) until at most ``keep`` remain."""
    while len(window) > keep or (window and window[0][2].done()):
        row, name, fut = window.popleft()
        img, avail[name][row] = fut.result()
        imgs[name].append(img)
