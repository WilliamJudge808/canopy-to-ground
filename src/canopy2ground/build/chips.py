"""Fetch the per-chip corpus: Pictometry photograph plus NOAA lidar label for every usable chip.

A chip is kept when, on the 1 m grid, at least 40% of its pixels carry lidar
(shoreline chips are wanted: sea cliffs are where the prior is worst, and the
loss is masked) with at least 1024 valid label pixels, and at least one
Pictometry tile covers it. Nothing else filters the chip set. Each kept chip
is written to ``<corpus>/<split>/<chip id>.npz``; rejected ids are appended
to ``<corpus>/rejected.txt``, so a rerun resumes where the last one stopped.
Network errors leave a chip unrecorded, to be retried on the next run.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .. import grid
from ..sources import Fetcher, SqliteCache, lidar, pictometry
from ..sources.cache import BlobCache
from ..sources.raster import gdal_env

MIN_LIDAR_COVERAGE = 0.40
MIN_LABEL_PX = 1024


def fetch_chip(cid: str, lidar_vrt: str, fetcher: Fetcher, cache: BlobCache | None = None,
               workers: int = 2) -> dict[str, np.ndarray] | None:
    """rgb (512, 512, 3) uint8, lidar float32 (0 where void) and lmask bool for a chip; None if unusable."""
    bounds = grid.chip_bounds(cid)
    elev, coverage = lidar.read(bounds, grid.CRS, lidar_vrt)
    if coverage < MIN_LIDAR_COVERAGE:
        return None
    valid = (elev > -10000) & (elev < 10000)
    if valid.sum() < MIN_LABEL_PX:
        return None
    rgb, rgb_coverage = pictometry.read(bounds, grid.CRS, fetcher, cache, workers=workers)
    if rgb_coverage == 0:
        return None
    elev[~valid] = 0.0
    return {"rgb": np.moveaxis(rgb, 0, -1), "lidar": elev, "lmask": valid}


def tile_cache(root: Path | None) -> SqliteCache | None:
    return SqliteCache(Path(root) / "pictometry_mauicounty282" / "z19.sqlite") if root else None


def fetch_corpus(corpus: Path, lidar_vrt: str, cache_root: Path | None = None, workers: int = 8) -> None:
    corpus = Path(corpus)
    for split in ("train", "val"):
        (corpus / split).mkdir(parents=True, exist_ok=True)
    rejected_file = corpus / "rejected.txt"
    rejected = set(rejected_file.read_text().split()) if rejected_file.exists() else set()
    todo = []
    for ix, iy in grid.chips_in_lonlat(*grid.CORPUS_LONLAT):
        cid = grid.chip_id(ix, iy)
        out = corpus / grid.chip_split(ix, iy) / f"{cid}.npz"
        if cid not in rejected and not out.exists():
            todo.append((cid, out))
    print(f"{len(todo)} chips to try ({len(rejected)} rejected earlier)", flush=True)

    fetcher, cache, lock = Fetcher(), tile_cache(cache_root), threading.Lock()

    def one(cid: str, out: Path) -> str:
        chip = fetch_chip(cid, lidar_vrt, fetcher, cache)
        if chip is None:
            with lock, open(rejected_file, "a") as f:
                f.write(cid + "\n")
            return "rejected"
        tmp = out.with_name(f".{out.name}.part")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **chip)
        os.replace(tmp, out)
        return "saved"

    counts = {"saved": 0, "rejected": 0, "error": 0}
    t0 = time.time()
    with gdal_env(), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, cid, out) for cid, out in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:                     # network or GDAL failure: retried next run
                counts["error"] += 1
                print(f"  [warn] {e}", flush=True)
            if k % 500 == 0:
                print(f"  {k}/{len(todo)} {counts}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {counts}", flush=True)
