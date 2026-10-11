"""Fetch a region's imagery into an offline store for ``canopy2ground.predict``.

    python -m canopy2ground.prefetch --preset east_maui --prior data/priors/copernicus \\
        --out maps/store_east_maui --max-rows 8

The region is cut into 512 m tiles overlapping by ``--overlap`` m, the grid
``predict`` blends. Each tile row becomes one ``row_NNNN.npz`` holding, per
tile, the base photograph and every extra layer with an ``_ok`` flag, plus
``skip`` for tiles whose prior is under half valid (open sea), which are not
fetched. ``grid.json`` records the grid. The base photograph is Pictometry
where its tiles cover more than half the footprint and Vivid elsewhere; an
extra layer counts only at coverage >= 0.5. Existing rows are kept, so the
command resumes; ``--max-rows`` bounds one process's lifetime on shared login
nodes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import grid
from .data.dataset import LAYER_CHANNELS
from .mapping import EAST_MAUI_BBOX
from .prior import PriorRaster
from .sources import LAYERS, Fetcher, MemoryCache, pictometry, read_layer, vivid
from .sources.cache import BlobCache
from .sources.raster import gdal_env

CHIP = grid.CHIP_M
PRESETS = {                                   # W, S, E, N
    "east_maui": EAST_MAUI_BBOX,
    "keanae": (-156.18, 20.83, -156.12, 20.88),
}
TILE_CACHE = 6000                             # Pictometry tiles kept in memory; neighbouring tiles share them


def region_grid(wsen: tuple[float, float, float, float], overlap: int, layers=LAYERS) -> dict:
    """Tile grid over a lon/lat box: top-left origin (x0, y1), nx x ny tiles, output width x height in m."""
    from rasterio.warp import transform_bounds

    x0, y0, x1, y1 = transform_bounds("EPSG:4326", grid.CRS, *wsen, densify_pts=21)
    width, height = int(round(x1 - x0)), int(round(y1 - y0))
    stride = CHIP - overlap
    return {"x0": x0, "y1": y1, "nx": max(1, -(-(width - overlap) // stride)),
            "ny": max(1, -(-(height - overlap) // stride)), "stride": stride, "chip": CHIP,
            "width": width, "height": height, "layers": list(layers), "crs": grid.CRS}


def tile_bounds(g: dict, ix: int, iy: int) -> tuple[float, float, float, float]:
    x, y = g["x0"] + ix * g["stride"], g["y1"] - iy * g["stride"]
    return x, y - CHIP, x + CHIP, y


def base_rgb(bounds, fetcher: Fetcher, tiles: BlobCache) -> np.ndarray:
    try:
        rgb, coverage = pictometry.read(bounds, grid.CRS, fetcher, tiles, workers=8)
        if coverage > 0.5:
            return np.moveaxis(rgb, 0, -1)
    except Exception as e:
        print(f"    [warn] pictometry: {e}", flush=True)
    rgb, _ = vivid.read(bounds, grid.CRS, fetcher)  # used even if empty, as for the published map
    return np.moveaxis(rgb, 0, -1)


def fetch_tile(bounds, layers, fetcher: Fetcher, tiles: BlobCache) -> tuple[np.ndarray, dict]:
    extra = {}
    for name in layers:
        try:
            img, coverage = read_layer(name, bounds, grid.CRS, fetcher)
            extra[name] = img if coverage >= 0.5 else None
        except Exception as e:
            print(f"    [warn] {name}: {e}", flush=True)
            extra[name] = None
    return base_rgb(bounds, fetcher, tiles), extra


def fetch_row(g: dict, iy: int, prior: PriorRaster, pool: ThreadPoolExecutor, fetcher: Fetcher,
              tiles: BlobCache) -> dict[str, np.ndarray]:
    nx, layers = g["nx"], g["layers"]
    row = {"skip": np.zeros(nx, bool), "rgb": np.zeros((nx, CHIP, CHIP, 3), np.uint8),
           "rgb_ok": np.zeros(nx, bool)}
    for name in layers:
        row[name] = np.zeros((nx, CHIP, CHIP, LAYER_CHANNELS[name]), np.uint8)
        row[name + "_ok"] = np.zeros(nx, bool)
    jobs = {}
    for ix in range(nx):
        bounds = tile_bounds(g, ix, iy)
        _, valid = prior.read(bounds, CHIP)          # rasterio datasets are not thread-safe: main thread only
        if valid.mean() < 0.5:
            row["skip"][ix] = True
        else:
            jobs[ix] = pool.submit(fetch_tile, bounds, layers, fetcher, tiles)
    for ix, fut in jobs.items():
        row["rgb"][ix], extra = fut.result()
        row["rgb_ok"][ix] = True
        for name, img in extra.items():
            if img is not None:
                row[name][ix], row[name + "_ok"][ix] = img, True
    return row


def prefetch(store: Path, g: dict, prior_path: Path, workers: int = 4, max_rows: int = 0) -> int:
    """Fetch missing rows into ``store``; returns how many rows the store holds afterwards."""
    store.mkdir(parents=True, exist_ok=True)
    grid_file = store / "grid.json"
    if grid_file.exists() and json.loads(grid_file.read_text()) != g:
        raise SystemExit(f"{grid_file} describes a different grid; use a new --out")
    grid_file.write_text(json.dumps(g))

    def path(iy: int) -> Path:
        return store / f"row_{iy:04d}.npz"

    prior, fetcher, tiles = PriorRaster(prior_path, grid.CRS), Fetcher(), MemoryCache(TILE_CACHE)
    t0, done = time.time(), 0
    try:
        with gdal_env(), ThreadPoolExecutor(max_workers=workers) as pool:
            for iy in range(g["ny"]):
                if path(iy).exists():
                    continue
                if max_rows and done >= max_rows:
                    break
                row = fetch_row(g, iy, prior, pool, fetcher, tiles)
                tmp = store / f".row_{iy:04d}.part"
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, **row)
                os.replace(tmp, path(iy))
                done += 1
                print(f"row {iy + 1}/{g['ny']}: rgb {int(row['rgb_ok'].sum())}/{g['nx']}, "
                      + ", ".join(f"{n} {int(row[n + '_ok'].sum())}" for n in g["layers"])
                      + f", {int(row['skip'].sum())} sea  {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        prior.close()
    have = sum(path(iy).exists() for iy in range(g["ny"]))
    print(f"{have}/{g['ny']} rows in {store}", flush=True)
    return have


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--preset", choices=sorted(PRESETS))
    where.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), help="lon/lat")
    ap.add_argument("--prior", required=True, type=Path, help="prior GeoTIFF or directory of tiles")
    ap.add_argument("--out", required=True, type=Path, help="store directory")
    ap.add_argument("--layers", nargs="*", default=list(LAYERS), choices=LAYERS,
                    help="extra layers (none: base photograph only); must match the checkpoints predict runs")
    ap.add_argument("--overlap", type=int, default=64, help="tile overlap in m")
    ap.add_argument("--workers", type=int, default=4, help="tiles fetched concurrently")
    ap.add_argument("--max-rows", type=int, default=0, help="stop after fetching N rows (0 = all)")
    a = ap.parse_args(argv)
    g = region_grid(PRESETS[a.preset] if a.preset else tuple(a.bbox), a.overlap, a.layers)
    print(f"region: {g['width']} x {g['height']} m -> {g['nx']} x {g['ny']} tiles (stride {g['stride']} m)")
    prefetch(a.out, g, a.prior, a.workers, a.max_rows)


if __name__ == "__main__":
    main()
