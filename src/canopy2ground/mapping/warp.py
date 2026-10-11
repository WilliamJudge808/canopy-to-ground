"""Warp a map onto the published viewer's EPSG:3857 grid.

    python -m canopy2ground.mapping.warp maps/east_maui_dtm_coast.tif maps/east_maui_dtm_coast_3857.tif

The default grid (1.0723 m, 39104 x 50192 px over East Maui) is the one the
surface tiles are built on, so a new map is a drop-in replacement for the
published one. Bilinear; water and no-data stay NaN. LERC with a 1 cm error
bound keeps the file small enough to move between machines.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np


class Grid(NamedTuple):
    transform: tuple[float, float, float, float, float, float]   # affine a, b, c, d, e, f
    width: int
    height: int


EAST_MAUI_3857 = Grid((1.0723361696418539, 0.0, -17400105.839794282,
                       0.0, -1.0723361696418539, 2391412.5552473925), 39104, 50192)


def warp_to_3857(src_path: Path, dst_path: Path, grid: Grid = EAST_MAUI_3857, block: int = 2048,
                 max_z_error: float = 0.01) -> None:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    tr = Affine(*grid.transform)
    profile = dict(driver="GTiff", dtype="float32", count=1, width=grid.width, height=grid.height,
                   crs="EPSG:3857", transform=tr, nodata=np.nan, tiled=True, blockxsize=512, blockysize=512,
                   compress="lerc_deflate", max_z_error=max_z_error, BIGTIFF="YES", num_threads="ALL_CPUS")
    t0 = time.time()
    with rasterio.open(src_path) as src, rasterio.open(dst_path, "w", **profile) as dst:
        for r0 in range(0, grid.height, block):
            h = min(block, grid.height - r0)
            win = Window(0, r0, grid.width, h)
            out = np.full((h, grid.width), np.nan, np.float32)
            reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=window_transform(win, tr), dst_crs="EPSG:3857",
                      dst_nodata=np.nan, resampling=Resampling.bilinear, num_threads=4)
            dst.write(out, 1, window=win)
            print(f"  rows {r0 + h}/{grid.height}  finite {np.isfinite(out).mean():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-z-error", type=float, default=0.01, help="LERC error bound, metres")
    a = ap.parse_args(argv)
    warp_to_3857(a.src, a.dst, max_z_error=a.max_z_error)
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
