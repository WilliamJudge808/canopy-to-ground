"""Maui County Pictometry 2023 aerial imagery (service ``Pictometry2023_MauiCounty282``), the base photograph.

Served as an ArcGIS tile cache in EPSG:3857. Tiles are fetched one by one at
zoom 19 (0.28 m ground pixels at 21 N), mosaicked and warped bilinearly onto
the 1 m grid. The tile URL puts the row before the column: ``tile/{z}/{y}/{x}``.
The service metadata advertises zoom levels it does not serve, so the zoom is
fixed rather than read from it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .cache import MISSING, OK, BlobCache
from .http import Fetcher
from .raster import decode_image, grid_of, warp_image

SERVICE = ("https://tiles.arcgis.com/tiles/fsrDo0QMPlK9CkZD/arcgis/rest/services/"
           "Pictometry2023_MauiCounty282/MapServer")
ZOOM = 19
TILE_PX = 256
ORIGIN = 20037508.342789244        # EPSG:3857 half-extent


def tile_size(zoom: int = ZOOM) -> float:
    return 2 * ORIGIN / 2 ** zoom


def tile_bounds(x: int, y: int, zoom: int = ZOOM) -> tuple[float, float, float, float]:
    """EPSG:3857 bounds of tile (x, y); y counts rows southward from the top."""
    ts = tile_size(zoom)
    xmin, ymax = -ORIGIN + x * ts, ORIGIN - y * ts
    return xmin, ymax - ts, xmin + ts, ymax


def tiles_covering(xmin: float, ymin: float, xmax: float, ymax: float,
                   zoom: int = ZOOM) -> list[tuple[int, int]]:
    ts = tile_size(zoom)
    x0, x1 = int((xmin + ORIGIN) // ts), int((xmax - 1e-6 + ORIGIN) // ts)
    y0, y1 = int((ORIGIN - ymax) // ts), int((ORIGIN - (ymin + 1e-6)) // ts)
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def mosaic(tiles: dict[tuple[int, int], np.ndarray], zoom: int = ZOOM):
    """Stitch (3 or 1, 256, 256) tiles into one (3, H, W) uint8 array over their bounding box.

    Returns (array, EPSG:3857 transform). Absent tiles are left at 0;
    single-band tiles are replicated to three bands.
    """
    from rasterio.transform import Affine

    xs, ys = [k[0] for k in tiles], [k[1] for k in tiles]
    x0, y0 = min(xs), min(ys)
    out = np.zeros((3, (max(ys) - y0 + 1) * TILE_PX, (max(xs) - x0 + 1) * TILE_PX), np.uint8)
    for (x, y), a in tiles.items():
        r, c = (y - y0) * TILE_PX, (x - x0) * TILE_PX
        out[:, r:r + TILE_PX, c:c + TILE_PX] = a[:3] if a.shape[0] >= 3 else a[0]
    xmin, _, _, ymax = tile_bounds(x0, y0, zoom)
    res = tile_size(zoom) / TILE_PX
    return out, Affine(res, 0, xmin, 0, -res, ymax)


def _tile(fetcher: Fetcher, cache: BlobCache | None, x: int, y: int) -> np.ndarray | None:
    key = f"{ZOOM}/{x}/{y}"
    hit = cache.get(key) if cache is not None else None
    if hit is None:
        resp = fetcher.get(f"{SERVICE}/tile/{ZOOM}/{y}/{x}")
        if resp.status_code == 200 and resp.content:
            hit = (OK, resp.content)
        elif resp.status_code == 404:
            hit = (MISSING, None)
        else:
            return None                      # transient: not cached, retried next time
        if cache is not None:
            cache.put(key, *hit)
    return decode_image(hit[1], rgb=True) if hit[0] == OK else None


def read(bounds, crs: str, fetcher: Fetcher, cache: BlobCache | None = None,
         workers: int = 2) -> tuple[np.ndarray, float]:
    """(3, H, W) uint8 RGB over ``bounds`` (in ``crs``) and the fraction of covering tiles with data."""
    from rasterio.warp import transform_bounds

    xy = tiles_covering(*transform_bounds(crs, "EPSG:3857", *bounds, densify_pts=21))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = dict(zip(xy, pool.map(lambda t: _tile(fetcher, cache, *t), xy)))
    tiles = {k: v for k, v in got.items() if v is not None}
    if not tiles:
        _, h, w = grid_of(bounds)
        return np.zeros((3, h, w), np.uint8), 0.0
    arr, transform = mosaic(tiles)
    return warp_image(arr, transform, "EPSG:3857", bounds, crs), len(tiles) / len(xy)

