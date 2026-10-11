"""GDAL settings and warp helpers shared by the source readers, which all return a 1 m grid over ``bounds``.

``gdal_env`` holds the GDAL options for reading remote VRTs and COGs. Callers
that read remote rasters enter it once, in the main thread, around the whole
job: GDAL config options are process-global, so worker threads see them,
whereas a ``rasterio.Env`` entered per thread would clear them for every
thread when the first one exits.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from .http import USER_AGENT

if TYPE_CHECKING:
    import rasterio

GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",       # a directory listing per open costs HTTP round trips
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",   # and so is probing for sidecar files
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
    "VSI_CACHE": "TRUE",
    "GDAL_CACHEMAX": 256 * 2**20,                       # bytes (rasterio wants an int), shared by all threads
}


def gdal_env() -> rasterio.Env:
    """``GDAL_OPTIONS`` for the duration of a ``with`` block."""
    import rasterio

    return rasterio.Env(**GDAL_OPTIONS)


RES_M = 1.0


def grid_of(bounds):
    """(transform, height, width) of the 1 m grid over ``bounds`` (xmin, ymin, xmax, ymax)."""
    from rasterio.transform import from_origin

    xmin, ymin, xmax, ymax = bounds
    width = max(1, round((xmax - xmin) / RES_M))
    height = max(1, round((ymax - ymin) / RES_M))
    return from_origin(xmin, ymax, RES_M, RES_M), height, width


def warp_image(arr: np.ndarray, src_transform, src_crs: str, bounds, crs: str) -> np.ndarray:
    """Bilinear warp of a (bands, h, w) image onto the 1 m grid; pixels with no source stay 0."""
    from rasterio.warp import Resampling, reproject

    transform, height, width = grid_of(bounds)
    dst = np.zeros((arr.shape[0], height, width), dtype=arr.dtype)
    reproject(source=arr, destination=dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear)
    return dst


def decode_image(data: bytes, rgb: bool = False) -> np.ndarray:
    """JPEG/PNG bytes -> (bands, h, w) uint8. ``rgb`` converts palette/alpha images to RGB first."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if rgb and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    a = np.asarray(img)
    return a[None] if a.ndim == 2 else np.moveaxis(a, -1, 0)
