"""Maxar Vivid 2022 (0.6 m RGB) from the State of Hawaii GIS ImageServer.

One ``exportImage`` request per window at the native 0.6 m in EPSG:3857,
warped bilinearly onto the 1 m grid. Coverage is the fraction of output
pixels that are not pure black, since the warp leaves uncovered pixels at 0.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .cache import OK, BlobCache
from .http import Fetcher
from .raster import decode_image, warp_image

SERVICE = "https://geodata.hawaii.gov/arcgis/rest/services/SoH_Imagery/Vivid_2022/ImageServer"
NATIVE_M = 0.6
MAX_PX = (15000, 4100)             # the service's export size limit (width, height)


def export_request(bounds, crs: str) -> tuple[str, str, tuple[float, float, float, float]]:
    """(url, cache key, EPSG:3857 bbox) of the export covering ``bounds``."""
    from rasterio.warp import transform_bounds

    bbox = transform_bounds(crs, "EPSG:3857", *bounds, densify_pts=21)
    xmin, ymin, xmax, ymax = bbox
    size = (min(MAX_PX[0], max(1, round((xmax - xmin) / NATIVE_M))),
            min(MAX_PX[1], max(1, round((ymax - ymin) / NATIVE_M))))
    key = f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f},{size}"
    key = "vivid_" + hashlib.sha256(key.encode()).hexdigest()[:24]
    url = (f"{SERVICE}/exportImage?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=3857"
           f"&size={size[0]},{size[1]}&imageSR=3857&format=jpg&f=image")
    return url, key, bbox


def read(bounds, crs: str, fetcher: Fetcher, cache: BlobCache | None = None) -> tuple[np.ndarray, float]:
    """(3, H, W) uint8 RGB over ``bounds`` and its pixel coverage. Raises on an HTTP failure."""
    from rasterio.transform import Affine

    url, key, (xmin, ymin, xmax, ymax) = export_request(bounds, crs)
    hit = cache.get(key) if cache is not None else None
    if hit is None or hit[0] != OK:
        resp = fetcher.get(url)
        if resp.status_code != 200 or not resp.content:
            raise OSError(f"Vivid export failed: HTTP {resp.status_code}")
        hit = (OK, resp.content)
        if cache is not None:
            cache.put(key, *hit)
    img = decode_image(hit[1])
    h, w = img.shape[1:]
    transform = Affine((xmax - xmin) / w, 0, xmin, 0, -(ymax - ymin) / h, ymax)
    out = warp_image(img, transform, "EPSG:3857", bounds, crs)
    coverage = float((out.sum(axis=0) > 0).mean())
    return out[:3], coverage
