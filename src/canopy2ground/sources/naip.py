"""NAIP Hawaii 2021 (0.6 m, R G B NIR) from the NOAA Digital Coast mosaic VRT (UTM 4N, NAD83)."""
from __future__ import annotations

import numpy as np

from .raster import grid_of

VRT = ("/vsicurl/https://coastalimagery.blob.core.windows.net/digitalcoast/HI_NAIP_2021_9668/"
       "HI_NAIP_2021_EPSG-26904.vrt")
NODATA = 0.0


def read(bounds, crs: str, vrt: str = VRT) -> tuple[np.ndarray, float]:
    """(4, H, W) float32 bilinear R, G, B, NIR over ``bounds`` and the fraction of pixels not all zero."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    transform, h, w = grid_of(bounds)
    dst = np.zeros((4, h, w), np.float32)
    with rasterio.open(vrt) as ds:
        for b in range(1, min(4, ds.count) + 1):
            reproject(source=rasterio.band(ds, b), destination=dst[b - 1], src_crs=ds.crs,
                      dst_transform=transform, dst_crs=crs, dst_nodata=NODATA,
                      src_nodata=ds.nodata if ds.nodata is not None else NODATA,
                      resampling=Resampling.bilinear)
    return dst, float((dst.sum(axis=0) != 0).mean())
