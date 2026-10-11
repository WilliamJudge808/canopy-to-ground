"""NOAA 2022 bare-earth lidar DTM for Maui County (0.5 m, EPSG:6634), the training target.

NOAA publishes the mosaic as a VRT whose tiles are referenced relative to it.
``fetch_vrt`` saves a local copy with those references rewritten as absolute
``/vsicurl/`` URLs (``localize_vrt``): opening the remote VRT from many
threads at once stalls GDAL. Read it inside ``raster.gdal_env``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .http import Fetcher
from .raster import grid_of

BASE = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/HI_Kahoo_Lanai_Maui_Molo_Oahu_DEM_2022_10335"
VRT_URL = f"{BASE}/HI_Kahoo_Lanai_Maui_Molo_Oahu_DEM_2022_m10335_EPSG-6634.vrt"
NODATA = -999999.0


def localize_vrt(text: str, base: str = BASE) -> str:
    """Point every relative source in a VRT at ``/vsicurl/<base>/``."""
    return re.sub(r'relativeToVRT="1">([^<]+)', lambda m: f'relativeToVRT="0">/vsicurl/{base}/{m[1]}', text)


def fetch_vrt(out: Path, fetcher: Fetcher | None = None) -> Path:
    """Download the NOAA mosaic VRT (~7 MB) to ``out`` with its tile paths made absolute."""
    resp = (fetcher or Fetcher(timeout=120)).get(VRT_URL)
    resp.raise_for_status()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(localize_vrt(resp.text), encoding="utf-8")
    return out


def read(bounds, crs: str, vrt: str) -> tuple[np.ndarray, float]:
    """(H, W) float32 bilinear elevation over ``bounds`` (``NODATA`` where void) and its valid fraction."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    transform, h, w = grid_of(bounds)
    dst = np.full((h, w), NODATA, np.float32)
    with rasterio.open(vrt) as ds:
        reproject(source=rasterio.band(ds, 1), destination=dst, src_crs=ds.crs, dst_transform=transform,
                  dst_crs=crs, dst_nodata=NODATA, src_nodata=ds.nodata if ds.nodata is not None else NODATA,
                  resampling=Resampling.bilinear)
    return dst, float((dst != NODATA).mean())
