"""The coarse prior DEM (Copernicus GLO-30), warped onto 1 m tiles.

Lanczos resampling, as in the training cache. Bilinear is discontinuous in its
first derivative at every source-cell edge, so hillshading a bilinear prior
prints the 30 m grid, and the prior reaches the output unchanged through
``elevation = prior + residual``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class PriorRaster:
    def __init__(self, path: str | Path, crs: str):
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.merge import merge

        self.crs, self._mem = crs, None
        p = Path(path)
        if p.is_dir():                               # a directory of 1-degree tiles
            tiles = [rasterio.open(t) for t in sorted(p.glob("*.tif"))]
            arr, transform = merge(tiles)
            profile = tiles[0].profile
            profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform, count=1,
                           driver="GTiff")
            self._mem = MemoryFile()
            self.src = self._mem.open(**profile)
            self.src.write(arr[0], 1)
            for t in tiles:
                t.close()
        else:
            self.src = rasterio.open(p)

    def read(self, bounds, size: int) -> tuple[np.ndarray, np.ndarray]:
        """(elevation with voids set to 0, validity mask) on a ``size`` x ``size`` grid over ``bounds``."""
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        dem = np.full((size, size), np.nan, np.float32)
        reproject(rasterio.band(self.src, 1), dem, dst_transform=from_bounds(*bounds, size, size),
                  dst_crs=self.crs, resampling=Resampling.lanczos,
                  src_nodata=self.src.nodata, dst_nodata=np.nan)
        valid = np.isfinite(dem)
        return np.where(valid, dem, 0.0).astype(np.float32), valid

    def __enter__(self) -> PriorRaster:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self):
        self.src.close()
        if self._mem is not None:
            self._mem.close()
