"""Coastline polygons -> per-tile land masks.

The County of Maui 2017 Detailed Coastline is used for both training
(``land.npy`` in the cache) and inference, rasterised the same way, so the
model's land/sea input matches between the two.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Coastline:
    """Land polygons in a projected CRS, pre-cut into ``piece``-metre tiles.

    The Maui island polygon alone has ~10^5 vertices, so rasterising it whole
    for every 512 m tile is slow; an STRtree over small pieces makes each
    query touch only the few pieces that intersect the tile.
    """

    def __init__(self, geojson: str | Path, crs: str, piece: float = 4096.0):
        from pyproj import Transformer
        from shapely.geometry import shape
        from shapely.ops import clip_by_rect, transform
        from shapely.strtree import STRtree

        to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        polys = []
        for feat in json.loads(Path(geojson).read_text(encoding="utf-8"))["features"]:
            if not feat.get("geometry"):
                continue
            g = transform(to_crs, shape(feat["geometry"]))
            g = g if g.is_valid else g.buffer(0)
            polys += list(g.geoms) if g.geom_type == "MultiPolygon" else [g]

        pieces = []
        for g in polys:
            x0, y0, x1, y1 = g.bounds
            if max(x1 - x0, y1 - y0) <= piece:
                pieces.append(g)
                continue
            for gx in np.arange(np.floor(x0 / piece) * piece, x1, piece):
                for gy in np.arange(np.floor(y0 / piece) * piece, y1, piece):
                    c = clip_by_rect(g, gx, gy, gx + piece, gy + piece)
                    if not c.is_empty:
                        pieces += list(c.geoms) if hasattr(c, "geoms") else [c]
        self.pieces = [p for p in pieces if p.geom_type == "Polygon" and p.area > 0]
        self.tree = STRtree(self.pieces)

    def land_mask(self, bounds, size: int) -> np.ndarray | None:
        """uint8 (size, size) land mask for ``bounds`` (x0, y0, x1, y1); None if the tile is all sea."""
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from shapely.geometry import box
        from shapely.ops import clip_by_rect

        hits = [self.pieces[i] for i in self.tree.query(box(*bounds))]
        x0, y0, x1, y1 = bounds
        hits = [h for h in (clip_by_rect(h, x0 - 2, y0 - 2, x1 + 2, y1 + 2) for h in hits) if not h.is_empty]
        if not hits:
            return None
        return rasterize([(h, 1) for h in hits], out_shape=(size, size),
                         transform=from_bounds(*bounds, size, size), fill=0, dtype="uint8")
