"""The outline of surveyed ground, as GeoJSON for the viewer.

    python -m canopy2ground.mapping.footprint --mosaic tiles/maui_lidar_2m.vrt \\
        --clip data/coast/mauico_detailedcoast_county.geojson --out site/lidar_coverage.geojson

The map merges lidar and prediction into one seamless surface, so this
outline is the disclaimer drawn on it: inside is measured, outside is
modelled. It must never claim ground the survey did not cover, so it is built
from the mosaic's valid pixels, not from the tile index: NOAA ships tiles that
are mostly nodata, and along the south shore the survey is a topobathy ribbon
a few hundred metres wide. Cells (100 m) count as surveyed at half coverage,
and the simplify tolerance stays well under the ribbon's width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DECIM = 50                       # 2 m -> 100 m cells
MIN_FRAC = 0.5                   # a cell counts as surveyed at half coverage
SIMPLIFY = 90.0                  # m; must stay under the south-shore ribbon's width
MIN_AREA = 1.0e6                 # drop specks under 1 km^2
FILL_HOLE = 5.0e5                # close interior gaps under 0.5 km^2 (real voids are larger)
CLIP_SIMPLIFY = 2.0              # m, after clipping to the coastline
CLIP_MIN_AREA = 2.0e3            # m^2


def coverage_polygons(mosaic: Path, clip_land: Path | None = None) -> tuple[list, str]:
    """Surveyed-ground polygons in the mosaic's CRS, largest first; and that CRS."""
    import rasterio
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from rasterio.windows import Window
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    with rasterio.open(mosaic) as src:
        h, w = src.height // DECIM, src.width // DECIM
        frac = np.empty((h, w), dtype="float32")
        for r0 in range(0, h, 64):
            hh = min(64, h - r0)
            blk = src.read(1, window=Window(0, r0 * DECIM, w * DECIM, hh * DECIM))
            frac[r0:r0 + hh] = np.isfinite(blk).reshape(hh, DECIM, w, DECIM).mean(axis=(1, 3))
        transform = src.transform * src.transform.scale(DECIM, DECIM)
        crs = src.crs
    mask = (frac >= MIN_FRAC).astype("uint8")
    merged = unary_union([shape(g) for g, v in shapes(mask, mask=mask == 1, transform=transform) if v == 1])

    # Close pinholes (single dropped tiles) but keep real voids, like the 14 km^2 one on Haleakala.
    parts = []
    for p in getattr(merged, "geoms", [merged]):
        if p.area < MIN_AREA:
            continue
        holes = [list(r.coords) for r in p.interiors if abs(Polygon(r.coords).area) >= FILL_HOLE]
        parts.append(Polygon(list(p.exterior.coords), holes))
    parts = [p.simplify(SIMPLIFY, preserve_topology=True) for p in parts]
    if clip_land is not None:
        # 100 m cells put whole coves inside the outline; the coastline makes it stop where the ground stops.
        gj = json.loads(Path(clip_land).read_text(encoding="utf-8"))
        land = unary_union([shape(transform_geom("EPSG:4326", crs, f["geometry"]))
                            for f in gj["features"] if f.get("geometry")]).buffer(0)
        clipped = []
        for p in parts:
            c = p.intersection(land)
            clipped += [g for g in getattr(c, "geoms", [c]) if g.geom_type == "Polygon"]
        parts = [p.simplify(CLIP_SIMPLIFY, preserve_topology=True)
                 for p in clipped if p.area >= CLIP_MIN_AREA]
    parts.sort(key=lambda p: -p.area)
    return parts, crs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mosaic", required=True, type=Path, help="2 m lidar mosaic from mapping.lidar_mosaic")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clip", type=Path, help="land polygons (lon/lat GeoJSON) to clip the outline to")
    a = ap.parse_args(argv)
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping

    parts, crs = coverage_polygons(a.mosaic, a.clip)
    print(f"{len(parts)} polygon(s), largest {parts[0].area / 1e6:,.0f} km2, "
          f"{sum(len(p.exterior.coords) for p in parts):,} exterior vertices", flush=True)
    feats = [{"type": "Feature", "properties": {"km2": round(p.area / 1e6, 1)},
              "geometry": transform_geom(crs, "EPSG:4326", mapping(p), precision=5)} for p in parts]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"type": "FeatureCollection", "name": "lidar_coverage",
                                 "note": "NOAA 2022 topobathy lidar, from valid pixels; outside this, "
                                         "elevation is modelled",
                                 "features": feats}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {a.out}  {a.out.stat().st_size / 1e3:.1f} kB", flush=True)


if __name__ == "__main__":
    main()
