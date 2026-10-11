"""Apply the coastline to a predicted DTM: the model on land, nothing on water.

    python -m canopy2ground.mapping.coast_mask --pred maps/east_maui_dtm.tif \\
        --county data/coast/mauico_detailedcoast_eastmaui.geojson \\
        --usace data/coast/usace2013_lidar_landpoly_eastmaui.geojson \\
        --prior data/priors/copernicus --out maps/east_maui_dtm_coast.tif

Land is the County of Maui 2017 Detailed Coastline united with the land
(above mean high water) of the 2013 USACE topobathy lidar, NOAA OCM dataset
9337. The lidar adds rocky points and stream mouths the County line cuts off,
but it also returns from whitewater, so lidar-only polygons under
``--min-blob`` m^2 that do not touch County land are dropped. Both vector
files must already be clipped to ``--clip-bbox`` (this tool does not clip
them); outside that box, less a 200 m margin, the prediction is left
untouched, since a clipped coastline says nothing about land there.

Per output pixel:

* water: ``--water`` (NaN, transparent in the viewer; or 0 m);
* land with a prediction: the prediction;
* land without one (the predictor skipped tiles where the prior is mostly
  void): the Copernicus prior, Lanczos-resampled as in training.

A companion ``<out>_source.tif`` (uint8) records which: 0 water, 1 model,
2 Copernicus fill. ``<out>_summary.json`` holds the pixel counts.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..prior import PriorRaster
from . import EAST_MAUI_BBOX

SOURCE_WATER, SOURCE_MODEL, SOURCE_PRIOR = 0, 1, 2
COVERAGE_MARGIN_M = 200.0


def read_polygons(path: str | Path, crs: str, keep: Callable[[dict], bool] = lambda props: True) -> list:
    """Polygons of a lon/lat GeoJSON, projected to ``crs``, multipolygons split into parts."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    out = []
    for feat in json.loads(Path(path).read_text(encoding="utf-8"))["features"]:
        if not feat.get("geometry") or not keep(feat.get("properties") or {}):
            continue
        g = transform(to_crs, shape(feat["geometry"]))
        out += list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    return out


def land_polygons(county: list, lidar: list, min_blob: float) -> tuple[list, int, int]:
    """County polygons plus every lidar polygon that is at least ``min_blob`` m^2 or touches County land.

    Returns (polygons, lidar polygons kept, lidar specks dropped).
    """
    from shapely.strtree import STRtree

    tree = STRtree(county)
    out, kept, dropped = list(county), 0, 0
    for g in lidar:
        if g.area >= min_blob:
            out.append(g)
            kept += 1
            continue
        if any(county[i].intersects(g) for i in tree.query(g)):
            out.append(g)
            kept += 1
        else:
            dropped += 1
    return out, kept, dropped


def mask_block(pred: np.ndarray, land: np.ndarray, covered: np.ndarray, water: float):
    """(output, source map, pixels that need the prior fill) for one block of rows."""
    is_land = land | ~covered                       # outside the vector coverage: untouched
    need = is_land & covered & ~np.isfinite(pred)
    out = np.where(is_land, pred, water).astype(np.float32)
    source = np.where(is_land & np.isfinite(pred), SOURCE_MODEL, SOURCE_WATER).astype(np.uint8)
    return out, source, need


def mask_window(src, win, land: list, cover, prior, water: float) -> tuple[np.ndarray, np.ndarray]:
    """(output, source map) for window ``win`` of the open prediction ``src``.

    ``land`` and ``cover`` (the vector coverage) are polygons in the
    prediction's CRS, rasterised for this window only; ``prior`` is the open
    Copernicus dataset that fills land the predictor skipped.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import transform as window_transform

    shape, tr = (win.height, win.width), window_transform(win, src.transform)
    is_land = rasterize([(g, 1) for g in land], out_shape=shape, transform=tr, fill=0, dtype="uint8")
    covered = rasterize([(cover, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8")
    out, source, need = mask_block(src.read(1, window=win).astype(np.float32), is_land.astype(bool),
                                   covered.astype(bool), water)
    if need.any():
        fill = np.full(shape, np.nan, np.float32)
        reproject(rasterio.band(prior, 1), fill, dst_transform=tr, dst_crs=src.crs,
                  resampling=Resampling.lanczos, src_nodata=prior.nodata, dst_nodata=np.nan)
        out[need] = fill[need]
        source[need] = SOURCE_PRIOR
    return out, source


def apply_coast_mask(pred_path: Path, out_path: Path, county: Path, usace: Path, prior: str | Path,
                     water: float = np.nan, min_blob: float = 50.0, clip_bbox=EAST_MAUI_BBOX,
                     block: int = 2048) -> dict:
    """Write ``out_path`` and its ``_source.tif``; returns the pixel-count summary (also written as JSON)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    t0 = time.time()
    counts = np.zeros(3, np.int64)
    source_path = out_path.with_name(f"{out_path.stem}_source.tif")
    with rasterio.open(pred_path) as src:
        crs = src.crs.to_string()
        land, kept, dropped = land_polygons(
            read_polygons(county, crs),
            read_polygons(usace, crs, lambda p: "land" in p.get("kind", "")), min_blob)
        print(f"county polys {len(land) - kept}, lidar land polys kept {kept}, dropped specks {dropped}",
              flush=True)
        to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        cover = shapely_transform(to_crs, box(*clip_bbox)).buffer(-COVERAGE_MARGIN_M)
        profile = dict(src.profile, dtype="float32", nodata=np.nan, compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        prior_raster = PriorRaster(prior, crs)
        try:
            with (rasterio.open(out_path, "w", **profile) as dst,
                  rasterio.open(source_path, "w", **dict(profile, dtype="uint8", nodata=255)) as sdst):
                for r0 in range(0, src.height, block):
                    win = Window(0, r0, src.width, min(block, src.height - r0))
                    out, source = mask_window(src, win, land, cover, prior_raster.src, water)
                    dst.write(out, 1, window=win)
                    sdst.write(source, 1, window=win)
                    counts += np.bincount(source.ravel(), minlength=3)[:3]
                    print(f"  rows {r0 + win.height}/{src.height}  ({time.time() - t0:.0f}s)", flush=True)
        finally:
            prior_raster.close()

    summary = {"water_or_empty_px": int(counts[0]), "model_px": int(counts[1]),
               "copernicus_fill_px": int(counts[2]),
               "copernicus_fill_frac_of_land": round(float(counts[2] / max(1, counts[1] + counts[2])), 5),
               "water": "nan" if np.isnan(water) else f"{water:g}", "min_blob_m2": min_blob,
               "pred": str(pred_path)}
    out_path.with_name(f"{out_path.stem}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, type=Path, help="predicted DTM from canopy2ground.predict")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--county", required=True, type=Path,
                    help="County of Maui 2017 Detailed Coastline (GeoJSON)")
    ap.add_argument("--usace", required=True, type=Path,
                    help="2013 USACE lidar land polygons (GeoJSON, features with kind 'land_above_mhw')")
    ap.add_argument("--prior", required=True, help="Copernicus GeoTIFF or directory of tiles")
    ap.add_argument("--water", default="nan", choices=["nan", "0"])
    ap.add_argument("--min-blob", type=float, default=50.0,
                    help="m^2; smaller lidar-only polygons not touching County land are dropped")
    ap.add_argument("--clip-bbox", type=float, nargs=4, default=EAST_MAUI_BBOX, metavar=("W", "S", "E", "N"),
                    help="lon/lat box the vector files were clipped to")
    a = ap.parse_args(argv)
    summary = apply_coast_mask(a.pred, a.out, a.county, a.usace, a.prior, water=float(a.water),
                               min_blob=a.min_blob, clip_bbox=tuple(a.clip_bbox))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
