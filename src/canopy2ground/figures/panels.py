"""Extract the arrays the publication figures are drawn from.

    python -m canopy2ground.figures.panels crop --store maps/store_east_maui \\
        --map maps/east_maui_dtm_coast.tif --prior data/priors/copernicus --out figs/hanawi_gorge.npz
    python -m canopy2ground.figures.panels overview --surface tiles/maui_surface_3857.tif \\
        --model tiles/east_maui_dtm_coast_3857.tif --footprint site/lidar_coverage.geojson \\
        --out figs/overview.npz

``crop``: NAIP photograph, Copernicus prior and the map on one 1 m grid
around a place with no lidar. The photo panel is NAIP (USDA, public domain);
the County's Pictometry imagery is not redistributed. The map's water (NaN)
is applied to both surfaces, so the two coasts match.

``overview``: a ~15 m multi-directional hillshade of the published surface
over the model's footprint, with lidar-coverage and sea masks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..grid import CHIP_M as CHIP
from ..grid import CRS
from ..prior import PriorRaster

FIGURE_PLACE = ("hanawi_gorge", -156.112, 20.818, 1800)   # name, lon, lat, side (m): Figs. 1 and 3
OVERVIEW_DECIM = 7                                        # 2.14 m surface -> ~15 m/px
OVERVIEW_LAT = 20.8                                       # Mercator scale correction for the hillshade


def place_bbox(lon: float, lat: float, size_m: float) -> tuple[float, float, float, float]:
    dlat = size_m / 2 / 111_000
    dlon = dlat / math.cos(math.radians(lat))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def crop_store(store: Path, bbox, prior: str | Path, layer: str = "naip"):
    """(image (H, W, C) uint8, prior (H, W) float32, transform) for a lon/lat ``bbox`` of a prefetch store."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    g = json.loads((store / "grid.json").read_text())
    to = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    xa, ya = to(bbox[0], bbox[1])
    xb, yb = to(bbox[2], bbox[3])
    stride = g["stride"]
    ix0 = max(0, math.floor((xa - g["x0"] - CHIP) / stride) + 1)
    ix1 = min(g["nx"] - 1, math.floor((xb - g["x0"]) / stride))
    iy0 = max(0, math.floor((g["y1"] - yb - CHIP) / stride) + 1)
    iy1 = min(g["ny"] - 1, math.floor((g["y1"] - ya) / stride))
    ox, oy = g["x0"] + ix0 * stride, g["y1"] - iy0 * stride
    img = None
    for iy in range(iy0, iy1 + 1):
        with np.load(store / f"row_{iy:04d}.npz") as z:
            arr, ok = z[layer], z[f"{layer}_ok"]
        if img is None:
            shape = ((iy1 - iy0) * stride + CHIP, (ix1 - ix0) * stride + CHIP, *arr.shape[3:])
            img = np.zeros(shape, np.uint8)
        for ix in range(ix0, ix1 + 1):
            if ok[ix]:
                y, x = (iy - iy0) * stride, (ix - ix0) * stride
                img[y:y + CHIP, x:x + CHIP] = arr[ix]
    c0, r0 = int(round(xa - ox)), int(round(oy - yb))
    w, h = int(round(xb - xa)), int(round(yb - ya))
    tr = Affine(1.0, 0, ox + c0, 0, -1.0, oy - r0)
    dem = np.full((h, w), np.nan, np.float32)
    pr = PriorRaster(prior, CRS)
    try:
        reproject(rasterio.band(pr.src, 1), dem, dst_transform=tr, dst_crs=CRS,
                  resampling=Resampling.lanczos, src_nodata=pr.src.nodata, dst_nodata=np.nan)
    finally:
        pr.close()
    return img[r0:r0 + h, c0:c0 + w], dem, tr


def crop_panel(store: Path, map_path: Path, prior: str | Path, out: Path, place=FIGURE_PLACE) -> dict:
    """Write ``out`` (npz: naip, sea, z_model, z_prior, transform) for ``place``; returns its summary."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    _, lon, lat, size = place
    naip, prior_z, tr = crop_store(store, place_bbox(lon, lat, size), prior)
    model = np.full(prior_z.shape, np.nan, np.float32)
    with rasterio.open(map_path) as ds:
        reproject(rasterio.band(ds, 1), model, dst_transform=tr, dst_crs=ds.crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    sea = ~np.isfinite(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, naip=naip[..., :3], sea=sea, z_model=np.where(sea, 0, model).astype(np.float32),
                        z_prior=np.where(sea, 0.0, prior_z).astype(np.float32),
                        transform=np.array(tuple(tr)[:6]))
    return {"lon": lon, "lat": lat, "size_m": size, "shape": list(model.shape),
            "elev_range_m": [float(np.nanmin(model)), float(np.nanmax(model))], "sea_frac": float(sea.mean()),
            "naip_empty_frac": float((naip[..., :3].sum(-1) == 0).mean())}


def hillshade(z: np.ndarray, az: float, alt: float = 40.0, spacing: float = 1.0) -> np.ndarray:
    gy, gx = np.gradient(z, spacing)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    return np.clip(np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def multi_hillshade(z: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Soft three-sun shade; reads better at phone size than a single sun."""
    return np.clip(0.55 * hillshade(z, 315, spacing=spacing) + 0.25 * hillshade(z, 270, spacing=spacing)
                   + 0.20 * hillshade(z, 360, spacing=spacing), 0, 1)


def overview(surface: Path, model: Path, footprint: Path, out: Path, decim: int = OVERVIEW_DECIM) -> None:
    """Write ``out`` (npz: shade uint8, lidar bool, sea bool, bounds of the model's 3857 grid)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.warp import transform_geom
    from rasterio.windows import from_bounds
    from rasterio.windows import transform as window_transform

    with rasterio.open(model) as m:
        bounds = m.bounds
    with rasterio.open(surface) as s:
        win = from_bounds(*bounds, s.transform).round_offsets().round_lengths()
        h, w = int(win.height // decim), int(win.width // decim)
        z = s.read(1, window=win, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
        tr = window_transform(win, s.transform) * Affine.scale(win.width / w, win.height / h)
        crs = s.crs
    shade = multi_hillshade(z, spacing=tr.a * np.cos(np.radians(OVERVIEW_LAT)))
    gj = json.loads(Path(footprint).read_text(encoding="utf-8"))
    lidar = rasterize([(transform_geom("EPSG:4326", crs, f["geometry"]), 1) for f in gj["features"]],
                      out_shape=(h, w), transform=tr, fill=0, dtype="uint8")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, shade=(shade * 255).astype(np.uint8), lidar=lidar.astype(bool), sea=z <= 0,
                        bounds=np.array(tuple(bounds)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crop", help="photo / prior / map crop around the figure's place")
    c.add_argument("--store", required=True, type=Path, help="imagery store from canopy2ground.prefetch")
    c.add_argument("--map", required=True, type=Path, help="coast-masked map (EPSG:6634)")
    c.add_argument("--prior", required=True)
    c.add_argument("--out", required=True, type=Path)
    o = sub.add_parser("overview", help="regional hillshade with lidar and sea masks")
    o.add_argument("--surface", required=True, type=Path, help="maui_surface_3857.tif from mapping.surface")
    o.add_argument("--model", required=True, type=Path, help="the coast-masked map on the 3857 grid")
    o.add_argument("--footprint", required=True, type=Path,
                   help="lidar_coverage.geojson from mapping.footprint")
    o.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "crop":
        print(json.dumps(crop_panel(a.store, a.map, a.prior, a.out)))
    else:
        overview(a.surface, a.model, a.footprint, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
