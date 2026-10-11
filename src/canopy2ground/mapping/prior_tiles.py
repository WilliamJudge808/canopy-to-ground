"""Tile the Copernicus GLO-30 prior for the viewer's before/after toggle.

    python -m canopy2ground.mapping.prior_tiles --prior data/priors/copernicus --work tiles --publish site

This is the raw prior with no vertical datum shift, exactly the surface the
model was fed; a toggle that silently corrected it would flatter the model.
30 m is ~32 m in Web Mercator at this latitude, so z12 (19.1 m/px) already
oversamples it and the viewer overzooms past that. ``prior_3857.tif`` is also
the gap filler of ``mapping.surface``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import MAUI_BBOX
from .terrain_rgb import build, publish

RES_3857 = 15.0                      # m, finer than z12's 19.1 m/px
MIN_Z, MAX_Z = 8, 12


def prior_3857(prior_dir: Path, out: Path, bbox=MAUI_BBOX) -> Path:
    """Merge the Copernicus tiles over ``bbox``, warp to EPSG:3857 at RES_3857, sea and voids to 0 m."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    lon0, lat0, lon1, lat1 = bbox
    srcs = []
    for p in sorted(prior_dir.glob("*.tif")):
        with rasterio.open(p) as s:
            b = s.bounds
            if b.right > lon0 and b.left < lon1 and b.top > lat0 and b.bottom < lat1:
                srcs.append(p)
    if not srcs:
        raise SystemExit(f"no Copernicus tiles overlap {bbox} in {prior_dir}")
    opened = [rasterio.open(p) for p in srcs]
    try:
        arr, transform = merge(opened, bounds=bbox)
        nodata, crs = opened[0].nodata, opened[0].crs
    finally:
        for s in opened:
            s.close()

    src = arr[0].astype("float32")
    if nodata is not None:
        src[src == nodata] = np.nan
    src[src < -500] = np.nan
    dst_transform, w, h = calculate_default_transform(crs, "EPSG:3857", src.shape[1], src.shape[0], left=lon0,
                                                      bottom=lat0, right=lon1, top=lat1, resolution=RES_3857)
    dst = np.full((h, w), np.nan, dtype="float32")
    reproject(source=src, destination=dst, src_transform=transform, src_crs=crs, dst_transform=dst_transform,
              dst_crs="EPSG:3857", src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    # Sea level, not nodata: NaN has no terrain-RGB encoding.
    dst[~np.isfinite(dst)] = 0.0
    np.clip(dst, 0.0, None, out=dst)

    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", dtype="float32", count=1, width=w, height=h, crs="EPSG:3857",
                       transform=dst_transform, tiled=True, compress="deflate", BIGTIFF="IF_SAFER") as d:
        d.write(dst, 1)
    print(f"wrote {out}  {w} x {h}, {src.shape} merged from {[p.name for p in srcs]}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prior", required=True, type=Path, help="directory of Copernicus GLO-30 tiles")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--publish", type=Path, help="site directory; receives maui_prior_z12.pmtiles.png")
    ap.add_argument("--format", default="png", choices=["png", "webp"])
    ap.add_argument("--skip-tiles", action="store_true", help="only write prior_3857.tif")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = prior_3857(a.prior, a.work / "prior_3857.tif")
    if not a.skip_tiles:
        pm = build(out, f"maui_prior_z{MAX_Z}", MIN_Z, MAX_Z, a.work, fmt=a.format)
        if a.publish is not None:
            publish(pm, a.publish, pm.stem)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
