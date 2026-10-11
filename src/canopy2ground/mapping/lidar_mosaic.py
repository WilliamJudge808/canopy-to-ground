"""Mosaic the NOAA 2022 lidar DEM over Maui island into one 2 m raster.

    python -m canopy2ground.mapping.lidar_mosaic --vrt data/lidar.vrt --out tiles/maui_lidar_2m.vrt

The tiles top out at z14/z15 (~2-4 m/px), so the 0.5 m source is far finer
than the viewer can show. Every source COG carries 2x and 4x overviews, and
the 4x level is exactly 2 m with no resampling, because each tile spans a
whole multiple of 4 source pixels (checked). Reading it through /vsicurl moves
~1/16 of the bytes.

* GDAL must not list the S3 prefix on every open (``GDAL_DISABLE_READDIR_ON_OPEN``, set by
  ``sources.raster.gdal_env``): that alone is ~15x slower per tile.
* The canvas is a raw memmap wrapped by a VRTRawRasterBand, not a GeoTIFF. Sources arrive in
  arbitrary order, and random windowed writes into a tiled GeoTIFF make GDAL re-read evicted blocks.

``--vrt`` is the local NOAA mosaic VRT written by ``python -m canopy2ground.build vrt``
(``sources.lidar.fetch_vrt``); it is fetched if missing.
"""
from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..grid import CRS
from ..sources import lidar
from ..sources.raster import gdal_env
from . import MAUI_BBOX

GRID_X0, GRID_Y0, GRID_RES = 579000.0, 2401500.0, 0.5   # the NOAA VRT's own grid
DECIM = 4                                               # 0.5 m -> 2 m, the 4x overview
OUT_RES = GRID_RES * DECIM

SRC_RE = re.compile(r"<SourceFilename[^>]*>([^<]+)</SourceFilename>.*?"
                    r'<DstRect xOff="(\d+)" yOff="(\d+)" xSize="(\d+)" ySize="(\d+)"', re.S)

RAW_VRT = """<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS>EPSG:6634</SRS>
  <GeoTransform>{x0}, {res}, 0.0, {y0}, 0.0, -{res}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{bin}</SourceFilename>
    <ImageOffset>0</ImageOffset>
    <PixelOffset>4</PixelOffset>
    <LineOffset>{line}</LineOffset>
    <ByteOrder>LSB</ByteOrder>
    <NoDataValue>nan</NoDataValue>
  </VRTRasterBand>
</VRTDataset>
"""


def sources_in_box(vrt_text: str, bbox=MAUI_BBOX) -> list[tuple[str, int, int, int, int]]:
    """(source, x, y, w, h) in 2 m output pixels for every VRT source intersecting the lon/lat ``bbox``.

    Maui island plus Kahoolawe by default. Molokai, Lanai and Oahu are in the same
    survey; they are left out only to keep the tile archive under GitHub's file cap.
    """
    from rasterio.warp import transform as warp_transform

    lon0, lat0, lon1, lat1 = bbox
    xs, ys = warp_transform("EPSG:4326", CRS, [lon0, lon1, lon1, lon0], [lat0, lat0, lat1, lat1])
    e0, e1, n0, n1 = min(xs), max(xs), min(ys), max(ys)
    keep = []
    for fn, x, y, w, h in SRC_RE.findall(vrt_text):
        x, y, w, h = int(x), int(y), int(w), int(h)
        if x % DECIM or y % DECIM or w % DECIM or h % DECIM:
            raise SystemExit(f"source {fn} is not on the {DECIM}x grid")
        se0, sn1 = GRID_X0 + x * GRID_RES, GRID_Y0 - y * GRID_RES
        if se0 + w * GRID_RES > e0 and se0 < e1 and sn1 > n0 and sn1 - h * GRID_RES < n1:
            keep.append((fn, x // DECIM, y // DECIM, w // DECIM, h // DECIM))
    return keep


def mosaic(srcs: list, out: Path, workers: int = 24) -> int:
    """Read each source's 4x overview into a raw memmap canvas; returns the number of failed sources."""
    import rasterio

    cx0, cy0 = min(s[1] for s in srcs), min(s[2] for s in srcs)
    width = max(s[1] + s[3] for s in srcs) - cx0
    height = max(s[2] + s[4] for s in srcs) - cy0
    out.parent.mkdir(parents=True, exist_ok=True)
    binpath = out.with_suffix(".bin")
    print(f"{len(srcs)} lidar tiles -> {width} x {height} @ {OUT_RES} m ({width * height * 4 / 1e9:.2f} GB)",
          flush=True)

    t0 = time.time()
    canvas = np.memmap(binpath, dtype="float32", mode="w+", shape=(height, width))
    for row in range(0, height, 4096):           # a fresh memmap is zeros, and 0 m is a legal elevation
        canvas[row:row + 4096, :] = np.nan
    lock = threading.Lock()
    done, failed = [0], [0]

    def fetch(item):
        fn, x, y, w, h = item
        try:
            with rasterio.open(fn) as src:
                arr = src.read(1, out_shape=(h, w)).astype("float32")
                nod = src.nodata
            if nod is not None:
                arr[arr == nod] = np.nan
            arr[arr < -500] = np.nan
            if np.isfinite(arr).any():
                canvas[y - cy0:y - cy0 + h, x - cx0:x - cx0 + w] = arr    # disjoint slices: no lock needed
            with lock:
                done[0] += 1
                n = done[0]
            if n % 500 == 0:
                print(f"  {n}/{len(srcs)} tiles  {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception as exc:                 # one bad tile must not stop the mosaic
            with lock:
                failed[0] += 1
                if failed[0] <= 10:
                    print(f"  FAIL {Path(fn).name}: {exc}", flush=True)

    with gdal_env(), ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, srcs))
    canvas.flush()
    out.write_text(RAW_VRT.format(w=width, h=height, x0=repr(GRID_X0 + cx0 * OUT_RES),
                                  y0=repr(GRID_Y0 - cy0 * OUT_RES), res=repr(OUT_RES), bin=binpath.name,
                                  line=width * 4), encoding="utf-8")
    print(f"wrote {out}  {done[0]} ok / {failed[0]} failed  {(time.time() - t0) / 60:.1f} min", flush=True)
    return failed[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vrt", required=True, type=Path, help="local NOAA mosaic VRT (fetched if missing)")
    ap.add_argument("--out", required=True, type=Path, help="output .vrt (with a .bin beside it)")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    if not a.vrt.exists():
        lidar.fetch_vrt(a.vrt)
    srcs = sources_in_box(a.vrt.read_text(encoding="utf-8", errors="replace"))
    if not srcs:
        raise SystemExit("no lidar sources in the target box")
    return int(mosaic(srcs, a.out, a.workers) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
