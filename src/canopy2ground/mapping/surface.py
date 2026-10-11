"""Merge surveyed lidar and the predicted DTM into one surface, and tile it.

    python -m canopy2ground.mapping.surface --model maps/east_maui_dtm_coast_3857.tif \\
        --lidar tiles/maui_lidar_2m.vrt --prior tiles/prior_3857.tif --work tiles --publish site

MapLibre's 3-D terrain reads exactly one raster-dem source, so lidar for the
rest of Maui and the prediction for East Maui have to be one raster before
they are one archive. Lidar wins wherever it exists.

The merged grid is exactly 2x the model's 3857 grid (2.1447 m), so the model
is downsampled by an integer 2x2 average with no resampling error, and it
still oversamples z14 (4.47 m/px) by 2x. The archive is z8-z14 everywhere
plus a z15 level (2.24 m/px, almost exactly native) over the model's
footprint only: z15 costs ~58 MB, and dropping z14 outside the footprint buys
~25 MB back, which keeps the archive under GitHub's 100 MiB file cap. The
lidar tops out a level earlier so the reconstruction can go a level deeper;
MapLibre falls back to the parent tile wherever a deeper one is absent.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from .terrain_rgb import GITHUB_FILE_LIMIT, build, fix_metadata, publish, rgbify, to_pmtiles

MIN_Z, MAX_Z, DETAIL_Z = 8, 14, 15
PRUNE_OUTSIDE = (14,)            # zooms kept only inside the model footprint
SEA = 0.5                        # m; everything below is drawn as flat water at 0
COARSE = 16                      # decimation of the gap-finding pass
GAP_MIN_PX = 20                  # coarse px, ~0.4 km^2
GAP_MIN_ELEV = 20.0              # m; keeps coastal ponds from counting as gaps
BLOCK = 1024                     # output rows per pass; caps the merge near 700 MB

FOOTPRINT_VRT = """<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS>EPSG:3857</SRS>
  <GeoTransform>{x0}, {res}, 0.0, {y0}, 0.0, -{res}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="{sx}" yOff="{sy}" xSize="{w}" ySize="{h}"/>
      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""


def find_gaps(lidar_vrt, model, prior_vrt, m_col: int, m_row: int, width: int, height: int, res: float):
    """Coarse mask of enclosed ground neither source covers, and the offset that levels the prior into it.

    The NOAA 2022 survey has a 14 km^2 hole on Haleakala's south flank, outside
    the model's footprint. Left at 0 m it would be a sea-level crater in a
    3,000 m mountain. Only enclosed voids are filled (open ocean must stay 0),
    and Copernicus is shifted by the median difference around each void's rim
    so the join does not show in the hillshade.
    """
    from rasterio.warp import Resampling
    from rasterio.windows import Window
    from scipy import ndimage

    hc, wc = height // COARSE, width // COARSE
    lid = lidar_vrt.read(1, out_shape=(hc, wc))
    pri = prior_vrt.read(1, out_shape=(hc, wc))
    mod = np.full((hc, wc), np.nan, dtype="float32")
    # Only whole coarse cells inside the model, so the read window never leaves the raster.
    c0 = max(0, -(-m_col // COARSE))
    c1 = min(wc, (m_col + model.width // 2) // COARSE)
    r0 = max(0, -(-m_row // COARSE))
    r1 = min(hc, (m_row + model.height // 2) // COARSE)
    if r1 > r0 and c1 > c0:
        w = Window((c0 * COARSE - m_col) * 2, (r0 * COARSE - m_row) * 2,
                   (c1 - c0) * COARSE * 2, (r1 - r0) * COARSE * 2)
        mod[r0:r1, c0:c1] = model.read(1, window=w, out_shape=(r1 - r0, c1 - c0),
                                       resampling=Resampling.average)

    elev = np.where(np.isfinite(lid), lid, np.where(np.isfinite(mod), mod, 0.0)).astype("float32")
    void = elev <= SEA
    lab, _ = ndimage.label(void)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    enclosed = void & ~np.isin(lab, edge[edge != 0])

    mask = np.zeros((hc, wc), dtype=bool)
    off = np.zeros((hc, wc), dtype="float32")
    hl, hn = ndimage.label(enclosed)
    filled = []
    for k in range(1, hn + 1):
        sel = hl == k
        if sel.sum() < GAP_MIN_PX or pri[sel].mean() < GAP_MIN_ELEV:
            continue
        ring = ndimage.binary_dilation(sel, iterations=3) & ~sel & ~void
        if ring.sum() < 20:
            continue
        d = float(np.median(elev[ring] - pri[ring]))
        grow = ndimage.binary_dilation(sel, iterations=2)          # covers the coarse cells' edges
        mask |= grow
        off[grow] = d
        filled.append((int(sel.sum()) * (COARSE * res) ** 2 / 1e6, d))
    for area, d in sorted(filled, reverse=True)[:4]:
        print(f"  gap fill {area:6.2f} km2, prior levelled {d:+.2f} m", flush=True)
    print(f"  {len(filled)} gap(s) filled from Copernicus", flush=True)
    return mask, off


def union_grid(model, lidar_bounds, res: float) -> tuple:
    """(transform, width, height, model col, model row) of the ``res`` grid over the model and the lidar.

    The union is snapped to the model's grid in whole output pixels, so every
    model window is an exact 2 x 2 block.
    """
    from rasterio.transform import from_origin

    mb, lb = model.bounds, lidar_bounds
    x0 = mb.left - math.ceil((mb.left - min(lb[0], mb.left)) / res) * res
    y0 = mb.top + math.ceil((max(lb[3], mb.top) - mb.top) / res) * res
    width = int(math.ceil((max(lb[2], mb.right) - x0) / res))
    height = int(math.ceil((y0 - min(lb[1], mb.bottom)) / res))
    m_col, m_row = round((mb.left - x0) / res), round((y0 - mb.top) / res)
    return from_origin(x0, y0, res, res), width, height, m_col, m_row


def merge_rows(dst, model, lidar_vrt, prior_vrt, gaps, m_col: int, m_row: int) -> dict[str, float]:
    """Write the merged surface into ``dst`` BLOCK rows at a time; returns pixel counts and seam stats."""
    from rasterio.warp import Resampling
    from rasterio.windows import Window

    t0 = time.time()
    gap_mask, gap_off = gaps
    width, height = dst.width, dst.height
    n = dict(model=0, lidar=0, both=0, gap=0, dz_sum=0.0, dz_abs=0.0)
    for row in range(0, height, BLOCK):
        h = min(BLOCK, height - row)
        buf = np.zeros((h, width), dtype="float32")

        # model, integer 2x2 average onto the output grid
        mr0, mc0 = row - m_row, -m_col
        r0, c0 = max(mr0, 0), max(mc0, 0)
        r1, c1 = min(mr0 + h, model.height // 2), min(mc0 + width, model.width // 2)
        model_here = np.zeros((h, width), dtype=bool)
        if r1 > r0 and c1 > c0:
            m = model.read(1, window=Window(c0 * 2, r0 * 2, (c1 - c0) * 2, (r1 - r0) * 2),
                           out_shape=(r1 - r0, c1 - c0), resampling=Resampling.average)
            good = np.isfinite(m)
            sl = (slice(r0 - mr0, r1 - mr0), slice(c0 - mc0, c1 - mc0))
            buf[sl] = np.where(good, m, 0.0)
            model_here[sl] = good
            n["model"] += int(good.sum())

        # lidar on top, where it was surveyed
        li = lidar_vrt.read(1, window=Window(0, row, width, h))
        lgood = np.isfinite(li)
        if lgood.any():
            overlap = lgood & model_here
            if overlap.any():
                d = li[overlap] - buf[overlap]
                n["both"] += int(overlap.sum())
                n["dz_sum"] += float(d.sum())
                n["dz_abs"] += float(np.abs(d).sum())
            buf[lgood] = li[lgood]
            n["lidar"] += int(lgood.sum())

        # A threshold, not a floor: the model predicts a 0.1-0.3 m "seafloor" offshore, which
        # hillshades into a dotted rectangle in the ocean. Flattens beach berms under 0.5 m.
        buf[buf < SEA] = 0.0

        # Enclosed voids: levelled Copernicus, nearest-upsampled from the coarse pass and re-gated
        # on the full-res void. The last COARSE-1 rows/columns clamp onto the last coarse cell.
        gi = np.minimum(np.arange(row, row + h) // COARSE, gap_mask.shape[0] - 1)
        gj = np.minimum(np.arange(width) // COARSE, gap_mask.shape[1] - 1)
        gsel = gap_mask[np.ix_(gi, gj)] & (buf < SEA)
        if gsel.any():
            pb = prior_vrt.read(1, window=Window(0, row, width, h))
            buf[gsel] = np.maximum(pb[gsel] + gap_off[np.ix_(gi, gj)][gsel], 0.0)
            n["gap"] += int(gsel.sum())
        dst.write(buf, 1, window=Window(0, row, width, h))
        if (row // BLOCK) % 4 == 0:
            print(f"  rows {row + h}/{height}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return n


def merge(model_path: Path, lidar_path: Path, prior_path: Path, out: Path) -> None:
    """Lidar over the 2x2-averaged model on one grid, enclosed voids filled from the levelled prior."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling, transform_bounds

    t0 = time.time()
    with (rasterio.open(model_path) as model, rasterio.open(lidar_path) as lidar,
          rasterio.open(prior_path) as prior):
        res = model.transform.a * 2
        transform, width, height, m_col, m_row = union_grid(
            model, transform_bounds(lidar.crs, model.crs, *lidar.bounds), res)
        print(f"output {width} x {height} @ {res:.4f} m ({width * height * 4 / 1e9:.2f} GB), "
              f"model at +({m_col},{m_row})", flush=True)
        grid = dict(crs=model.crs, transform=transform, width=width, height=height)
        profile = dict(driver="GTiff", dtype="float32", count=1, nodata=None, tiled=True, blockxsize=512,
                       blockysize=512, compress="deflate", zlevel=1, predictor=3, BIGTIFF="YES",
                       num_threads="ALL_CPUS", **grid)
        with (WarpedVRT(lidar, resampling=Resampling.bilinear, src_nodata=np.nan, nodata=np.nan,
                        **grid) as lidar_vrt,
              WarpedVRT(prior, resampling=Resampling.bilinear, **grid) as prior_vrt):
            gaps = find_gaps(lidar_vrt, model, prior_vrt, m_col, m_row, width, height, res)
            with rasterio.open(out, "w", **profile) as dst:
                n = merge_rows(dst, model, lidar_vrt, prior_vrt, gaps, m_col, m_row)

    tot = width * height
    print(f"wrote {out}  {out.stat().st_size / 1e9:.2f} GB ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"  lidar {n['lidar'] / tot:6.1%} of grid, model {n['model'] / tot:6.1%}, "
          f"overlap {n['both'] / tot:6.1%}, gap-filled {n['gap'] / tot:6.2%} "
          f"({n['gap'] * res * res / 1e6:.1f} km2)", flush=True)
    if n["both"]:
        # A large bias here would show as a cliff along the seam.
        print(f"  model - lidar where both exist: bias {-n['dz_sum'] / n['both']:+.3f} m, "
              f"MAE {n['dz_abs'] / n['both']:.3f} m over {n['both']:,} px", flush=True)


def tms_range(w: float, s: float, e: float, n: float, z: int) -> tuple[int, int, int, int]:
    """Lon/lat box -> (col0, col1, tms_row0, tms_row1) at zoom z (mbtiles rows count from the south)."""
    k = 2 ** z

    def xy(lon, lat):
        row = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * k
        return int((lon + 180) / 360 * k), int(row)

    x0, y_n = xy(w, n)
    x1, y_s = xy(e, s)
    return x0, x1, k - 1 - y_s, k - 1 - y_n


def add_detail_zoom(merged: Path, model_path: Path, base_mbtiles: Path, work: Path, stem: str,
                    fmt: str = "webp", workers: int = 4) -> Path:
    """Add DETAIL_Z over the model's footprint to ``base_mbtiles``; prune PRUNE_OUTSIDE outside it."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(model_path) as m:
        bounds = m.bounds
        wgs = transform_bounds(m.crs, "EPSG:4326", *m.bounds)
    with rasterio.open(merged) as s:
        win = from_bounds(*bounds, s.transform).round_offsets().round_lengths()
        sx, sy = max(0, int(win.col_off)), max(0, int(win.row_off))
        w, h = min(int(win.width), s.width - sx), min(int(win.height), s.height - sy)
        x0, y0 = s.transform * (sx, sy)
        res = s.transform.a
    clip = work / "model_footprint_3857.vrt"
    clip.write_text(FOOTPRINT_VRT.format(w=w, h=h, x0=repr(x0), y0=repr(y0), res=repr(res),
                                         src=merged.resolve(), sx=sx, sy=sy), encoding="utf-8")
    print(f"model footprint {w} x {h} px @ {res:.4f} m", flush=True)

    detail = work / f"{stem}_only.mbtiles"
    rgbify(clip, detail, DETAIL_Z, DETAIL_Z, fmt=fmt, workers=workers)
    out = work / f"{stem}.mbtiles"
    shutil.copyfile(base_mbtiles, out)
    con = sqlite3.connect(out)
    con.execute("ATTACH ? AS d", (str(detail),))
    con.execute("INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) "
                "SELECT zoom_level, tile_column, tile_row, tile_data FROM d.tiles")
    for z in PRUNE_OUTSIDE:
        c0, c1, r0, r1 = tms_range(*wgs, z)
        gone = con.execute("DELETE FROM tiles WHERE zoom_level=? AND NOT "
                           "(tile_column BETWEEN ? AND ? AND tile_row BETWEEN ? AND ?)",
                           (z, c0, c1, r0, r1)).rowcount
        print(f"  pruned {gone} z{z} tiles outside the model footprint", flush=True)
    con.commit()
    con.execute("VACUUM")
    for z, n, b in con.execute("SELECT zoom_level, count(*), sum(length(tile_data)) FROM tiles "
                               "GROUP BY zoom_level ORDER BY zoom_level").fetchall():
        print(f"    z{z:2d} {n:5d} tiles {b / 1e6:7.1f} MB", flush=True)
    con.execute("DETACH d")
    con.close()
    fix_metadata(out, MIN_Z, DETAIL_Z, fmt, src=merged)
    pm = work / f"{stem}.pmtiles"
    to_pmtiles(out, pm, DETAIL_Z)
    return pm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, type=Path, help="coast-masked map on the viewer's 3857 grid")
    ap.add_argument("--lidar", required=True, type=Path, help="2 m lidar mosaic from mapping.lidar_mosaic")
    ap.add_argument("--prior", required=True, type=Path, help="prior_3857.tif from mapping.prior_tiles")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--publish", type=Path, help="site directory; receives maui_surface_z15.pmtiles.png")
    ap.add_argument("--format", default="webp", choices=["webp", "png"],
                    help="tile encoding; PNG puts the East Maui archive over GitHub's 100 MiB cap")
    ap.add_argument("--skip-merge", action="store_true", help="reuse work/maui_surface_3857.tif")
    a = ap.parse_args(argv)

    t0 = time.time()
    a.work.mkdir(parents=True, exist_ok=True)
    merged = a.work / "maui_surface_3857.tif"
    if not a.skip_merge:
        merge(a.model, a.lidar, a.prior, merged)
    build(merged, f"maui_surface_z{MAX_Z}", MIN_Z, MAX_Z, a.work, fmt=a.format)
    pm = add_detail_zoom(merged, a.model, a.work / f"maui_surface_z{MAX_Z}.mbtiles", a.work,
                         f"maui_surface_z{DETAIL_Z}", fmt=a.format)
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.publish is not None:
        publish(pm, a.publish, pm.stem)
    return int(pm.stat().st_size > GITHUB_FILE_LIMIT)


if __name__ == "__main__":
    raise SystemExit(main())
