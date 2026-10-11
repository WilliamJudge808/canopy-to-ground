"""EPSG:3857 elevation raster -> terrain-RGB PMTiles archive.

Encoding is Mapbox terrain-RGB, ``elevation = -10000 + (R * 65536 + G * 256 + B) * 0.1``, which is what
the viewer's ``encoding: "mapbox"`` expects. rio-rgbify truncates rather than rounds, so a decoded value
is at most 0.1 m below the true one and never above it. NaN is not representable (it decodes as
-10000 m), so every builder writes the sea as 0 m.

Three sharp edges are handled here:

* rio-rgbify writes no ``minzoom`` or ``bounds`` into the mbtiles metadata, and the PMTiles converter
  reads ``minzoom`` unconditionally.
* The mbtiles ``metadata`` table has no unique index on ``name``, so INSERT OR REPLACE appends a
  duplicate row instead of replacing; keys are deleted before they are written.
* GitHub Pages gzips ``application/octet-stream`` and then applies HTTP Range requests to the
  compressed stream, so every PMTiles offset lands on the wrong bytes. The published copy is named
  ``.pmtiles.png`` so Pages serves it as ``image/png``, untouched; PMTiles readers ignore the extension.

Archives should be built with WebP tiles: PNG puts the East Maui surface archive over GitHub's
100 MiB per-file cap.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE, INTERVAL = -10000.0, 0.1
GITHUB_FILE_LIMIT = 100 * 1024 * 1024


def encode(elev: np.ndarray) -> np.ndarray:
    """(H, W) metres -> (3, H, W) uint8 terrain-RGB, bit-identical to rio-rgbify's encoder."""
    d = (elev.astype(np.float64) - BASE) / INTERVAL
    rgb = np.zeros((3, *d.shape), np.uint8)
    rgb[2] = ((d / 256) - (d // 256)) * 256
    rgb[1] = (((d // 256) / 256) - ((d // 256) // 256)) * 256
    rgb[0] = ((((d // 256) // 256) / 256) - (((d // 256) // 256) // 256)) * 256
    return rgb


def decode(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) uint8 terrain-RGB -> (H, W) metres."""
    r, g, b = rgb.astype(np.float64)
    return BASE + (r * 65536 + g * 256 + b) * INTERVAL


def _rio() -> str:
    # The interpreter's own environment first, so an unactivated venv still finds its plugin.
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    exe = shutil.which("rio", path=path)
    if exe is None:
        raise SystemExit("`rio` not found: pip install rio-rgbify")
    return exe


def rgbify(src: Path, mbtiles: Path, min_z: int, max_z: int, fmt: str = "webp", workers: int = 4) -> None:
    mbtiles.unlink(missing_ok=True)
    cmd = [_rio(), "rgbify", "-b", f"{BASE:g}", "-i", f"{INTERVAL:g}",
           "--min-z", str(min_z), "--max-z", str(max_z), "--format", fmt,
           "-j", str(workers), str(src), str(mbtiles)]
    print("  " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"rio rgbify failed, exit code {r.returncode}")
    print(f"  mbtiles {mbtiles.stat().st_size / 1e6:.1f} MB ({(time.time() - t0) / 60:.1f} min)", flush=True)


def fix_metadata(mbtiles: Path, min_z: int, max_z: int, fmt: str, src: Path | None = None) -> None:
    """Write the metadata keys rio-rgbify omits; ``bounds`` and ``center`` come from ``src`` if given."""
    want = {"name": mbtiles.stem, "format": fmt, "type": "baselayer",
            "minzoom": str(min_z), "maxzoom": str(max_z)}
    if src is not None:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(src) as s:
            w, s_, e, n = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
        want.update(bounds=f"{w:.6f},{s_:.6f},{e:.6f},{n:.6f}",
                    center=f"{(w + e) / 2:.6f},{(s_ + n) / 2:.6f},{min_z}")
    con = sqlite3.connect(mbtiles)
    con.execute(f"DELETE FROM metadata WHERE name IN ({','.join('?' * len(want))})", tuple(want))
    con.executemany("INSERT INTO metadata (name, value) VALUES (?,?)", list(want.items()))
    con.commit()
    con.close()


def to_pmtiles(mbtiles: Path, out: Path, max_z: int) -> None:
    from pmtiles.convert import mbtiles_to_pmtiles

    out.unlink(missing_ok=True)
    t0 = time.time()
    mbtiles_to_pmtiles(str(mbtiles), str(out), str(max_z))
    print(f"  -> {out.name}  {out.stat().st_size / 1e6:.1f} MB ({(time.time() - t0) / 60:.1f} min)",
          flush=True)


def publish(pmtiles: Path, publish_dir: Path, name: str) -> Path:
    """Copy to ``publish_dir/<name>.pmtiles.png``; warns past GitHub's per-file cap."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    dest = publish_dir / f"{name}.pmtiles.png"
    shutil.copyfile(pmtiles, dest)
    n = dest.stat().st_size
    print(f"  published {dest}  {n / 2**20:.1f} MiB ({n / GITHUB_FILE_LIMIT:.0%} of GitHub's cap)",
          flush=True)
    if n > GITHUB_FILE_LIMIT:
        print("  WARNING: over GitHub's 100 MiB per-file limit; git will refuse the push", flush=True)
    return dest


def build(src: Path, stem: str, min_z: int, max_z: int, work: Path, fmt: str = "webp",
          workers: int = 4) -> Path:
    """``src`` -> ``work/<stem>.mbtiles`` -> ``work/<stem>.pmtiles``."""
    work.mkdir(parents=True, exist_ok=True)
    mb, pm = work / f"{stem}.mbtiles", work / f"{stem}.pmtiles"
    rgbify(src, mb, min_z, max_z, fmt=fmt, workers=workers)
    fix_metadata(mb, min_z, max_z, fmt, src=src)
    to_pmtiles(mb, pm, max_z)
    return pm
