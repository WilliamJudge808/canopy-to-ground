"""Command line for the cache build; see the package docstring for the steps in order."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..sources import LAYERS, lidar
from . import __doc__ as steps_doc
from .arrays import build_land, build_layers, build_prior
from .cache import SPLITS, assemble
from .chips import fetch_corpus
from .regions import build_regions


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m canopy2ground.build", description=steps_doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)

    p = sub.add_parser("vrt", help="local copy of the NOAA lidar VRT")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("chips", help="fetch the per-chip corpus (rgb + lidar)")
    p.add_argument("--lidar-vrt", required=True, help="VRT written by the vrt step")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tile-cache", type=Path, help="directory for the SQLite tile cache (default: none)")
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("cache", help="assemble the corpus into the memory-mapped cache")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    def cache_step(name: str, text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=text)
        p.add_argument("--cache", type=Path, required=True)
        p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
        return p

    p = cache_step("prior", "Copernicus prior, lanczos-warped to each chip")
    p.add_argument("--dem", type=Path, required=True, help="GeoTIFF or directory of Copernicus tiles")

    p = cache_step("layers", "extra imagery layers")
    p.add_argument("--out", type=Path, required=True, help="layer root (may equal --cache)")
    p.add_argument("--layers", nargs="+", default=list(LAYERS), choices=LAYERS)
    p.add_argument("--tile-cache", type=Path)
    p.add_argument("--workers", type=int, default=8)

    p = cache_step("land", "coastline land mask")
    p.add_argument("--coastline", type=Path, required=True)

    p = sub.add_parser("regions", help="island / region labels")
    p.add_argument("--cache", type=Path, required=True)

    a = ap.parse_args(argv)
    if a.step == "vrt":
        print(f"wrote {lidar.fetch_vrt(a.out)}")
    elif a.step == "chips":
        fetch_corpus(a.corpus, a.lidar_vrt, a.tile_cache, a.workers)
    elif a.step == "cache":
        assemble(a.corpus, a.out)
    elif a.step == "prior":
        build_prior(a.cache, a.dem, a.splits)
    elif a.step == "layers":
        build_layers(a.cache, a.out, a.layers, a.splits, a.workers, a.tile_cache)
    elif a.step == "land":
        build_land(a.cache, a.coastline, a.splits)
    elif a.step == "regions":
        build_regions(a.cache)


if __name__ == "__main__":
    main()
