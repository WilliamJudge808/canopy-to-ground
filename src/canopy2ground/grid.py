"""The 512 m chip grid and the train/val split.

Chip ``(ix, iy)`` covers ``[512 ix, 512 (ix + 1)) x [512 iy, 512 (iy + 1))`` in
EPSG:6634 (NAD83(PA11) / UTM 4N, the CRS of the NOAA lidar), so a chip id is
computable from any coordinate with no registry, and two runs always agree on
chip boundaries. Ids look like ``maui-+01544-+04489``; the prefix names Maui
County, not the island the chip sits on (see ``build.regions``).

The split is spatial: 20 x 20 chip (10.24 km) blocks are assigned to val with
probability 0.2 by an MD5 hash of the block index, so it is stable across
machines and Python hash seeds. Training additionally drops every train chip
within one chip of a val block.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

CRS = "EPSG:6634"     # working CRS of the chips, the prior warp and the output map
CHIP_M = 512          # chip side in metres, and in pixels: every raster is gridded at 1 m
PREFIX = "maui"
CORPUS_LONLAT = (-157.35, 20.45, -155.95, 21.25)   # W, S, E, N: Maui, Molokai, Lanai, Kahoolawe
BLOCK_CHIPS = 20
VAL_PERCENT = 20
BUFFER_CHIPS = 1


class Bounds(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def chip_index(easting: float, northing: float) -> tuple[int, int]:
    """Projected coordinate -> (ix, iy) of the chip containing it."""
    return int(easting // CHIP_M), int(northing // CHIP_M)


def chip_id(ix: int, iy: int) -> str:
    return f"{PREFIX}-{ix:+06d}-{iy:+06d}"


def parse_chip_id(cid: str) -> tuple[int, int]:
    prefix, ix, iy = cid.rsplit("-", 2)
    if prefix != PREFIX:
        raise ValueError(f"malformed chip id {cid!r}")
    return int(ix), int(iy)


def chip_bounds(cid: str) -> Bounds:
    ix, iy = parse_chip_id(cid)
    return Bounds(float(ix * CHIP_M), float(iy * CHIP_M), float((ix + 1) * CHIP_M), float((iy + 1) * CHIP_M))


def chips_in_lonlat(west: float, south: float, east: float, north: float) -> list[tuple[int, int]]:
    """Every (ix, iy) between the chips holding the projected SW and NE corners, inclusive."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x0, y0 = tf.transform(west, south)
    x1, y1 = tf.transform(east, north)
    ix0, iy0 = chip_index(min(x0, x1), min(y0, y1))
    ix1, iy1 = chip_index(max(x0, x1), max(y0, y1))
    return [(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)]


def block_split(bx: int, by: int) -> str:
    h = int(hashlib.md5(f"{PREFIX}_{bx}_{by}".encode()).hexdigest(), 16)
    return "val" if h % 100 < VAL_PERCENT else "train"


def chip_split(ix: int, iy: int) -> str:
    return block_split(ix // BLOCK_CHIPS, iy // BLOCK_CHIPS)


def in_buffer(ix: int, iy: int, ring: int = BUFFER_CHIPS) -> bool:
    """True for a train chip within ``ring`` chips (8-neighbourhood) of a val block."""
    if chip_split(ix, iy) != "train":
        return False
    return any(chip_split(ix + dx, iy + dy) == "val"
               for dx in range(-ring, ring + 1) for dy in range(-ring, ring + 1) if dx or dy)
