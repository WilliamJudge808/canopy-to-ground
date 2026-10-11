"""Label every cache chip with its region, for hold-out experiments.

Chip ids do not say which island a chip is on, so islands are recovered from
geometry: the occupied 512 m cells of both splits form 8-connected
components, the inter-island channels are far wider than a chip, so each
component is an island, named after the nearest reference centroid. West and
East Maui come apart at the isthmus wherever no chip bridges it. Molokai is
cut at 157.00 W into the dry leeward west and the wet windward Kamakou massif
(the closest analogue in the state to the East Maui rainforest), with a
``molokai_buffer`` band of 0.02 degrees (~2 km) on the dry side that belongs to
neither, so a ``molokai_wet`` hold-out never touches its training data.

Writes ``<cache>/island_labels.json``: ``{split: {str(row): region}}``.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .. import grid
from .cache import SPLITS, chip_ids

CENTROIDS = {                       # lon, lat
    "molokai": (-157.02, 21.13),
    "lanai": (-156.92, 20.83),
    "west_maui": (-156.62, 20.90),
    "east_maui": (-156.25, 20.75),
    "kahoolawe": (-156.61, 20.55),
}
WET_MOLOKAI_LON = -157.00
BUFFER_DEG = 0.02


def components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """8-connected components of a set of grid cells."""
    left, out = set(cells), []
    while left:
        seed = min(left)
        left.discard(seed)
        comp, queue = [seed], deque([seed])
        while queue:
            ix, iy = queue.popleft()
            for nb in ((ix + dx, iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                if nb in left:
                    left.discard(nb)
                    comp.append(nb)
                    queue.append(nb)
        out.append(comp)
    return out


def label_cells(cells: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", grid.CRS, always_xy=True)
    to_lonlat = Transformer.from_crs(grid.CRS, "EPSG:4326", always_xy=True)
    refs = {k: to_utm.transform(*v) for k, v in CENTROIDS.items()}
    labels = {}
    for comp in components(cells):
        cx = float(np.mean([c[0] for c in comp])) * grid.CHIP_M
        cy = float(np.mean([c[1] for c in comp])) * grid.CHIP_M
        name = min(refs, key=lambda k: (refs[k][0] - cx) ** 2 + (refs[k][1] - cy) ** 2)
        for ix, iy in comp:
            if name == "molokai":
                lon, _ = to_lonlat.transform((ix + 0.5) * grid.CHIP_M, (iy + 0.5) * grid.CHIP_M)
                labels[ix, iy] = ("molokai_wet" if lon > WET_MOLOKAI_LON else
                                  "molokai_buffer" if lon > WET_MOLOKAI_LON - BUFFER_DEG else "molokai_dry")
            else:
                labels[ix, iy] = name
    return labels


def build_regions(cache: Path) -> dict[str, dict[str, str]]:
    ids = {split: chip_ids(Path(cache) / split) for split in SPLITS}
    cells = {s: [grid.parse_chip_id(c) for c in v] for s, v in ids.items()}
    labels = label_cells({c for v in cells.values() for c in v})
    out = {s: {str(row): labels[c] for row, c in enumerate(v)} for s, v in cells.items()}
    (Path(cache) / "island_labels.json").write_text(json.dumps(out))
    for s, v in out.items():
        counts = {r: list(v.values()).count(r) for r in sorted(set(v.values()))}
        print(f"{s}: {counts}", flush=True)
    return out
