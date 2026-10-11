"""Assemble the per-chip corpus into the memory-mapped training cache.

Row order is fixed by the chip set alone: chip ids sorted, then shuffled with
``SEED``, so a rebuild from the same corpus reproduces every row index (and
with it every paired comparison between models). Train chips within
``grid.BUFFER_CHIPS`` of a val block are dropped after the shuffle.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import grid

CHIP = grid.CHIP_M
SEED = 1234
SPLITS = ("train", "val")


class NpyWriter:
    """Stream ``shape[0]`` rows into a ``.npy`` file; it appears under its final name only once complete."""

    def __init__(self, path: Path, shape: tuple[int, ...], dtype):
        self.path, self.shape, self.dtype = Path(path), tuple(shape), np.dtype(dtype)
        self.tmp = self.path.with_name(f".{self.path.name}.part")
        self.n = 0
        self.f = open(self.tmp, "wb")
        np.lib.format.write_array_header_1_0(self.f, {"descr": np.lib.format.dtype_to_descr(self.dtype),
                                                      "fortran_order": False, "shape": self.shape})

    def append(self, row: np.ndarray) -> None:
        row = np.ascontiguousarray(row, dtype=self.dtype)
        if row.shape != self.shape[1:]:
            raise ValueError(f"{self.path.name}: row shape {row.shape}, expected {self.shape[1:]}")
        self.f.write(row.tobytes())
        self.n += 1

    def __enter__(self) -> NpyWriter:
        return self

    def __exit__(self, exc_type, *_):
        self.f.close()
        if exc_type is None and self.n == self.shape[0]:
            os.replace(self.tmp, self.path)
            return
        self.tmp.unlink(missing_ok=True)
        if exc_type is None:
            raise ValueError(f"{self.path.name}: wrote {self.n} of {self.shape[0]} rows")


def chip_ids(split_dir: Path) -> list[str]:
    """Chip id of every cache row, in row order."""
    meta = json.loads((Path(split_dir) / "meta.json").read_text(encoding="utf-8-sig"))
    chips = meta["chips"][:meta["n"]]
    if [c["i"] for c in chips] != list(range(meta["n"])):
        raise ValueError(f"{split_dir}/meta.json: chips are not in row order")
    # caches built before ids were stored record the corpus file path instead
    return [c["id"] if "id" in c else Path(c["path"].replace("\\", "/")).stem for c in chips]


def row_order(ids: list[str], split: str, seed: int = SEED) -> tuple[list[str], int]:
    """Cache row order of a split's corpus chips, and how many train chips the buffer removed."""
    order = sorted(ids)
    np.random.default_rng(seed).shuffle(order)
    if split != "train":
        return order, 0
    kept = [c for c in order if not grid.in_buffer(*grid.parse_chip_id(c))]
    return kept, len(order) - len(kept)


def assemble(corpus: Path, out: Path) -> None:
    """``corpus/<split>/<chip>.npz`` -> ``out/<split>/{rgb,lidar,lmask}.npy`` and ``meta.json``."""
    for split in SPLITS:
        ids, n_buffer = row_order([p.stem for p in (Path(corpus) / split).glob("*.npz")], split)
        d = Path(out) / split
        d.mkdir(parents=True, exist_ok=True)
        n, chips = len(ids), []
        with (NpyWriter(d / "rgb.npy", (n, CHIP, CHIP, 3), np.uint8) as rgb,
              NpyWriter(d / "lidar.npy", (n, CHIP, CHIP), np.float32) as lidar,
              NpyWriter(d / "lmask.npy", (n, CHIP, CHIP), np.bool_) as lmask):
            for k, cid in enumerate(ids):
                with np.load(Path(corpus) / split / f"{cid}.npz") as z:
                    img, elev, m = z["rgb"], z["lidar"].astype(np.float32), z["lmask"].astype(bool)
                rgb.append(img[..., :3])
                lidar.append(np.where(m, elev, np.float32(0)))
                lmask.append(m)
                v = elev[m]
                chips.append({"i": k, "id": cid, "relief": float(v.max() - v.min()), "std": float(v.std()),
                              "lidar_cov": float(m.mean())})
                if (k + 1) % 1000 == 0:
                    print(f"  {split}: {k + 1}/{n}", flush=True)
        (d / "meta.json").write_text(json.dumps({"n": n, "shape": [CHIP, CHIP], "seed": SEED,
                                                 "buffer_chips": grid.BUFFER_CHIPS,
                                                 "n_buffer_excluded": n_buffer, "chips": chips}))
        print(f"{split}: {n} chips -> {d} ({n_buffer} train chips in the val buffer dropped)", flush=True)
