"""Block bootstrap of an ``external`` comparison: intervals that respect the spatial split.

    python -m canopy2ground.evaluation.blocks --result results/external_val.json --cache data/cache \\
        --split val --out results/block_bootstrap_val.json

The chip bootstrap ``external score`` reports treats every chip as
independent, but the chips come from a handful of 20 x 20 chip split blocks
and neighbours within a block share terrain and canopy, so it is
anti-conservative. This reads the per-chip 1 m MAE stored in an ``external``
result, maps cache rows to blocks through the split's ``meta.json``, and
resamples whole blocks for each product minus the model (positive: the model
is better).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import grid
from ..build.cache import chip_ids
from .stats import block_bootstrap


def chip_blocks(ids: list[str], rows: list[int]) -> np.ndarray:
    """(n, 2) split-block index ``(bx, by)`` of each cache row in ``rows``."""
    return np.array([[v // grid.BLOCK_CHIPS for v in grid.parse_chip_id(ids[r])] for r in rows])


def products_vs_model(per_chip: dict[str, dict], ids: list[str], n_boot: int = 4000, seed: int = 0) -> dict:
    """Block bootstrap of ``product - model`` per-chip values for every product in ``per_chip``.

    ``per_chip`` is ``{surface: {row: value}}`` with a ``model`` entry, as in
    ``per_chip_mae1`` of an ``external`` result (row keys may be strings).
    """
    model = {int(r): v for r, v in per_chip["model"].items()}
    rows = list(model)
    diffs = {}
    for s, vals in per_chip.items():
        if s != "model":
            by_row = {int(r): v for r, v in vals.items()}
            diffs[s] = np.array([by_row[r] - model[r] for r in rows])
    return block_bootstrap(diffs, chip_blocks(ids, rows), n_boot, seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result", required=True, type=Path, help="JSON written by `external score`")
    ap.add_argument("--cache", required=True, type=Path, help="cache root the result was scored on")
    ap.add_argument("--split", help="cache split (default: the result's)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    res = json.loads(a.result.read_text(encoding="utf-8"))
    if "model" not in res["per_chip_mae1"]:
        raise SystemExit(f"{a.result} scores no model; nothing to pair")
    out = products_vs_model(res["per_chip_mae1"], chip_ids(a.cache / (a.split or res["split"])))
    for s, r in out.items():
        print(f"  {s:10s} - model: {r['mean']:+.3f} m [{r['lo']:+.3f}, {r['hi']:+.3f}], "
              f"model better in {r['blocks_better']}/{r['n_blocks']} blocks")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
