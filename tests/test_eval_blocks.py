import json
import os
from pathlib import Path

import numpy as np
import pytest

from canopy2ground import grid
from canopy2ground.evaluation import blocks
from canopy2ground.evaluation.stats import block_bootstrap

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _cache(root: Path, ids: list[str], split: str = "val") -> Path:
    d = root / split
    d.mkdir(parents=True)
    chips = [{"i": i, "id": c} for i, c in enumerate(ids)]
    (d / "meta.json").write_text(json.dumps({"n": len(ids), "chips": chips}))
    return root


def test_chip_blocks_follow_the_split_grid():
    ids = [grid.chip_id(1439, 4579), grid.chip_id(1440, 4579), grid.chip_id(1440, 4580)]
    np.testing.assert_array_equal(blocks.chip_blocks(ids, [2, 0, 1]), [[72, 229], [71, 228], [72, 228]])


def test_cli_pairs_every_product_against_the_model(tmp_path):
    rng = np.random.default_rng(0)
    ids = [grid.chip_id(ix, iy) for ix in range(0, 80, 4) for iy in (0, 25)]     # 4 x 2 blocks
    rows = list(range(0, len(ids), 2)) + [1, 3]          # a subset, out of order: blocks (0..3, 0) and (0, 1)
    model = rng.uniform(0.3, 1.0, len(rows))
    per = {"model": dict(zip(map(str, rows), model)),
           "copernicus": dict(zip(map(str, rows), model + rng.normal(0.8, 0.3, len(rows)))),
           "fabdem": dict(zip(map(str, rows), model + rng.normal(0.5, 0.3, len(rows))))}
    (tmp_path / "ext.json").write_text(json.dumps({"split": "val", "per_chip_mae1": per}))
    cache = _cache(tmp_path / "cache", ids)
    out = tmp_path / "blocks.json"
    blocks.main(["--result", str(tmp_path / "ext.json"), "--cache", str(cache), "--out", str(out)])
    got = json.loads(out.read_text())

    want = block_bootstrap({s: np.array([per[s][str(r)] - per["model"][str(r)] for r in rows])
                            for s in ("copernicus", "fabdem")}, blocks.chip_blocks(ids, rows))
    assert got == want and list(got) == ["copernicus", "fabdem"]
    assert got["copernicus"]["n_blocks"] == 5 and got["copernicus"]["lo"] > 0


@pytest.mark.data
@pytest.mark.parametrize("result, split, frozen", [("external_val.json", "val", "block_bootstrap_val.json"),
                                                  ("external_molokai.json", "train",
                                                   "block_bootstrap_molokai.json")])
def test_reproduces_the_published_block_intervals(result, split, frozen):
    if not os.environ.get("C2G_CACHE"):
        pytest.skip("set C2G_CACHE to the training cache root")
    from canopy2ground.build.cache import chip_ids

    res = json.loads((RESULTS / result).read_text(encoding="utf-8"))
    got = blocks.products_vs_model(res["per_chip_mae1"], chip_ids(Path(os.environ["C2G_CACHE"]) / split))
    assert got == json.loads((RESULTS / frozen).read_text(encoding="utf-8"))
