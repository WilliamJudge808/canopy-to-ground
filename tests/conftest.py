import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from canopy2ground.data import DataConfig


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tiny_cache(tmp_path_factory):
    """Read-only 6-chip cache with the real layout (512 px chips, two imagery layers, coastline)."""
    tmp_path = tmp_path_factory.mktemp("tiny")
    rng = np.random.default_rng(0)
    n, s = 6, 512
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    root = tmp_path / "cache"
    labels = {}
    for split in ("train", "val"):
        d = root / split
        d.mkdir(parents=True)
        terrain = np.stack([100 + 0.2 * xx + 5 * np.sin(yy / (20 + 5 * i)) for i in range(n)])
        terrain = terrain.astype(np.float32)
        np.save(d / "lidar.npy", terrain)
        np.save(d / "dem_copernicus.npy", terrain + rng.normal(0, 2, terrain.shape).astype(np.float32))
        np.save(d / "rgb.npy", rng.integers(0, 255, (n, s, s, 3), dtype=np.uint8))
        lmask = np.ones((n, s, s), bool)
        lmask[0, :, :400] = False          # chip 0 is mostly void
        lmask[0, :, 450:] = False
        np.save(d / "lmask.npy", lmask)
        np.save(d / "dmask_copernicus.npy", np.ones((n, s, s), bool))
        land = np.ones((n, s, s), np.uint8)
        land[:, :, :64] = 0
        np.save(d / "land.npy", land)
        chips = [{"i": i, "id": f"maui-{i}"} for i in range(n)]
        (d / "meta.json").write_text(json.dumps({"n": n, "chips": chips}))
        for name, c in (("vivid", 3), ("naip", 4)):
            np.save(d / f"img_{name}.npy", rng.integers(0, 255, (n, s, s, c), dtype=np.uint8))
            avail = np.ones(n, np.float32)
            avail[2] = 0.0                  # chip 2 has no coverage for either layer
            np.save(d / f"avail_{name}.npy", avail)
        labels[split] = {str(i): ("molokai_wet" if i >= 4 else "maui_west") for i in range(n)}
    (root / "island_labels.json").write_text(json.dumps(labels))
    return root


@pytest.fixture
def writable_cache(tiny_cache, tmp_path):
    """A private copy of ``tiny_cache`` for tests that add or overwrite arrays."""
    return Path(shutil.copytree(tiny_cache, tmp_path / "cache"))


@pytest.fixture
def data_cfg(tiny_cache):
    return DataConfig(root=str(tiny_cache), layers=("vivid", "naip"), coast=True)
