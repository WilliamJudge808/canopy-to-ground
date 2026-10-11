import json

import numpy as np
import pytest
import torch

from canopy2ground.data import ChipDataset, DataConfig
from canopy2ground.evaluation import external
from canopy2ground.evaluation.ensemble import Ensemble
from canopy2ground.evaluation.external import block_mae, chip_metrics, score, summarise

IX0, IY = 1427, 4559                       # chip ids maui-+01427-+04559 ... on the 512 m grid


class Const(torch.nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, batch):
        return self.c * batch["in_mask"]


def test_block_mae_uses_half_valid_blocks_only():
    t = np.zeros((60, 60))
    p = t + 1.0
    p[:30, :30] = 5.0                                           # one block is far off ...
    v = np.ones((60, 60), bool)
    assert block_mae(p, t, v) == pytest.approx((5 + 1 + 1 + 1) / 4)
    v[:30, :30] = False
    v[:10, :30] = True                                          # ... and only a third valid: dropped
    assert block_mae(p, t, v) == pytest.approx(1.0)
    assert np.isnan(block_mae(p, t, np.zeros_like(v)))


def test_chip_metrics_slope_strata():
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    prior = np.tan(np.radians(15.0)) * xx                       # a uniform 15 degree slope
    lidar = prior + 0.3
    valid = np.ones_like(prior, bool)
    m = chip_metrics({"a": lidar + 2.0, "b": lidar - 1.0}, lidar, valid, prior)
    assert m["a"]["mae1"] == pytest.approx(2.0) and m["b"]["mae1"] == pytest.approx(1.0)
    assert m["a"]["mae30"] == pytest.approx(2.0)
    assert m["a"]["bin10_20"] == pytest.approx(2.0)
    assert np.isnan(m["a"]["bin0_5"]) and np.isnan(m["a"]["bin20_90"])


def _products(cache, split="val"):
    d = cache / split
    lidar = np.load(d / "lidar.npy")
    fab, gedtm = lidar + 1.0, lidar - 2.0
    fmask = np.ones(lidar.shape, bool)
    fmask[3, :, :256] = False                                   # a void: every surface loses these pixels
    np.save(d / "dem_fabdem.npy", fab.astype(np.float32))
    np.save(d / "dmask_fabdem.npy", fmask)
    np.save(d / "dem_gedtm30.npy", gedtm.astype(np.float32))
    np.save(d / "dmask_gedtm30.npy", np.ones(lidar.shape, bool))
    return {p: (np.load(d / f"dem_{p}.npy", mmap_mode="r"), np.load(d / f"dmask_{p}.npy", mmap_mode="r"))
            for p in external.PRODUCTS}


def test_score_products_only(writable_cache):
    products = _products(writable_cache)
    ds = ChipDataset(DataConfig(root=str(writable_cache), layers=()), "val")
    rows, per = score(ds, products, device="cpu", log_every=0)
    assert rows == [1, 2, 3, 4, 5]                              # chip 0 is too sparse to batch
    assert list(per) == ["copernicus", "fabdem", "gedtm30"]
    assert per["fabdem"]["mae1"] == pytest.approx([1.0] * 5, abs=1e-4)
    assert per["gedtm30"]["mae30"] == pytest.approx([2.0] * 5, abs=1e-4)
    res = summarise(rows, per)
    assert res["n_chips"] == 5 and res["paired_product_minus_model"] == {}
    assert res["surfaces"]["copernicus"]["mae1"]["median"] == pytest.approx(1.6, abs=0.05)   # |N(0, 2)|


def test_score_with_model_pairs_against_products(writable_cache):
    products = _products(writable_cache)
    ds = ChipDataset(DataConfig(root=str(writable_cache), layers=("vivid", "naip"), coast=True), "val")
    rows, per = score(ds, products, Ensemble([Const(0.5), Const(-0.5)]), device="cpu", log_every=0)
    res = summarise(rows, per)
    assert list(res["surfaces"]) == ["model", "copernicus", "fabdem", "gedtm30"]
    q = res["paired_product_minus_model"]
    assert q["copernicus"]["mae1"]["mean"] == pytest.approx(0.0, abs=1e-6)     # zero residual: the prior
    assert q["gedtm30"]["mae1"]["mean"] > 0 and q["gedtm30"]["mae1"]["frac_positive"] == 1.0
    assert set(res["per_chip_mae1"]["model"]) == set(rows)
    assert "block_bootstrap_mae1" not in res


def test_summarise_adds_the_block_bootstrap_given_chip_ids(writable_cache):
    _grid_cache(writable_cache)
    products = _products(writable_cache)
    ds = ChipDataset(DataConfig(root=str(writable_cache), layers=("vivid", "naip"), coast=True), "val")
    rows, per = score(ds, products, Ensemble([Const(0.25)]), device="cpu", log_every=0)
    res = summarise(rows, per, external.chip_ids(writable_cache / "val"))
    blocks = res["block_bootstrap_mae1"]
    assert list(blocks) == ["copernicus", "fabdem", "gedtm30"]
    for name, r in blocks.items():                     # all six chips sit in split block (71, 227)
        chip = res["paired_product_minus_model"][name]["mae1"]
        assert r["n_blocks"] == 1 and r["mean"] == pytest.approx(chip["mean"])
        assert r["lo"] == pytest.approx(r["mean"]) and r["hi"] == pytest.approx(r["mean"])


def _grid_cache(cache):
    """Give the tiny cache real chip ids, one row of chips along x, in the ``build cache`` meta format."""
    for split in ("train", "val"):
        meta = json.loads((cache / split / "meta.json").read_text())
        meta["chips"] = [{"i": i, "id": f"maui-+{IX0 + i:05d}-+{IY:05d}"} for i in range(meta["n"])]
        (cache / split / "meta.json").write_text(json.dumps(meta))


def _plane_source(path=None):
    """A 4 m plane over the six chips, as a file (``path``) or an in-memory dataset."""
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    x0, y1, res = IX0 * 512 - 100.0, (IY + 1) * 512 + 100.0, 4.0
    h, w = int((512 + 200) / res), int((6 * 512 + 200) / res)
    rows, cols = np.mgrid[0:h, 0:w]
    x, y = x0 + (cols + 0.5) * res, y1 - (rows + 0.5) * res
    z = (500.0 + 0.02 * x - 0.01 * y - 14000.0).astype(np.float32)
    tr = from_origin(x0, y1, res, res)
    if path is None:
        return external._float_dataset(z, tr, "EPSG:6634")
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs="EPSG:6634", transform=tr) as dst:
        dst.write(z, 1)
    return None


def _plane_on_chip(i):
    rows, cols = np.mgrid[0:512, 0:512]
    x = (IX0 + i) * 512 + cols + 0.5
    y = (IY + 1) * 512 - rows - 0.5
    return 500.0 + 0.02 * x - 0.01 * y - 14000.0


def test_build_warps_only_the_region_rows(writable_cache, tmp_path, monkeypatch):
    _grid_cache(writable_cache)
    monkeypatch.setattr(external, "open_product", lambda product, priors: _plane_source())
    external.build("fabdem", writable_cache, tmp_path, "train", regions=["molokai_wet"])
    dem = np.load(writable_cache / "train" / "dem_fabdem.npy")
    mask = np.load(writable_cache / "train" / "dmask_fabdem.npy")
    assert dem.shape == (6, 512, 512) and dem.dtype == np.float32
    assert mask[4:].all() and not mask[:4].any()                # rows 4-5 are the molokai_wet chips
    for i in (4, 5):
        assert np.abs(dem[i] - _plane_on_chip(i)).max() < 0.05
    assert not (writable_cache / "train" / ".dem_fabdem.building.npy").exists()


def test_verify_grid_reproduces_the_stored_prior(writable_cache, tmp_path):
    _grid_cache(writable_cache)
    (tmp_path / "copernicus").mkdir()
    _plane_source(tmp_path / "copernicus" / "tile.tif")
    from canopy2ground.prior import PriorRaster

    prior = PriorRaster(tmp_path / "copernicus", "EPSG:6634")
    stems = [f"maui-+{IX0 + i:05d}-+{IY:05d}" for i in range(6)]
    warped = [external.warp_chip(prior.src, s) for s in stems]
    prior.close()
    np.save(writable_cache / "val" / "dem_copernicus.npy", np.stack([np.where(g, d, 0) for d, g in warped]))
    np.save(writable_cache / "val" / "dmask_copernicus.npy", np.stack([g for _, g in warped]))
    assert external.verify_grid(writable_cache, tmp_path, k=3) == 0.0
