import numpy as np
import pytest
import torch

from canopy2ground.data import BatchOptions, ChipDataset, DataConfig, extra_channels_for, make_batch


def test_extra_channels():
    assert extra_channels_for(("vivid", "naip")) == 3 + 4 + 2
    assert extra_channels_for(()) == 0


def test_batch_keys_shapes_and_normalisation(data_cfg):
    ds = ChipDataset(data_cfg, "train")
    b = make_batch(ds, [1, 2, 3], BatchOptions(side=256), device="cpu")
    assert b["rgb"].shape == (3, 3, 256, 256)
    assert b["extra_img"].shape == (3, 7, 256, 256)
    assert b["extra_avail"].shape == (3, 2, 256, 256)
    for k in ("dem", "lidar", "mask", "dem_z", "slope_mag", "hillshade", "land", "sea", "in_mask", "dem_abs"):
        assert b[k].shape == (3, 256, 256), k
    assert b["chip_ids"] == [1, 2, 3]
    assert torch.allclose(b["land"] + b["sea"], b["in_mask"])
    assert torch.allclose(b["target_res"], (b["lidar"] - b["dem"]) * b["mask"])


def test_missing_layer_is_exactly_zero(data_cfg):
    ds = ChipDataset(data_cfg, "train")
    b = make_batch(ds, [1, 2], device="cpu")
    assert (b["extra_img"][1] == 0).all() and (b["extra_avail"][1] == 0).all()   # chip 2: no coverage
    assert b["extra_img"][0].abs().sum() > 0 and (b["extra_avail"][0] == 1).all()


def test_near_empty_chips_are_dropped(data_cfg):
    ds = ChipDataset(data_cfg, "train")
    b = make_batch(ds, [0, 1], device="cpu")          # chip 0 has ~10% valid lidar at full size
    assert b["chip_ids"] == [1]
    assert make_batch(ds, [0], BatchOptions(side=128), device="cpu") is None  # centre crop is all void


def test_random_crops_are_reproducible(data_cfg):
    ds = ChipDataset(data_cfg, "train")
    opts = BatchOptions(side=128, augment=True)
    a = make_batch(ds, [1, 3], opts, rng=np.random.default_rng(7), device="cpu")
    b = make_batch(ds, [1, 3], opts, rng=np.random.default_rng(7), device="cpu")
    for k in ("rgb", "dem", "lidar", "extra_img"):
        assert torch.equal(a[k], b[k])


def test_decoys_swap_one_modality(data_cfg):
    ds = ChipDataset(data_cfg, "train")
    clean = make_batch(ds, [1], device="cpu")
    partner = make_batch(ds, [4], device="cpu")
    img = make_batch(ds, [1], BatchOptions(decoy="scramble_rgb", decoy_src=[4]), device="cpu")
    assert torch.equal(img["rgb"], partner["rgb"]) and torch.equal(img["extra_img"], partner["extra_img"])
    assert torch.equal(img["dem"], clean["dem"]) and torch.equal(img["lidar"], clean["lidar"])
    dem = make_batch(ds, [1], BatchOptions(decoy="scramble_dem", decoy_src=[4]), device="cpu")
    assert torch.equal(dem["dem"], partner["dem"]) and torch.equal(dem["rgb"], clean["rgb"])
    assert torch.equal(dem["lidar"], clean["lidar"])


def test_region_filter(tiny_cache):
    cfg = DataConfig(root=str(tiny_cache), layers=(), regions=("molokai_wet",))
    assert ChipDataset(cfg, "train").rows == [4, 5]
    with pytest.raises(ValueError):
        ChipDataset(DataConfig(root=str(tiny_cache), layers=(), regions=("oahu",)), "train")
