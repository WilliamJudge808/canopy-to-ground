import argparse

import pytest
import torch

from canopy2ground.data import ChipDataset, extra_channels_for, make_batch
from canopy2ground.models import ModelConfig, TerrainFusionNet, load_checkpoint, save_checkpoint


def _batch(data_cfg, side=128):
    from canopy2ground.data import BatchOptions
    return make_batch(ChipDataset(data_cfg, "train"), [1, 2], BatchOptions(side=side), device="cpu")


@pytest.mark.parametrize("decoder_v2", [False, True])
@pytest.mark.parametrize("coast", [False, True])
def test_forward_shape_and_gate(data_cfg, coast, decoder_v2):
    cfg = ModelConfig(extra_channels=extra_channels_for(data_cfg.layers), coast_inputs=coast,
                      decoder_v2=decoder_v2)
    model = TerrainFusionNet(cfg, encoder_weights=None).eval()
    b = _batch(data_cfg)
    gate = b["in_mask"] if coast else b["mask"]
    gate[:, :8] = 0
    with torch.no_grad():
        res = model(b)
    assert res.shape == b["dem"].shape
    assert torch.isfinite(res).all()
    assert (res[:, :8] == 0).all(), "residual must be zero outside the valid mask"


def test_checkpoint_roundtrip(tmp_path, data_cfg):
    cfg = ModelConfig(extra_channels=extra_channels_for(data_cfg.layers), coast_inputs=True, decoder_v2=True)
    model = TerrainFusionNet(cfg, encoder_weights=None).eval()
    save_checkpoint(str(tmp_path / "m.pt"), model, seed=3)
    loaded, meta = load_checkpoint(str(tmp_path / "m.pt"))
    assert meta["seed"] == 3 and loaded.cfg == cfg
    b = _batch(data_cfg)
    with torch.no_grad():
        assert torch.equal(model(b), loaded(b))


def test_loads_original_checkpoint_format(tmp_path, data_cfg):
    """Checkpoints of the published map: no model_config, plus two unused Sobel buffers."""
    cfg = ModelConfig(extra_channels=9)
    model = TerrainFusionNet(cfg, encoder_weights=None)
    state = dict(model.state_dict(), kx=torch.zeros(1, 1, 3, 3), ky=torch.zeros(1, 1, 3, 3))
    torch.save({"state_dict": state, "arch": "unified_geotrans", "extra_channels": 9,
                "config": {"layers": ("vivid", "naip"), "prior": "copernicus"}}, tmp_path / "old.pt")
    loaded, _ = load_checkpoint(str(tmp_path / "old.pt"))
    assert loaded.cfg == cfg


def test_checkpoint_with_arbitrary_objects_needs_trusted(tmp_path):
    model = TerrainFusionNet(ModelConfig(), encoder_weights=None)
    save_checkpoint(tmp_path / "m.pt", model, args=argparse.Namespace(lr=1e-3))
    with pytest.raises(ValueError, match="trusted=True"):
        load_checkpoint(tmp_path / "m.pt")
    _, meta = load_checkpoint(tmp_path / "m.pt", trusted=True)
    assert meta["args"].lr == 1e-3
