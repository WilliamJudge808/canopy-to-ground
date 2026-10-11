import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from canopy2ground.models import ModelConfig, TerrainFusionNet
from canopy2ground.train import (
    EpochSampler,
    OptimConfig,
    TrainConfig,
    apply_freeze,
    encoder_stages,
    trainable_stages,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_configs_parse():
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = TrainConfig.from_yaml(path)
        assert cfg.model.extra_channels == 9, path
        assert cfg.model.coast_inputs == cfg.data.coast, path


@pytest.mark.parametrize("extra", ["lr: 0.1", "optim: {learning_rate: 0.1}"])
def test_unknown_config_keys_are_rejected(tmp_path, extra):
    path = tmp_path / "bad.yaml"
    path.write_text(f"data: {{root: x}}\n{extra}\n")
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_yaml(path)


def test_unfreeze_schedule():
    o = OptimConfig(freeze_epochs=2, unfreeze_every=1)
    assert [trainable_stages(e, o, 6) for e in range(1, 10)] == [0, 0, 1, 2, 3, 4, 5, 6, 6]
    assert trainable_stages(1, dataclasses.replace(o, freeze_epochs=0), 6) == 6


def test_freeze_releases_deepest_stages_first():
    model = TerrainFusionNet(ModelConfig(), encoder_weights=None)
    mit, resnet = encoder_stages(model)
    assert len(mit) == 4 and len(resnet) == 6
    apply_freeze(model, 1)
    assert all(p.requires_grad for m in mit[-1] for p in m.parameters())
    assert not any(p.requires_grad for m in mit[0] for p in m.parameters())
    assert all(p.requires_grad for p in model.synthesis_head.parameters())   # the head is never frozen
    apply_freeze(model, 6)
    assert all(p.requires_grad for p in model.parameters())


def test_sampler_sees_every_chip_once_per_pass():
    s = EpochSampler(10, np.random.default_rng(0))
    first, second = s.draw(10), s.draw(10)
    assert sorted(first) == list(range(10)) and sorted(second) == list(range(10))
    assert sorted(s.draw(4) + s.draw(6)) == list(range(10))


def test_one_optimisation_step_reduces_loss_on_a_fixed_batch(data_cfg):
    """Smoke test of the training step on CPU: a few steps on one batch must lower the objective."""
    from canopy2ground.data import BatchOptions, ChipDataset, make_batch
    from canopy2ground.losses import LossWeights, total_loss

    torch.manual_seed(0)
    cfg = ModelConfig(extra_channels=9, coast_inputs=True, decoder_v2=True)
    model = TerrainFusionNet(cfg, encoder_weights=None)
    b = make_batch(ChipDataset(data_cfg, "train"), [1, 3], BatchOptions(side=128), device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        loss = total_loss(b["dem"] + model(b) * b["in_mask"], b, LossWeights(water=0.3))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)` before")   # no step was taken
def test_non_finite_losses_are_counted_and_stop_the_run(tmp_path, data_cfg, monkeypatch):
    """Every step non-finite: the epoch is logged with its skip count, then the run stops."""
    import canopy2ground.train as train_mod

    monkeypatch.setattr(train_mod, "TerrainFusionNet",
                        lambda cfg: TerrainFusionNet(cfg, encoder_weights=None))
    monkeypatch.setattr(train_mod, "total_loss", lambda pred, batch, w: pred.sum() * float("nan"))
    model = ModelConfig(rgb_encoder="mit_b1", extra_channels=9, coast_inputs=True)
    optim = OptimConfig(epochs=2, batch_size=2, scales=(4.0,), steps_per_epoch=3)
    cfg = TrainConfig(data=data_cfg, model=model, optim=optim, probe_chips=2)
    with pytest.raises(RuntimeError, match="3 of 3 steps"):
        train_mod.train(cfg, seed=0, out_dir=tmp_path / "run", device="cpu")
    (rec,) = [json.loads(line) for line in (tmp_path / "run" / "curves.jsonl").read_text().splitlines()]
    assert rec["epoch"] == 1 and rec["skipped_steps"] == 3
