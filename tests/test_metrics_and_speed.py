import numpy as np
import torch

from canopy2ground.data import ChipDataset
from canopy2ground.metrics import decoy_partners, evaluate, nmad


class Zero(torch.nn.Module):
    def forward(self, b):
        return torch.zeros_like(b["dem"])


class Oracle(torch.nn.Module):
    def forward(self, b):
        return b["target_res"]


def test_evaluate_prior_and_oracle(data_cfg):
    ds = ChipDataset(data_cfg, "val")
    prior = evaluate(Zero(), ds, device="cpu")
    oracle = evaluate(Oracle(), ds, device="cpu")
    assert prior["n_chips"] == 5                                 # chip 0 is dropped as near-empty
    assert 1.4 < prior["mae_median_m"] < 1.8                     # N(0, 2) noise: E|x| = 1.6
    assert oracle["mae_median_m"] < 1e-5 and oracle["ridge_r"] > 0.999


def test_prior_decoy_hurts_a_prior_only_model(data_cfg):
    ds = ChipDataset(data_cfg, "val")
    clean = evaluate(Zero(), ds, device="cpu")
    swapped = evaluate(Zero(), ds, decoy="scramble_dem", device="cpu")   # each chip gets another chip's prior
    assert swapped["mae_median_m"] > clean["mae_median_m"] + 1.0


def test_decoy_partners_have_no_fixed_points_and_are_far():
    for n in (2, 3, 7, 320, 1107):
        p = decoy_partners(n)
        assert all(j != i for i, j in enumerate(p))
        assert min(abs(j - i) for i, j in enumerate(p)) >= (n - 1) // 2


def test_nmad_of_gaussian():
    x = np.random.default_rng(0).normal(0, 2.0, 200_000)
    assert abs(nmad(x) - 2.0) < 0.02


def test_fused_attention_matches_reference():
    import segmentation_models_pytorch as smp

    from canopy2ground.models.speed import use_fused_mit_attention

    enc = smp.encoders.get_encoder("mit_b0", in_channels=3, depth=5, weights=None).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        ref = enc(x)
        use_fused_mit_attention()
        fused = enc(x)
    for a, b in zip(ref, fused):
        assert torch.allclose(a, b, atol=1e-5)
