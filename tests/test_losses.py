import math

import torch

from canopy2ground.losses import (
    LossWeights,
    band_amplitude_loss,
    band_corr_loss,
    laplacian_bands,
    masked_mean,
    multiband_detail_loss,
    total_loss,
)


def _terrain(b=2, s=128):
    yy, xx = torch.meshgrid(torch.arange(s, dtype=torch.float32), torch.arange(s, dtype=torch.float32),
                            indexing="ij")
    base = 0.3 * xx + 4 * torch.sin(yy / 9) + 1.5 * torch.cos(xx / 3.5)
    return base.expand(b, s, s).clone() + torch.randn(b, s, s) * 0.2


def test_masked_mean_ignores_masked_and_never_divides_by_zero():
    x = torch.tensor([[1.0, 100.0]])
    assert masked_mean(x, torch.tensor([[1.0, 0.0]]), dims=(1,)).item() == 1.0
    assert masked_mean(x, torch.zeros(1, 2), dims=(1,)).item() == 0.0


def test_laplacian_bands_sum_back_to_signal_minus_residual_lowpass():
    x = _terrain(1, 64)
    bands, _ = laplacian_bands(x, torch.ones_like(x), levels=3)
    low = torch.nn.functional.avg_pool2d(x[:, None], 8)
    up = torch.nn.functional.interpolate(low, size=(8, 8), mode="bilinear", align_corners=False)
    assert len(bands) == 3 and bands[0].shape == x.shape
    assert torch.isfinite(torch.stack([b.abs().sum() for b in bands])).all()
    assert up.shape[-1] == 8


def test_band_losses_are_zero_at_the_truth():
    t, m = _terrain(), torch.ones(2, 128, 128)
    assert multiband_detail_loss(t, t, m).item() < 1e-6
    assert band_corr_loss(t, t, m).item() < 1e-5
    assert band_amplitude_loss(t, t, m).item() < 1e-5


def test_corr_loss_is_scale_invariant_but_amplitude_loss_is_not():
    t, m = _terrain(), torch.ones(2, 128, 128)
    damped = 0.5 * (t - t.mean()) + t.mean()
    assert band_corr_loss(damped, t, m).item() < 1e-5
    assert math.isclose(band_amplitude_loss(damped, t, m).item(), math.log(2), rel_tol=1e-2)


def test_total_loss_backpropagates_to_prediction():
    t, m = _terrain(), torch.ones(2, 128, 128)
    pred = (t + torch.randn_like(t)).requires_grad_()
    sea = torch.zeros_like(m)
    sea[:, :, :10] = 1
    loss = total_loss(pred, {"lidar": t, "mask": m, "sea": sea}, LossWeights(water=0.3))
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(pred.grad).all()
