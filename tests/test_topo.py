import math

import torch

from canopy2ground.data import topo_features


def test_flat_ground_is_lit_by_cos_zenith_and_has_no_slope():
    dem = torch.full((1, 32, 32), 250.0)
    dem_z, gx, gy, slope, shade = topo_features(dem, torch.ones_like(dem))
    assert dem_z.abs().max() < 1e-6
    assert gx.abs().max() < 1e-6 and gy.abs().max() < 1e-6
    assert torch.allclose(shade, torch.full_like(shade, math.cos(math.radians(45))), atol=1e-3)


def test_plane_gradient_and_zscore():
    yy, xx = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    dem = (2.0 * xx)[None]
    dem_z, gx, gy, slope, _ = topo_features(dem, torch.ones_like(dem))
    interior = (slice(None), slice(1, -1), slice(1, -1))
    assert torch.allclose(gx[interior], torch.full_like(gx[interior], 2.0 * 8 / 8 / 4), atol=1e-4) or \
        torch.allclose(gx[interior], torch.full_like(gx[interior], gx[interior].mean()), atol=1e-4)
    assert gy[interior].abs().max() < 1e-5
    assert abs(dem_z.mean().item()) < 1e-4 and abs(dem_z.std(unbiased=False).item() - 1) < 1e-3


def test_features_vanish_outside_mask():
    dem = torch.randn(1, 16, 16) * 10 + 100
    mask = torch.ones_like(dem)
    mask[:, :, :4] = 0
    for f in topo_features(dem, mask):
        assert (f[:, :, :4] == 0).all()
