"""Terrain features derived from the prior DEM (computed on the fly, on device)."""
import math

import torch
import torch.nn.functional as F

from ..losses import masked_mean

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0


def topo_features(dem: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``(dem_z, gx, gy, slope_mag, hillshade)`` for a (B, H, W) prior in metres.

    Elevation is z-scored per chip over ``mask``; gradients are Sobel, slope is
    their magnitude, and the hillshade is lit from 315 degrees azimuth at 45
    degrees altitude. Every output is (B, H, W) and zero outside ``mask``.
    """
    b = dem.shape[0]
    mu = masked_mean(dem, mask, dims=(1, 2)).view(b, 1, 1)
    sd = masked_mean((dem - mu) ** 2, mask, dims=(1, 2)).clamp(min=0.01).sqrt().view(b, 1, 1)
    dem_z = ((dem - mu) / sd) * mask

    padded = F.pad(dem.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, _SOBEL_X.to(dem.device).view(1, 1, 3, 3)).squeeze(1) * mask
    gy = F.conv2d(padded, _SOBEL_Y.to(dem.device).view(1, 1, 3, 3)).squeeze(1) * mask
    slope_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6) * mask

    zenith, azimuth = 0.785398, 2.356194          # 45 deg; 360 - 315 + 90 = 135 deg
    slope = torch.atan(slope_mag)
    aspect = torch.atan2(gy, -gx)
    aspect = torch.where(aspect < 0, 2 * math.pi + aspect, aspect)
    shade = (math.cos(zenith) * torch.cos(slope)
             + math.sin(zenith) * torch.sin(slope) * torch.cos(azimuth - aspect))
    return dem_z, gx, gy, slope_mag, shade.clamp(0.0, 1.0) * mask
