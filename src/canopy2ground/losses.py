"""Training objective.

A lidar chip's variance is dominated by hillside-scale relief: on this corpus
98.5% of it sits at wavelengths above 128 m and the 4-32 m band (ridges,
gulches, benches) holds under 0.1%. A plain L1 loss on elevation is therefore
almost entirely a loss on the overall slope, and a smooth surface with no
ridges is close to optimal under it. The terms below put the short
wavelengths back into the objective:

* ``multiband_detail_loss``: L1 per Laplacian octave, each octave normalised
  by the target's own standard deviation in that octave.
* ``band_corr_loss``: 1 - Pearson r per octave. Scale-invariant, so it cannot
  be lowered by shrinking the output toward the mean.
* ``band_amplitude_loss``: symmetric log-ratio of per-octave amplitude.
  Under uncertainty an L1-optimal model emits r times the true relief, where
  r is its correlation with the truth; this term pays for keeping amplitude.
* ``fft_log_magnitude_loss``: matches the log power spectrum.
* ``sea_loss``: pulls pixels seaward of the coastline to sea level, with a
  dead zone for tide and datum.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

N_BANDS = 6                                    # 2-4, 4-8, ..., 64-128 m at 1 m/px
BAND_WEIGHTS = (0.15, 1.0, 1.0, 1.0, 0.5, 0.25)  # the 2-4 m band is mostly texture and residual canopy


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dims) -> torch.Tensor:
    """Mean of ``x`` over ``mask`` along ``dims``. Always reduced in float32."""
    x, mask = x.float(), mask.float()
    return (x * mask).sum(dim=dims) / mask.sum(dim=dims).clamp(min=1.0)


def laplacian_bands(x: torch.Tensor, mask: torch.Tensor, levels: int = N_BANDS):
    """Split (B, H, W) into Laplacian-pyramid bands, finest first.

    Each band gets a mask eroded by the pyramid, so a band is only scored where
    every contributing pixel was valid.
    """
    cur, cur_m = x.unsqueeze(1).float(), mask.unsqueeze(1).float()
    bands, masks = [], []
    for _ in range(levels):
        if min(cur.shape[-2:]) < 4:
            break
        down, down_m = F.avg_pool2d(cur, 2), F.avg_pool2d(cur_m, 2)
        up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        up_m = F.interpolate(down_m, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        bands.append((cur - up).squeeze(1))
        masks.append(((cur_m * up_m) > 0.999).squeeze(1).float())
        cur, cur_m = down, down_m
    return bands, masks


def multiband_detail_loss(pred, target, mask, band_weights=BAND_WEIGHTS, eps=1e-4):
    bands_p, _ = laplacian_bands(pred, mask)
    bands_t, masks = laplacian_bands(target, mask)
    total = pred.new_zeros(())
    for bp, bt, bm, w in zip(bands_p, bands_t, masks, band_weights):
        if bm.sum() < 1:
            continue
        mu = masked_mean(bt, bm, dims=(1, 2)).view(-1, 1, 1)
        sigma = masked_mean((bt - mu) ** 2, bm, dims=(1, 2)).clamp(min=eps).sqrt()
        sigma = sigma.detach().view(-1, 1, 1)
        total = total + w * masked_mean((bp - bt).abs() / sigma, bm, dims=(1, 2)).mean()
    return total / max(1, len(bands_t))


def band_corr_loss(pred, target, mask, band_weights=BAND_WEIGHTS, eps=1e-8):
    bands_p, _ = laplacian_bands(pred, mask)
    bands_t, masks = laplacian_bands(target, mask)
    total, used = pred.new_zeros(()), 0.0
    for bp, bt, bm, w in zip(bands_p, bands_t, masks, band_weights):
        if bm.sum() < 1:
            continue
        pc = (bp - masked_mean(bp, bm, dims=(1, 2)).view(-1, 1, 1)) * bm
        tc = (bt - masked_mean(bt, bm, dims=(1, 2)).view(-1, 1, 1)) * bm
        num = (pc * tc).sum(dim=(1, 2))
        den = (pc.pow(2).sum(dim=(1, 2)).clamp(min=eps) * tc.pow(2).sum(dim=(1, 2)).clamp(min=eps)).sqrt()
        total = total + w * (1.0 - num / den).mean()
        used += w
    return total / max(used, 1e-6)


def band_amplitude_loss(pred, target, mask, band_weights=BAND_WEIGHTS, eps=1e-4):
    bands_p, _ = laplacian_bands(pred, mask)
    bands_t, masks = laplacian_bands(target, mask)
    total, used = pred.new_zeros(()), 0.0
    for bp, bt, bm, w in zip(bands_p, bands_t, masks, band_weights):
        if bm.sum() < 64 or w == 0:
            continue
        mp = masked_mean(bp, bm, dims=(1, 2)).view(-1, 1, 1)
        mt = masked_mean(bt, bm, dims=(1, 2)).view(-1, 1, 1)
        sp = masked_mean((bp - mp) ** 2, bm, dims=(1, 2)).clamp(min=eps).sqrt()
        st = masked_mean((bt - mt) ** 2, bm, dims=(1, 2)).clamp(min=eps).sqrt()
        total = total + w * torch.log(sp / st.detach()).abs().mean()
        used += w
    return total / max(used, 1e-6)


def fft_log_magnitude_loss(pred, target, mask):
    pf = torch.fft.rfft2(pred * mask, dim=(-2, -1), norm="ortho")
    tf = torch.fft.rfft2(target * mask, dim=(-2, -1), norm="ortho")
    return F.l1_loss(torch.log1p(pf.abs()), torch.log1p(tf.abs()))


def sea_loss(pred, sea_mask, sea_level=0.0, deadzone=0.5):
    excess = torch.relu((pred - sea_level).abs() - deadzone)
    return masked_mean(excess, sea_mask, dims=(1, 2)).mean()


@dataclass(frozen=True)
class LossWeights:
    mae: float = 0.3
    detail: float = 0.5
    corr: float = 1.0
    amp: float = 0.5
    fft: float = 0.2
    water: float = 0.0
    sea_level: float = 0.0
    sea_deadzone: float = 0.5


def total_loss(pred_elev: torch.Tensor, batch: dict, w: LossWeights) -> torch.Tensor:
    """Weighted sum of the terms above. Land terms are masked to lidar ground."""
    lidar, mask = batch["lidar"], batch["mask"]
    loss = w.mae * masked_mean((pred_elev - lidar).abs(), mask, dims=(1, 2)).mean()
    loss = loss + w.detail * multiband_detail_loss(pred_elev, lidar, mask)
    loss = loss + w.corr * band_corr_loss(pred_elev, lidar, mask)
    if w.amp > 0:
        loss = loss + w.amp * band_amplitude_loss(pred_elev, lidar, mask)
    if w.water > 0 and "sea" in batch:
        loss = loss + w.water * sea_loss(pred_elev, batch["sea"], w.sea_level, w.sea_deadzone)
    if w.fft > 0:
        loss = loss + w.fft * fft_log_magnitude_loss(pred_elev, lidar, mask)
    return loss
