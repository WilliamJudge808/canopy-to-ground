"""Per-chip evaluation against lidar.

Every statistic is computed per chip and then summarised over chips (median
by default), and the per-chip values are returned keyed by cache row so two
models can be compared paired on the same chips.
"""
from __future__ import annotations

import numpy as np
import torch

from .data.dataset import BatchOptions, ChipDataset, make_batch
from .losses import laplacian_bands

BANDS = ("2-4", "4-8", "8-16", "16-32", "32-64", "64-128")   # metres, Laplacian octaves at 1 m/px
RIDGE_BANDS = ("4-8", "8-16", "16-32")                        # ridges, gulches, benches


def _band_r_amp(p: torch.Tensor, t: torch.Tensor):
    pv, tv = p.double(), t.double()
    ps, ts = pv.std(), tv.std()
    if ts < 1e-9:
        return None
    r = 0.0 if ps < 1e-12 else float(((pv - pv.mean()) * (tv - tv.mean())).mean() / (ps * ts))
    return r, float(ps / ts)


def decoy_partners(n: int) -> list[int]:
    """Partner for each of ``n`` chips: the chip half the list away.

    Chips are stored in scan order, so this pairs each chip with one far from it
    and never with itself (a permutation inside a small batch would leave about
    one chip per batch paired with itself, under-reporting the decoy effect).
    """
    return [(i + n // 2) % n for i in range(n)]


def nmad(x: np.ndarray) -> float:
    """Normalised median absolute deviation, a robust spread (1.4826 * MAD)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@torch.no_grad()
def predict_batch(model, batch: dict) -> torch.Tensor:
    """Residual for one batch under fp16 autocast, redone in fp32 if fp16 overflows."""
    with torch.autocast(batch["dem"].device.type, dtype=torch.float16,
                        enabled=batch["dem"].is_cuda):
        res = model(batch)
    if not torch.isfinite(res).all():
        res = model(batch)
    return res.float()


@torch.no_grad()
def evaluate(model, ds: ChipDataset, n_eval: int | None = None, batch_size: int = 4,
             decoy: str | None = None, device="cuda") -> dict:
    """Score ``model`` on the first ``n_eval`` chips of ``ds``.

    ``decoy`` ("scramble_rgb" / "scramble_dem") pairs each chip with the one
    half the dataset away and feeds that chip's imagery (or prior) instead.
    The rise in error over the clean run is the skill attributable to that
    input.
    """
    model.eval()
    n = min(n_eval or len(ds), len(ds))
    idx = list(range(n))
    partner = decoy_partners(n)
    per_chip = {"mae": {}, "rmse": {}, "bias": {}, "ridge_r": {}}
    band_r = {b: [] for b in BANDS}
    band_amp = {b: [] for b in BANDS}
    errors = []

    for i in range(0, n, batch_size):
        opts = BatchOptions(decoy=decoy, decoy_src=partner[i:i + batch_size])
        batch = make_batch(ds, idx[i:i + batch_size], opts, device=device)
        if batch is None:
            continue
        mask = batch["mask"]
        pred = batch["dem"] + predict_batch(model, batch) * mask

        bp, _ = laplacian_bands(pred, mask)
        bt, bm = laplacian_bands(batch["lidar"], mask)
        chip_band_r = [{} for _ in batch["chip_ids"]]
        for name, p, t, m in zip(BANDS, bp, bt, bm):
            for b in range(p.shape[0]):
                sel = m[b] > 0.5
                if int(sel.sum()) < 256:
                    continue
                ra = _band_r_amp(p[b][sel], t[b][sel])
                if ra:
                    band_r[name].append(ra[0])
                    band_amp[name].append(ra[1])
                    chip_band_r[b][name] = ra[0]

        for b, cid in enumerate(batch["chip_ids"]):
            m = mask[b] > 0.5
            if m.sum() < 256:
                continue
            d = (pred[b][m] - batch["lidar"][b][m]).cpu().numpy()
            per_chip["mae"][cid] = float(np.abs(d).mean())
            per_chip["rmse"][cid] = float(np.sqrt((d ** 2).mean()))
            per_chip["bias"][cid] = float(np.median(d))
            if all(k in chip_band_r[b] for k in RIDGE_BANDS):
                per_chip["ridge_r"][cid] = float(np.mean([chip_band_r[b][k] for k in RIDGE_BANDS]))
            errors.append(d[:: max(1, d.size // 2048)])      # thinned pixel sample for NMAD/LE90

    pix = np.concatenate(errors) if errors else np.zeros(1)
    octave_r = {b: float(np.median(v)) if v else float("nan") for b, v in band_r.items()}
    return {
        "n_chips": len(per_chip["mae"]),
        "mae_median_m": float(np.median(list(per_chip["mae"].values()))),
        "mae_mean_m": float(np.mean(list(per_chip["mae"].values()))),
        "rmse_median_m": float(np.median(list(per_chip["rmse"].values()))),
        "nmad_m": nmad(pix),
        "le90_m": float(np.percentile(np.abs(pix), 90)),
        "ridge_r": float(np.mean([octave_r[b] for b in RIDGE_BANDS])),
        "octave_r": octave_r,
        "octave_amp": {b: float(np.median(v)) if v else float("nan") for b, v in band_amp.items()},
        "per_chip": per_chip,
    }

