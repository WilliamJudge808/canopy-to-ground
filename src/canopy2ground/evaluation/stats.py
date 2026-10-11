"""Bootstrap confidence intervals for paired comparisons.

Three resampling units are used. ``bootstrap_mean`` resamples chips, which
treats chip-paired differences as independent. They are not quite: the
validation chips come from a few 10.24 km split blocks (``grid.BLOCK_CHIPS``)
whose chips share terrain, canopy and imagery, so ``block_bootstrap``
resamples whole blocks for the honest interval. ICESat-2 segments along one
overpass share that pass's geolocation and atmosphere errors, so
``cluster_bootstrap`` resamples whole overpasses.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_mean(d: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    """Percentile 95% CI of the mean of paired differences ``d``, resampling elements.

    Non-finite entries (a chip a metric could not score) are dropped first.
    ``frac_positive`` is the share of pairs where the first surface had the larger value.
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    reps = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean()), "lo": float(np.percentile(reps, 2.5)),
            "hi": float(np.percentile(reps, 97.5)), "median": float(np.median(d)),
            "frac_positive": float((d > 0).mean()), "n": int(len(d))}


def cluster_bootstrap(delta: np.ndarray, clusters: np.ndarray, n_boot: int = 10_000, seed: int = 42,
                      stat: Callable[[np.ndarray], float] = np.median) -> dict:
    """Percentile 95% CI of ``stat(delta)``, resampling whole clusters with replacement.

    ``p`` is the two-sided bootstrap p-value for ``stat(delta) == 0``.
    """
    labels, inv = np.unique(clusters, return_inverse=True)
    groups = [delta[inv == i] for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(len(labels), len(labels), replace=True)
        reps[k] = stat(np.concatenate([groups[c] for c in pick]))
    return {"point": float(stat(delta)), "lo": float(np.percentile(reps, 2.5)),
            "hi": float(np.percentile(reps, 97.5)),
            "p": float(2 * min((reps <= 0).mean(), (reps >= 0).mean())),
            "n_segments": int(len(delta)), "n_overpasses": int(len(labels))}


def block_bootstrap(diffs: dict[str, np.ndarray], blocks: np.ndarray, n_boot: int = 4000,
                    seed: int = 0) -> dict[str, dict]:
    """Percentile 95% CI of the mean paired difference per entry of ``diffs``, resampling whole blocks.

    ``diffs`` maps a name to per-chip differences aligned with ``blocks``, the
    chips' block labels (1-D, or one row per chip such as ``(bx, by)``).
    Blocks are numbered in sorted label order and one generator is drawn from
    across the names in ``diffs`` order, so a result is reproducible only with
    the same names in the same order. Non-finite differences are dropped.
    ``blocks_better`` counts blocks whose mean difference is positive.
    """
    _, inv = np.unique(np.asarray(blocks), axis=0, return_inverse=True)
    inv = inv.ravel()
    rng = np.random.default_rng(seed)
    out = {}
    for name, d in diffs.items():
        d = np.asarray(d, dtype=np.float64)
        ok = np.isfinite(d)
        d, labels = d[ok], inv[ok]
        groups = [d[labels == b] for b in np.unique(labels)]
        n = len(groups)
        reps = [np.concatenate([groups[b] for b in rng.integers(0, n, n)]).mean() for _ in range(n_boot)]
        out[name] = {"mean": float(d.mean()), "lo": float(np.percentile(reps, 2.5)),
                     "hi": float(np.percentile(reps, 97.5)),
                     "blocks_better": int(sum(g.mean() > 0 for g in groups)), "n_blocks": n}
    return out
