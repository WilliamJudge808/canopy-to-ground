import numpy as np
import pytest
import torch

from canopy2ground.data import ChipDataset, make_batch
from canopy2ground.evaluation.ensemble import Ensemble, paired, score_set
from canopy2ground.evaluation.stats import block_bootstrap, bootstrap_mean, cluster_bootstrap


class Const(torch.nn.Module):
    """Stand-in model: a constant residual inside the valid mask."""

    def __init__(self, c: float):
        super().__init__()
        self.c = c
        self.w = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        return (self.c + self.w) * batch["mask"]


def test_bootstrap_mean_matches_reference_loop():
    d = np.random.default_rng(1).normal(0.3, 1.0, 200)
    d[5] = np.nan
    got = bootstrap_mean(d, n_boot=500, seed=0)
    x = d[np.isfinite(d)]
    rng = np.random.default_rng(0)
    reps = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(500)]
    assert got["n"] == 199
    assert got["mean"] == pytest.approx(x.mean())
    assert got["lo"] == pytest.approx(np.percentile(reps, 2.5))
    assert got["hi"] == pytest.approx(np.percentile(reps, 97.5))
    assert got["lo"] < got["mean"] < got["hi"]
    assert got["frac_positive"] == pytest.approx((x > 0).mean())


def test_cluster_bootstrap_resamples_whole_clusters():
    rng = np.random.default_rng(0)
    clusters = np.repeat([f"{t}_{c}" for t in (10, 20, 30) for c in (1, 2)], 50)
    delta = rng.normal(0.5, 0.2, clusters.size)
    got = cluster_bootstrap(delta, clusters, n_boot=300, seed=42)

    # same draws as picking cluster labels directly
    labels = np.unique(clusters)
    by = {c: delta[clusters == c] for c in labels}
    ref_rng = np.random.default_rng(42)
    reps = [np.median(np.concatenate([by[c] for c in ref_rng.choice(labels, len(labels), replace=True)]))
            for _ in range(300)]
    assert got["lo"] == pytest.approx(np.percentile(reps, 2.5))
    assert got["hi"] == pytest.approx(np.percentile(reps, 97.5))
    assert got["point"] == pytest.approx(np.median(delta))
    assert got["n_overpasses"] == 6 and got["n_segments"] == 300
    assert got["p"] == 0.0


def test_cluster_bootstrap_ci_reflects_between_cluster_spread():
    # each cluster is internally constant: a per-segment bootstrap would give a near-zero CI
    clusters = np.repeat(np.arange(8), 100)
    delta = np.repeat(np.linspace(-1.0, 1.4, 8), 100)
    got = cluster_bootstrap(delta, clusters, n_boot=400, seed=0)
    assert got["hi"] - got["lo"] > 0.5
    assert got["p"] > 0.05


def test_block_bootstrap_matches_reference_scheme_with_a_shared_generator():
    rng = np.random.default_rng(3)
    blocks = np.array([(7, 2), (3, 9), (7, 2), (5, 5), (3, 9), (5, 5), (7, 2), (3, 9)] * 10)
    diffs = {"a": rng.normal(1.0, 1.0, len(blocks)), "b": rng.normal(-0.2, 1.0, len(blocks))}
    got = block_bootstrap(diffs, blocks, n_boot=300, seed=0)

    # blocks numbered in sorted order, one generator drawn from for "a" and then "b"
    g = np.array([{(3, 9): 0, (5, 5): 1, (7, 2): 2}[tuple(b)] for b in blocks])
    ref_rng = np.random.default_rng(0)
    for name, d in diffs.items():
        reps = [np.concatenate([d[g == k] for k in ref_rng.integers(0, 3, 3)]).mean() for _ in range(300)]
        r = got[name]
        assert r["mean"] == d.mean() and r["n_blocks"] == 3
        assert r["lo"] == np.percentile(reps, 2.5) and r["hi"] == np.percentile(reps, 97.5)
        assert r["blocks_better"] == sum(d[g == k].mean() > 0 for k in range(3))
    assert list(got["a"]) == ["mean", "lo", "hi", "blocks_better", "n_blocks"]


def test_block_bootstrap_is_wider_than_the_chip_bootstrap_for_correlated_chips():
    blocks = np.repeat(np.arange(6), 50)
    d = np.repeat([0.1, 1.9, 0.4, 1.2, -0.3, 0.9], 50) + np.random.default_rng(0).normal(0, 0.05, 300)
    by_block = block_bootstrap({"d": d}, blocks, n_boot=400)["d"]
    by_chip = bootstrap_mean(d, n_boot=400)
    assert by_block["hi"] - by_block["lo"] > 5 * (by_chip["hi"] - by_chip["lo"])
    assert by_block["blocks_better"] == 5


def test_block_bootstrap_drops_non_finite_and_empty_blocks():
    d = np.array([1.0, 2.0, np.nan, np.nan, -1.0])
    r = block_bootstrap({"d": d}, np.array([0, 0, 1, 1, 2]), n_boot=50)["d"]
    assert r["n_blocks"] == 2 and r["mean"] == pytest.approx(2 / 3) and r["blocks_better"] == 1


def test_ensemble_averages_residuals(tiny_cache, data_cfg):
    ds = ChipDataset(data_cfg, "val")
    b = make_batch(ds, [1, 2], device="cpu")
    out = Ensemble([Const(1.0), Const(3.0)])(b)
    assert torch.allclose(out, 2.0 * b["mask"])


def test_score_set_and_pairing(tiny_cache, data_cfg):
    ds = ChipDataset(data_cfg, "val")
    zero, ens = Const(0.0), Ensemble([Const(-1.0), Const(1.0)])      # the ensemble is the prior itself
    a = score_set(zero, ds, attribution=False, device="cpu")
    b = score_set(ens, ds, attribution=True, device="cpu")
    assert a["n_chips"] == b["n_chips"] == 5                         # chip 0 has too little lidar
    assert a["mae_median_m"] == pytest.approx(b["mae_median_m"], abs=1e-6)
    assert "imagery_attr" in b and abs(b["imagery_attr"]) < 1e-6     # constant residual ignores imagery
    q = paired(a["per_chip_mae"], b["per_chip_mae"])
    assert q["n"] == 5 and abs(q["mean"]) < 1e-6

    worse = score_set(Const(2.0), ds, attribution=False, device="cpu")
    q = paired(worse["per_chip_mae"], a["per_chip_mae"])
    assert q["mean"] > 0 and q["frac_positive"] == 1.0
