import numpy as np
import pytest
import torch

from canopy2ground.predict import ensemble_residual, feather


class LatticeModel(torch.nn.Module):
    """Returns a smooth function of the input plus a 4-pixel lattice fixed to the input grid,
    which is the artefact shift-TTA exists to remove."""

    def forward(self, b):
        h, w = b["dem"].shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        lattice = torch.cos(np.pi / 2 * xx) + torch.cos(np.pi * yy)
        return 0.1 * b["dem"] + lattice.to(b["dem"])


def _tile(n=64):
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32),
                            indexing="ij")
    dem = (50 + 0.5 * xx + 3 * torch.sin(yy / 7))[None]
    return {"rgb": torch.zeros(1, 3, n, n), "dem": dem, "mask": torch.ones(1, n, n)}


def test_shift_tta_cancels_grid_locked_lattice():
    b = _tile()
    one = ensemble_residual([LatticeModel()], b, n_shift=1)
    tta = ensemble_residual([LatticeModel()], b, n_shift=4)
    inner = (slice(None), slice(6, -6), slice(6, -6))
    target = 0.1 * b["dem"]
    assert (one - target)[inner].abs().max() > 0.9
    assert (tta - target)[inner].abs().max() < 1e-4


def test_ensemble_averages_members():
    class Const(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, b):
            return torch.full_like(b["dem"], self.c)

    r = ensemble_residual([Const(1.0), Const(3.0)], _tile(32), n_shift=1)
    assert torch.allclose(r, torch.full_like(r, 2.0))


@pytest.mark.parametrize("overlap", [1, 2, 64, 256])
def test_feather_is_a_partition_of_unity_with_no_zero_weights(overlap):
    n = 512
    w = feather(n, overlap)
    assert w.min() > 0, "edge pixels must keep some weight, or the map border goes to nodata"
    stride = n - overlap
    total = np.zeros(3 * stride + overlap)
    for k in range(3):
        total[k * stride:k * stride + n] += w
    assert np.allclose(total[overlap:-overlap], 1.0, atol=1e-6)
    assert np.array_equal(feather(n, 0), np.ones(n, np.float32))


def test_ensemble_redoes_a_member_whose_first_pass_overflows():
    class OverflowOnce(torch.nn.Module):
        """Non-finite on the first call, as an fp16 overflow would be, then finite."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, b):
            self.calls += 1
            return torch.full_like(b["dem"], float("inf") if self.calls == 1 else 1.0)

    class Const(torch.nn.Module):
        def forward(self, b):
            return torch.full_like(b["dem"], 3.0)

    flaky = OverflowOnce()
    r = ensemble_residual([flaky, Const()], _tile(16), n_shift=1)
    assert flaky.calls == 2
    assert torch.allclose(r, torch.full_like(r, 2.0))
