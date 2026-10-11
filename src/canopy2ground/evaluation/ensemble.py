"""Score checkpoint ensembles on the cache chips and pair them chip by chip.

Members are averaged at the predicted residual, so the prior is added once and
an ensemble has the interface of a single model. Averaging weights would be
meaningless: independently seeded runs sit in different basins with permuted
channels. Evaluation is deployment-style: terrain inputs are computed where
the prior is valid, as in a map tile, never on the lidar footprint. No
shift-TTA is applied (that is a map-time fix for a decoder lattice; it is not
part of the chip scores).

    python -m canopy2ground.evaluation.ensemble --cache data/cache --layer-root data/layers \\
        --set prev4 b2_s0.pt b2_s1.pt b5_s0.pt b5_s1.pt \\
        --set six b2_s0.pt b2_s1.pt b5_s0.pt b5_s1.pt b5_coast_s0.pt b5_coast_s1.pt \\
        --baseline prev4 --attribution --fused-attention --out results/ensembles.json

Paired differences are ``baseline - set`` per-chip MAE (positive: the set is
better), with a bootstrap over chips.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import ChipDataset, DataConfig
from ..metrics import evaluate
from ..models import load_checkpoint
from ..models.speed import use_fused_mit_attention
from ..predict import checkpoint_layers
from .stats import bootstrap_mean


class Ensemble(torch.nn.Module):
    """Mean of the members' residuals."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, batch: dict) -> torch.Tensor:
        acc = None
        for m in self.members:
            r = m(batch)
            acc = r if acc is None else acc + r
        return acc / len(self.members)


def load_members(paths, device="cuda") -> tuple[list[torch.nn.Module], tuple[str, ...]]:
    """Load checkpoints in eval mode; they must share one imagery layer stack."""
    models, layers = [], set()
    for p in paths:
        m, meta = load_checkpoint(p, map_location=device)
        models.append(m.to(device).eval())
        layers.add(checkpoint_layers(meta))
    if len(layers) != 1:
        raise ValueError(f"members were trained on different imagery layers: {layers}")
    return models, layers.pop()


def score_set(model, ds: ChipDataset, attribution: bool, device="cuda") -> dict:
    """Headline metrics of one model or ensemble, plus imagery-attributable ridge skill if asked."""
    full = evaluate(model, ds, device=device)
    row = {"n_chips": full["n_chips"], "mae_median_m": full["mae_median_m"], "mae_mean_m": full["mae_mean_m"],
           "ridge_r": full["ridge_r"], "fine_2_4_r": full["octave_r"]["2-4"],
           "per_chip_mae": full["per_chip"]["mae"]}
    if attribution:
        scrambled = evaluate(model, ds, decoy="scramble_rgb", device=device)
        row["imagery_attr"] = full["ridge_r"] - scrambled["ridge_r"]
    return row


def paired(base: dict, other: dict) -> dict:
    """Bootstrap of ``base - other`` per-chip MAE over the chips both score."""
    ids = sorted(set(base) & set(other))
    return bootstrap_mean(np.array([base[i] - other[i] for i in ids]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="cache root with <split>/ subdirectories")
    ap.add_argument("--layer-root", help="root of the extra imagery layers (default: --cache)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--set", nargs="+", action="append", required=True, metavar=("NAME", "CKPT"),
                    help="NAME ckpt [ckpt ...], once per ensemble; one checkpoint scores a single model")
    ap.add_argument("--baseline", help="set every other set is paired against")
    ap.add_argument("--attribution", action="store_true", help="also score with scrambled imagery")
    ap.add_argument("--fused-attention", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sets = {s[0]: s[1:] for s in a.set}
    if any(not v for v in sets.values()):
        ap.error("every --set needs a name and at least one checkpoint")
    if a.baseline and a.baseline not in sets:
        ap.error(f"--baseline {a.baseline!r} is not a --set name")

    if a.fused_attention:
        use_fused_mit_attention()
    paths = sorted({p for v in sets.values() for p in v})
    models, layers = load_members(paths)
    by_path = dict(zip(paths, models))
    ds = ChipDataset(DataConfig(root=a.cache, layer_root=a.layer_root, layers=layers, coast=True), a.split)

    res = {}
    for name, ckpts in sets.items():
        model = by_path[ckpts[0]] if len(ckpts) == 1 else Ensemble([by_path[p] for p in ckpts])
        r = res[name] = {"members": [Path(p).name for p in ckpts], **score_set(model, ds, a.attribution)}
        attr = f"  imagery-attr {r['imagery_attr']:+.4f}" if a.attribution else ""
        print(f"{name:14s} MAE med {r['mae_median_m']:.4f} mean {r['mae_mean_m']:.4f}  "
              f"ridge {r['ridge_r']:+.4f}  2-4m {r['fine_2_4_r']:+.4f}{attr}", flush=True)
    if a.baseline:
        base = res[a.baseline]["per_chip_mae"]
        for name in sets:
            if name == a.baseline:
                continue
            q = res[name]["paired_vs_baseline"] = paired(base, res[name]["per_chip_mae"])
            print(f"  {a.baseline} - {name}: {q['mean']:+.4f} m [{q['lo']:+.4f}, {q['hi']:+.4f}], "
                  f"{name} better on {q['frac_positive']:.0%} of {q['n']} chips")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"split": a.split, "baseline": a.baseline, "sets": res}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
