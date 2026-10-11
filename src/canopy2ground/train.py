"""Train one TerrainFusionNet from a YAML config.

    python -m canopy2ground.train configs/b5_coast.yaml --seed 0 --out runs/b5_coast_s0

Recipe (shared by every published model, see configs/):

* AdamW, cosine schedule to 1e-5 over the whole run, gradient clip 1.0, fp16
  autocast with a GradScaler.
* Two learning rates: the ImageNet encoders train at ``backbone_lr_mult`` of
  the head's rate, and are frozen for the first ``freeze_epochs`` epochs, then
  released one stage per epoch, deepest first. Frozen stages are also held in
  eval mode so their BatchNorm statistics do not drift.
* Multi-scale crops: the crop side cycles through 512 / s for s in ``scales``,
  so the network sees the same metre-scale terrain at several token sizes.
* Sampling without replacement: every chip is seen once per epoch.
* Steps with a non-finite loss are skipped (on GPU by the GradScaler, which
  also lowers its scale). Training stops after 50 in a row, or when more than
  5% of an epoch's steps were skipped; ``curves.jsonl`` records the count.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.dataset import CHIP, BatchOptions, ChipDataset, DataConfig, extra_channels_for, make_batch
from .losses import LossWeights, masked_mean, total_loss
from .metrics import evaluate
from .models import ModelConfig, TerrainFusionNet, save_checkpoint
from .models.speed import use_fused_mit_attention


@dataclass
class OptimConfig:
    epochs: int = 66
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    backbone_lr_mult: float = 0.1
    freeze_epochs: int = 2
    unfreeze_every: int = 1
    scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    fused_adam: bool = False
    fused_attention: bool = False
    channels_last: bool = True
    prefetch: int = 2
    steps_per_epoch: int | None = None    # default: one pass over the training chips


@dataclass
class TrainConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    optim: OptimConfig = field(default_factory=OptimConfig)
    val_split: str = "val"
    val_regions: tuple[str, ...] | None = None
    probe_chips: int = 160

    @classmethod
    def from_yaml(cls, path: str | Path, **data_overrides) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        def build(dc, d):
            names = {f.name for f in fields(dc)}
            unknown = set(d) - names
            if unknown:
                raise ValueError(f"{path}: unknown {dc.__name__} keys {sorted(unknown)}")
            return dc(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

        data = build(DataConfig, {**raw.get("data", {}), **data_overrides})
        model = build(ModelConfig, raw.get("model", {}))
        if not model.extra_channels:
            model = ModelConfig(**{**asdict(model), "extra_channels": extra_channels_for(data.layers)})
        sections = ("data", "model", "loss", "optim")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{path}: unknown {cls.__name__} keys {sorted(unknown)}")
        top = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items() if k not in sections}
        return cls(data=data, model=model, loss=build(LossWeights, raw.get("loss", {})),
                   optim=build(OptimConfig, raw.get("optim", {})), **top)


# ---------------------------------------------------------------------------
# encoder freezing

def encoder_stages(model: TerrainFusionNet) -> list[list[list[torch.nn.Module]]]:
    """Per encoder, its stages shallow -> deep, each a list of modules."""
    mit = []
    for i in (1, 2, 3, 4):
        mods = [getattr(model.enc_rgb, f"{k}{i}") for k in ("patch_embed", "block", "norm")
                if hasattr(model.enc_rgb, f"{k}{i}")]
        if mods:
            mit.append(mods)
    resnet = [[getattr(model.enc_topo, n)] for n in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
              if hasattr(model.enc_topo, n)]
    return [mit or [[model.enc_rgb]], resnet or [[model.enc_topo]]]


def trainable_stages(epoch: int, opt: OptimConfig, n_stages: int) -> int:
    """Number of (deepest) encoder stages trainable in 1-based ``epoch``."""
    if opt.freeze_epochs <= 0:
        return n_stages
    if epoch <= opt.freeze_epochs:
        return 0
    return min(n_stages, 1 + (epoch - opt.freeze_epochs - 1) // max(1, opt.unfreeze_every))


def apply_freeze(model: TerrainFusionNet, n_trainable: int) -> list[torch.nn.Module]:
    """Freeze all but the deepest ``n_trainable`` stages of each encoder; return the frozen modules."""
    frozen = []
    for stages in encoder_stages(model):
        cut = len(stages) - n_trainable
        for i, stage in enumerate(stages):
            for mod in stage:
                for p in mod.parameters():
                    p.requires_grad_(i >= cut)
                if i < cut:
                    frozen.append(mod)
    return frozen


# ---------------------------------------------------------------------------
# data feeding

class Prefetcher:
    """Build batches on one background thread.

    A single producer makes the same RNG calls in the same order as an inline
    loop, so prefetching does not change which crops a run sees.
    """

    def __init__(self, make, n_steps: int, depth: int):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()

        def run():
            for step in range(n_steps):
                if self._stop.is_set():
                    break
                try:
                    item = make(step)
                except Exception as exc:       # re-raised on the consumer side
                    item = exc
                while not self._stop.is_set():
                    try:
                        self.q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            if not self._stop.is_set():
                self.q.put(None)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def __iter__(self):
        while (item := self.q.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        """Stop the producer and drop queued batches; safe to call more than once."""
        self._stop.set()
        while self._thread.is_alive() or not self.q.empty():
            try:
                self.q.get(timeout=0.1)
            except queue.Empty:
                pass

    def __enter__(self) -> Prefetcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class EpochSampler:
    """Shuffled sampling without replacement; the pool carries across epochs."""

    def __init__(self, n: int, rng: np.random.Generator):
        self.n, self.rng, self.pool = n, rng, []

    def draw(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if not self.pool:
                self.pool = self.rng.permutation(self.n).tolist()
            take = min(k - len(out), len(self.pool))
            out.extend(self.pool[:take])
            self.pool = self.pool[take:]
        return out


@torch.no_grad()
def val_skill(model, ds: ChipDataset, n: int, batch_size: int, device) -> float:
    """Mean metres of error removed from the prior on the first ``n`` val chips (a cheap probe)."""
    model.eval()
    gains = []
    for i in range(0, min(n, len(ds)), batch_size):
        b = make_batch(ds, list(range(i, min(i + batch_size, n, len(ds)))), device=device)
        if b is None:
            continue
        with torch.autocast(b["dem"].device.type, dtype=torch.float16, enabled=b["dem"].is_cuda):
            pe = b["dem"] + model(b) * b["mask"]
        gains.append(masked_mean((b["dem"] - b["lidar"]).abs() - (pe - b["lidar"]).abs(),
                                 b["mask"], dims=(1, 2)).mean().item())
    model.train()
    return float(np.mean(gains)) if gains else float("nan")


# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, seed: int, out_dir: Path, device: str | torch.device = "cuda") -> Path:
    """Train one model; writes ``config.json``, ``curves.jsonl``, ``model.pt`` and ``val_metrics.json``."""
    torch.backends.cudnn.benchmark = True          # fixed crop sizes
    torch.backends.cuda.matmul.allow_tf32 = True   # Ampere: only affects ops left in fp32 under autocast
    torch.backends.cudnn.allow_tf32 = True
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"seed": seed, **asdict(cfg)}, indent=2))
    o = cfg.optim

    train_ds = ChipDataset(cfg.data, "train")
    val_ds = ChipDataset(cfg.data, cfg.val_split, regions=cfg.val_regions)
    if cfg.val_split == "train" and set(train_ds.rows) & set(val_ds.rows):
        raise ValueError("hold-out regions overlap the training regions")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if o.fused_attention:
        use_fused_mit_attention()
    model = TerrainFusionNet(cfg.model).to(device)
    if o.channels_last:
        model = model.to(memory_format=torch.channels_last)

    stages = encoder_stages(model)
    backbone = {id(p) for enc in stages for stage in enc for m in stage for p in m.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in backbone], "lr": o.lr * o.backbone_lr_mult},
        {"params": [p for p in model.parameters() if id(p) not in backbone], "lr": o.lr},
    ]
    optim = torch.optim.AdamW(groups, lr=o.lr, weight_decay=o.weight_decay,
                              **({"fused": True} if o.fused_adam else {}))
    steps_per_epoch = o.steps_per_epoch or len(train_ds) // o.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=o.epochs * steps_per_epoch, eta_min=1e-5)
    dev = torch.device(device).type
    amp = dev == "cuda"                   # fp16 autocast + loss scaling on GPU, plain fp32 elsewhere
    scaler = torch.amp.GradScaler(dev, enabled=amp)
    rng = np.random.default_rng(seed)
    sampler = EpochSampler(len(train_ds), rng)
    n_stages = max(len(s) for s in stages)

    def next_batch(step: int):
        idx = sampler.draw(o.batch_size)
        side = CHIP
        if o.scales:
            s = float(o.scales[step % len(o.scales)])
            side = max(64, min(CHIP, int(round(CHIP / s)) // 16 * 16))
        return make_batch(train_ds, idx, BatchOptions(side=side, augment=True), rng=rng, device=device)

    t0, n_bad = time.time(), 0
    with open(out_dir / "curves.jsonl", "w", encoding="utf-8", buffering=1) as log:
        for epoch in range(1, o.epochs + 1):
            model.train()
            for m in apply_freeze(model, trainable_stages(epoch, o, n_stages)):
                m.eval()
            losses, n_steps, n_skipped = [], 0, 0
            optim.zero_grad()
            feed = (Prefetcher(next_batch, steps_per_epoch, o.prefetch) if o.prefetch
                    else contextlib.nullcontext(next_batch(i) for i in range(steps_per_epoch)))
            with feed as batches:
                for batch in batches:
                    if batch is None:
                        continue
                    with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                        gate = batch["in_mask"] if cfg.model.coast_inputs else batch["mask"]
                        pred = batch["dem"] + model(batch) * gate
                        loss = total_loss(pred, batch, cfg.loss)
                    n_steps += 1
                    finite = bool(torch.isfinite(loss.detach()))
                    if finite:
                        n_bad = 0
                        losses.append(float(loss.item()))
                    else:
                        n_bad, n_skipped = n_bad + 1, n_skipped + 1
                        if n_bad >= 50:
                            raise RuntimeError(f"50 consecutive non-finite losses at epoch {epoch}")
                    if finite or amp:     # under amp the GradScaler sees the non-finite gradients and skips
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optim)
                        scaler.update()
                    optim.zero_grad()
                    sched.step()

            rec = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
                   "skipped_steps": n_skipped, "lr": optim.param_groups[-1]["lr"],
                   "val_skill_m": val_skill(model, val_ds, cfg.probe_chips, o.batch_size, device),
                   "minutes": (time.time() - t0) / 60}
            log.write(json.dumps(rec) + "\n")
            print(f"epoch {epoch:3d}/{o.epochs}  loss {rec['loss']:.4f}  skipped {n_skipped}  "
                  f"val skill {rec['val_skill_m']:+.3f} m  ({rec['minutes']:.0f} min)", flush=True)
            if n_skipped > 0.05 * n_steps:
                raise RuntimeError(f"{n_skipped} of {n_steps} steps had a non-finite loss in epoch {epoch}")

    ckpt = out_dir / "model.pt"
    save_checkpoint(str(ckpt), model, seed=seed, train_config=asdict(cfg))
    report = evaluate(model, val_ds, device=device)
    report.pop("per_chip")
    (out_dir / "val_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"val: median MAE {report['mae_median_m']:.3f} m over {report['n_chips']} chips -> {ckpt}")
    return ckpt


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-root", help="override data.root from the config")
    ap.add_argument("--layer-root", help="override data.layer_root from the config")
    a = ap.parse_args(argv)
    overrides = {k: v for k, v in (("root", a.data_root), ("layer_root", a.layer_root)) if v}
    train(TrainConfig.from_yaml(a.config, **overrides), a.seed, a.out)


if __name__ == "__main__":
    main()
