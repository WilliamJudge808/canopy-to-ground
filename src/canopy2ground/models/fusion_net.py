"""TerrainFusionNet: imagery + coarse DEM -> 1 m bare-earth elevation correction.

Two encoders, fused at four scales, one decoder::

    optical stack (1 m)  --SegFormer MiT-->  f_img[2..5]  --+
                                                            |  4 m, 8 m : GatedFusionBlock
    prior DEM features   --ResNet-18------>  f_topo[2..5] --+  16 m, 32 m: SpatialCrossAttention
                                                            v
                                 U-Net decoder (32 m -> 1 m) + 1 m HighFreqFilterBank
                                                            v
                                 head -> residual r;   elevation = prior + r

The network never predicts absolute elevation. It predicts a correction to
the 30 m prior, which keeps the long wavelengths (where the prior is good)
anchored and spends capacity on the short ones (where it is not).

Attribute names match the checkpoints the published map was built from, so
they load with ``strict=True`` (see :func:`load_checkpoint`).
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .blocks import GatedFusionBlock, HighFreqFilterBank, SpatialCrossAttention


@dataclass(frozen=True)
class ModelConfig:
    rgb_encoder: str = "mit_b2"     # any of mit_b1..mit_b5 (same feature widths)
    topo_encoder: str = "resnet18"
    extra_channels: int = 0         # extra imagery bands + one availability plane per layer
    coast_inputs: bool = False      # land/sea channel + absolute elevation (dem / 1000)
    decoder_v2: bool = False        # bilinear upsampling + learned H/2 stem skip

    @property
    def optical_channels(self) -> int:
        return 3 + self.extra_channels + int(self.coast_inputs)

    @property
    def topo_channels(self) -> int:
        return 3 + 2 * int(self.coast_inputs)


class TerrainFusionNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, encoder_weights: str | None = "imagenet"):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        opt_ch = cfg.optical_channels

        self.enc_rgb = smp.encoders.get_encoder(
            cfg.rgb_encoder, in_channels=opt_ch, depth=5, weights=encoder_weights)
        self.enc_topo = smp.encoders.get_encoder(
            cfg.topo_encoder, in_channels=cfg.topo_channels, depth=5, weights=encoder_weights)
        img_ch = self.enc_rgb.out_channels     # (C, 0, 64, 128, 320, 512) for MiT
        topo_ch = self.enc_topo.out_channels   # (C, 64, 64, 128, 256, 512) for ResNet-18

        self.fuse2 = GatedFusionBlock(img_ch[2], topo_ch[2], img_ch[2])
        self.fuse3 = GatedFusionBlock(img_ch[3], topo_ch[3], img_ch[3])
        self.cross_attn4 = SpatialCrossAttention(img_ch[4], topo_ch[4], embed_dim=128, num_heads=4)
        self.cross_attn5 = SpatialCrossAttention(img_ch[5], topo_ch[5], embed_dim=128, num_heads=4)

        if cfg.decoder_v2:
            # MiT emits nothing at H/2, so the original decoder had an empty
            # skip there and upsampled with nearest-neighbour. Together those
            # drew a 2 m / 4 m lattice into the output. A learned stride-2
            # stem fills the skip and bilinear upsampling removes the lattice.
            stem_ch = 48
            self.stem = nn.Sequential(
                nn.Conv2d(opt_ch, stem_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_ch), nn.SiLU(inplace=True),
                nn.Conv2d(stem_ch, stem_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(stem_ch), nn.SiLU(inplace=True))
            enc_ch = [img_ch[0], stem_ch, *img_ch[2:]]
            dec_ch, hf_ch, head_ch, interp = (256, 128, 64, 64, 48), 48, 32, "bilinear"
        else:
            enc_ch, dec_ch, hf_ch, head_ch, interp = img_ch, (256, 128, 64, 32, 16), 16, 16, "nearest"

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_ch, decoder_channels=dec_ch, n_blocks=5,
            interpolation_mode=interp)
        self.hf_extractor = HighFreqFilterBank(in_channels=opt_ch, out_channels=hf_ch)
        self.synthesis_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1] + hf_ch, head_ch, 3, padding=1),
            nn.BatchNorm2d(head_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_ch, 1, 1))

    def inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the optical and terrain input stacks from a batch dict."""
        optical = batch["rgb"]
        if self.cfg.extra_channels:
            optical = torch.cat([optical, batch["extra_img"], batch["extra_avail"]], dim=1)
        topo = [batch["dem_z"], batch["slope_mag"], batch["hillshade"]]
        if self.cfg.coast_inputs:
            optical = torch.cat([optical, batch["land"].unsqueeze(1).to(optical.dtype)], dim=1)
            topo += [batch["land"], batch["dem_abs"]]
        return optical, torch.stack(topo, dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        """Return the elevation residual, shape (B, H, W), zero outside the valid mask."""
        optical, topo = self.inputs(batch)
        fi = self.enc_rgb(optical)
        ft = self.enc_topo(topo)
        skips = [
            fi[0],
            self.stem(optical) if self.cfg.decoder_v2 else fi[1],
            self.fuse2(fi[2], ft[2]),
            self.fuse3(fi[3], ft[3]),
            self.cross_attn4(fi[4], ft[4]),
            self.cross_attn5(fi[5], ft[5]),
        ]
        dec = self.decoder(skips)
        res = self.synthesis_head(torch.cat([dec, self.hf_extractor(optical)], dim=1)).squeeze(1)
        gate = batch["in_mask"] if self.cfg.coast_inputs else batch["mask"]
        return res * gate


# Buffers present in checkpoints trained before the hillshade-rendering loss
# was removed (that loss had weight 0 in every published model).
_UNUSED_KEYS = ("kx", "ky")


def _legacy_model_config(ck: dict) -> dict:
    """ModelConfig fields from a checkpoint written by the original training script."""
    c = ck.get("config", {})
    out = {"extra_channels": int(ck.get("extra_channels", 0))}
    for k in ("rgb_encoder", "topo_encoder", "coast_inputs", "decoder_v2"):
        if c.get(k):
            out[k] = c[k]
    return out


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu",
                    trusted: bool = False) -> tuple[TerrainFusionNet, dict]:
    """Rebuild a model from a training checkpoint. Returns (model, metadata).

    The file is read with ``weights_only=True`` (tensors and plain containers
    only). ``trusted=True`` permits a full unpickle for a checkpoint that holds
    other objects; that can run arbitrary code, so use it only on your own files.
    """
    try:
        ck = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        if not trusted:
            raise ValueError(f"{path} holds objects beyond tensors and plain containers; "
                             "pass trusted=True to unpickle it if you trust its source") from exc
        ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**(ck.get("model_config") or _legacy_model_config(ck)))
    model = TerrainFusionNet(cfg, encoder_weights=None)
    state = {k: v for k, v in ck["state_dict"].items() if k not in _UNUSED_KEYS}
    model.load_state_dict(state, strict=True)
    return model.eval(), {k: v for k, v in ck.items() if k != "state_dict"}


def save_checkpoint(path: str | Path, model: TerrainFusionNet, **meta) -> None:
    torch.save({"model_config": asdict(model.cfg), "state_dict": model.state_dict(), **meta}, path)
