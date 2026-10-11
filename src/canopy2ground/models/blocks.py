"""Fusion and detail blocks used by :class:`~canopy2ground.models.TerrainFusionNet`."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionBlock(nn.Module):
    """Mix imagery and terrain features, gated by the terrain stream.

    ``out = act(proj(f_img) + sigmoid(W f_topo) * conv([f_img, f_topo]))``. The
    gate lets the terrain encoder decide, per location and channel, how much
    of the joint feature to admit; with the gate closed the block is an
    identity on the imagery path.
    """

    def __init__(self, in_img: int, in_topo: int, out_channels: int):
        super().__init__()
        self.conv_fused = nn.Sequential(
            nn.Conv2d(in_img + in_topo, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.gate_topo = nn.Sequential(nn.Conv2d(in_topo, out_channels, 1), nn.Sigmoid())
        self.proj_rgb = (nn.Conv2d(in_img, out_channels, 1)
                         if in_img != out_channels else nn.Identity())
        self.act = nn.SiLU(inplace=True)

    def forward(self, f_img: torch.Tensor, f_topo: torch.Tensor) -> torch.Tensor:
        fused = self.conv_fused(torch.cat([f_img, f_topo], dim=1))
        return self.act(self.proj_rgb(f_img) + self.gate_topo(f_topo) * fused)


class SpatialCrossAttention(nn.Module):
    """Terrain features query imagery features (global attention over the map).

    Used at the two coarsest scales (16 m and 32 m per token for a 512 m chip),
    where drainage structure spans the whole chip and a convolution's
    receptive field is too local. Residual on the imagery stream.
    """

    def __init__(self, in_img: int, in_topo: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_q = nn.Conv2d(in_topo, embed_dim, 1)
        self.proj_k = nn.Conv2d(in_img, embed_dim, 1)
        self.proj_v = nn.Conv2d(in_img, embed_dim, 1)
        self.proj_out = nn.Sequential(nn.Conv2d(embed_dim, in_img, 1), nn.BatchNorm2d(in_img))
        self.norm_rgb = nn.BatchNorm2d(in_img)
        self.norm_topo = nn.BatchNorm2d(in_topo)
        self.act = nn.SiLU(inplace=True)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return x.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

    def forward(self, f_img: torch.Tensor, f_topo: torch.Tensor) -> torch.Tensor:
        b, _, h, w = f_img.shape
        img_n = self.norm_rgb(f_img)
        q = self._heads(self.proj_q(self.norm_topo(f_topo)))
        k = self._heads(self.proj_k(img_n))
        v = self._heads(self.proj_v(img_n))
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(2, 3).reshape(b, self.embed_dim, h, w)
        return self.act(f_img + self.proj_out(attn))


class HighFreqFilterBank(nn.Module):
    """Full-resolution (1 m) image-detail branch.

    The decoder reaches 1 m only through five 2x upsamplings, so fine photometric
    edges (stream banks, road cuts, terrace risers) are re-injected here from
    the raw optical stack plus a fixed Laplacian response.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3).repeat(1, in_channels, 1, 1))
        self.conv_hf = nn.Sequential(
            nn.Conv2d(in_channels + 1, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(x, self.lap_kernel, padding=1)
        return self.conv_hf(torch.cat([x, lap], dim=1))
