"""Fused attention for the SegFormer (MiT) encoder.

segmentation_models_pytorch writes MiT attention as an explicit softmax(QK^T)V,
which materialises the full attention matrix. This swaps in
``F.scaled_dot_product_attention`` (flash / memory-efficient kernels). With
attention dropout at 0, as in every MiT here, the result is the same up to
floating-point reordering; tests/test_metrics_and_speed.py checks it.
"""
import torch
import torch.nn.functional as F

_MAX_QUERIES = 32768   # larger query counts overflow a CUDA grid dimension; queries are independent, so chunk
_patched = False


def _forward(self, x, height, width):
    b, n, c = x.shape
    h = self.num_heads
    q = self.q(x).reshape(b, n, h, c // h).permute(0, 2, 1, 3)
    if self.sr_ratio > 1:
        x = x.permute(0, 2, 1).reshape(b, c, height, width)
        x = self.norm(self.sr(x).reshape(b, c, -1).permute(0, 2, 1))
    k, v = self.kv(x).reshape(b, -1, 2, h, c // h).permute(2, 0, 3, 1, 4)
    p = self.attn_drop.p if self.training else 0.0
    out = torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + _MAX_QUERIES], k, v,
                                                    dropout_p=p, scale=self.scale)
                     for i in range(0, n, _MAX_QUERIES)], dim=2)
    return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, c)))


def use_fused_mit_attention() -> None:
    """Patch smp's MiT ``Attention.forward`` in place. Idempotent."""
    global _patched
    if _patched:
        return
    import segmentation_models_pytorch.encoders.mix_transformer as mit
    mit.Attention.forward = _forward
    _patched = True
