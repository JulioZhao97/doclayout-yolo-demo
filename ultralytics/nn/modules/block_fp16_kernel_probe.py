"""Speed-only probe: the BF16-autocast block with an fp16 FlashAttention kernel.

Identical to ``HWAttentionBF16Autocast`` except that ``_flash`` feeds the kernel
float16 instead of bfloat16.  Everything else -- the module structure, the
absence of a narrowing cast on the kernel result, the dtype reconciliation
before ``proj`` -- is unchanged, so a speed difference against the bf16-kernel
variant isolates the cost of the kernel dtype alone.

MEASUREMENT ONLY.  Feeding the kernel float16 restores the 65504 overflow that
made the original HWAttention diverge, so this must not be used for a real
training run.
"""

import torch
import torch.nn as nn

from . import block as _block
from .block import C3k, HWBlock, HWC2f
from .block_bf16_autocast import HWAttentionBF16Autocast
from .conv import Conv


class HWAttentionFP16Kernel(HWAttentionBF16Autocast):
    """Same graph as the BF16-autocast attention, fp16 FlashAttention kernel."""

    @staticmethod
    def _flash(q, k, v):
        return _block.flash_attn_func(
            q.contiguous().to(dtype=torch.float16),
            k.contiguous().to(dtype=torch.float16),
            v.contiguous().to(dtype=torch.float16),
        )


class HWBlockFP16Kernel(HWBlock):
    """HWBlock using :class:`HWAttentionFP16Kernel`."""

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        nn.Module.__init__(self)
        self.attn = HWAttentionFP16Kernel(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)


class HWC2fFP16Kernel(HWC2f):
    """HWC2f built from fp16-kernel attention blocks; interface unchanged."""

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        nn.Module.__init__(self)
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(HWBlockFP16Kernel(c_, num_heads, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )
