"""Restormer-style transformer with optional SAR cross-attention.

``@register_model("restormer")``.

Implements the **Restormer** restoration core (Zamir et al., CVPR 2022): a
multi-scale U-shaped encoder-decoder of transformer blocks whose attention is
computed over the **channel** dimension (linear in the number of pixels, so it
scales to full tiles without window seams). Two ingredients:

* **MDTA** — Multi-Dconv-head Transposed Attention: depth-wise-conv Q/K/V
  projections, attention over channels (``softmax(Q Kᵀ / τ)``), restoring
  locality cheaply.
* **GDFN** — Gated-Dconv Feed-forward Network: a depth-wise-conv gated FFN that
  controls information flow.

An **optional GLF-CR-style SAR cross-attention branch** (Xu et al., 2022) is
inserted at the bottleneck when ``use_sar`` is set: SAR features provide the
keys/values and the optical features provide the queries, letting the model pull
all-weather structure into cloud-occluded regions. The attention is implemented
with plain ``torch`` tensor ops (a tiny lazy ``einops`` shim is provided but never
required), so the module imports and runs on the CPU-smoke stack.

Loss: Charbonnier + SAM + SSIM (restoration fidelity + spectral safety).

Extension hook: an *Align-CR-style deformable alignment* of SAR-to-optical could
be added before the cross-attention branch (see comment in :class:`_SARFusion`).

Discovery note: imported by name (``transformer_restormer``) from
``cloudremoval.models.__init__`` (B0); ``@register_model`` runs on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.losses import charbonnier_loss, sam_loss, ssim_loss
from cloudremoval.models.registry import register_model
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import ModelConfig

__all__ = ["RestormerModel", "MDTA", "GDFN", "TransformerBlock"]

_log = get_logger(__name__)


def _num_groups(channels: int, max_groups: int = 4) -> int:
    """GroupNorm group count that divides ``channels`` (tiny-config safe)."""
    for g in (max_groups, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for ``[B, C, H, W]`` (bias-free, Restormer-style)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MDTA(nn.Module):
    """Multi-Dconv-head Transposed Attention (Restormer).

    Attention is taken over the channel dimension: Q/K are L2-normalised along
    channels and ``softmax(Q Kᵀ * temperature)`` forms a ``[heads, C', C']``
    attention matrix applied to V. Depth-wise convs in the Q/K/V projection inject
    spatial locality. Cost is linear in ``H*W``.
    """

    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        num_heads = max(1, min(num_heads, channels))
        while channels % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads

        def reshape_heads(t: Tensor) -> Tensor:
            # [B, C, H, W] -> [B, heads, head_dim, H*W]
            return t.reshape(b, self.num_heads, head_dim, h * w)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, heads, hd, hd]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, heads, head_dim, H*W]
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-forward Network (Restormer)."""

    def __init__(self, channels: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """Restormer block: x + MDTA(LN(x)); x + GDFN(LN(x))."""

    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.norm1 = _LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = _LayerNorm2d(channels)
        self.ffn = GDFN(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _SARFusion(nn.Module):
    """GLF-CR-style SAR->optical cross-attention (channel-wise, Restormer-cheap).

    Optical features form the queries; SAR features (projected to the same width)
    form keys/values. Attention is over the channel dimension (consistent with
    MDTA) and the result is residually added to the optical features, so SAR
    structure is pulled in where it helps and ignored where speckle would hurt.

    Extension hook (Align-CR, Xu et al.): insert a deformable-conv alignment of
    ``sar_feat`` to the optical grid *before* the projection below to correct
    residual SAR-optical mis-registration. Omitted here to keep the CPU-smoke
    path dependency-free (``torchvision.ops.deform_conv2d`` would be required).
    """

    def __init__(self, channels: int, sar_in: int = 2, num_heads: int = 1) -> None:
        super().__init__()
        num_heads = max(1, min(num_heads, channels))
        while channels % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sar_proj = nn.Sequential(
            nn.Conv2d(sar_in, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.norm = _LayerNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, optical_feat: Tensor, sar: Tensor) -> Tensor:
        b, c, h, w = optical_feat.shape
        # Resize SAR to the feature grid, then project to feature width.
        if sar.shape[-2:] != (h, w):
            sar = F.interpolate(sar, size=(h, w), mode="bilinear", align_corners=False)
        sar_feat = self.sar_proj(sar)

        q = self.q(self.norm(optical_feat))
        k, v = self.kv(sar_feat).chunk(2, dim=1)
        head_dim = c // self.num_heads

        def reshape_heads(t: Tensor) -> Tensor:
            return t.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(reshape_heads(q), dim=-1)
        k = F.normalize(reshape_heads(k), dim=-1)
        v = reshape_heads(v)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return optical_feat + self.project_out(out)


@register_model("restormer")
class RestormerModel(BaseCloudRemovalModel):
    """Channel-attention transformer restoration core for cloud removal.

    Config fields (``getattr`` with defaults):

    * ``in_channels`` (3) / ``out_channels`` (3) — optical bands.
    * ``base_channels`` (32) — width of level 0.
    * ``depth`` (2) — number of down/up levels (channels double each level).
    * ``num_blocks`` (2) — transformer blocks per level.
    * ``num_heads`` (1) — MDTA heads (clamped to divide the channel count).
    * ``use_sar`` (False) — enable the GLF-CR SAR cross-attention bottleneck.

    Predicts a long-skip residual over ``optical_cloudy``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.in_channels = int(getattr(cfg, "in_channels", 3))
        self.out_channels = int(getattr(cfg, "out_channels", 3))
        base_channels = int(getattr(cfg, "base_channels", 32))
        depth = max(1, int(getattr(cfg, "depth", 2)))
        num_blocks = max(1, int(getattr(cfg, "num_blocks", 2)))
        num_heads = max(1, int(getattr(cfg, "num_heads", 1)))
        self.use_sar = bool(getattr(cfg, "use_sar", False))

        chans = [base_channels * (2**i) for i in range(depth + 1)]
        self.patch_embed = nn.Conv2d(self.in_channels, chans[0], kernel_size=3, padding=1)

        # Encoder: blocks at each level + strided downsample.
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(*[TransformerBlock(chans[i], num_heads) for _ in range(num_blocks)])
            )
            self.downs.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1))

        # Bottleneck.
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(chans[depth], num_heads) for _ in range(num_blocks)]
        )
        self.sar_fusion = (
            _SARFusion(chans[depth], sar_in=2, num_heads=num_heads) if self.use_sar else None
        )

        # Decoder: upsample + concat skip + reduce + blocks.
        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(chans[i], chans[i - 1], kernel_size=4, stride=2, padding=1)
            )
            self.reduces.append(nn.Conv2d(chans[i - 1] * 2, chans[i - 1], kernel_size=1))
            self.decoders.append(
                nn.Sequential(
                    *[TransformerBlock(chans[i - 1], num_heads) for _ in range(num_blocks)]
                )
            )
        self.output = nn.Conv2d(chans[0], self.out_channels, kernel_size=3, padding=1)
        _log.debug(
            "RestormerModel: in=%d out=%d base=%d depth=%d blocks=%d heads=%d use_sar=%s",
            self.in_channels,
            self.out_channels,
            base_channels,
            depth,
            num_blocks,
            num_heads,
            self.use_sar,
        )

    def _get_sar(self, sample: dict[str, Any], ref: Tensor) -> Tensor:
        sar = sample.get("sar")
        if sar is None or sar.numel() == 0:
            sar = ref.new_zeros((ref.shape[0], 2, ref.shape[2], ref.shape[3]))
        return sar.to(ref.dtype)

    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Transformer encoder-decoder with optional SAR cross-attention."""
        x = sample["optical_cloudy"]
        h = self.patch_embed(x)

        skips: list[Tensor] = []
        for enc, down in zip(self.encoders, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)
        if self.sar_fusion is not None:
            h = self.sar_fusion(h, self._get_sar(sample, x))

        for idx, (up, reduce, dec) in enumerate(zip(self.ups, self.reduces, self.decoders)):
            skip = skips[-(idx + 1)]
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(torch.cat([h, skip], dim=1))
            h = dec(h)

        residual = self.output(h)
        return ModelOutput(reconstruction=x + residual)

    def loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """Charbonnier + SAM + SSIM."""
        pred = output.reconstruction
        target = sample["optical_clear"]
        losses: dict[str, Tensor] = {}
        losses["charbonnier"] = charbonnier_loss(pred, target)
        losses["sam"] = sam_loss(pred, target)
        losses["ssim"] = ssim_loss(pred, target)
        losses["total"] = losses["charbonnier"] + 0.1 * losses["sam"] + 0.1 * losses["ssim"]
        return losses
