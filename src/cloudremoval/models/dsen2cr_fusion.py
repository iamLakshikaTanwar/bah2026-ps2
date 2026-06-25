"""DSen2-CR-style SAR-optical fusion ResNet — ``@register_model("dsen2cr")``.

A faithful, lightweight re-implementation of the DSen2-CR architecture (Meraner
et al., *ISPRS J. P&RS* 2020; ISPRS Best-Paper / U.V. Helava Award). The original
fuses Sentinel-1 SAR with cloudy Sentinel-2 through a **deep residual network
with a long global skip connection** and trains with the **Cloud-Adaptive
Regularised L1 (CARL) loss** that weights cloudy vs. clear pixels by a cloud mask.

Here the optical (G/R/NIR), SAR (VV/VH) and optional DEM modalities are fused by
**early channel concatenation**; a stack of residual blocks predicts a *residual*
that is added back to the cloudy optical (the long global skip), so already-clear
pixels are preserved and the network focuses on filling cloud-occluded regions —
exactly where SAR's all-weather structure is the only surface evidence.

This is the **thick-cloud workhorse** of the model zoo: non-adversarial (stable,
low hallucination) and SAR-conditioned. It honours the ``use_sar`` / ``use_dem``
flags and degrades gracefully (zero-filled modality) when an enabled input is
absent in the batch. Pure ``torch``; CPU-smoke runnable.

Discovery note: ``cloudremoval.models.__init__`` (B0) imports this module by name
(``dsen2cr_fusion``), so ``@register_model`` runs on package import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.losses import carl_loss, sam_loss
from cloudremoval.models.registry import register_model
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import ModelConfig

__all__ = ["DSen2CRModel", "ResidualBlock"]

_log = get_logger(__name__)


class ResidualBlock(nn.Module):
    """DSen2-CR residual block: Conv -> ReLU -> Conv, scaled skip add.

    The original scales the residual branch by a constant (``0.1``) before the
    skip add to stabilise very deep stacks; we keep that convention.
    """

    def __init__(self, channels: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv2(self.act(self.conv1(x)))
        return x + self.res_scale * h


@register_model("dsen2cr")
class DSen2CRModel(BaseCloudRemovalModel):
    """Residual SAR-optical fusion network with a long global skip.

    Config fields (``getattr`` with defaults):

    * ``in_channels`` (3) — optical bands.
    * ``out_channels`` (3) — reconstructed bands.
    * ``base_channels`` (32) — feature width of the residual trunk.
    * ``depth`` (3) — used as ``num_res_blocks = max(4, 2*depth)`` for a
      meaningfully deep trunk while staying CPU-cheap at tiny configs.
    * ``num_res_blocks`` (optional) — explicit override of the block count.
    * ``use_sar`` (True) — concatenate the 2-band SAR.
    * ``use_dem`` (False) — concatenate the 1-band DEM.

    The fused input is ``optical_cloudy`` (+ SAR (+ DEM)); the trunk predicts a
    residual added to ``optical_cloudy``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.in_channels = int(getattr(cfg, "in_channels", 3))
        self.out_channels = int(getattr(cfg, "out_channels", 3))
        base_channels = int(getattr(cfg, "base_channels", 32))
        depth = int(getattr(cfg, "depth", 3))
        num_res_blocks = int(getattr(cfg, "num_res_blocks", max(4, 2 * depth)))
        self.use_sar = bool(getattr(cfg, "use_sar", True))
        self.use_dem = bool(getattr(cfg, "use_dem", False))

        fused_in = self.in_channels
        if self.use_sar:
            fused_in += 2  # VV, VH
        if self.use_dem:
            fused_in += 1

        self.head = nn.Sequential(
            nn.Conv2d(fused_in, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_res_blocks)])
        self.body_tail = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.tail = nn.Conv2d(base_channels, self.out_channels, kernel_size=3, padding=1)
        _log.debug(
            "DSen2CRModel: fused_in=%d base=%d blocks=%d use_sar=%s use_dem=%s",
            fused_in,
            base_channels,
            num_res_blocks,
            self.use_sar,
            self.use_dem,
        )

    def _gather_inputs(self, sample: dict[str, Any]) -> Tensor:
        """Channel-concatenate enabled modalities, zero-filling absent ones."""
        x = sample["optical_cloudy"]
        b, _, h, w = x.shape
        parts = [x]
        if self.use_sar:
            sar = sample.get("sar")
            if sar is None or sar.numel() == 0:
                sar = x.new_zeros((b, 2, h, w))
            parts.append(sar.to(x.dtype))
        if self.use_dem:
            dem = sample.get("dem")
            if dem is None or dem.numel() == 0:
                dem = x.new_zeros((b, 1, h, w))
            parts.append(dem.to(x.dtype))
        return torch.cat(parts, dim=1)

    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Fuse modalities and predict a residual over the cloudy optical."""
        optical = sample["optical_cloudy"]
        fused = self._gather_inputs(sample)
        feat = self.head(fused)
        feat = feat + self.body_tail(self.body(feat))  # residual trunk skip
        residual = self.tail(feat)
        reconstruction = optical + residual  # long global skip (DSen2-CR)
        return ModelOutput(reconstruction=reconstruction)

    def loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """CARL (mask-weighted L1, the DSen2-CR objective) + SAM spectral term."""
        pred = output.reconstruction
        target = sample["optical_clear"]
        cloud_mask = sample.get("cloud_mask")
        if cloud_mask is None:
            cloud_mask = pred.new_ones((pred.shape[0], 1, pred.shape[2], pred.shape[3]))

        losses: dict[str, Tensor] = {}
        losses["carl"] = carl_loss(pred, target, cloud_mask)
        losses["sam"] = sam_loss(pred, target)
        losses["total"] = losses["carl"] + 0.1 * losses["sam"]
        return losses
