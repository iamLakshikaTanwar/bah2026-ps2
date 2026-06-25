"""Uncertainty-aware cloud removal — ``@register_model("uncertainty")``.

Wraps a reconstruction backbone (a small U-Net feature trunk, reusing
:class:`cloudremoval.models.unet._UNetBackbone`) with **two heads**: a mean head
predicting the clear reconstruction and a **log-variance head** predicting
per-pixel aleatoric uncertainty. The model is trained with a heteroscedastic
**Gaussian negative log-likelihood** (``gaussian_nll_loss``) so it learns *where*
its reconstruction is unreliable — the UnCRtainTS recipe (Ebel et al., CVPRW
2023), which turns a research demo into a credible operational product by
shipping a per-pixel confidence map alongside the reconstruction.

``ModelOutput.uncertainty`` is populated with the predicted **variance**
(``exp(log_var)``); ``aux["log_var"]`` carries the raw log-variance used by the
NLL loss.

The backbone is configurable via ``backbone`` (only ``"unet"`` is bundled here to
stay dependency-free on the CPU-smoke path; other registered models could be
composed by name in a richer build). Pure ``torch``.

Discovery note: imported by name (``uncertainty``) from
``cloudremoval.models.__init__`` (B0); ``@register_model`` runs on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.losses import gaussian_nll_loss, sam_loss
from cloudremoval.models.registry import register_model
from cloudremoval.models.unet import _UNetBackbone
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import ModelConfig

__all__ = ["UncertaintyModel"]

_log = get_logger(__name__)


@register_model("uncertainty")
class UncertaintyModel(BaseCloudRemovalModel):
    """Aleatoric-uncertainty cloud-removal model (mean + log-variance heads).

    Config fields (``getattr`` with defaults):

    * ``in_channels`` (3) / ``out_channels`` (3) — optical bands.
    * ``base_channels`` (32) / ``depth`` (3) — backbone capacity.
    * ``use_cloud_mask`` (True) — concat the cloud mask to the input.
    * ``backbone`` ("unet") — feature trunk (only ``"unet"`` bundled).
    * ``uncertainty_per_band`` (False) — if ``True`` predict ``out_channels``
      log-variance channels, else a single shared-across-channels map.
    * ``logvar_min`` (-7.0) / ``logvar_max`` (3.0) — clamp range for stability.

    The mean head predicts a long-skip residual over ``optical_cloudy``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.in_channels = int(getattr(cfg, "in_channels", 3))
        self.out_channels = int(getattr(cfg, "out_channels", 3))
        base_channels = int(getattr(cfg, "base_channels", 32))
        depth = int(getattr(cfg, "depth", 3))
        self.use_cloud_mask = bool(getattr(cfg, "use_cloud_mask", True))
        self.per_band = bool(getattr(cfg, "uncertainty_per_band", False))
        self.logvar_min = float(getattr(cfg, "logvar_min", -7.0))
        self.logvar_max = float(getattr(cfg, "logvar_max", 3.0))
        self.lambda_sam = float(getattr(cfg, "lambda_sam", 0.1))

        backbone = str(getattr(cfg, "backbone", "unet"))
        if backbone != "unet":  # only the bundled trunk is supported on CPU-smoke
            _log.warning("uncertainty: backbone '%s' not bundled; using 'unet'.", backbone)

        extra_in = 1 if self.use_cloud_mask else 0
        trunk_out = max(base_channels, self.out_channels * 2)
        # Trunk produces a shared feature map; two light heads branch off it.
        self.trunk = _UNetBackbone(
            in_ch=self.in_channels + extra_in,
            out_ch=trunk_out,
            base_channels=base_channels,
            depth=depth,
        )
        self.mean_head = nn.Conv2d(trunk_out, self.out_channels, kernel_size=3, padding=1)
        logvar_ch = self.out_channels if self.per_band else 1
        self.logvar_head = nn.Conv2d(trunk_out, logvar_ch, kernel_size=3, padding=1)
        _log.debug(
            "UncertaintyModel: in=%d(+%d) trunk_out=%d out=%d per_band=%s",
            self.in_channels,
            extra_in,
            trunk_out,
            self.out_channels,
            self.per_band,
        )

    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Predict reconstruction (mean) and per-pixel variance (uncertainty)."""
        x = sample["optical_cloudy"]
        inputs = [x]
        if self.use_cloud_mask:
            cloud_mask = sample.get("cloud_mask")
            if cloud_mask is None:
                cloud_mask = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            inputs.append(cloud_mask.to(x.dtype))

        feat = self.trunk(torch.cat(inputs, dim=1))
        residual = self.mean_head(feat)
        reconstruction = x + residual

        log_var = self.logvar_head(feat).clamp(self.logvar_min, self.logvar_max)
        variance = torch.exp(log_var)
        return ModelOutput(
            reconstruction=reconstruction,
            uncertainty=variance,
            aux={"log_var": log_var},
        )

    def loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """Gaussian NLL (heteroscedastic) + a light SAM spectral term."""
        pred = output.reconstruction
        target = sample["optical_clear"]
        log_var = output.aux["log_var"]

        losses: dict[str, Tensor] = {}
        losses["nll"] = gaussian_nll_loss(pred, target, log_var)
        total = losses["nll"]
        if self.lambda_sam > 0:
            losses["sam"] = sam_loss(pred, target)
            total = total + self.lambda_sam * losses["sam"]
        losses["total"] = total
        return losses
