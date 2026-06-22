"""Clean U-Net CNN baseline — ``@register_model("unet")``.

A straightforward encoder-decoder convolutional network with skip connections,
``GroupNorm`` + ``SiLU`` activations and bilinear up-sampling. It consumes only
the cloudy optical image (optionally with the cloud mask concatenated as an extra
input channel) and predicts the clear optical reconstruction directly.

This is the **hallucination / control yardstick** for the project: an optical-only
model with no auxiliary conditioning (no SAR, DEM or temporal references). Any
gain a fusion / diffusion / GAN model shows over this baseline is the value added
by its extra machinery, and any *spectral drift* this model exhibits bounds the
"how much can a plain CNN invent" question.

Architecture lineage: the canonical U-Net encoder-decoder (Ronneberger et al.,
MICCAI 2015) modernised with the residual-block + GroupNorm/SiLU conventions used
by restoration networks (e.g. the DDPM / DSen2-CR families). Pure ``torch`` so it
imports and runs on the CPU-smoke stack.

Discovery note: ``cloudremoval.models.__init__`` (owned by B0) imports this module
by name (``unet``), so the ``@register_model`` side effect runs automatically when
``cloudremoval.models`` is imported. No manual import hook is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.losses import carl_loss, charbonnier_loss, sam_loss, ssim_loss
from cloudremoval.models.registry import register_model
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import ModelConfig

__all__ = ["UNetModel", "ConvBlock", "DoubleConv"]

_log = get_logger(__name__)


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Pick a ``GroupNorm`` group count that divides ``channels``.

    Falls back gracefully for tiny channel counts (CPU-smoke configs use as few
    as 8 channels) so ``GroupNorm`` never raises.
    """
    for g in (max_groups, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two stacked :class:`ConvBlock`\\ s with a residual 1x1 shortcut."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.block2(self.block1(x)) + self.skip(x)


class _UNetBackbone(nn.Module):
    """Reusable U-Net encoder-decoder mapping ``in_ch -> out_ch`` channels.

    Exposed separately so other models (e.g. the uncertainty wrapper) can embed a
    small U-Net as a feature trunk without depending on the
    :class:`BaseCloudRemovalModel` wiring.
    """

    def __init__(self, in_ch: int, out_ch: int, base_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        depth = max(1, int(depth))
        chans = [base_channels * (2**i) for i in range(depth + 1)]

        self.in_conv = DoubleConv(in_ch, chans[0])

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            self.pools.append(nn.MaxPool2d(2))
            self.downs.append(DoubleConv(chans[i], chans[i + 1]))

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(nn.Conv2d(chans[i], chans[i - 1], kernel_size=1))
            # After up + skip concat we have 2 * chans[i-1] channels.
            self.up_convs.append(DoubleConv(chans[i - 1] * 2, chans[i - 1]))

        self.out_conv = nn.Conv2d(chans[0], out_ch, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        h = self.in_conv(x)
        skips.append(h)
        for pool, down in zip(self.pools, self.downs):
            h = down(pool(h))
            skips.append(h)

        # skips[-1] is the bottleneck; decode upward.
        h = skips[-1]
        for idx, (up, up_conv) in enumerate(zip(self.ups, self.up_convs)):
            skip = skips[-(idx + 2)]
            h = up(h)
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(torch.cat([h, skip], dim=1))
        return self.out_conv(h)


@register_model("unet")
class UNetModel(BaseCloudRemovalModel):
    """Optical-only U-Net cloud-removal baseline.

    Config fields (read via ``getattr`` with defaults so the permissive
    :class:`~cloudremoval.config.ModelConfig` works unchanged):

    * ``in_channels`` (default 3) — optical bands (G, R, NIR).
    * ``out_channels`` (default 3) — reconstructed bands.
    * ``base_channels`` (default 32) — width of the first encoder stage.
    * ``depth`` (default 3) — number of down/up stages.
    * ``use_cloud_mask`` (default ``True``) — concatenate the cloud mask as an
      extra input channel (a cheap, informative hint for an optical-only model).

    The reconstruction is a long-skip residual added to ``optical_cloudy`` so the
    network only has to predict the *correction*, which stabilises training and
    preserves already-clear pixels.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.in_channels = int(getattr(cfg, "in_channels", 3))
        self.out_channels = int(getattr(cfg, "out_channels", 3))
        base_channels = int(getattr(cfg, "base_channels", 32))
        depth = int(getattr(cfg, "depth", 3))
        self.use_cloud_mask = bool(getattr(cfg, "use_cloud_mask", True))

        extra_in = 1 if self.use_cloud_mask else 0
        self.backbone = _UNetBackbone(
            in_ch=self.in_channels + extra_in,
            out_ch=self.out_channels,
            base_channels=base_channels,
            depth=depth,
        )
        _log.debug(
            "UNetModel: in=%d(+%d mask) out=%d base=%d depth=%d",
            self.in_channels,
            extra_in,
            self.out_channels,
            base_channels,
            depth,
        )

    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Predict the clear optical image from the cloudy optical (+ mask)."""
        x = sample["optical_cloudy"]
        inputs = [x]
        if self.use_cloud_mask:
            cloud_mask = sample.get("cloud_mask")
            if cloud_mask is None:
                cloud_mask = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            inputs.append(cloud_mask.to(x.dtype))
        residual = self.backbone(torch.cat(inputs, dim=1))
        reconstruction = x + residual
        return ModelOutput(reconstruction=reconstruction)

    def loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """Charbonnier + SSIM + SAM (+ CARL when a cloud mask is present).

        SSIM enforces structural similarity, SAM penalises spectral-angle drift
        (guarding NDVI/NIR fidelity), and the optional CARL term focuses error on
        the cloud-occluded pixels (DSen2-CR-style mask weighting).
        """
        pred = output.reconstruction
        target = sample["optical_clear"]
        cloud_mask = sample.get("cloud_mask")

        losses: dict[str, Tensor] = {}
        losses["charbonnier"] = charbonnier_loss(pred, target)
        losses["ssim"] = ssim_loss(pred, target)
        losses["sam"] = sam_loss(pred, target)
        total = losses["charbonnier"] + 0.1 * losses["ssim"] + 0.1 * losses["sam"]

        if cloud_mask is not None:
            losses["carl"] = carl_loss(pred, target, cloud_mask)
            total = total + 0.5 * losses["carl"]

        losses["total"] = total
        return losses
