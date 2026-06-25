"""SpA-GAN / Pix2Pix-style adversarial cloud removal — ``@register_model("spagan")``.

Generator follows the **SpA-GAN** recipe (Pan, arXiv 2009.13015): a U-Net-like
encoder-decoder augmented with a **Spatial Attention Network (SPANet-style)
block** so the network focuses on cloud-occluded pixels. The discriminator is a
**PatchGAN** (Isola et al., Pix2Pix, CVPR 2017) classifying local patches as
real/fake. Training combines an adversarial term with a strong pixel L1 /
Charbonnier term, an SSIM structural term and an optional perceptual term — the
adversarial signal supplies the fine 5.8 m texture that an L1-only objective
blurs.

GAN integration with the trainer (BUILD_PLAN §3.1 optional hooks, detected via
``hasattr``):

* :meth:`generator_parameters` / :meth:`discriminator_parameters` — disjoint,
  non-empty parameter groups for the two optimizers.
* :meth:`discriminator_loss` — the D step (real ``optical_clear`` vs. detached
  fake reconstruction).
* :meth:`loss` — the G step (adversarial + L1 + SSIM + optional perceptual).

:meth:`forward` returns the generator's reconstruction and stashes the
discriminator logits on the fake sample in ``aux`` (useful for logging /
debugging). Pure ``torch``; CPU-smoke runnable.

Discovery note: imported by name (``gan_spagan``) from
``cloudremoval.models.__init__`` (B0), so ``@register_model`` runs on import.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.losses import (
    adversarial_loss,
    charbonnier_loss,
    perceptual_loss,
    ssim_loss,
)
from cloudremoval.models.registry import register_model
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import ModelConfig

__all__ = ["SpAGANModel", "SpatialAttentionBlock", "PatchGANDiscriminator"]

_log = get_logger(__name__)


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """GroupNorm group count that divides ``channels`` (tiny-config safe)."""
    for g in (max_groups, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SpatialAttentionBlock(nn.Module):
    """SPANet-style spatial-attention residual block.

    Computes a single-channel spatial attention map from channel-pooled
    statistics (mean + max over channels, a CBAM-style spatial-attention recipe)
    and multiplies the residual branch by it, so the block emphasises
    (cloud-occluded) regions the network should rewrite. The attention map is
    also returned for optional visualisation.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.SiLU()
        # 2-channel (mean,max) -> 1-channel spatial attention.
        self.attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.act(self.norm(self.conv1(x)))
        h = self.conv2(h)
        avg_pool = h.mean(dim=1, keepdim=True)
        max_pool = h.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.attn(torch.cat([avg_pool, max_pool], dim=1)))
        return x + h * attn, attn


class _AttnUNetGenerator(nn.Module):
    """Small U-Net generator with spatial-attention blocks at each scale."""

    def __init__(self, in_ch: int, out_ch: int, base_channels: int = 32, depth: int = 2) -> None:
        super().__init__()
        depth = max(1, int(depth))
        chans = [base_channels * (2**i) for i in range(depth + 1)]

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, chans[0], kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(chans[0]), chans[0]),
            nn.SiLU(),
        )
        self.enc_attn = nn.ModuleList([SpatialAttentionBlock(chans[0])])
        self.downs = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(_num_groups(chans[i + 1]), chans[i + 1]),
                    nn.SiLU(),
                )
            )
            self.down_attn.append(SpatialAttentionBlock(chans[i + 1]))

        self.ups = nn.ModuleList()
        self.up_merge = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(chans[i], chans[i - 1], kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(_num_groups(chans[i - 1]), chans[i - 1]),
                    nn.SiLU(),
                )
            )
            self.up_merge.append(
                nn.Sequential(
                    nn.Conv2d(chans[i - 1] * 2, chans[i - 1], kernel_size=3, padding=1),
                    nn.GroupNorm(_num_groups(chans[i - 1]), chans[i - 1]),
                    nn.SiLU(),
                )
            )
        self.out_conv = nn.Conv2d(chans[0], out_ch, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.in_conv(x)
        h, attn0 = self.enc_attn[0](h)
        skips = [h]
        last_attn = attn0
        for down, attn in zip(self.downs, self.down_attn, strict=True):
            h = down(h)
            h, last_attn = attn(h)
            skips.append(h)

        h = skips[-1]
        for idx, (up, merge) in enumerate(zip(self.ups, self.up_merge, strict=True)):
            skip = skips[-(idx + 2)]
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = merge(torch.cat([h, skip], dim=1))
        return self.out_conv(h), last_attn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Pix2Pix).

    Conditioned: takes the (input, output) pair channel-concatenated, classifying
    overlapping patches as real/fake. Returns raw logits (no sigmoid) so it pairs
    with ``adversarial_loss(..., mode="bce")`` (BCE-with-logits) or hinge.
    """

    def __init__(self, in_ch: int, base_channels: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for _ in range(1, n_layers):
            nxt = min(ch * 2, base_channels * 8)
            layers += [
                nn.Conv2d(ch, nxt, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(_num_groups(nxt), nxt),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = nxt
        layers += [nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


@register_model("spagan")
class SpAGANModel(BaseCloudRemovalModel):
    """Spatial-attention GAN cloud-removal model with a PatchGAN critic.

    Config fields (``getattr`` with defaults):

    * ``in_channels`` (3) / ``out_channels`` (3) — optical bands.
    * ``base_channels`` (32) / ``depth`` (2) — generator capacity.
    * ``use_cloud_mask`` (True) — concat cloud mask to the generator input.
    * ``gan_lambda_l1`` (100.0) — weight on the pixel L1/Charbonnier term.
    * ``gan_lambda_perceptual`` (0.0) — weight on the perceptual term.
    * ``gan_lambda_ssim`` (1.0) — weight on the SSIM term.
    * ``gan_mode`` ("bce") — ``"bce"`` or ``"hinge"`` adversarial loss.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.in_channels = int(getattr(cfg, "in_channels", 3))
        self.out_channels = int(getattr(cfg, "out_channels", 3))
        base_channels = int(getattr(cfg, "base_channels", 32))
        depth = int(getattr(cfg, "depth", 2))
        self.use_cloud_mask = bool(getattr(cfg, "use_cloud_mask", True))
        self.lambda_l1 = float(getattr(cfg, "gan_lambda_l1", 100.0))
        self.lambda_perceptual = float(getattr(cfg, "gan_lambda_perceptual", 0.0))
        self.lambda_ssim = float(getattr(cfg, "gan_lambda_ssim", 1.0))
        self.lambda_adv = float(getattr(cfg, "gan_lambda_adv", 1.0))
        self.gan_mode = str(getattr(cfg, "gan_mode", "bce"))

        gen_in = self.in_channels + (1 if self.use_cloud_mask else 0)
        self.generator = _AttnUNetGenerator(
            in_ch=gen_in,
            out_ch=self.out_channels,
            base_channels=base_channels,
            depth=depth,
        )
        # PatchGAN sees the conditional input (cloudy optical) + the target/fake.
        self.discriminator = PatchGANDiscriminator(
            in_ch=self.in_channels + self.out_channels,
            base_channels=base_channels,
        )
        _log.debug(
            "SpAGANModel: gen_in=%d out=%d base=%d depth=%d lambda_l1=%.1f mode=%s",
            gen_in,
            self.out_channels,
            base_channels,
            depth,
            self.lambda_l1,
            self.gan_mode,
        )

    # ------------------------------------------------------------------ #
    # Core forward
    # ------------------------------------------------------------------ #
    def _gen_input(self, sample: dict[str, Any]) -> Tensor:
        x = sample["optical_cloudy"]
        if not self.use_cloud_mask:
            return x
        cloud_mask = sample.get("cloud_mask")
        if cloud_mask is None:
            cloud_mask = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, cloud_mask.to(x.dtype)], dim=1)

    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Run the generator; stash the critic's logits on the fake in ``aux``."""
        x = sample["optical_cloudy"]
        recon, attn = self.generator(self._gen_input(sample))
        aux: dict[str, Tensor] = {"attention": attn}
        # Discriminator logits on the fake pair (kept in graph for the G step).
        aux["disc_fake"] = self.discriminator(torch.cat([x, recon], dim=1))
        return ModelOutput(reconstruction=recon, aux=aux)

    # ------------------------------------------------------------------ #
    # GAN hooks (probed via hasattr by B3's trainer)
    # ------------------------------------------------------------------ #
    def generator_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the generator (and nothing from the discriminator)."""
        return self.generator.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the PatchGAN discriminator only."""
        return self.discriminator.parameters()

    def discriminator_loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """Discriminator step: push real toward 1, detached fake toward 0."""
        x = sample["optical_cloudy"]
        target = sample["optical_clear"]
        fake = output.reconstruction.detach()

        logits_real = self.discriminator(torch.cat([x, target], dim=1))
        logits_fake = self.discriminator(torch.cat([x, fake], dim=1))
        loss_real = adversarial_loss(logits_real, is_real=True, mode=self.gan_mode)
        loss_fake = adversarial_loss(logits_fake, is_real=False, mode=self.gan_mode)
        d_total = 0.5 * (loss_real + loss_fake)
        return {"d_real": loss_real, "d_fake": loss_fake, "total": d_total}

    def loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """Generator step: adversarial (fool D) + L1/Charbonnier + SSIM (+ perc.)."""
        x = sample["optical_cloudy"]
        pred = output.reconstruction
        target = sample["optical_clear"]

        # Adversarial term: prefer the logits already computed in forward().
        logits_fake = output.aux.get("disc_fake")
        if logits_fake is None:
            logits_fake = self.discriminator(torch.cat([x, pred], dim=1))
        g_adv = adversarial_loss(logits_fake, is_real=True, mode=self.gan_mode)

        losses: dict[str, Tensor] = {}
        losses["g_adv"] = g_adv
        losses["l1"] = charbonnier_loss(pred, target)
        total = self.lambda_adv * g_adv + self.lambda_l1 * losses["l1"]

        if self.lambda_ssim > 0:
            losses["ssim"] = ssim_loss(pred, target)
            total = total + self.lambda_ssim * losses["ssim"]
        if self.lambda_perceptual > 0:
            losses["perceptual"] = perceptual_loss(pred, target)
            total = total + self.lambda_perceptual * losses["perceptual"]

        losses["total"] = total
        return losses
