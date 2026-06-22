"""Conditional DDPM cloud-removal with SR3 conditioning — ``@register_model("diffusion")``.

A pixel-space conditional denoising diffusion probabilistic model (DDPM; Ho et
al., NeurIPS 2020) with **SR3-style concatenation conditioning** (Saharia et al.,
TPAMI 2022): the cloudy optical image (and optionally SAR) is concatenated to the
noisy target as the input to a time-conditioned UNet that predicts the added
noise (``epsilon``-prediction). Training minimises the ``L_simple`` MSE between
predicted and true noise. Inference overrides :meth:`predict` to run **DDIM**
(Song et al., ICLR 2021) deterministic sampling with a small step budget, so it
finishes quickly on CPU at 64x64.

Design notes / extension hooks (documented for the build, not implemented to keep
the CPU path light):

* **EMRDM** (mean-reverting diffusion, Liu et al., CVPR 2025): instead of
  diffusing the clear image to pure Gaussian noise, the forward endpoint reverts
  toward the *cloudy* image, giving a direct cloudy->clear trajectory with better
  spectral fidelity. Toggleable via ``mean_reverting`` — the hook lives in
  :meth:`_q_sample` / :meth:`predict` (see comments). The base (non-mean-reverting)
  path is the default and fully implemented.
* **CM-CR** (consistency-model distillation, *Remote Sensing* 2025): distil this
  diffusion teacher into a few-step consistency student for near-real-time CR.
  The hook is :meth:`predict` — a consistency student would replace the DDIM loop
  with a single (or 2-4 step) consistency evaluation.

Noise schedules: ``linear`` (Ho et al.) and ``cosine`` (Nichol & Dhariwal 2021).

Discovery note: imported by name (``diffusion``) from
``cloudremoval.models.__init__`` (B0); ``@register_model`` runs on import.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.registry import register_model
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import ModelConfig

__all__ = ["DiffusionModel", "TimeUNet"]

_log = get_logger(__name__)


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """GroupNorm group count that divides ``channels`` (tiny-config safe)."""
    for g in (max_groups, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def _make_beta_schedule(kind: str, timesteps: int) -> Tensor:
    """Return ``betas`` ``[timesteps]`` for a ``linear`` or ``cosine`` schedule."""
    if kind == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule.
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-4, 0.999)
    # Linear schedule (Ho et al. 2020), scaled for arbitrary T.
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps).clamp(1e-4, 0.999)


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal timestep embedding ``[B] -> [B, dim]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # zero-pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class _ResBlockT(nn.Module):
    """Residual block with a FiLM-style additive timestep embedding."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class TimeUNet(nn.Module):
    """Compact time-conditioned UNet (epsilon predictor).

    Input is the noisy target concatenated with the conditioning image(s); output
    has ``out_ch`` channels (the predicted noise on the target). Down/up sampling
    via strided / transposed convs; a timestep MLP feeds every residual block.
    """

    def __init__(
        self, in_ch: int, out_ch: int, base_channels: int = 32, depth: int = 2, t_dim: int = 64
    ) -> None:
        super().__init__()
        depth = max(1, int(depth))
        chans = [base_channels * (2**i) for i in range(depth + 1)]
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, chans[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(_ResBlockT(chans[i], chans[i], t_dim))
            self.downs.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1))

        self.mid = _ResBlockT(chans[depth], chans[depth], t_dim)

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(chans[i], chans[i - 1], kernel_size=4, stride=2, padding=1)
            )
            self.up_blocks.append(_ResBlockT(chans[i - 1] * 2, chans[i - 1], t_dim))

        self.out_norm = nn.GroupNorm(_num_groups(chans[0]), chans[0])
        self.out_conv = nn.Conv2d(chans[0], out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.t_mlp(_sinusoidal_embedding(t, self.t_dim))
        h = self.in_conv(x)
        skips: list[Tensor] = []
        for block, down in zip(self.down_blocks, self.downs):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, t_emb)
        for idx, (up, block) in enumerate(zip(self.ups, self.up_blocks)):
            skip = skips[-(idx + 1)]
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1), t_emb)
        return self.out_conv(self.act(self.out_norm(h)))


@register_model("diffusion")
class DiffusionModel(BaseCloudRemovalModel):
    """Conditional DDPM (train) / DDIM (sample) cloud-removal model.

    Config fields (``getattr`` with defaults):

    * ``in_channels`` (3) / ``out_channels`` (3) — optical bands (the target).
    * ``base_channels`` (32) / ``depth`` (2) — UNet capacity.
    * ``timesteps`` (1000) — training diffusion steps.
    * ``sampling_steps`` (defaults to ``sample_steps`` config, else 10) — DDIM
      steps at inference (kept small for CPU).
    * ``schedule`` ("linear") — ``"linear"`` or ``"cosine"`` noise schedule.
    * ``use_sar`` (False) — concatenate SAR to the conditioning.
    * ``mean_reverting`` (False) — EMRDM hook (documented; base path used).

    The model conditions on ``optical_cloudy`` (+ SAR) by SR3 concatenation; the
    UNet predicts the noise added to ``optical_clear``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.in_channels = int(getattr(cfg, "in_channels", 3))
        self.out_channels = int(getattr(cfg, "out_channels", 3))
        base_channels = int(getattr(cfg, "base_channels", 32))
        depth = int(getattr(cfg, "depth", 2))
        self.timesteps = int(getattr(cfg, "timesteps", 1000))
        # Prefer an explicit ``sampling_steps``; fall back to the contract's
        # ``sample_steps`` field; default to a small CPU-friendly value.
        self.sampling_steps = int(
            getattr(cfg, "sampling_steps", getattr(cfg, "sample_steps", 10))
        )
        self.schedule = str(getattr(cfg, "schedule", "linear"))
        self.use_sar = bool(getattr(cfg, "use_sar", False))
        self.mean_reverting = bool(getattr(cfg, "mean_reverting", False))

        cond_ch = self.in_channels + (2 if self.use_sar else 0)
        unet_in = self.out_channels + cond_ch  # noisy target + conditioning
        self.unet = TimeUNet(
            in_ch=unet_in,
            out_ch=self.out_channels,
            base_channels=base_channels,
            depth=depth,
        )

        betas = _make_beta_schedule(self.schedule, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Buffers move with .to(device) and are excluded from the optimizer.
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_acp", torch.sqrt(1.0 - alphas_cumprod), persistent=False
        )
        if self.mean_reverting:
            _log.debug("DiffusionModel: mean_reverting=True requested (EMRDM hook).")
        _log.debug(
            "DiffusionModel: unet_in=%d out=%d base=%d depth=%d T=%d steps=%d sched=%s sar=%s",
            unet_in,
            self.out_channels,
            base_channels,
            depth,
            self.timesteps,
            self.sampling_steps,
            self.schedule,
            self.use_sar,
        )

    # ------------------------------------------------------------------ #
    # Conditioning + forward diffusion
    # ------------------------------------------------------------------ #
    def _condition(self, sample: dict[str, Any]) -> Tensor:
        """Build the SR3 conditioning stack (cloudy optical [+ SAR])."""
        x = sample["optical_cloudy"]
        if not self.use_sar:
            return x
        sar = sample.get("sar")
        if sar is None or sar.numel() == 0:
            sar = x.new_zeros((x.shape[0], 2, x.shape[2], x.shape[3]))
        return torch.cat([x, sar.to(x.dtype)], dim=1)

    def _q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion ``q(x_t | x_0)`` (closed form).

        EMRDM hook: a mean-reverting variant would interpolate the endpoint
        toward the cloudy image rather than zero-mean noise here, e.g.
        ``mean = sqrt_acp*x0 + (1-sqrt_acp)*cloudy``. The base DDPM path below is
        the implemented default.
        """
        sa = self.sqrt_alphas_cumprod[t][:, None, None, None]
        soma = self.sqrt_one_minus_acp[t][:, None, None, None]
        return sa * x0 + soma * noise

    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Training forward: noise the target, predict the noise.

        Returns a :class:`ModelOutput` whose ``reconstruction`` is the one-step
        ``x0`` estimate (handy for logging) and whose ``aux`` carries the
        predicted and true noise for the MSE loss.
        """
        x0 = sample["optical_clear"]
        b = x0.shape[0]
        device = x0.device
        cond = self._condition(sample)

        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self._q_sample(x0, t, noise)
        pred_noise = self.unet(torch.cat([x_t, cond], dim=1), t)

        # One-step x0 estimate from the predicted noise (for visualisation only).
        sa = self.sqrt_alphas_cumprod[t][:, None, None, None]
        soma = self.sqrt_one_minus_acp[t][:, None, None, None]
        x0_hat = (x_t - soma * pred_noise) / sa.clamp_min(1e-8)

        return ModelOutput(
            reconstruction=x0_hat,
            aux={"pred_noise": pred_noise, "true_noise": noise},
        )

    def loss(self, sample: dict[str, Any], output: ModelOutput) -> dict[str, Tensor]:
        """``L_simple``: MSE between predicted and true noise (stored in 'total')."""
        pred_noise = output.aux["pred_noise"]
        true_noise = output.aux["true_noise"]
        mse = F.mse_loss(pred_noise, true_noise)
        return {"mse": mse, "total": mse}

    # ------------------------------------------------------------------ #
    # DDIM sampling (inference)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, sample: dict[str, Any]) -> ModelOutput:
        """DDIM deterministic sampling with ``sampling_steps`` steps.

        Starts from Gaussian noise and iteratively denoises, conditioned on the
        cloudy optical (+ SAR) via concatenation. Few-step DDIM keeps CPU 64x64
        inference fast. Override of the base single-forward ``predict``.

        CM-CR hook: replace this DDIM loop with a consistency-model student for
        single/few-step sampling.
        """
        self.eval()
        cond = self._condition(sample)
        x_shape = (
            sample["optical_cloudy"].shape[0],
            self.out_channels,
            sample["optical_cloudy"].shape[2],
            sample["optical_cloudy"].shape[3],
        )
        device = sample["optical_cloudy"].device
        x_t = torch.randn(x_shape, device=device)

        steps = max(1, min(self.sampling_steps, self.timesteps))
        # Evenly spaced timesteps from T-1 down to 0.
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long()
        acp = self.alphas_cumprod

        for i in range(steps):
            t = ts[i]
            t_batch = t.expand(x_shape[0])
            pred_noise = self.unet(torch.cat([x_t, cond], dim=1), t_batch)

            alpha_t = acp[t]
            sqrt_at = torch.sqrt(alpha_t)
            sqrt_1mat = torch.sqrt(1.0 - alpha_t)
            x0_pred = ((x_t - sqrt_1mat * pred_noise) / sqrt_at.clamp_min(1e-8)).clamp(-1.0, 2.0)

            if i < steps - 1:
                t_next = ts[i + 1]
                alpha_next = acp[t_next]
                # Deterministic DDIM update (eta = 0).
                x_t = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1.0 - alpha_next) * pred_noise
            else:
                x_t = x0_pred

        return ModelOutput(reconstruction=x_t)
