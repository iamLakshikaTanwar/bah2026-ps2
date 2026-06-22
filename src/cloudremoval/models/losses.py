"""Shared, pure-PyTorch loss functions and the weighted :class:`LossBundle`.

Owns the **frozen loss contract** (``docs/BUILD_PLAN.md`` §3.6). Concrete models
import these; weights are supplied by
:class:`~cloudremoval.config.LossConfig`. Each function returns a **scalar**
``Tensor`` so it can be summed and back-propagated. All implementations are pure
``torch`` (the perceptual loss lazily imports ``torchvision`` and degrades to a
zero/L1 surrogate if unavailable) so the module imports on the CPU-smoke stack.

Conventions
-----------
* Image tensors are ``[B, C, H, W]`` (or ``[C, H, W]``) ``float32``.
* ``mask`` (when given) is ``[B, 1, H, W]`` / ``[*, 1, H, W]`` float in ``{0,1}``
  selecting the pixels that contribute; it broadcasts over channels.
* For LISS-IV, channel order is Green=0, Red=1, NIR=2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from cloudremoval.config import LossConfig
    from cloudremoval.models.base import LossDict, ModelOutput

__all__ = [
    "l1_loss",
    "mse_loss",
    "charbonnier_loss",
    "carl_loss",
    "ssim_loss",
    "sam_loss",
    "perceptual_loss",
    "adversarial_loss",
    "gaussian_nll_loss",
    "total_variation_loss",
    "LossBundle",
]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #
def _masked_mean(per_pixel: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of ``per_pixel`` over (optionally) ``mask``-selected pixels.

    ``per_pixel`` is an elementwise error map; ``mask`` (``[*,1,H,W]``)
    broadcasts over channels. If ``mask`` is ``None`` the plain mean is returned.
    A fully-empty mask yields ``0.0`` (no contribution) rather than NaN.
    """
    if mask is None:
        return per_pixel.mean()
    mask = mask.to(per_pixel.dtype)
    # Broadcast mask over channel dim.
    weighted = per_pixel * mask
    denom = mask.expand_as(per_pixel).sum().clamp_min(_EPS)
    return weighted.sum() / denom


# --------------------------------------------------------------------------- #
# Reconstruction losses
# --------------------------------------------------------------------------- #
def l1_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean absolute error, optionally restricted to ``mask`` pixels."""
    return _masked_mean((pred - target).abs(), mask)


def mse_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared error, optionally restricted to ``mask`` pixels."""
    return _masked_mean((pred - target) ** 2, mask)


def charbonnier_loss(
    pred: Tensor, target: Tensor, mask: Tensor | None = None, eps: float = 1e-3
) -> Tensor:
    """Charbonnier (smooth-L1-like) loss ``sqrt((pred-target)^2 + eps^2)``.

    A robust reconstruction loss popular in restoration transformers.
    """
    per_pixel = torch.sqrt((pred - target) ** 2 + eps * eps)
    return _masked_mean(per_pixel, mask)


def carl_loss(
    pred: Tensor,
    target: Tensor,
    cloud_mask: Tensor,
    clear_weight: float = 0.1,
) -> Tensor:
    """Cloud-Adaptive Regularized L1 loss (DSen2-CR).

    Weights L1 error by ``1.0`` inside cloudy pixels and by ``clear_weight`` in
    clear pixels, so the model focuses on reconstructing occluded regions while
    only lightly regularising already-visible ones.

    Args:
        pred: Predicted image ``[B, C, H, W]``.
        target: Clear target ``[B, C, H, W]``.
        cloud_mask: ``[B, 1, H, W]`` float; ``1.0`` = cloud, ``0.0`` = clear.
        clear_weight: Weight applied to clear (non-cloud) pixels.

    Returns:
        Scalar weighted-L1 loss.
    """
    cloud_mask = cloud_mask.to(pred.dtype)
    weight = cloud_mask + clear_weight * (1.0 - cloud_mask)  # [B,1,H,W]
    per_pixel = (pred - target).abs() * weight  # broadcast over channels
    denom = weight.expand_as(per_pixel).sum().clamp_min(_EPS)
    return per_pixel.sum() / denom


# --------------------------------------------------------------------------- #
# SSIM (pure torch, used for both the loss and as a building block)
# --------------------------------------------------------------------------- #
def _gaussian_window(window_size: int, sigma: float, device, dtype) -> Tensor:
    """1-D Gaussian kernel normalised to sum 1."""
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    return g / g.sum()


def _ssim_map(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> Tensor:
    """Per-pixel SSIM map for ``[B, C, H, W]`` inputs (depthwise Gaussian)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    channels = pred.shape[1]
    k1d = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    k2d = (k1d[:, None] @ k1d[None, :]).expand(channels, 1, window_size, window_size)
    pad = window_size // 2

    def filt(x: Tensor) -> Tensor:
        return F.conv2d(x, k2d, padding=pad, groups=channels)

    mu_x = filt(pred)
    mu_y = filt(target)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(target * target) - mu_y2
    sigma_xy = filt(pred * target) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return ssim_n / (ssim_d + _EPS)


def ssim_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Structural-dissimilarity loss ``1 - mean(SSIM)`` (pure torch)."""
    return 1.0 - _ssim_map(pred, target).mean()


# --------------------------------------------------------------------------- #
# Spectral losses
# --------------------------------------------------------------------------- #
def sam_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean Spectral Angle Mapper loss in **radians** (lower is better).

    Computes, per pixel, the angle between predicted and target spectral vectors
    across channels, then averages over (optionally masked) pixels.

    Args:
        pred: ``[B, C, H, W]`` or ``[C, H, W]``.
        target: Same shape as ``pred``.
        mask: Optional ``[*, 1, H, W]`` pixel selector.

    Returns:
        Scalar mean spectral angle (radians).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(0)
    dot = (pred * target).sum(dim=1)  # [B, H, W]
    denom = pred.norm(dim=1) * target.norm(dim=1) + _EPS
    cos = (dot / denom).clamp(-1.0 + _EPS, 1.0 - _EPS)
    angle = torch.acos(cos).unsqueeze(1)  # [B, 1, H, W]
    return _masked_mean(angle, mask)


def perceptual_loss(pred: Tensor, target: Tensor) -> Tensor:
    """VGG-feature perceptual loss on an RGB render (lazy ``torchvision``).

    The 3 LISS-IV bands (G/R/NIR) are mapped to RGB channels as a stand-in. If
    ``torchvision`` is unavailable (CPU-smoke minimal stack), this gracefully
    falls back to a plain L1 loss so the call never crashes — the contract only
    promises a scalar.

    Args:
        pred: ``[B, C, H, W]`` with ``C >= 3``.
        target: Same shape as ``pred``.

    Returns:
        Scalar perceptual (or fallback L1) loss.
    """
    try:  # lazy, optional
        extractor = _get_vgg_extractor(pred.device)
    except Exception:  # noqa: BLE001 - any failure -> graceful fallback
        return l1_loss(pred, target)

    def to_rgb(x: Tensor) -> Tensor:
        x3 = x[:, :3] if x.shape[1] >= 3 else x.repeat(1, 3, 1, 1)[:, :3]
        return x3.clamp(0.0, 1.0)

    fp = extractor(to_rgb(pred))
    ft = extractor(to_rgb(target))
    return F.l1_loss(fp, ft)


_VGG_CACHE: dict[str, nn.Module] = {}


def _get_vgg_extractor(device: Any) -> nn.Module:
    """Build (and cache) a frozen VGG16 feature extractor. Raises on failure."""
    key = str(device)
    if key in _VGG_CACHE:
        return _VGG_CACHE[key]
    from torchvision import models  # type: ignore  # lazy import

    vgg = models.vgg16(weights=None)  # no network download; random init is fine
    extractor = nn.Sequential(*list(vgg.features.children())[:16]).to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    _VGG_CACHE[key] = extractor
    return extractor


# --------------------------------------------------------------------------- #
# Adversarial & uncertainty
# --------------------------------------------------------------------------- #
def adversarial_loss(disc_logits: Tensor, is_real: bool, mode: str = "bce") -> Tensor:
    """Adversarial loss for a discriminator output.

    Args:
        disc_logits: Raw discriminator logits (any shape; reduced by mean).
        is_real: Target label — ``True`` to push logits toward "real".
        mode: ``"bce"`` (binary cross-entropy with logits) or ``"hinge"``.

    Returns:
        Scalar adversarial loss.
    """
    if mode == "hinge":
        if is_real:
            return F.relu(1.0 - disc_logits).mean()
        return F.relu(1.0 + disc_logits).mean()
    target = torch.ones_like(disc_logits) if is_real else torch.zeros_like(disc_logits)
    return F.binary_cross_entropy_with_logits(disc_logits, target)


def gaussian_nll_loss(
    pred: Tensor,
    target: Tensor,
    log_var: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Heteroscedastic Gaussian negative log-likelihood (aleatoric uncertainty).

    ``0.5 * (exp(-log_var) * (pred-target)^2 + log_var)`` per element, averaged
    over (optionally masked) pixels. ``log_var`` may be ``[B, 1, H, W]`` (shared
    across channels) or ``[B, C, H, W]``.

    Args:
        pred: Predicted mean ``[B, C, H, W]``.
        target: Target ``[B, C, H, W]``.
        log_var: Predicted log-variance, broadcastable to ``pred``.
        mask: Optional pixel selector.

    Returns:
        Scalar NLL loss.
    """
    log_var = log_var.expand_as(pred) if log_var.shape[1] == 1 else log_var
    per_pixel = 0.5 * (torch.exp(-log_var) * (pred - target) ** 2 + log_var)
    return _masked_mean(per_pixel, mask)


def total_variation_loss(pred: Tensor) -> Tensor:
    """Anisotropic total-variation regulariser (encourages spatial smoothness)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    dh = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
    dw = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    return dh + dw


# --------------------------------------------------------------------------- #
# Weighted bundle
# --------------------------------------------------------------------------- #
class LossBundle:
    """Sum weighted loss terms from a :class:`LossConfig` into a :data:`LossDict`.

    This is the default object a model's ``loss()`` can delegate to. Only terms
    with a non-zero weight are computed. The returned dict always contains
    ``"total"`` (the weighted sum) plus each active term under its own key.

    Terms read from the ``SAMPLE``/``ModelOutput``:
        * ``l1``         — L1 on ``optical_clear``.
        * ``carl``       — cloud-weighted L1 using ``cloud_mask``.
        * ``ssim``       — ``1 - SSIM``.
        * ``sam``        — spectral angle (radians).
        * ``perceptual`` — VGG features (lazy; falls back to L1).
        * ``nll``        — Gaussian NLL using ``output.uncertainty`` as log-var.
        * ``tv``         — total variation on the reconstruction.

    ``adversarial`` is intentionally **not** computed here — GAN models own their
    discriminator wiring and add the adversarial term themselves.
    """

    def __init__(self, cfg: LossConfig) -> None:
        """Store loss weights.

        Args:
            cfg: The ``train.loss`` section of the global config.
        """
        self.cfg = cfg

    def __call__(self, sample: dict, output: ModelOutput) -> LossDict:
        """Compute the weighted loss dict for one batch.

        Args:
            sample: The ``SAMPLE`` dict (must contain ``optical_clear``; may
                contain ``cloud_mask``).
            output: The model's :class:`ModelOutput`.

        Returns:
            A :data:`LossDict` with ``"total"`` and each active term.
        """
        cfg = self.cfg
        pred = output.reconstruction
        target = sample["optical_clear"]
        cloud_mask = sample.get("cloud_mask")

        losses: dict[str, Tensor] = {}
        total = pred.new_zeros(())

        if cfg.l1 > 0:
            losses["l1"] = l1_loss(pred, target)
            total = total + cfg.l1 * losses["l1"]
        if cfg.carl > 0 and cloud_mask is not None:
            losses["carl"] = carl_loss(pred, target, cloud_mask)
            total = total + cfg.carl * losses["carl"]
        if cfg.ssim > 0:
            losses["ssim"] = ssim_loss(pred, target)
            total = total + cfg.ssim * losses["ssim"]
        if cfg.sam > 0:
            losses["sam"] = sam_loss(pred, target)
            total = total + cfg.sam * losses["sam"]
        if cfg.perceptual > 0:
            losses["perceptual"] = perceptual_loss(pred, target)
            total = total + cfg.perceptual * losses["perceptual"]
        if cfg.nll > 0 and output.uncertainty is not None:
            # Treat the uncertainty head as log-variance for the NLL term.
            losses["nll"] = gaussian_nll_loss(pred, target, output.uncertainty)
            total = total + cfg.nll * losses["nll"]
        if getattr(cfg, "tv", 0.0) > 0:
            losses["tv"] = total_variation_loss(pred)
            total = total + cfg.tv * losses["tv"]

        # Guarantee at least one term so 'total' is a real graph node even when
        # every configured weight is zero (keeps backward() valid in smoke runs).
        if not losses:
            losses["l1"] = l1_loss(pred, target)
            total = total + losses["l1"]

        losses["total"] = total
        return losses
