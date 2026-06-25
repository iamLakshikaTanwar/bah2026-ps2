"""Sample transforms applied consistently across all ``SAMPLE`` modalities.

Geometric augmentations (flips, 90-degree rotations, random crops) must be applied
**identically** to every spatial modality in a ``SAMPLE`` -- ``optical_cloudy``,
``optical_clear``, ``cloud_mask``, ``shadow_mask``, ``sar``, ``dem`` and each
``temporal_refs[t]`` -- or pixels stop corresponding across modalities. This
module builds a single callable ``transform(sample) -> sample`` that draws one set
of random parameters per call and threads them through all tensors.

Spectral / photometric augmentation is kept minimal to protect radiometric
fidelity (``research/05`` §B.4). Albumentations is supported via a lazy import but
is **not** required: the default path is pure torch/NumPy and runs on the
CPU-smoke stack.

``build_transforms(cfg)`` reads the relevant knobs from a :class:`Config` (or a
``DataConfig``) and returns the composed callable. At evaluation/inference time it
returns an identity transform (no augmentation, deterministic).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cloudremoval.config import Config, DataConfig

__all__ = [
    "Compose",
    "RandomFlip",
    "RandomRot90",
    "RandomCrop",
    "Identity",
    "build_transforms",
]

_log = get_logger(__name__)

# Spatial modalities that must receive the *same* geometric transform. ``meta`` and
# zero-length ``temporal_refs`` are handled specially.
_SPATIAL_KEYS = (
    "optical_cloudy",
    "optical_clear",
    "cloud_mask",
    "shadow_mask",
    "sar",
    "dem",
)


# --------------------------------------------------------------------------- #
# Base transform protocol
# --------------------------------------------------------------------------- #
class _SampleTransform:
    """Base class: a callable mapping a ``SAMPLE`` dict to a ``SAMPLE`` dict."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


def _apply_spatial(
    sample: dict[str, Any],
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    """Apply ``fn`` to every spatial tensor in ``sample`` (including temporal refs).

    ``fn`` operates on a ``[C, H, W]`` tensor. ``temporal_refs`` is ``[T, C, H, W]``;
    each frame is transformed and re-stacked (a ``[0, C, H, W]`` empty stack passes
    through untouched). ``meta`` is left as-is.
    """
    out = dict(sample)
    for key in _SPATIAL_KEYS:
        if key in out and isinstance(out[key], torch.Tensor) and out[key].ndim == 3:
            out[key] = fn(out[key])
    refs = out.get("temporal_refs")
    if isinstance(refs, torch.Tensor) and refs.ndim == 4 and refs.shape[0] > 0:
        out["temporal_refs"] = torch.stack([fn(refs[t]) for t in range(refs.shape[0])], dim=0)
    return out


# --------------------------------------------------------------------------- #
# Concrete transforms
# --------------------------------------------------------------------------- #
class Identity(_SampleTransform):
    """No-op transform (used at eval/inference)."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return sample


class RandomFlip(_SampleTransform):
    """Random horizontal/vertical flips applied consistently to all modalities."""

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5) -> None:
        """Args: ``p_horizontal``/``p_vertical`` flip probabilities in ``[0, 1]``."""
        self.p_h = p_horizontal
        self.p_v = p_vertical

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        do_h = torch.rand(()).item() < self.p_h
        do_v = torch.rand(()).item() < self.p_v
        if not (do_h or do_v):
            return sample

        def fn(t: torch.Tensor) -> torch.Tensor:
            if do_h:
                t = torch.flip(t, dims=[-1])
            if do_v:
                t = torch.flip(t, dims=[-2])
            return t

        return _apply_spatial(sample, fn)


class RandomRot90(_SampleTransform):
    """Random 0/90/180/270-degree rotation applied consistently to all modalities."""

    def __init__(self, p: float = 0.5) -> None:
        """Args: ``p`` probability of applying a (non-zero) rotation."""
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()).item() >= self.p:
            return sample
        k = int(torch.randint(1, 4, ()).item())  # 1..3 quarter turns

        def fn(t: torch.Tensor) -> torch.Tensor:
            return torch.rot90(t, k, dims=[-2, -1])

        return _apply_spatial(sample, fn)


class RandomCrop(_SampleTransform):
    """Random square crop to ``size`` applied consistently to all modalities.

    If the sample is already at or below ``size`` the transform is a no-op (the
    synthetic dataset already emits ``tile_size`` tiles, so cropping is optional).
    """

    def __init__(self, size: int) -> None:
        """Args: ``size`` target crop height/width in pixels."""
        self.size = int(size)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        ref = sample.get("optical_cloudy")
        if not isinstance(ref, torch.Tensor) or ref.ndim != 3:
            return sample
        _, h, w = ref.shape
        if h <= self.size or w <= self.size:
            return sample
        top = int(torch.randint(0, h - self.size + 1, ()).item())
        left = int(torch.randint(0, w - self.size + 1, ()).item())

        def fn(t: torch.Tensor) -> torch.Tensor:
            return t[..., top : top + self.size, left : left + self.size]

        return _apply_spatial(sample, fn)


class Compose(_SampleTransform):
    """Sequentially apply a list of sample transforms."""

    def __init__(self, transforms: list[_SampleTransform]) -> None:
        """Args: ``transforms`` ordered list of sample transforms."""
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for tr in self.transforms:
            sample = tr(sample)
        return sample


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_transforms(cfg: Config | DataConfig, split: str = "train") -> _SampleTransform:
    """Build the ``transform(sample) -> sample`` callable for a split.

    Training returns flips + rot90 (and a crop if the data is larger than the
    tile size); validation/test/inference return :class:`Identity` so evaluation
    is deterministic and radiometry is untouched.

    Args:
        cfg: A :class:`~cloudremoval.config.Config` or its ``data`` section.
        split: ``"train"`` enables augmentation; anything else is identity.

    Returns:
        A composed :class:`_SampleTransform`.
    """
    data_cfg = getattr(cfg, "data", cfg)
    if split != "train":
        return Identity()

    transforms: list[_SampleTransform] = [
        RandomFlip(p_horizontal=0.5, p_vertical=0.5),
        RandomRot90(p=0.5),
    ]
    tile = int(getattr(data_cfg, "tile_size", 0) or 0)
    if tile > 0:
        transforms.append(RandomCrop(tile))
    _log.debug("built %d train transforms (tile=%d)", len(transforms), tile)
    return Compose(transforms)
