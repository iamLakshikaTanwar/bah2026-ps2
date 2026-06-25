"""Abstract base class and standard output for all cloud-removal models.

This module owns the **frozen model interface contract** (``docs/BUILD_PLAN.md``
§3.1). The trainer, evaluator, benchmark runner and inference engine are written
*only* against :class:`BaseCloudRemovalModel` and :class:`ModelOutput` — they
never import a concrete model. Every concrete model (B2) subclasses
:class:`BaseCloudRemovalModel`, registers itself via
``models.registry.register_model``, and reads the keys it needs from the
``SAMPLE`` dict (``docs/BUILD_PLAN.md`` §2).

Pure-``torch`` module: importable on the CPU-smoke stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # avoid a hard import cycle on the smoke path
    from cloudremoval.config import ModelConfig

__all__ = ["ModelOutput", "BaseCloudRemovalModel", "LossDict"]

# Type alias used across the codebase. A model's ``loss`` MUST return a mapping
# containing the key ``"total"`` (the scalar that is back-propagated).
LossDict = dict[str, Tensor]


@dataclass
class ModelOutput:
    """Standard return value of every model's :meth:`forward`.

    Attributes:
        reconstruction: ``[B, C, H, W]`` predicted clear optical image in the
            normalized space the dataset uses (per-band reflectance norm).
        uncertainty: Optional ``[B, 1, H, W]`` per-pixel aleatoric std/variance
            (populated by uncertainty-aware models; ``None`` otherwise).
        aux: Free-form mapping of auxiliary tensors (attention maps,
            discriminator logits, intermediate predictions, ...). Consumers must
            tolerate arbitrary / missing keys.
    """

    reconstruction: Tensor  # [B, C, H, W] predicted clear optical (normalized)
    uncertainty: Tensor | None = None  # [B, 1, H, W] per-pixel aleatoric std/var
    aux: dict[str, Tensor] = field(default_factory=dict)


class BaseCloudRemovalModel(nn.Module, ABC):
    """Abstract base every cloud-removal model implements.

    The trainer / evaluator / inference engine use **only** this API. Subclasses
    MUST:

    * accept a parsed :class:`~cloudremoval.config.ModelConfig` (pydantic) as the
      single positional argument to ``__init__``;
    * register via ``@register_model("<name>")``;
    * set :attr:`name` (the decorator does this for you);
    * implement :meth:`forward` (differentiable, reads the ``SAMPLE`` dict) and
      :meth:`loss` (returns a :data:`LossDict` containing ``"total"``).

    :meth:`predict` has a default (``self.eval()`` + a single ``forward``) and may
    be overridden — e.g. diffusion overrides it to run DDIM sampling rather than
    a single forward pass.
    """

    #: Registry name; overwritten by ``@register_model``.
    name: str = "base"

    def __init__(self, cfg: ModelConfig) -> None:
        """Store the model configuration.

        Args:
            cfg: The ``model`` section of the global :class:`Config`.
        """
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Differentiable forward pass used during training.

        Args:
            sample: The ``SAMPLE`` dict (``docs/BUILD_PLAN.md`` §2). Image
                tensors are ``float32``, channel-first, batched on dim 0.

        Returns:
            A :class:`ModelOutput` whose ``reconstruction`` is ``[B, C, H, W]``.
        """
        ...

    @abstractmethod
    def loss(self, sample: dict[str, Any], output: ModelOutput) -> LossDict:
        """Compute the training loss for one batch.

        Args:
            sample: The ``SAMPLE`` dict (must include ``optical_clear`` target).
            output: The :class:`ModelOutput` from :meth:`forward`.

        Returns:
            A :data:`LossDict` mapping term names to scalar tensors. The key
            ``"total"`` is required and is the value back-propagated.
        """
        ...

    @torch.no_grad()
    def predict(self, sample: dict[str, Any]) -> ModelOutput:
        """Inference entry point (no gradients).

        Default implementation switches to ``eval`` mode and runs a single
        :meth:`forward`. Override for iterative samplers (e.g. diffusion DDIM).

        Args:
            sample: The ``SAMPLE`` dict (``optical_clear`` may be a zero target).

        Returns:
            A :class:`ModelOutput`.
        """
        self.eval()
        return self.forward(sample)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield the parameters with ``requires_grad=True`` (optimizer target)."""
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    # Optional hooks for GAN-style models (B3's trainer probes via hasattr).
    # They are intentionally *not* defined here so ``hasattr`` is False for
    # non-adversarial models; the signatures below document the contract:
    #
    #   def generator_parameters(self) -> Iterable[nn.Parameter]: ...
    #   def discriminator_parameters(self) -> Iterable[nn.Parameter]: ...
    #   def discriminator_loss(self, sample, output) -> LossDict: ...
    # ------------------------------------------------------------------ #
