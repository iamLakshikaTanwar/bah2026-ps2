"""Optional PyTorch Lightning wrapper around a :class:`BaseCloudRemovalModel`.

This is the ``cfg.train.use_lightning`` path. Lightning is an **optional** heavy
dependency: it is imported *lazily* inside :func:`build_lightning_module` /
:func:`get_lightning_base` so that importing this module never requires Lightning
and the CPU-smoke stack stays minimal. When Lightning is absent, those functions
raise a clear, actionable :class:`ImportError`.

The wrapper reuses the model's own ``forward`` / ``loss`` (which already delegates
to :class:`~cloudremoval.models.losses.LossBundle`), so training is identical to
the pure-torch :class:`~cloudremoval.training.trainer.Trainer` — Lightning only
provides the engine. GAN models are *not* fully supported here (they need
``automatic_optimization=False`` plumbing); use the pure-torch trainer for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cloudremoval.evaluation import metrics as M
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import Config
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = [
    "lightning_available",
    "get_lightning_base",
    "build_lightning_module",
    "build_trainer",
]

_log = get_logger(__name__)

_LIGHTNING_HELP = (
    "PyTorch Lightning is not installed. Install it with "
    "`pip install lightning` (or `pip install pytorch-lightning`) to use the "
    "cfg.train.use_lightning path, or set use_lightning=false to use the "
    "pure-torch cloudremoval.training.trainer.Trainer."
)


def lightning_available() -> bool:
    """Return ``True`` if a Lightning import (``lightning`` or legacy) succeeds."""
    return _import_lightning(raise_on_missing=False) is not None


def _import_lightning(raise_on_missing: bool = True) -> Any | None:
    """Import the Lightning module (new ``lightning.pytorch`` or legacy package).

    Returns the imported ``pl`` namespace (with ``LightningModule`` / ``Trainer``)
    or ``None``/raises when unavailable.
    """
    try:
        import lightning.pytorch as pl  # type: ignore

        return pl
    except Exception:  # noqa: BLE001 - fall through to legacy package name
        pass
    try:
        import pytorch_lightning as pl  # type: ignore

        return pl
    except Exception as exc:  # noqa: BLE001
        if raise_on_missing:
            raise ImportError(_LIGHTNING_HELP) from exc
        return None


def get_lightning_base() -> type:
    """Return a concrete ``CloudRemovalLightningModule`` class (lazy).

    The class is **defined inside this function** so that subclassing
    ``pl.LightningModule`` only happens when Lightning is importable — module
    import never touches Lightning.

    Returns:
        A ``LightningModule`` subclass wrapping a :class:`BaseCloudRemovalModel`.

    Raises:
        ImportError: If Lightning is not installed.
    """
    pl = _import_lightning(raise_on_missing=True)
    import torch

    class CloudRemovalLightningModule(pl.LightningModule):  # type: ignore[misc]
        """LightningModule delegating to a wrapped :class:`BaseCloudRemovalModel`.

        ``training_step`` / ``validation_step`` call the model's ``forward`` and
        ``loss`` (so the loss weighting is shared with the pure-torch path);
        ``configure_optimizers`` builds an Adam/AdamW from ``cfg.train``.
        """

        def __init__(self, model: BaseCloudRemovalModel, cfg: Config) -> None:
            super().__init__()
            self.model = model
            self.cfg = cfg
            self._is_gan = hasattr(model, "discriminator_parameters")
            if self._is_gan:
                _log.warning(
                    "GAN model under Lightning: only the generator loss is "
                    "optimised here. Use the pure-torch Trainer for full G/D "
                    "alternation."
                )

        def forward(self, sample: dict[str, Any]) -> Any:  # noqa: D102
            return self.model.forward(sample)

        def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:  # noqa: D102
            output = self.model.forward(batch)
            loss_dict = self.model.loss(batch, output)
            for key, value in loss_dict.items():
                self.log(f"train_{key}", value, prog_bar=(key == "total"), on_step=True)
            return loss_dict["total"]

        def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:  # noqa: D102
            output = self.model.forward(batch)
            loss_dict = self.model.loss(batch, output)
            pred, target = output.reconstruction, batch["optical_clear"]
            self.log("val_loss", loss_dict["total"], prog_bar=True)
            self.log("val_psnr", M.psnr(pred, target))
            self.log("val_ssim", M.ssim(pred, target))
            self.log("val_sam", M.sam(pred, target))
            return loss_dict["total"]

        def configure_optimizers(self) -> Any:  # noqa: D102
            tcfg = self.cfg.train
            name = str(getattr(tcfg, "optimizer", "adamw")).lower()
            params = list(self.model.trainable_parameters())
            if name == "adam":
                return torch.optim.Adam(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
            return torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    return CloudRemovalLightningModule


def build_lightning_module(model: BaseCloudRemovalModel, cfg: Config) -> Any:
    """Instantiate the Lightning wrapper for ``model`` (lazy; raises if absent).

    Args:
        model: The model to wrap.
        cfg: The global config (its ``train`` section configures the optimizer).

    Returns:
        A ``LightningModule`` instance.

    Raises:
        ImportError: If Lightning is not installed.
    """
    cls = get_lightning_base()
    return cls(model, cfg)


def build_trainer(cfg: Config, **trainer_kwargs: Any) -> Any:
    """Build a ``lightning.Trainer`` honoring ``cfg.train`` (lazy; raises if absent).

    Maps ``cfg.train`` onto the Lightning ``Trainer`` (max epochs/steps, device,
    precision, gradient clipping). Extra keyword arguments override the defaults.

    Raises:
        ImportError: If Lightning is not installed.
    """
    pl = _import_lightning(raise_on_missing=True)
    tcfg = cfg.train
    accelerator = "gpu" if tcfg.device == "cuda" else "cpu"
    kwargs: dict[str, Any] = {
        "max_epochs": max(int(tcfg.epochs), 1),
        "accelerator": accelerator,
        "devices": 1,
        "gradient_clip_val": tcfg.grad_clip,
        "enable_checkpointing": True,
        "logger": False,
    }
    if tcfg.max_steps is not None:
        kwargs["max_steps"] = int(tcfg.max_steps)
    if accelerator == "gpu" and str(tcfg.precision) in {"16", "bf16"}:
        kwargs["precision"] = "16-mixed" if str(tcfg.precision) == "16" else "bf16-mixed"
    kwargs.update(trainer_kwargs)
    return pl.Trainer(**kwargs)
