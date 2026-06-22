"""Pure-torch training loop for cloud-removal models (BUILD_PLAN §3.7).

The :class:`Trainer` is written **only** against the frozen
:class:`~cloudremoval.models.base.BaseCloudRemovalModel` API and the ``SAMPLE``
dict — it never imports a concrete model. It supports:

* one optimizer on ``model.trainable_parameters()`` (Adam / AdamW from cfg);
* **GAN models** — detected via ``hasattr(model, "discriminator_parameters")`` —
  trained with **two optimizers** alternating a discriminator step and a
  generator step using the optional ``discriminator_parameters`` /
  ``generator_parameters`` / ``discriminator_loss`` hooks;
* **uncertainty models** — handled transparently: the model's own ``loss`` (via
  ``LossBundle``) consumes ``output.uncertainty`` as a log-variance for the NLL
  term, so the trainer needs no special case beyond detecting/logging it;
* cosine / no LR schedule (lazy), gradient clipping, optional AMP (off on CPU);
* an ``cfg.train.max_steps`` cap (CPU-smoke), periodic validation that computes a
  val loss plus a couple of metrics from :mod:`cloudremoval.evaluation.metrics`;
* checkpoint save/load (``torch.save`` of model/optimizer(s)/epoch/cfg);
* structured logging and a returned ``history`` dict.

The module is pure ``torch`` + stdlib; nothing here imports a heavy optional
dependency, so it runs on the CPU-smoke stack.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from cloudremoval.evaluation import metrics as M
from cloudremoval.models.losses import LossBundle
from cloudremoval.training.callbacks import (
    Callback,
    CallbackList,
    LRSchedulerCallback,
    MetricLogger,
    ModelCheckpoint,
)
from cloudremoval.utils.io import move_to_device
from cloudremoval.utils.logging import get_logger
from cloudremoval.utils.seed import seed_everything

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from cloudremoval.config import Config
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = ["Trainer", "train"]

_log = get_logger(__name__)


class Trainer:
    """Model-agnostic training loop honoring the BUILD_PLAN §3.7 contract.

    Args:
        cfg: The global :class:`~cloudremoval.config.Config`. The ``train`` and
            ``eval`` sections drive the loop; ``train.device`` selects the device
            (``"cpu"`` default).
        callbacks: Optional list of :class:`Callback` objects. When ``None``, a
            default :class:`ModelCheckpoint` + :class:`MetricLogger` pair is
            attached (writing under ``cfg.train.ckpt_dir``).
    """

    def __init__(self, cfg: Config, callbacks: list[Callback] | None = None) -> None:
        """Initialise the trainer (does not yet build optimizers — see :meth:`fit`)."""
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.loss_bundle = LossBundle(cfg.train.loss)
        self.should_stop = False
        self.epoch = 0
        self.global_step = 0

        # State populated in fit().
        self.model: BaseCloudRemovalModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.gen_optimizer: torch.optim.Optimizer | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.is_gan = False
        self.has_uncertainty = False

        self.callbacks = CallbackList(
            callbacks if callbacks is not None else self._default_callbacks()
        )

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _default_callbacks(self) -> list[Callback]:
        """Default checkpoint + CSV-metric callbacks under ``cfg.train.ckpt_dir``."""
        ckpt_dir = Path(self.cfg.train.ckpt_dir)
        return [
            ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_last=True),
            MetricLogger(csv_path=ckpt_dir / "metrics.csv"),
        ]

    def _build_optimizer(self, params: Any) -> torch.optim.Optimizer:
        """Build an Adam/AdamW optimizer from ``cfg.train`` over ``params``."""
        tcfg = self.cfg.train
        name = str(getattr(tcfg, "optimizer", "adamw")).lower()
        kwargs = {"lr": tcfg.lr, "weight_decay": tcfg.weight_decay}
        params = list(params)
        if not params:
            # No trainable params (e.g. a frozen dummy) — give the optimizer a
            # harmless empty group so .step() is a no-op rather than crashing.
            params = [torch.nn.Parameter(torch.zeros(1))]
        if name == "adam":
            return torch.optim.Adam(params, **kwargs)
        if name == "sgd":
            return torch.optim.SGD(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay, momentum=0.9)
        return torch.optim.AdamW(params, **kwargs)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Any | None:
        """Build a cosine LR schedule when requested (``train.scheduler='cosine'``).

        Returns ``None`` for the default ('none') schedule. Lazy: only constructs a
        scheduler if asked, so CPU-smoke stays minimal.
        """
        sched = str(getattr(self.cfg.train, "scheduler", "none")).lower()
        if sched in {"none", "", "constant"}:
            return None
        if sched == "cosine":
            t_max = max(int(getattr(self.cfg.train, "scheduler_t_max", total_steps)), 1)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        if sched == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        _log.warning("unknown scheduler '%s'; using a constant LR.", sched)
        return None

    def _amp_enabled(self) -> bool:
        """AMP only on CUDA with a 16/bf16 precision setting."""
        return self.device.type == "cuda" and str(self.cfg.train.precision) in {"16", "bf16"}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        model: BaseCloudRemovalModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list]:
        """Train ``model`` on ``train_loader`` and return a history dict.

        Detects GAN vs. single-optimizer models and uncertainty-aware models,
        builds the optimizer(s) + optional scheduler, then runs the epoch/step
        loop (capped by ``cfg.train.max_steps``), validating after each epoch when
        ``val_loader`` is given. Checkpoints/metrics are written via the attached
        callbacks.

        Args:
            model: A :class:`BaseCloudRemovalModel` subclass instance.
            train_loader: Yields ``SAMPLE`` batches.
            val_loader: Optional validation loader.

        Returns:
            ``history`` — a mapping of metric name to a per-record list, e.g.
            ``{"step": [...], "train_loss": [...], "val_loss": [...], ...}``.
        """
        seed_everything(self.cfg.train.seed)
        self.model = model.to(self.device)
        self.is_gan = hasattr(model, "discriminator_parameters")
        self.has_uncertainty = self._probe_uncertainty(model, train_loader)

        epochs = max(int(self.cfg.train.epochs), 1)
        steps_per_epoch = self._safe_len(train_loader)
        total_steps = steps_per_epoch * epochs
        if self.cfg.train.max_steps is not None:
            total_steps = min(total_steps, int(self.cfg.train.max_steps))

        self._setup_optimizers(model, total_steps)

        _log.info(
            "Trainer.fit | model=%s device=%s gan=%s uncertainty=%s epochs=%d max_steps=%s",
            getattr(model, "name", model.__class__.__name__),
            self.device,
            self.is_gan,
            self.has_uncertainty,
            epochs,
            self.cfg.train.max_steps,
        )

        history: dict[str, list] = {}
        self.should_stop = False
        self.global_step = 0
        self.callbacks.on_train_begin(self, {})

        for epoch in range(epochs):
            self.epoch = epoch
            self.callbacks.on_epoch_begin(self, {"epoch": epoch})
            epoch_logs = self._run_epoch(model, train_loader)

            if val_loader is not None:
                val_logs = self.validate(model, val_loader)
                epoch_logs.update(val_logs)
                self.callbacks.on_validation_end(self, {**epoch_logs, "epoch": epoch})

            epoch_logs["epoch"] = float(epoch)
            epoch_logs["step"] = float(self.global_step)
            self._append_history(history, epoch_logs)
            self.callbacks.on_epoch_end(self, epoch_logs)

            if self.should_stop or self._reached_max_steps():
                _log.info("Stopping at epoch %d (step %d).", epoch, self.global_step)
                break

        self.callbacks.on_train_end(self, {"step": float(self.global_step)})
        return history

    @torch.no_grad()
    def validate(self, model: BaseCloudRemovalModel, val_loader: DataLoader) -> dict[str, float]:
        """Compute mean validation loss + a couple of metrics over ``val_loader``.

        Uses ``model.predict`` (no grad) and a small subset of metrics (PSNR /
        SSIM / SAM, whole-image and cloud-masked) from
        :mod:`cloudremoval.evaluation.metrics`. Returns a flat ``{name: float}``
        dict prefixed with ``val_``.
        """
        was_training = model.training
        model.eval()
        n = 0
        acc: dict[str, float] = {}
        max_val_batches = int(getattr(self.cfg.eval, "max_val_batches", 0)) or None

        for i, batch in enumerate(val_loader):
            if max_val_batches is not None and i >= max_val_batches:
                break
            batch = move_to_device(batch, self.device)
            output = model.predict(batch)
            loss_dict = model.loss(batch, output)
            pred = output.reconstruction
            target = batch["optical_clear"]
            cloud_mask = batch.get("cloud_mask")

            row = {
                "val_loss": float(loss_dict["total"].detach().cpu()),
                "val_psnr": M.psnr(pred, target),
                "val_ssim": M.ssim(pred, target),
                "val_sam": M.sam(pred, target),
            }
            if cloud_mask is not None:
                row["val_psnr_cloud"] = M.psnr(pred, target, cloud_mask)
                row["val_sam_cloud"] = M.sam(pred, target, cloud_mask)
            for key, value in row.items():
                if value == value:  # skip nan
                    acc[key] = acc.get(key, 0.0) + value
            n += 1

        if was_training:
            model.train()
        if n == 0:
            return {}
        return {key: value / n for key, value in acc.items()}

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> str:
        """Save model/optimizer(s)/epoch/cfg state to ``path`` via ``torch.save``.

        Args:
            path: Output file path (parent dirs created).
            extra: Optional extra payload merged into the saved blob (e.g. logs).

        Returns:
            The path written (as a string).
        """
        assert self.model is not None, "save_checkpoint called before fit()"
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, Any] = {
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_name": getattr(self.model, "name", self.model.__class__.__name__),
            "config": self.cfg.model_dump(mode="json"),
            "is_gan": self.is_gan,
        }
        if self.optimizer is not None:
            blob["optimizer"] = self.optimizer.state_dict()
        if self.gen_optimizer is not None:
            blob["gen_optimizer"] = self.gen_optimizer.state_dict()
        if self.disc_optimizer is not None:
            blob["disc_optimizer"] = self.disc_optimizer.state_dict()
        if extra:
            blob["extra"] = extra
        torch.save(blob, out)
        return str(out)

    def load_checkpoint(
        self,
        model: BaseCloudRemovalModel,
        path: str | Path,
        load_optimizer: bool = True,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint into ``model`` (and optimizer(s) if available).

        Args:
            model: The model to load weights into.
            path: Checkpoint path written by :meth:`save_checkpoint`.
            load_optimizer: Restore optimizer state when present and built.
            map_location: ``torch.load`` map location (defaults to the cfg device).

        Returns:
            The raw checkpoint blob (so callers can read ``epoch`` etc.).
        """
        blob = torch.load(
            Path(path),
            map_location=map_location or self.device,
            weights_only=False,
        )
        model.load_state_dict(blob["model"])
        self.model = model.to(self.device)
        self.epoch = int(blob.get("epoch", 0))
        self.global_step = int(blob.get("global_step", 0))
        if load_optimizer:
            if self.optimizer is not None and "optimizer" in blob:
                self.optimizer.load_state_dict(blob["optimizer"])
            if self.gen_optimizer is not None and "gen_optimizer" in blob:
                self.gen_optimizer.load_state_dict(blob["gen_optimizer"])
            if self.disc_optimizer is not None and "disc_optimizer" in blob:
                self.disc_optimizer.load_state_dict(blob["disc_optimizer"])
        return blob

    # ------------------------------------------------------------------ #
    # Internal loop
    # ------------------------------------------------------------------ #
    def _setup_optimizers(self, model: BaseCloudRemovalModel, total_steps: int) -> None:
        """Build one or two optimizers (+ schedulers) depending on GAN detection."""
        if self.is_gan:
            gen_params = (
                model.generator_parameters()
                if hasattr(model, "generator_parameters")
                else model.trainable_parameters()
            )
            self.gen_optimizer = self._build_optimizer(gen_params)
            self.disc_optimizer = self._build_optimizer(model.discriminator_parameters())
            self.optimizer = self.gen_optimizer  # primary, for checkpoint convenience
            gen_sched = self._build_scheduler(self.gen_optimizer, total_steps)
            if gen_sched is not None:
                self.callbacks.append(LRSchedulerCallback(gen_sched, interval="epoch"))
        else:
            self.optimizer = self._build_optimizer(model.trainable_parameters())
            sched = self._build_scheduler(self.optimizer, total_steps)
            if sched is not None:
                self.callbacks.append(LRSchedulerCallback(sched, interval="epoch"))

    def _run_epoch(
        self, model: BaseCloudRemovalModel, train_loader: DataLoader
    ) -> dict[str, float]:
        """Run one training epoch; returns mean train-loss terms for the epoch."""
        model.train()
        running: dict[str, float] = {}
        count = 0
        t0 = time.time()

        for batch in train_loader:
            if self._reached_max_steps():
                break
            batch = move_to_device(batch, self.device)
            step_logs = (
                self._gan_step(model, batch) if self.is_gan else self._standard_step(model, batch)
            )
            for key, value in step_logs.items():
                running[key] = running.get(key, 0.0) + value
            count += 1
            self.global_step += 1

            if self.global_step % max(int(self.cfg.train.log_every), 1) == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in step_logs.items())
                _log.info("epoch %d step %d | %s", self.epoch, self.global_step, msg)
            self.callbacks.on_step_end(self, {**step_logs, "step": float(self.global_step)})

        if count == 0:
            return {}
        epoch_logs = {f"train_{k}": v / count for k, v in running.items()}
        epoch_logs["epoch_time_s"] = time.time() - t0
        return epoch_logs

    def _standard_step(
        self, model: BaseCloudRemovalModel, batch: dict[str, Any]
    ) -> dict[str, float]:
        """One single-optimizer training step. Returns scalar loss terms."""
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        output = model.forward(batch)
        loss_dict = model.loss(batch, output)
        total = loss_dict["total"]
        total.backward()
        self._clip_grads(model.parameters())
        self.optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in loss_dict.items()}

    def _gan_step(self, model: BaseCloudRemovalModel, batch: dict[str, Any]) -> dict[str, float]:
        """One alternating discriminator-then-generator GAN step.

        Uses the model's optional hooks: ``discriminator_loss`` for the D update
        and ``model.loss`` (which includes the adversarial generator term) for the
        G update. Both forwards re-run the generator so the graphs are independent.
        """
        assert self.disc_optimizer is not None and self.gen_optimizer is not None
        logs: dict[str, float] = {}

        # ----- Discriminator step ----- #
        if hasattr(model, "discriminator_loss"):
            self.disc_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                d_output = model.forward(batch)
            d_loss_dict = model.discriminator_loss(batch, d_output)
            d_total = d_loss_dict["total"]
            d_total.backward()
            self._clip_grads(model.discriminator_parameters())
            self.disc_optimizer.step()
            logs.update({f"d_{k}": float(v.detach().cpu()) for k, v in d_loss_dict.items()})

        # ----- Generator step ----- #
        self.gen_optimizer.zero_grad(set_to_none=True)
        g_output = model.forward(batch)
        g_loss_dict = model.loss(batch, g_output)
        g_total = g_loss_dict["total"]
        g_total.backward()
        gen_params = (
            model.generator_parameters()
            if hasattr(model, "generator_parameters")
            else model.parameters()
        )
        self._clip_grads(gen_params)
        self.gen_optimizer.step()
        logs.update({f"g_{k}": float(v.detach().cpu()) for k, v in g_loss_dict.items()})
        # Expose a unified "total" so callbacks/monitors have a single train loss.
        logs["total"] = logs.get("g_total", 0.0)
        return logs

    def _clip_grads(self, params: Any) -> None:
        """Clip gradient norm when ``cfg.train.grad_clip`` is set."""
        clip = self.cfg.train.grad_clip
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(list(params), max_norm=float(clip))

    # ------------------------------------------------------------------ #
    # Small utilities
    # ------------------------------------------------------------------ #
    def _reached_max_steps(self) -> bool:
        return (
            self.cfg.train.max_steps is not None
            and self.global_step >= int(self.cfg.train.max_steps)
        )

    @staticmethod
    def _safe_len(loader: DataLoader) -> int:
        try:
            return max(len(loader), 1)
        except TypeError:  # IterableDataset without __len__
            return 1

    @staticmethod
    def _append_history(history: dict[str, list], logs: dict[str, float]) -> None:
        for key, value in logs.items():
            history.setdefault(key, []).append(value)

    def _probe_uncertainty(
        self, model: BaseCloudRemovalModel, loader: DataLoader
    ) -> bool:
        """Detect whether the model emits an uncertainty map (transparent handling).

        Runs a single no-grad forward on one batch and checks
        ``output.uncertainty is not None``. Purely informational — the model's own
        ``loss`` already consumes the uncertainty; the trainer needs no branch.
        Any failure is swallowed (returns ``False``).
        """
        if bool(getattr(self.cfg.model, "uncertainty", False)):
            return True
        try:
            batch = next(iter(loader))
        except (StopIteration, TypeError):
            return False
        try:
            with torch.no_grad():
                out = model.to(self.device).forward(move_to_device(batch, self.device))
            return out.uncertainty is not None
        except Exception:  # noqa: BLE001 - probing must never break training
            return False


# --------------------------------------------------------------------------- #
# Convenience entry point (used by scripts/train.py and the CLI)
# --------------------------------------------------------------------------- #
def train(cfg: Config, ckpt: str | Path | None = None) -> dict[str, Any]:
    """Build the model + dataloaders from ``cfg`` and run :meth:`Trainer.fit`.

    Lazily imports :func:`cloudremoval.data.datasets.build_dataloader` (owned by
    B1) so this module imports without the data layer present. When the data
    layer is unavailable the error is surfaced clearly.

    Args:
        cfg: The global config.
        ckpt: Optional checkpoint to resume model weights from before training.

    Returns:
        A small summary dict: ``{"history", "best_ckpt", "final_ckpt"}``.
    """
    from cloudremoval.models.registry import build_model

    seed_everything(cfg.train.seed)
    model = build_model(cfg)

    train_loader, val_loader = _build_loaders(cfg)

    trainer = Trainer(cfg)
    if ckpt is not None:
        # Optimizers are built inside fit(); load just the weights here.
        blob = torch.load(Path(ckpt), map_location=cfg.train.device, weights_only=False)
        model.load_state_dict(blob["model"])
        _log.info("Resumed model weights from %s", ckpt)

    history = trainer.fit(model, train_loader, val_loader)

    final_path = Path(cfg.train.ckpt_dir) / "final.pt"
    trainer.save_checkpoint(final_path)

    best_ckpt: str | None = None
    for cb in trainer.callbacks.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path is not None:
            best_ckpt = str(cb.best_model_path)
            break

    return {
        "history": history,
        "best_ckpt": best_ckpt,
        "final_ckpt": str(final_path),
    }


def _build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader | None]:
    """Construct train (+ best-effort val) dataloaders via the B1 data layer."""
    try:
        from cloudremoval.data.datasets import build_dataloader
    except ImportError as exc:  # pragma: no cover - only without the data layer
        raise ImportError(
            "cloudremoval.data.datasets.build_dataloader is unavailable; the data "
            "layer (B1) must be installed to use train(cfg)."
        ) from exc

    train_loader = build_dataloader(cfg.data, "train")
    val_loader: DataLoader | None
    try:
        val_loader = build_dataloader(cfg.data, "val")
    except Exception as exc:  # noqa: BLE001 - val split is optional
        _log.warning("no validation loader (%s); training without validation.", exc)
        val_loader = None
    return train_loader, val_loader
