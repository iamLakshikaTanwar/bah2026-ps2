"""Pure-torch training callbacks for the :class:`~cloudremoval.training.trainer.Trainer`.

A small, dependency-free callback protocol that the :class:`Trainer` invokes at
fixed hook points (``on_train_begin`` / ``on_epoch_begin`` / ``on_step_end`` /
``on_epoch_end`` / ``on_validation_end`` / ``on_train_end``). Callbacks may signal
early termination by setting ``trainer.should_stop = True``.

Provided callbacks
------------------
* :class:`ModelCheckpoint` — save best/last checkpoints by a monitored metric.
* :class:`EarlyStopping` — stop when a monitored metric stops improving.
* :class:`LRSchedulerCallback` — step a torch LR scheduler each epoch/step.
* :class:`MetricLogger` — append metrics to a CSV (and, lazily, TensorBoard /
  MLflow if those optional deps are installed).

Everything here is pure ``torch`` + stdlib; the optional loggers are imported
lazily inside the methods that use them and degrade to no-ops when absent, so the
module imports cleanly on the CPU-smoke stack.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:  # avoid import cycles on the smoke path
    from cloudremoval.training.trainer import Trainer

__all__ = [
    "Callback",
    "CallbackList",
    "ModelCheckpoint",
    "EarlyStopping",
    "LRSchedulerCallback",
    "MetricLogger",
]

_log = get_logger(__name__)


def _is_better(current: float, best: float, mode: str, min_delta: float) -> bool:
    """Return ``True`` if ``current`` improves on ``best`` under ``mode``."""
    if math.isnan(current):
        return False
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


# --------------------------------------------------------------------------- #
# Protocol + container
# --------------------------------------------------------------------------- #
class Callback:
    """Base callback. Subclasses override the hooks they care about.

    Every hook receives the owning :class:`Trainer` and a ``logs`` mapping (the
    latest scalar metrics: train loss terms, validation metrics, ``"epoch"``,
    ``"step"``). Hooks return ``None``; to halt training, set
    ``trainer.should_stop = True``.
    """

    def on_train_begin(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        ...

    def on_epoch_begin(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        ...

    def on_step_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        ...

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        ...

    def on_validation_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        ...

    def on_train_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        ...


class CallbackList(Callback):
    """Fan a single hook call out to an ordered list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        """Store the callbacks (``None`` → empty list)."""
        self.callbacks: list[Callback] = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        """Append a callback to the list."""
        self.callbacks.append(callback)

    def _dispatch(self, hook: str, trainer: Trainer, logs: dict[str, Any]) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(trainer, logs)

    def on_train_begin(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._dispatch("on_train_begin", trainer, logs)

    def on_epoch_begin(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._dispatch("on_epoch_begin", trainer, logs)

    def on_step_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._dispatch("on_step_end", trainer, logs)

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._dispatch("on_epoch_end", trainer, logs)

    def on_validation_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._dispatch("on_validation_end", trainer, logs)

    def on_train_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._dispatch("on_train_end", trainer, logs)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
class ModelCheckpoint(Callback):
    """Save the *best* (by a monitored metric) and *last* model checkpoints.

    Uses the trainer's :meth:`Trainer.save_checkpoint` so the saved blob includes
    model/optimizer/epoch/config state (load-compatible). The "best" file is
    overwritten whenever the monitored metric improves; the "last" file is written
    every epoch.
    """

    def __init__(
        self,
        dirpath: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        best_filename: str = "best.pt",
        last_filename: str = "last.pt",
        min_delta: float = 0.0,
    ) -> None:
        """Configure where/how checkpoints are written.

        Args:
            dirpath: Directory to write checkpoints into (created if needed).
            monitor: Key in ``logs`` to monitor (e.g. ``"val_loss"``).
            mode: ``"min"`` (lower is better) or ``"max"``.
            save_last: Whether to also write a ``last`` checkpoint each epoch.
            best_filename: Filename for the best checkpoint.
            last_filename: Filename for the last checkpoint.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best_filename = best_filename
        self.last_filename = last_filename
        self.min_delta = min_delta
        self.best_value: float = math.inf if mode == "min" else -math.inf
        self.best_path: Path | None = None

    @property
    def best_model_path(self) -> Path | None:
        """Path to the best checkpoint written so far (``None`` until first save)."""
        return self.best_path

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if self.save_last:
            trainer.save_checkpoint(self.dirpath / self.last_filename, extra={"logs": logs})
        if self.monitor in logs:
            current = float(logs[self.monitor])
            if _is_better(current, self.best_value, self.mode, self.min_delta):
                self.best_value = current
                self.best_path = self.dirpath / self.best_filename
                trainer.save_checkpoint(self.best_path, extra={"logs": logs})
                _log.info(
                    "ModelCheckpoint: new best %s=%.6f -> %s",
                    self.monitor,
                    current,
                    self.best_path,
                )


class EarlyStopping(Callback):
    """Stop training when a monitored metric has not improved for ``patience`` epochs."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        """Configure the early-stopping criterion.

        Args:
            monitor: Key in ``logs`` to monitor.
            mode: ``"min"`` or ``"max"``.
            patience: Epochs with no improvement before stopping.
            min_delta: Minimum change counted as an improvement.
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float = math.inf if mode == "min" else -math.inf
        self.wait = 0

    def on_validation_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        if self.monitor not in logs:
            return
        current = float(logs[self.monitor])
        if _is_better(current, self.best_value, self.mode, self.min_delta):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                _log.info(
                    "EarlyStopping: %s did not improve for %d epochs — stopping.",
                    self.monitor,
                    self.patience,
                )
                trainer.should_stop = True


class LRSchedulerCallback(Callback):
    """Step a torch LR scheduler at the configured interval.

    Args:
        scheduler: Any object with a ``.step()`` (e.g. a
            ``torch.optim.lr_scheduler`` instance), or ``None`` (no-op).
        interval: ``"epoch"`` (step after each epoch) or ``"step"`` (after each
            optimizer step).
        monitor: If given and the scheduler's ``.step`` accepts a metric
            (``ReduceLROnPlateau``), the monitored value is forwarded.
    """

    def __init__(
        self,
        scheduler: Any | None,
        interval: str = "epoch",
        monitor: str | None = None,
    ) -> None:
        """Store the scheduler and stepping policy."""
        self.scheduler = scheduler
        self.interval = interval
        self.monitor = monitor

    def _step(self, logs: dict[str, Any]) -> None:
        if self.scheduler is None:
            return
        if self.monitor is not None and self.monitor in logs:
            try:
                self.scheduler.step(float(logs[self.monitor]))
                return
            except TypeError:
                pass
        self.scheduler.step()

    def on_step_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        if self.interval == "step":
            self._step(logs)

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        if self.interval == "epoch":
            self._step(logs)


class MetricLogger(Callback):
    """Append epoch/step metrics to a CSV (+ optional TensorBoard / MLflow).

    The CSV is the always-available, dependency-free record. TensorBoard
    (``torch.utils.tensorboard``) and MLflow are imported lazily and only used if
    explicitly enabled *and* installed; any import error degrades to a no-op with
    a single warning.
    """

    def __init__(
        self,
        csv_path: str | Path,
        use_tensorboard: bool = False,
        use_mlflow: bool = False,
        tb_logdir: str | Path | None = None,
    ) -> None:
        """Configure logging sinks.

        Args:
            csv_path: Destination CSV file (parent dirs created).
            use_tensorboard: Attempt to also log scalars to TensorBoard.
            use_mlflow: Attempt to also log scalars/metrics to MLflow.
            tb_logdir: TensorBoard log directory (defaults next to ``csv_path``).
        """
        self.csv_path = Path(csv_path)
        self.use_tensorboard = use_tensorboard
        self.use_mlflow = use_mlflow
        self.tb_logdir = Path(tb_logdir) if tb_logdir else self.csv_path.parent / "tb"
        self._fieldnames: list[str] = []
        self._tb_writer: Any | None = None
        self._mlflow: Any | None = None
        self._global_step = 0

    # -- optional sinks ---------------------------------------------------- #
    def _ensure_tb(self) -> None:
        if self._tb_writer is not None or not self.use_tensorboard:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.tb_logdir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(self.tb_logdir))
        except Exception as exc:  # noqa: BLE001
            _log.warning("TensorBoard unavailable (%s); CSV logging only.", exc)
            self.use_tensorboard = False

    def _ensure_mlflow(self) -> None:
        if self._mlflow is not None or not self.use_mlflow:
            return
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001
            _log.warning("MLflow unavailable (%s); CSV logging only.", exc)
            self.use_mlflow = False

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _scalars(logs: dict[str, Any]) -> dict[str, float]:
        """Keep only scalar (int/float) entries for tabular/scalar logging."""
        out: dict[str, float] = {}
        for key, value in logs.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out[key] = float(value)
        return out

    def _write_csv_row(self, row: dict[str, float]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._rewrite_csv_with_header(row)
            return
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    def _rewrite_csv_with_header(self, latest: dict[str, float]) -> None:
        """Rewrite the CSV with the (possibly extended) header preserving prior rows."""
        prior: list[dict[str, str]] = []
        if self.csv_path.exists():
            with self.csv_path.open("r", newline="", encoding="utf-8") as fh:
                prior = list(csv.DictReader(fh))
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writeheader()
            for prow in prior:
                writer.writerow({k: prow.get(k, "") for k in self._fieldnames})
            writer.writerow({k: latest.get(k, "") for k in self._fieldnames})

    # -- hooks ------------------------------------------------------------- #
    def _log_row(self, logs: dict[str, Any]) -> None:
        scalars = self._scalars(logs)
        if not scalars:
            return
        self._write_csv_row(scalars)
        self._ensure_tb()
        if self._tb_writer is not None:
            step = int(scalars.get("step", self._global_step))
            for key, value in scalars.items():
                self._tb_writer.add_scalar(key, value, step)
        self._ensure_mlflow()
        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics(scalars, step=int(scalars.get("step", self._global_step)))
            except Exception as exc:  # noqa: BLE001
                _log.warning("mlflow.log_metrics failed (%s).", exc)
        self._global_step += 1

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        self._log_row(logs)

    def on_train_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:  # noqa: D102
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
