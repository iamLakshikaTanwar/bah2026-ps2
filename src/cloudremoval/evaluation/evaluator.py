"""Model-agnostic evaluator: run a model over a loader and aggregate MVES metrics.

Implements the BUILD_PLAN §3.5 / §B3 evaluation surface. The :class:`Evaluator`
iterates a dataloader, calls ``model.predict`` (no grad), builds the stratum masks
(``whole`` / ``cloud`` / ``shadow`` / ``thin`` / ``thick`` / ``clear``) from the
``SAMPLE`` dict, computes the metric panel per stratum via
:mod:`cloudremoval.evaluation.metrics`, and averages over the dataset. Optionally
saves per-sample metrics to JSON/CSV.

Pure ``torch`` + stdlib; no heavy deps (the metrics module keeps Tier-0 pure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from cloudremoval.evaluation import metrics as M
from cloudremoval.utils.io import move_to_device
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from cloudremoval.config import Config, EvalConfig
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = ["Evaluator", "evaluate", "build_stratum_masks"]

_log = get_logger(__name__)

# Scalar metrics aggregated per stratum (list-valued ones are handled separately).
_SCALAR_METRICS = (
    "psnr",
    "ssim",
    "ms_ssim",
    "sam",
    "ergas",
    "sid",
    "rmse",
    "mae",
    "ndvi_mae",
    "spectral_correlation",
)


def build_stratum_masks(
    batch: dict[str, Any],
    strata: list[str],
    thin_threshold: float = 0.5,
) -> dict[str, Tensor | None]:
    """Build per-stratum region masks from a ``SAMPLE`` batch.

    Strata semantics:
        * ``whole``  — ``None`` (the full image).
        * ``cloud``  — ``cloud_mask > 0.5``.
        * ``shadow`` — ``shadow_mask > 0.5``.
        * ``clear``  — complement of cloud ∪ shadow (the easy, already-visible
          pixels — useful to show the model does not damage them).
        * ``thin`` / ``thick`` — cloud pixels split by opacity. When the batch
          carries a soft cloud map (values in ``(0,1)``) the split uses
          ``thin_threshold``; otherwise it falls back to ``meta["cloud_type"]``
          per sample (a whole-tile thin/thick label), and if neither is available
          the stratum mask is ``None`` (skipped gracefully).

    Args:
        batch: A batched ``SAMPLE`` dict (tensors ``[B,1,H,W]`` for the masks).
        strata: Which strata to materialise.
        thin_threshold: Soft-cloud opacity boundary between thin and thick.

    Returns:
        Mapping ``stratum -> mask`` (``float`` ``[B,1,H,W]`` or ``None``). Masks
        with no selected pixels are returned as-is (the metrics emit ``nan``).
    """
    cloud = batch.get("cloud_mask")
    shadow = batch.get("shadow_mask")
    cloud_bin = (cloud > 0.5).float() if cloud is not None else None
    shadow_bin = (shadow > 0.5).float() if shadow is not None else None

    out: dict[str, Tensor | None] = {}
    for stratum in strata:
        if stratum == "whole":
            out["whole"] = None
        elif stratum == "cloud":
            out["cloud"] = cloud_bin
        elif stratum == "shadow":
            out["shadow"] = shadow_bin
        elif stratum == "clear":
            if cloud_bin is None and shadow_bin is None:
                out["clear"] = None
            else:
                occluded = torch.zeros_like(
                    cloud_bin if cloud_bin is not None else shadow_bin
                )
                if cloud_bin is not None:
                    occluded = occluded + cloud_bin
                if shadow_bin is not None:
                    occluded = occluded + shadow_bin
                out["clear"] = (occluded <= 0.5).float()
        elif stratum in {"thin", "thick"}:
            out[stratum] = _thin_thick_mask(batch, stratum, cloud, cloud_bin, thin_threshold)
        # Unknown strata are silently ignored (forward-compatible).
    return out


def _thin_thick_mask(
    batch: dict[str, Any],
    stratum: str,
    cloud: Tensor | None,
    cloud_bin: Tensor | None,
    thin_threshold: float,
) -> Tensor | None:
    """Derive a thin/thick cloud mask from soft opacity or per-sample meta."""
    if cloud is None or cloud_bin is None:
        return None
    # Soft cloud map present (not strictly binary) → split by opacity.
    is_soft = bool(((cloud > 0.0) & (cloud < 1.0)).any().item())
    if is_soft:
        if stratum == "thin":
            return ((cloud > 0.0) & (cloud <= thin_threshold)).float()
        return (cloud > thin_threshold).float()

    # Otherwise use a per-sample whole-tile thin/thick label from meta.
    meta = batch.get("meta")
    if not isinstance(meta, list):
        return None
    keep = torch.zeros_like(cloud_bin)
    for i, m in enumerate(meta):
        ctype = str(m.get("cloud_type", "mixed")).lower() if isinstance(m, dict) else "mixed"
        # Only pure-stratum tiles contribute; "mixed" tiles are excluded to keep
        # the thin/thick columns clean.
        if ctype == stratum:
            keep[i] = cloud_bin[i]
    if keep.sum() <= 0:
        return None
    return keep


class Evaluator:
    """Run a model over a dataloader and aggregate masked/stratified MVES metrics.

    Args:
        cfg: The global :class:`Config`. ``cfg.eval`` controls the strata and the
            output directory; ``cfg.train.device`` selects the device.
    """

    def __init__(self, cfg: Config) -> None:
        """Store config and resolve the evaluation device."""
        self.cfg = cfg
        self.eval_cfg: EvalConfig = cfg.eval
        self.device = torch.device(cfg.train.device)
        # Always include 'whole' so there is a reference column even if omitted.
        strata = list(self.eval_cfg.strata)
        if "whole" not in strata:
            strata = ["whole", *strata]
        self.strata = strata

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseCloudRemovalModel,
        loader: DataLoader,
        save: bool = False,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate ``model`` on ``loader`` and return an aggregated results dict.

        Args:
            model: A :class:`BaseCloudRemovalModel` instance.
            loader: Yields ``SAMPLE`` batches.
            save: If ``True``, write ``metrics.json`` and ``per_sample.csv`` under
                ``cfg.eval.out_dir``.
            max_batches: Optional cap on the number of batches (CPU-smoke).

        Returns:
            A dict::

                {
                  "model": <name>,
                  "num_samples": int,
                  "metrics": {stratum: {metric: mean_float}},
                  "per_band_bias": {stratum: [floats]},
                  "per_sample": [ {...}, ... ],   # only when save=True
                }
        """
        model = model.to(self.device)
        model.eval()
        name = getattr(model, "name", model.__class__.__name__)

        # Accumulators: sums and counts per (stratum, metric) ignoring NaNs.
        sums: dict[str, dict[str, float]] = {s: {} for s in self.strata}
        counts: dict[str, dict[str, int]] = {s: {} for s in self.strata}
        band_bias_sums: dict[str, list[float]] = {}
        band_bias_counts: dict[str, int] = {}
        per_sample: list[dict[str, Any]] = []
        n_samples = 0

        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            batch = move_to_device(batch, self.device)
            output = model.predict(batch)
            pred = output.reconstruction
            target = batch["optical_clear"]
            masks = build_stratum_masks(batch, self.strata)
            batch_size = pred.shape[0]
            n_samples += batch_size

            table = M.compute_stratified(pred, target, masks, cfg=None)
            # ``compute_stratified`` keys by metric; restrict to our strata.
            for metric in _SCALAR_METRICS:
                for stratum, value in table.get(metric, {}).items():
                    if stratum not in sums:
                        continue
                    if value == value:  # not NaN
                        sums[stratum][metric] = sums[stratum].get(metric, 0.0) + value * batch_size
                        counts[stratum][metric] = counts[stratum].get(metric, 0) + batch_size

            # Per-band bias per stratum (list-valued).
            for stratum, mask in masks.items():
                bias = M.per_band_bias(pred, target, mask)
                if all(v == v for v in bias):  # no NaNs
                    if stratum not in band_bias_sums:
                        band_bias_sums[stratum] = [0.0] * len(bias)
                    band_bias_sums[stratum] = [
                        a + v * batch_size
                        for a, v in zip(band_bias_sums[stratum], bias, strict=False)
                    ]
                    band_bias_counts[stratum] = band_bias_counts.get(stratum, 0) + batch_size

            if save:
                per_sample.extend(self._per_sample_rows(batch, pred, target, masks))

        metrics_out = {
            stratum: {
                metric: sums[stratum][metric] / counts[stratum][metric]
                for metric in sums[stratum]
            }
            for stratum in self.strata
        }
        bias_out = {
            stratum: [v / band_bias_counts[stratum] for v in band_bias_sums[stratum]]
            for stratum in band_bias_sums
        }

        results: dict[str, Any] = {
            "model": name,
            "num_samples": n_samples,
            "metrics": metrics_out,
            "per_band_bias": bias_out,
        }
        if save:
            results["per_sample"] = per_sample
            self._save(results)

        self._log_summary(name, metrics_out)
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _per_sample_rows(
        batch: dict[str, Any],
        pred: Tensor,
        target: Tensor,
        masks: dict[str, Tensor | None],
    ) -> list[dict[str, Any]]:
        """Compute a flat per-sample metric row (whole + cloud) for each item."""
        rows: list[dict[str, Any]] = []
        meta = batch.get("meta")
        cloud_mask = masks.get("cloud")
        for i in range(pred.shape[0]):
            p_i = pred[i : i + 1]
            t_i = target[i : i + 1]
            m_i = cloud_mask[i : i + 1] if cloud_mask is not None else None
            scene = ""
            if isinstance(meta, list) and i < len(meta) and isinstance(meta[i], dict):
                scene = str(meta[i].get("scene_id", ""))
            row: dict[str, Any] = {
                "scene_id": scene,
                "psnr_whole": M.psnr(p_i, t_i),
                "ssim_whole": M.ssim(p_i, t_i),
                "sam_whole": M.sam(p_i, t_i),
                "ndvi_mae_whole": M.ndvi_mae(p_i, t_i),
            }
            if m_i is not None:
                row["psnr_cloud"] = M.psnr(p_i, t_i, m_i)
                row["ssim_cloud"] = M.ssim(p_i, t_i, m_i)
                row["sam_cloud"] = M.sam(p_i, t_i, m_i)
                row["ndvi_mae_cloud"] = M.ndvi_mae(p_i, t_i, m_i)
            rows.append(row)
        return rows

    def _save(self, results: dict[str, Any]) -> None:
        """Write ``metrics.json`` and ``per_sample.csv`` under ``cfg.eval.out_dir``."""
        out_dir = Path(self.eval_cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in results.items() if k != "per_sample"}
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        per_sample = results.get("per_sample") or []
        if per_sample:
            import csv

            fieldnames: list[str] = []
            for row in per_sample:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_sample:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
        _log.info("Saved evaluation results to %s", out_dir)

    @staticmethod
    def _log_summary(name: str, metrics_out: dict[str, dict[str, float]]) -> None:
        """Log a compact whole/cloud summary line for the headline metrics."""
        whole = metrics_out.get("whole", {})
        cloud = metrics_out.get("cloud", {})

        def fmt(d: dict[str, float], key: str) -> str:
            return f"{d[key]:.4f}" if key in d else "n/a"

        _log.info(
            "[eval %s] whole: PSNR=%s SSIM=%s SAM=%s ERGAS=%s NDVI_MAE=%s | "
            "cloud: PSNR=%s SSIM=%s SAM=%s",
            name,
            fmt(whole, "psnr"),
            fmt(whole, "ssim"),
            fmt(whole, "sam"),
            fmt(whole, "ergas"),
            fmt(whole, "ndvi_mae"),
            fmt(cloud, "psnr"),
            fmt(cloud, "ssim"),
            fmt(cloud, "sam"),
        )


def evaluate(
    model: BaseCloudRemovalModel,
    loader: DataLoader,
    cfg: Config,
    save: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Functional wrapper around :meth:`Evaluator.evaluate` (BUILD_PLAN §B3).

    Args:
        model: The model to evaluate.
        loader: A ``SAMPLE`` dataloader.
        cfg: The global config.
        save: Whether to persist per-sample + summary metrics.
        max_batches: Optional batch cap.

    Returns:
        The aggregated results dict (see :meth:`Evaluator.evaluate`).
    """
    return Evaluator(cfg).evaluate(model, loader, save=save, max_batches=max_batches)
