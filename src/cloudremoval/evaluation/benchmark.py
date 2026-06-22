"""Comparative-assessment runner: evaluate many models on identical data → leaderboard.

This is the required **comparative assessment** deliverable. :func:`run_benchmark`
evaluates several models (by registry name or by per-model :class:`Config`) on the
*same* dataloader and seed, measures speed/compute, ranks each model per metric,
and returns a structured leaderboard (a list of per-model dicts). Results are
saved to JSON and CSV.

Pure ``torch`` + stdlib; reuses :class:`cloudremoval.evaluation.evaluator.Evaluator`
so the metric definitions and stratification are identical to single-model eval.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloudremoval.evaluation.evaluator import Evaluator
from cloudremoval.utils.logging import get_logger
from cloudremoval.utils.seed import seed_everything

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from cloudremoval.config import Config
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = ["run_benchmark", "rank_leaderboard", "ModelSpec"]

_log = get_logger(__name__)

# Lower-is-better metrics (for ranking direction).
_LOWER_BETTER = {"sam", "ergas", "sid", "rmse", "mae", "ndvi_mae", "lpips", "latency_ms"}
# Higher-is-better metrics.
_HIGHER_BETTER = {"psnr", "ssim", "ms_ssim", "spectral_correlation", "throughput_sps"}

# A type alias documenting accepted model specifications.
ModelSpec = "str | Config | BaseCloudRemovalModel"


def run_benchmark(
    models: list[Any],
    loader: DataLoader,
    cfg: Config,
    out_dir: str | Path | None = None,
    rank_metric: str = "psnr",
    rank_stratum: str = "cloud",
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate multiple models on identical data and produce a ranked leaderboard.

    Each entry of ``models`` may be:
        * a **registry name** (``str``) — built with a copy of ``cfg`` whose
          ``model.name`` is overridden;
        * a :class:`Config` — built as-is (lets each model use its own config);
        * an already-constructed :class:`BaseCloudRemovalModel` instance.

    All models see the *same* ``loader`` and the same seed before each evaluation,
    so the comparison is apples-to-apples. Inference wall-time → latency/throughput
    columns; parameter count → a compute column.

    Args:
        models: The models to compare (see above).
        loader: A shared ``SAMPLE`` dataloader.
        cfg: The global config (drives evaluation strata / device).
        out_dir: Where to write ``leaderboard.json`` / ``leaderboard.csv``
            (defaults to ``cfg.eval.out_dir / "benchmark"``).
        rank_metric: Metric used for the overall ``rank`` column.
        rank_stratum: Stratum the ranking metric is read from (``"cloud"`` is the
            real score; falls back to ``"whole"`` if absent).
        max_batches: Optional per-model batch cap (CPU-smoke).

    Returns:
        The leaderboard: a list of per-model dicts, sorted best→worst by
        ``rank_metric`` at ``rank_stratum``, each containing flattened metrics plus
        ``params``, ``latency_ms``, ``throughput_sps`` and ``rank``.
    """
    out_path = Path(out_dir) if out_dir else Path(cfg.eval.out_dir) / "benchmark"
    rows: list[dict[str, Any]] = []

    for spec in models:
        name, model, used_cfg = _resolve_model(spec, cfg)
        _log.info("Benchmarking model '%s' ...", name)
        seed_everything(cfg.train.seed)

        evaluator = Evaluator(used_cfg)
        t0 = time.time()
        results = evaluator.evaluate(model, loader, save=False, max_batches=max_batches)
        elapsed = time.time() - t0

        n = max(int(results.get("num_samples", 0)), 1)
        row: dict[str, Any] = {
            "model": name,
            "params": _count_params(model),
            "latency_ms": 1000.0 * elapsed / n,
            "throughput_sps": n / elapsed if elapsed > 0 else float("nan"),
        }
        row.update(_flatten_metrics(results["metrics"]))
        rows.append(row)

    leaderboard = rank_leaderboard(rows, rank_metric=rank_metric, rank_stratum=rank_stratum)
    _save_leaderboard(leaderboard, out_path)
    _log_leaderboard(leaderboard, rank_metric, rank_stratum)
    return leaderboard


def run_benchmark_all(
    loader: DataLoader,
    cfg: Config,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convenience: benchmark **every registered model** (BUILD_PLAN §B3 done-when).

    Args:
        loader: A shared ``SAMPLE`` dataloader.
        cfg: The global config.
        **kwargs: Forwarded to :func:`run_benchmark` (e.g. ``rank_metric``).

    Returns:
        The ranked leaderboard over all models in the registry.
    """
    from cloudremoval.models.registry import list_models

    names = list_models()
    if not names:
        _log.warning("No models registered; leaderboard will be empty.")
    return run_benchmark(names, loader, cfg, **kwargs)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def rank_leaderboard(
    rows: list[dict[str, Any]],
    rank_metric: str = "psnr",
    rank_stratum: str = "cloud",
) -> list[dict[str, Any]]:
    """Sort rows best→worst and add per-metric ranks + an overall ``rank``.

    For each numeric metric column, a ``<col>_rank`` (1 = best) is added using the
    metric's known direction (lower-better vs higher-better; whole-image scalar
    metrics inherit the direction of their base name). The overall ``rank`` is by
    ``<rank_metric>`` at ``rank_stratum`` (column ``"<stratum>/<metric>"``).

    Args:
        rows: Per-model metric dicts (flattened, ``"<stratum>/<metric>"`` keys).
        rank_metric: Base metric name for the overall rank.
        rank_stratum: Preferred stratum for the overall rank.

    Returns:
        A new sorted list with rank columns added.
    """
    rows = [copy.deepcopy(r) for r in rows]
    if not rows:
        return rows

    # Per-column ranks.
    numeric_cols = _numeric_columns(rows)
    for col in numeric_cols:
        higher = _is_higher_better(col)
        valid = [(r[col], idx) for idx, r in enumerate(rows) if _is_number(r.get(col))]
        ordered = sorted(valid, key=lambda kv: kv[0], reverse=higher)
        for position, (_, idx) in enumerate(ordered, start=1):
            rows[idx][f"{col}_rank"] = position

    overall_col = _resolve_rank_column(rows, rank_metric, rank_stratum)
    higher = _is_higher_better(rank_metric)

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        value = row.get(overall_col)
        if not _is_number(value):
            return (1, 0.0)  # missing → sorted last
        return (0, -float(value) if higher else float(value))

    rows.sort(key=sort_key)
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
        row["rank_by"] = overall_col
    return rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_model(spec: Any, cfg: Config) -> tuple[str, BaseCloudRemovalModel, Config]:
    """Resolve a model spec into ``(name, model, config_used)``."""
    from cloudremoval.config import Config as ConfigCls
    from cloudremoval.models.base import BaseCloudRemovalModel
    from cloudremoval.models.registry import build_model

    if isinstance(spec, BaseCloudRemovalModel):
        return getattr(spec, "name", spec.__class__.__name__), spec, cfg
    if isinstance(spec, ConfigCls):
        model = build_model(spec)
        return spec.model.name, model, spec
    if isinstance(spec, str):
        used = cfg.model_copy(deep=True)
        used.model.name = spec
        model = build_model(used)
        return spec, model, used
    raise TypeError(f"unsupported model spec type: {type(spec)!r}")


def _count_params(model: BaseCloudRemovalModel) -> int:
    """Total number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten ``{stratum: {metric: value}}`` to ``{"stratum/metric": value}``."""
    flat: dict[str, float] = {}
    for stratum, panel in metrics.items():
        for metric, value in panel.items():
            flat[f"{stratum}/{metric}"] = value
    return flat


def _numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Column names that hold numeric values in at least one row (excl. ranks)."""
    cols: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.endswith("_rank") or key in {"rank", "model", "rank_by"}:
                continue
            if _is_number(value) and key not in cols:
                cols.append(key)
    return cols


def _is_number(value: Any) -> bool:
    """True for a finite int/float (not bool, not NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value  # excludes NaN


def _base_metric(col: str) -> str:
    """Strip a ``"stratum/"`` prefix to get the base metric name."""
    return col.split("/", 1)[1] if "/" in col else col


def _is_higher_better(col: str) -> bool:
    """Whether a column ranks higher=better (default True for unknown metrics)."""
    base = _base_metric(col)
    if base in _LOWER_BETTER:
        return False
    if base in _HIGHER_BETTER:
        return True
    return True


def _resolve_rank_column(rows: list[dict[str, Any]], rank_metric: str, rank_stratum: str) -> str:
    """Pick the flattened column for the overall rank, falling back to 'whole'."""
    preferred = f"{rank_stratum}/{rank_metric}"
    if any(_is_number(r.get(preferred)) for r in rows):
        return preferred
    fallback = f"whole/{rank_metric}"
    if any(_is_number(r.get(fallback)) for r in rows):
        return fallback
    return preferred


def _save_leaderboard(leaderboard: list[dict[str, Any]], out_dir: Path) -> None:
    """Write the leaderboard to ``leaderboard.json`` and ``leaderboard.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    if leaderboard:
        import csv

        fieldnames: list[str] = []
        for row in leaderboard:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in leaderboard:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    _log.info("Saved leaderboard to %s", out_dir)


def _log_leaderboard(
    leaderboard: list[dict[str, Any]], rank_metric: str, rank_stratum: str
) -> None:
    """Log a compact ranked summary."""
    if not leaderboard:
        _log.info("Leaderboard is empty.")
        return
    col = leaderboard[0].get("rank_by", f"{rank_stratum}/{rank_metric}")
    _log.info("Leaderboard (ranked by %s):", col)
    for row in leaderboard:
        value = row.get(col)
        vtxt = f"{value:.4f}" if _is_number(value) else "n/a"
        _log.info("  #%s %-22s %s=%s", row.get("rank", "?"), row.get("model", "?"), col, vtxt)


# Backwards/forward-friendly alias used by the CLI/script layer.
run_benchmark.all = run_benchmark_all  # type: ignore[attr-defined]
