"""Runnable benchmark entry point — ``scripts/benchmark.py``.

Thin wrapper the CLI delegates to: ``cloudremoval benchmark --config ...`` calls
:func:`main(cfg)` here. Evaluates **every registered model** on one shared
dataloader, ranks them into a leaderboard, and writes a comparative-assessment
Markdown report.

Run directly::

    python scripts/benchmark.py --config configs/cpu_smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if TYPE_CHECKING:
    from cloudremoval.config import Config


def main(
    cfg: Config,
    models: list[str] | None = None,
    rank_metric: str = "psnr",
    rank_stratum: str = "cloud",
    **_: Any,
) -> list[dict[str, Any]]:
    """Benchmark models into a ranked leaderboard + comparative report.

    Args:
        cfg: The loaded config.
        models: Explicit list of registry names to compare. ``None`` → every
            registered model.
        rank_metric: Metric for the overall ranking column.
        rank_stratum: Stratum the ranking metric is read from (``"cloud"`` =
            the real score).
        **_: Ignored extra kwargs.

    Returns:
        The leaderboard (list of per-model dicts).
    """
    from cloudremoval.evaluation.benchmark import run_benchmark
    from cloudremoval.evaluation.report import generate_report
    from cloudremoval.models.registry import list_models
    from cloudremoval.utils.logging import get_logger
    from cloudremoval.utils.seed import seed_everything

    log = get_logger("scripts.benchmark")
    seed_everything(cfg.train.seed)

    names = models if models is not None else list_models()
    if not names:
        log.warning("No models registered — nothing to benchmark.")
        return []

    loader = _build_eval_loader(cfg)
    out_dir = Path(cfg.eval.out_dir) / "benchmark"
    leaderboard = run_benchmark(
        names,
        loader,
        cfg,
        out_dir=out_dir,
        rank_metric=rank_metric,
        rank_stratum=rank_stratum,
    )

    report_path = out_dir / "comparative_assessment.md"
    generate_report(leaderboard, report_path, title="Comparative Assessment — Cloud Removal")
    log.info("Leaderboard + report written under %s", out_dir)
    return leaderboard


def _build_eval_loader(cfg: Config) -> Any:
    """Build a shared evaluation dataloader via the B1 data layer (lazy import)."""
    from cloudremoval.data.datasets import build_dataloader

    for split in ("val", "test", "train"):
        try:
            return build_dataloader(cfg.data, split)
        except Exception:  # noqa: BLE001 - try the next split
            continue
    raise RuntimeError("could not build any evaluation dataloader (val/test/train).")


def _cli() -> None:
    """Standalone argparse entry."""
    import argparse

    from cloudremoval.config import load_config
    from cloudremoval.utils.seed import seed_everything

    parser = argparse.ArgumentParser(description="Benchmark cloud-removal models.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Registry names to compare (default: all registered).",
    )
    parser.add_argument("--rank-metric", default="psnr", help="Overall ranking metric.")
    parser.add_argument("--rank-stratum", default="cloud", help="Stratum for ranking.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    main(cfg, models=args.models, rank_metric=args.rank_metric, rank_stratum=args.rank_stratum)


if __name__ == "__main__":
    _cli()
