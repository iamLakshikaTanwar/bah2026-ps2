"""Runnable evaluation entry point — ``scripts/eval.py``.

Thin wrapper the CLI delegates to: ``cloudremoval eval --config ... [--ckpt ...]``
calls :func:`main(cfg, ckpt=...)` here. Builds the model + a val dataloader,
optionally loads a checkpoint, runs the masked/stratified evaluator, and writes a
Markdown report next to the metrics.

Run directly::

    python scripts/eval.py --config configs/cpu_smoke.yaml --ckpt outputs/smoke/ckpt/final.pt
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


def main(cfg: Config, ckpt: Any | None = None, **_: Any) -> dict[str, Any]:
    """Evaluate the configured model with masked/stratified MVES metrics.

    Args:
        cfg: The loaded config.
        ckpt: Optional checkpoint to load weights from before evaluating.
        **_: Ignored extra kwargs.

    Returns:
        The aggregated results dict from
        :func:`cloudremoval.evaluation.evaluator.evaluate`.
    """
    import torch

    from cloudremoval.evaluation.evaluator import evaluate
    from cloudremoval.evaluation.report import generate_report
    from cloudremoval.models.registry import build_model
    from cloudremoval.utils.logging import get_logger
    from cloudremoval.utils.seed import seed_everything

    log = get_logger("scripts.eval")
    seed_everything(cfg.train.seed)

    model = build_model(cfg)
    if ckpt is not None:
        blob = torch.load(Path(ckpt), map_location=cfg.train.device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(state)
        log.info("Loaded checkpoint %s", ckpt)

    loader = _build_eval_loader(cfg)
    results = evaluate(model, loader, cfg, save=True)

    report_path = Path(cfg.eval.out_dir) / "report.md"
    generate_report(results, report_path, title=f"Evaluation — {results.get('model','model')}")
    log.info("Evaluation report at %s", report_path)
    return results


def _build_eval_loader(cfg: Config) -> Any:
    """Build a validation/test dataloader via the B1 data layer (lazy import)."""
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

    parser = argparse.ArgumentParser(description="Evaluate a cloud-removal model.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint to evaluate.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    main(cfg, ckpt=args.ckpt)


if __name__ == "__main__":
    _cli()
