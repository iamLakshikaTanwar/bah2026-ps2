"""Runnable training entry point — ``scripts/train.py``.

Thin wrapper the CLI delegates to: ``cloudremoval train --config ... [--ckpt ...]``
calls :func:`main(cfg, ckpt=...)` here (see ``cloudremoval.cli``). It bootstraps
``src/`` onto ``sys.path`` (so it runs without ``pip install -e .``), then calls
:func:`cloudremoval.training.trainer.train`.

Run directly::

    python scripts/train.py --config configs/cpu_smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# --------------------------------------------------------------------------- #
# sys.path bootstrap (no scripts/__init__.py by design; B1 owns it if needed).
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if TYPE_CHECKING:
    from cloudremoval.config import Config


def main(cfg: Config, ckpt: Any | None = None, **_: Any) -> dict[str, Any]:
    """Train the configured model.

    Args:
        cfg: The loaded :class:`~cloudremoval.config.Config`.
        ckpt: Optional checkpoint path to resume model weights from.
        **_: Ignored extra kwargs (forward-compatible with the CLI).

    Returns:
        The summary dict from :func:`cloudremoval.training.trainer.train`
        (``{"history", "best_ckpt", "final_ckpt"}``).
    """
    from cloudremoval.training.trainer import train as _train
    from cloudremoval.utils.logging import get_logger

    log = get_logger("scripts.train")
    summary = _train(cfg, ckpt=ckpt)
    log.info(
        "Training complete. final=%s best=%s",
        summary.get("final_ckpt"),
        summary.get("best_ckpt"),
    )
    return summary


def _cli() -> None:
    """Standalone argparse entry (so the script is runnable on its own)."""
    import argparse

    from cloudremoval.config import load_config
    from cloudremoval.utils.seed import seed_everything

    parser = argparse.ArgumentParser(description="Train a cloud-removal model.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    parser.add_argument("--ckpt", default=None, help="Resume from checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    main(cfg, ckpt=args.ckpt)


if __name__ == "__main__":
    _cli()
