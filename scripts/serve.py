"""Runnable wrapper to launch the serving API (``cloudremoval serve``).

Invoked by ``cli.py`` as ``scripts.serve.main(cfg)`` (BUILD_PLAN §3.10), and
runnable directly as ``python scripts/serve.py --config configs/serve/serve.yaml``.
It builds the FastAPI app (``serving/app.py:create_app``) bound to ``cfg`` and
serves it with uvicorn on ``cfg.serve.host:cfg.serve.port``. The app imports and
answers ``/health`` on the minimal CPU stack (no heavy deps, no GPU).

A ``sys.path`` bootstrap makes ``cloudremoval`` importable without an editable
install and without a ``scripts`` package (B1 owns ``scripts/__init__.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# --- sys.path bootstrap (run before any cloudremoval import) --------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudremoval.config import Config, load_config  # noqa: E402
from cloudremoval.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


def main(cfg: Config, **kwargs: Any) -> None:
    """Launch the uvicorn server hosting the cloud-removal API.

    Args:
        cfg: The loaded :class:`Config` (uses ``serve.host`` / ``serve.port`` and
            passes the whole config to the app factory).
        **kwargs: Unused; accepted for the ``scripts.<name>.main(cfg, **kwargs)``
            contract.
    """
    import uvicorn

    from cloudremoval.serving.app import create_app

    application = create_app(cfg)
    host, port = cfg.serve.host, cfg.serve.port
    _log.info("Serving cloudremoval API on http://%s:%d (cache=%s)", host, port, cfg.serve.cache)
    uvicorn.run(application, host=host, port=port, log_level="info")


def _cli() -> None:
    """Standalone CLI (``python scripts/serve.py --config configs/serve/serve.yaml``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch the cloud-removal serving API.")
    parser.add_argument("--config", "-c", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)


if __name__ == "__main__":
    _cli()
