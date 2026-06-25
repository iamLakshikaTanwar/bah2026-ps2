"""Runnable pipeline scripts invoked by the ``cloudremoval`` CLI.

``cloudremoval.cli`` lazily imports ``scripts.<name>`` and calls its
``main(cfg, **kwargs)`` (BUILD_PLAN §3.10). Making ``scripts`` a package gives that
import a stable target. Each module also bootstraps ``sys.path`` so it can be run
directly (``python scripts/<name>.py``) outside an editable install.

B1 owns ``download``, ``preprocess`` and ``simulate``; B3/B4 own the rest.
"""

from __future__ import annotations

__all__: list[str] = []
