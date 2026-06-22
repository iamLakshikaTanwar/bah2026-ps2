"""Serving layer (owned by B4): FastAPI app, XYZ tiles, request/response schemas.

Implemented by B4 against the serving contract (BUILD_PLAN §3.9). The app boots
on CPU with ``cache="memory"`` and no GPU; rio-tiler / Redis / FAISS are optional.
Empty-but-valid namespace until those modules land.
"""

from __future__ import annotations

__all__: list[str] = []
