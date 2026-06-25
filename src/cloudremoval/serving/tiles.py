"""Dynamic XYZ/WMTS tile router with an O(1) content-addressed cache + FAISS hook.

This is the cloud-native serve path (``research/05`` §C). The design goal is
**O(1)-per-tile** serving, achieved by three composable pieces — every one of which
degrades gracefully when its optional library is absent, so the router *mounts and
imports on the minimal stack*:

1. **Dynamic tiling (rio-tiler / TiTiler, lazy).** ``GET /tiles/{z}/{x}/{y}.png``
   renders a single XYZ tile straight from a COG using ``rio-tiler``, which issues
   an HTTP **ranged GET** for just that tile's bytes — *O(1) read* regardless of
   scene size. If ``rio-tiler`` is not installed the endpoint returns ``503`` with
   a JSON explanation instead of crashing.

2. **Content-addressed tile cache (Redis, lazy; in-memory fallback).** The cache
   key is ``hash(tile_identity + model_version + params)`` (``research/05`` §C.1):
   identical inputs are never recomputed, so a cache hit is an *O(1)* dictionary /
   Redis ``GET``. Redis is used when reachable (hot tiles shared across workers /
   fronted by a CDN for edge O(1) repeats); otherwise a process-local ``dict``
   bounded by an LRU keeps the same semantics for a single node.

3. **FAISS nearest clear-reference hook (lazy).** ``retrieve_clear_reference``
   ANN-searches an index of historical clear-patch embeddings to find the best
   cloud-free analog to condition the fill. Approximate-NN is sublinear → *O(1)-ish*
   per query at our scale. Without ``faiss`` it returns ``None`` (the model simply
   runs unconditioned).

The router never imports a heavy dependency at module load; all of them are
imported inside the function that needs them.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "router",
    "TileCache",
    "content_key",
    "retrieve_clear_reference",
    "get_tile_cache",
]

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Content-addressed cache (O(1) lookup): Redis when available, else in-memory LRU
# --------------------------------------------------------------------------- #
def content_key(*parts: Any) -> str:
    """Hash request identity into a stable cache key (``research/05`` §C.1).

    The key combines whatever uniquely identifies a tile's *output*: the scene /
    tile bytes (or its id), the model version, and the render params. Identical
    inputs therefore collide to the same key and reuse the cached tile — an O(1)
    lookup that never recomputes.

    Args:
        *parts: Any hashable-as-string identity components (scene id, z/x/y,
            model name+version, params).

    Returns:
        A hex SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


class TileCache:
    """Content-addressed byte cache with a Redis backend and in-memory fallback.

    ``get``/``set`` are O(1): a Redis ``GET``/``SETEX`` when a server is reachable,
    otherwise a bounded process-local ``OrderedDict`` (LRU). The two backends share
    one interface so callers (and tests) are backend-agnostic; the minimal stack
    transparently uses the in-memory map.

    Args:
        backend: ``"memory"`` or ``"redis"``.
        redis_url: Redis DSN (used only for ``backend="redis"``).
        max_items: LRU capacity for the in-memory fallback.
        ttl: Time-to-live (seconds) for Redis entries.
    """

    def __init__(
        self,
        backend: str = "memory",
        redis_url: str | None = None,
        max_items: int = 256,
        ttl: int = 3600,
    ) -> None:
        self.backend = backend
        self.redis_url = redis_url
        self.max_items = max_items
        self.ttl = ttl
        self._mem: OrderedDict[str, bytes] = OrderedDict()
        self._redis = self._connect_redis() if backend == "redis" else None

    def _connect_redis(self) -> Any | None:
        """Try to connect to Redis (lazy import); fall back to memory on failure."""
        try:  # pragma: no cover - requires the optional 'serve' extra + a server
            import redis

            client = redis.Redis.from_url(self.redis_url or "redis://localhost:6379/0")
            client.ping()
            _log.info("TileCache: connected to Redis at %s", self.redis_url)
            return client
        except Exception as exc:  # noqa: BLE001 - any failure -> memory fallback
            _log.warning("TileCache: Redis unavailable (%s); using in-memory LRU.", exc)
            self.backend = "memory"
            return None

    def get(self, key: str) -> bytes | None:
        """Return cached bytes for ``key`` (O(1)), or ``None`` on a miss."""
        if self._redis is not None:  # pragma: no cover - needs a live server
            return self._redis.get(key)
        if key in self._mem:
            self._mem.move_to_end(key)  # LRU touch
            return self._mem[key]
        return None

    def set(self, key: str, value: bytes) -> None:
        """Store ``value`` under ``key`` (O(1)); evicts LRU in memory mode."""
        if self._redis is not None:  # pragma: no cover - needs a live server
            self._redis.setex(key, self.ttl, value)
            return
        self._mem[key] = value
        self._mem.move_to_end(key)
        while len(self._mem) > self.max_items:
            self._mem.popitem(last=False)

    def clear(self) -> None:
        """Drop all in-memory entries (Redis is left untouched)."""
        self._mem.clear()


# Process-wide default cache (memory backend) so the router works with no config.
_DEFAULT_CACHE: TileCache | None = None


def get_tile_cache(backend: str = "memory", redis_url: str | None = None) -> TileCache:
    """Return a process-wide :class:`TileCache`, creating it on first use."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = TileCache(backend=backend, redis_url=redis_url)
    return _DEFAULT_CACHE


# --------------------------------------------------------------------------- #
# FAISS nearest clear-reference retrieval (lazy)
# --------------------------------------------------------------------------- #
def retrieve_clear_reference(
    query_embedding: np.ndarray,
    index: Any | None = None,
    k: int = 1,
) -> np.ndarray | None:
    """ANN-retrieve the nearest clear-reference embedding(s) via FAISS (lazy).

    At inference, the best cloud-free analog of the query patch can condition /
    guide the fill (``research/05`` §C.3). FAISS approximate-NN is sublinear, so
    this is effectively O(1) per query at our scale. Returns ``None`` when
    ``faiss`` is unavailable or no index is supplied (the caller then runs the
    model unconditioned), so the smoke path is unaffected.

    Args:
        query_embedding: ``[D]`` or ``[N, D]`` float query vector(s).
        index: A pre-built ``faiss.Index`` (or ``None`` for a no-op).
        k: Number of neighbours to return.

    Returns:
        ``[N, k]`` neighbour indices as a NumPy array, or ``None`` if retrieval
        is unavailable.
    """
    if index is None:
        return None
    try:  # pragma: no cover - requires the optional 'accel' extra
        import faiss  # noqa: F401
        import numpy as np

        q = np.asarray(query_embedding, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]
        _distances, indices = index.search(q, k)
        return indices
    except ImportError:
        _log.debug("faiss not installed; clear-reference retrieval disabled.")
        return None


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
router = APIRouter(prefix="/tiles", tags=["tiles"])


def _placeholder_png() -> bytes:
    """Return a tiny 1x1 PNG (used when rio-tiler is absent but a tile is asked).

    Built byte-for-byte with the stdlib ``zlib``/``struct`` so it needs no Pillow
    — keeps the tiles router functional on the minimal stack for smoke checks.
    """
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    width = height = 1
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"\x00" + b"\x00\x00\x00"  # one filtered scanline: filter byte + 1 RGB px
    idat = zlib.compress(raw)
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    )


@router.get("/{z}/{x}/{y}.png", summary="Dynamic XYZ tile (rio-tiler, lazy)")
def get_tile(
    z: int,
    x: int,
    y: int,
    scene: str | None = None,
    model_name: str = "unet",
    rescale: str | None = None,
) -> Response:
    """Serve one XYZ tile, cached by content hash; renders from a COG via rio-tiler.

    Flow: build the O(1) content key, return the cached PNG on a hit, else (lazily)
    render the tile from ``scene``'s COG with ``rio-tiler`` (a ranged GET — O(1)
    read), cache, and return it. If ``rio-tiler`` is missing, respond ``503`` with
    JSON (or a 1x1 placeholder when no scene is given) rather than failing.

    Args:
        z: Tile zoom.
        x: Tile column.
        y: Tile row.
        scene: COG path/URL or scene id to render from.
        model_name: Model whose output is being tiled (part of the cache key).
        rescale: ``"min,max"`` display rescale passed to rio-tiler.

    Returns:
        A ``image/png`` :class:`fastapi.Response`, or a JSON ``503``.
    """
    cache = get_tile_cache()
    key = content_key("tile", scene, model_name, z, x, y, rescale)
    hit = cache.get(key)
    if hit is not None:
        return Response(content=hit, media_type="image/png", headers={"X-Cache": "HIT"})

    if scene is None:
        # No source COG to read — return a valid placeholder so viewers don't break.
        png = _placeholder_png()
        cache.set(key, png)
        return Response(content=png, media_type="image/png", headers={"X-Cache": "MISS"})

    try:  # pragma: no cover - requires the optional 'serve' (rio-tiler) extra
        from rio_tiler.io import Reader

        with Reader(scene) as cog:
            img = cog.tile(x, y, z)
            if rescale:
                lo, hi = (float(v) for v in rescale.split(","))
                img.rescale(in_range=((lo, hi),))
            png = img.render(img_format="PNG")
        cache.set(key, png)
        return Response(content=png, media_type="image/png", headers={"X-Cache": "MISS"})
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "rio-tiler not installed; dynamic tiling unavailable.",
                "hint": "pip install 'cloudremoval[serve]'",
                "tile": {"z": z, "x": x, "y": y, "scene": scene},
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface read/render errors as JSON
        _log.warning("tile render failed for z=%d x=%d y=%d: %s", z, x, y, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"tile render failed: {exc}"},
        )


@router.get("/info", summary="Tile subsystem capabilities")
def tiles_info() -> dict[str, Any]:
    """Report which optional tile-serving backends are available (O(1) design)."""

    def _have(mod: str) -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec(mod) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    cache = get_tile_cache()
    return {
        "dynamic_tiling": _have("rio_tiler"),
        "cache_backend": cache.backend,
        "redis_available": _have("redis"),
        "faiss_available": _have("faiss"),
        "design": "COG ranged-GET reads + content-addressed cache + FAISS ANN = O(1)/tile",
    }
