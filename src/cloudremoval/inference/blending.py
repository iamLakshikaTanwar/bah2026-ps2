"""Tile-overlap blending windows for seam-free reassembly.

When a large scene is reconstructed tile-by-tile (``inference/tiled.py``), abutting
tiles disagree slightly at their shared edges, producing visible seams. The fix is
**overlap (halo) + feathered weighting**: tiles overlap by ``halo`` pixels and each
tile's contribution is down-weighted toward its border by a smooth 2-D window
(Hann / cosine, Gaussian, or a linear ramp). Overlapping windows sum to a positive
weight everywhere, so a weighted average produces a continuous mosaic
(``research/05`` §B.4 / §C.3).

This module is **pure NumPy/torch** so the CPU-smoke path needs no heavy deps. The
window cache keeps repeated tiles cheap, and :func:`blend_tiles` accumulates
``sum(weight * tile)`` and ``sum(weight)`` in two buffers, dividing once at the end
(memory-safe, single normalization).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "hann_window_1d",
    "hann_window_2d",
    "gaussian_window_2d",
    "linear_window_2d",
    "make_window",
    "BlendAccumulator",
    "blend_tiles",
]

# Window kinds accepted by :func:`make_window` (matches ``InferConfig.blend``
# plus a couple of useful extras).
_WINDOW_KINDS = ("hann", "cosine", "gaussian", "linear", "none")


# --------------------------------------------------------------------------- #
# 1-D windows
# --------------------------------------------------------------------------- #
def hann_window_1d(size: int) -> np.ndarray:
    """Return a 1-D raised-cosine (Hann) window of length ``size``.

    The window is strictly positive in its interior (endpoints are nudged off
    zero) so that, combined with overlap, the summed weight never vanishes and
    the final division is safe.

    Args:
        size: Window length in pixels (``>= 1``).

    Returns:
        ``float32`` array of shape ``[size]`` in ``(0, 1]``.
    """
    if size <= 1:
        return np.ones(max(size, 1), dtype=np.float32)
    n = np.arange(size, dtype=np.float32)
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (size - 1))
    # Keep endpoints strictly positive to guarantee full coverage.
    win = np.clip(win, 1e-3, None)
    return win.astype(np.float32)


def _gaussian_window_1d(size: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Return a 1-D Gaussian window (std = ``sigma_frac * size``)."""
    if size <= 1:
        return np.ones(max(size, 1), dtype=np.float32)
    n = np.arange(size, dtype=np.float32)
    center = (size - 1) / 2.0
    sigma = max(sigma_frac * size, 1e-3)
    win = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    return np.clip(win, 1e-3, None).astype(np.float32)


def _linear_window_1d(size: int, ramp: int) -> np.ndarray:
    """Return a 1-D trapezoid: linear ramp-up of ``ramp`` px, flat, ramp-down."""
    if size <= 1:
        return np.ones(max(size, 1), dtype=np.float32)
    win = np.ones(size, dtype=np.float32)
    ramp = int(max(0, min(ramp, size // 2)))
    if ramp > 0:
        edge = (np.arange(ramp, dtype=np.float32) + 1.0) / (ramp + 1.0)
        win[:ramp] = edge
        win[-ramp:] = edge[::-1]
    return np.clip(win, 1e-3, None).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2-D windows (separable outer products)
# --------------------------------------------------------------------------- #
def hann_window_2d(height: int, width: int) -> np.ndarray:
    """Return a separable 2-D Hann window ``[height, width]`` (``float32``)."""
    wy = hann_window_1d(height)
    wx = hann_window_1d(width)
    return np.outer(wy, wx).astype(np.float32)


def gaussian_window_2d(height: int, width: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Return a separable 2-D Gaussian window ``[height, width]`` (``float32``)."""
    wy = _gaussian_window_1d(height, sigma_frac)
    wx = _gaussian_window_1d(width, sigma_frac)
    return np.outer(wy, wx).astype(np.float32)


def linear_window_2d(height: int, width: int, ramp: int) -> np.ndarray:
    """Return a separable 2-D trapezoid (feather) window ``[height, width]``."""
    wy = _linear_window_1d(height, ramp)
    wx = _linear_window_1d(width, ramp)
    return np.outer(wy, wx).astype(np.float32)


@lru_cache(maxsize=64)
def _cached_window(kind: str, height: int, width: int, ramp: int) -> np.ndarray:
    """Build and cache a 2-D window so identical tile sizes reuse one array."""
    if kind in ("hann", "cosine"):
        win = hann_window_2d(height, width)
    elif kind == "gaussian":
        win = gaussian_window_2d(height, width)
    elif kind == "linear":
        win = linear_window_2d(height, width, ramp)
    elif kind == "none":
        win = np.ones((height, width), dtype=np.float32)
    else:  # pragma: no cover - guarded by make_window
        raise ValueError(f"unknown blend window '{kind}'")
    win.setflags(write=False)  # protect the cached array from mutation
    return win


def make_window(
    height: int,
    width: int,
    kind: str = "hann",
    ramp: int | None = None,
) -> np.ndarray:
    """Return a 2-D blend window for a tile of shape ``[height, width]``.

    Args:
        height: Tile height in pixels.
        width: Tile width in pixels.
        kind: One of ``{"hann", "cosine", "gaussian", "linear", "none"}``.
            ``"hann"``/``"cosine"`` are equivalent (raised cosine). ``"none"``
            yields a uniform window (simple averaging in overlaps).
        ramp: Feather width for ``kind="linear"`` (defaults to ``min(h, w)//4``).

    Returns:
        A read-only ``float32`` array ``[height, width]`` with strictly positive
        values (safe to use as blend weights).

    Raises:
        ValueError: If ``kind`` is not recognised.
    """
    if kind not in _WINDOW_KINDS:
        raise ValueError(f"unknown blend window '{kind}'; expected one of {_WINDOW_KINDS}")
    if ramp is None:
        ramp = max(1, min(height, width) // 4)
    return _cached_window(kind, int(height), int(width), int(ramp))


# --------------------------------------------------------------------------- #
# Weighted-overlap accumulation
# --------------------------------------------------------------------------- #
class BlendAccumulator:
    """Accumulate weighted tiles into a full canvas, normalising once at the end.

    Holds two buffers — ``num = sum(weight * tile)`` and ``den = sum(weight)`` —
    and divides them in :meth:`result`. This is the memory-safe heart of tiled
    reassembly: only the full-scene buffers (not all tiles) live at once, and a
    single division avoids per-tile normalization error.

    Args:
        channels: Number of channels ``C`` in the output.
        height: Full canvas height ``H``.
        width: Full canvas width ``W``.
        dtype: Accumulation dtype (``float32`` by default).
    """

    def __init__(self, channels: int, height: int, width: int, dtype: type = np.float32) -> None:
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self._num = np.zeros((self.channels, self.height, self.width), dtype=dtype)
        self._den = np.zeros((1, self.height, self.width), dtype=dtype)

    def add(self, tile: np.ndarray, top: int, left: int, weight: np.ndarray) -> None:
        """Add one ``[C, h, w]`` tile at ``(top, left)`` with a ``[h, w]`` weight.

        Tiles are clipped to the canvas, so windows that hang over an edge (from
        reflect-padding) contribute only their in-bounds portion.

        Args:
            tile: Channel-first tile ``[C, h, w]``.
            top: Row of the tile's top-left corner in the canvas.
            left: Column of the tile's top-left corner in the canvas.
            weight: 2-D blend window ``[h, w]`` (e.g. from :func:`make_window`).
        """
        tile = np.asarray(tile, dtype=self._num.dtype)
        if tile.ndim == 2:
            tile = tile[None, ...]
        _, th, tw = tile.shape

        # Intersect the tile footprint with the canvas bounds.
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + th, self.height), min(left + tw, self.width)
        if y1 <= y0 or x1 <= x0:
            return  # fully outside the canvas

        ty0, tx0 = y0 - top, x0 - left
        ty1, tx1 = ty0 + (y1 - y0), tx0 + (x1 - x0)

        w2d = np.asarray(weight, dtype=self._num.dtype)[ty0:ty1, tx0:tx1]
        self._num[:, y0:y1, x0:x1] += tile[:, ty0:ty1, tx0:tx1] * w2d[None, ...]
        self._den[:, y0:y1, x0:x1] += w2d[None, ...]

    def result(self, eps: float = 1e-8) -> np.ndarray:
        """Return the normalised canvas ``[C, H, W]`` (``num / den``).

        Args:
            eps: Floor added to the denominator to avoid division by zero in any
                pixel no tile covered (should not happen with full tiling).

        Returns:
            The blended full-scene array, ``float32``.
        """
        den = np.maximum(self._den, eps)
        return (self._num / den).astype(np.float32)

    @property
    def coverage(self) -> np.ndarray:
        """Per-pixel summed weight ``[1, H, W]`` (zero where uncovered)."""
        return self._den


def blend_tiles(
    tiles: list[np.ndarray],
    positions: list[tuple[int, int]],
    canvas_shape: tuple[int, int, int],
    kind: str = "hann",
) -> np.ndarray:
    """Reassemble overlapping ``tiles`` into a seam-free canvas.

    Convenience wrapper over :class:`BlendAccumulator`: builds a window per tile
    size (cached) and accumulates a weighted average.

    Args:
        tiles: List of channel-first tiles ``[C, h, w]`` (sizes may vary).
        positions: ``(top, left)`` canvas coordinate of each tile's corner.
        canvas_shape: Target full-scene shape ``(C, H, W)``.
        kind: Blend window kind (see :func:`make_window`).

    Returns:
        The blended canvas ``[C, H, W]`` (``float32``).

    Raises:
        ValueError: If ``tiles`` and ``positions`` differ in length.
    """
    if len(tiles) != len(positions):
        raise ValueError("tiles and positions must have equal length")
    c, h, w = canvas_shape
    acc = BlendAccumulator(c, h, w)
    for tile, (top, left) in zip(tiles, positions, strict=True):
        arr = np.asarray(tile, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        win = make_window(arr.shape[1], arr.shape[2], kind=kind)
        acc.add(arr, top, left, win)
    return acc.result()
