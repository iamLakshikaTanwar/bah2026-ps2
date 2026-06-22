"""Tiling / patchification with overlap, windowed reads and COG creation.

Whole LISS-IV scenes are processed as overlapping tiles with a ``halo`` and
reassembled with feathered (Hann) blending to avoid seam discontinuities
(``research/05`` §B.4). The same sliding-window+halo machinery is reused at
inference (BUILD_PLAN §3.8 / §4 B4 integrates this).

Public API
----------
:func:`tile_array` / :func:`untile_array`
    Pure-NumPy patchification of a ``[C, H, W]`` array into overlapping tiles and
    blended reconstruction back to the full scene.
:class:`TileIndex`
    A lightweight record of tile positions for a scene (the "simple tile index").
:func:`windowed_read`
    Read a single window from a raster -- lazy ``rasterio`` windowed read, NumPy
    slice fallback for in-memory arrays / ``.npy``.
:func:`create_cog`
    Write a Cloud-Optimized GeoTIFF -- lazy ``rio-cogeo``/``rasterio``; ``.npy``
    fallback (with a JSON profile sidecar) on the minimal stack.

Only NumPy is imported at module load; all geospatial deps are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cloudremoval.utils.logging import get_logger

__all__ = [
    "TilePosition",
    "TileIndex",
    "tile_array",
    "untile_array",
    "hann_window_2d",
    "windowed_read",
    "create_cog",
]

_log = get_logger(__name__)
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Tile index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TilePosition:
    """Location of one tile within a scene.

    Attributes:
        row: Top pixel row of the tile in the source scene.
        col: Left pixel column of the tile in the source scene.
        height: Tile height in pixels.
        width: Tile width in pixels.
    """

    row: int
    col: int
    height: int
    width: int


@dataclass
class TileIndex:
    """Simple index of tile positions for a scene of shape ``(H, W)``.

    Attributes:
        scene_height: Source scene height.
        scene_width: Source scene width.
        tile_size: Nominal tile size (interior, excluding halo overlap).
        halo: Overlap halo in pixels.
        positions: Ordered list of :class:`TilePosition`.
    """

    scene_height: int
    scene_width: int
    tile_size: int
    halo: int
    positions: list[TilePosition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.positions)


def _tile_positions(h: int, w: int, tile: int, halo: int) -> list[TilePosition]:
    """Compute overlapping tile origins covering an ``(h, w)`` scene.

    Tiles step by ``tile`` (the interior stride) and each tile spans
    ``tile + 2*halo``, clamped to the scene; the last row/column is snapped so the
    scene edge is fully covered.
    """
    span = tile + 2 * halo
    span = min(span, max(h, w)) if (h < span or w < span) else span
    rows = list(range(0, max(1, h - halo), tile)) if h > span else [0]
    cols = list(range(0, max(1, w - halo), tile)) if w > span else [0]

    positions: list[TilePosition] = []
    seen: set[tuple[int, int]] = set()
    for r in rows:
        r0 = min(max(0, r - halo), max(0, h - span))
        for c in cols:
            c0 = min(max(0, c - halo), max(0, w - span))
            th = min(span, h)
            tw = min(span, w)
            key = (r0, c0)
            if key in seen:
                continue
            seen.add(key)
            positions.append(TilePosition(row=r0, col=c0, height=th, width=tw))
    return positions


# --------------------------------------------------------------------------- #
# Tiling / untiling (pure NumPy)
# --------------------------------------------------------------------------- #
def tile_array(
    array: np.ndarray,
    tile_size: int,
    halo: int = 0,
) -> tuple[list[np.ndarray], TileIndex]:
    """Split a ``[C, H, W]`` (or ``[H, W]``) array into overlapping tiles.

    Args:
        array: Channel-first image (or 2-D single band).
        tile_size: Interior stride between tiles (tiles span ``tile_size+2*halo``).
        halo: Overlap halo in pixels (0 = non-overlapping).

    Returns:
        ``(tiles, index)`` -- ``tiles`` is a list of ``[C, th, tw]`` arrays in
        index order; ``index`` is the :class:`TileIndex` describing positions.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, ...]
    _, h, w = arr.shape
    positions = _tile_positions(h, w, int(tile_size), int(halo))
    tiles = [arr[:, p.row : p.row + p.height, p.col : p.col + p.width].copy() for p in positions]
    index = TileIndex(
        scene_height=h,
        scene_width=w,
        tile_size=int(tile_size),
        halo=int(halo),
        positions=positions,
    )
    return tiles, index


def hann_window_2d(height: int, width: int) -> np.ndarray:
    """Separable 2-D Hann (cosine) window for feathered tile blending.

    The window is floored at a small positive value so fully-covered interior
    pixels still receive non-zero weight (avoids divide-by-zero in seams).

    Args:
        height: Window height.
        width: Window width.

    Returns:
        ``[height, width]`` weight array in ``(0, 1]`` (``float32``).
    """
    wy = np.hanning(height + 2)[1:-1] if height > 1 else np.ones(1)
    wx = np.hanning(width + 2)[1:-1] if width > 1 else np.ones(1)
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 1e-3, 1.0)


def untile_array(
    tiles: list[np.ndarray],
    index: TileIndex,
    blend: str = "hann",
) -> np.ndarray:
    """Reassemble overlapping tiles into a full ``[C, H, W]`` scene.

    Overlap regions are combined with feathered Hann weights (``blend="hann"``) or
    simple averaging (``blend="mean"``/``"none"``) to avoid seam artefacts
    (``research/05`` §B.4).

    Args:
        tiles: Tiles in the same order as ``index.positions``.
        index: The :class:`TileIndex` from :func:`tile_array`.
        blend: ``"hann"`` (feathered) or ``"mean"``/``"none"`` (uniform average).

    Returns:
        The reconstructed scene ``[C, H, W]`` (``float32``).

    Raises:
        ValueError: If the number of tiles does not match the index.
    """
    if len(tiles) != len(index.positions):
        raise ValueError(f"got {len(tiles)} tiles for {len(index.positions)} positions")
    c = np.asarray(tiles[0]).shape[0] if tiles[0].ndim == 3 else 1
    h, w = index.scene_height, index.scene_width
    acc = np.zeros((c, h, w), dtype=np.float64)
    wsum = np.zeros((1, h, w), dtype=np.float64)

    for tile, pos in zip(tiles, index.positions, strict=True):
        t = np.asarray(tile, dtype=np.float64)
        if t.ndim == 2:
            t = t[None, ...]
        th, tw = t.shape[1], t.shape[2]
        if blend == "hann":
            weight = hann_window_2d(th, tw)[None, ...]
        else:
            weight = np.ones((1, th, tw), dtype=np.float32)
        acc[:, pos.row : pos.row + th, pos.col : pos.col + tw] += t * weight
        wsum[:, pos.row : pos.row + th, pos.col : pos.col + tw] += weight

    out = acc / np.maximum(wsum, _EPS)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Windowed read (lazy rasterio; NumPy fallback)
# --------------------------------------------------------------------------- #
def windowed_read(
    source: Any,
    row: int,
    col: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Read a ``[C, height, width]`` window at ``(row, col)`` from a raster/array.

    For a file path with a ``.tif``/``.tiff`` extension this performs a true
    windowed read via lazy ``rasterio`` (HTTP range / block read -- the literal
    "O(1) tile read", ``research/05`` §C.1). For an in-memory NumPy array (or a
    ``.npy`` path) it slices the array. Out-of-bounds windows are zero-padded.

    Args:
        source: A ``[C, H, W]`` NumPy array, or a path to a GeoTIFF/``.npy``.
        row: Top row of the window.
        col: Left column of the window.
        height: Window height.
        width: Window width.

    Returns:
        The window ``[C, height, width]`` (``float32``).
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() in {".tif", ".tiff"}:
            window = _rasterio_window(p, row, col, height, width)
            if window is not None:
                return window
            p = p.with_suffix(".npy")
        arr = np.load(p)
    else:
        arr = np.asarray(source)
    if arr.ndim == 2:
        arr = arr[None, ...]

    c, h, w = arr.shape
    out = np.zeros((c, height, width), dtype=np.float32)
    r1, c1 = min(row + height, h), min(col + width, w)
    r0, c0 = max(row, 0), max(col, 0)
    if r1 > r0 and c1 > c0:
        out[:, r0 - row : r1 - row, c0 - col : c1 - col] = arr[:, r0:r1, c0:c1]
    return out


def _rasterio_window(
    path: Path,
    row: int,
    col: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    """True windowed GeoTIFF read via lazy ``rasterio``; ``None`` if unavailable."""
    try:
        import rasterio  # type: ignore  (lazy, optional)
        from rasterio.windows import Window  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        with rasterio.open(path) as src:
            data = src.read(window=Window(col, row, width, height), boundless=True, fill_value=0)
        return np.asarray(data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        _log.warning("rasterio windowed read failed (%s); falling back", exc)
        return None


# --------------------------------------------------------------------------- #
# COG creation (lazy rio-cogeo / rasterio; .npy fallback)
# --------------------------------------------------------------------------- #
def create_cog(
    array: np.ndarray,
    out_path: str | Path,
    profile: dict[str, Any] | None = None,
    blocksize: int = 512,
    overviews: bool = True,
) -> str:
    """Write a ``[C, H, W]`` array as a Cloud-Optimized GeoTIFF (lazy deps).

    Tries ``rio-cogeo`` first, then a tiled ``rasterio`` GeoTIFF with internal
    overviews; if neither is installed (the CPU-smoke stack) it falls back to a
    ``.npy`` plus a JSON profile sidecar so the pipeline still produces an output.

    Args:
        array: Channel-first image.
        out_path: Output ``.tif`` path (``.npy`` written on fallback).
        profile: Optional rasterio profile (crs, transform, nodata, ...).
        blocksize: Internal tile size for the COG.
        overviews: Whether to build internal overviews.

    Returns:
        The path actually written.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, ...]
    out = Path(out_path)

    written = _create_cog_rio_cogeo(arr, out, profile, blocksize, overviews)
    if written is not None:
        return written
    written = _create_cog_rasterio(arr, out, profile, blocksize, overviews)
    if written is not None:
        return written

    # Pure-NumPy fallback (no geospatial deps available).
    from cloudremoval.utils.io import save_npy

    npy = out.with_suffix(".npy")
    save_npy(arr, npy)
    if profile:
        _write_profile_sidecar(profile, out.with_suffix(".json"))
    _log.info("COG deps unavailable; wrote NumPy fallback %s", npy)
    return str(npy)


def _create_cog_rio_cogeo(
    arr: np.ndarray,
    out: Path,
    profile: dict[str, Any] | None,
    blocksize: int,
    overviews: bool,
) -> str | None:
    """COG via ``rio-cogeo`` if installed, else ``None``."""
    try:
        import rasterio  # type: ignore
        from rasterio.io import MemoryFile  # type: ignore
        from rio_cogeo.cogeo import cog_translate  # type: ignore
        from rio_cogeo.profiles import cog_profiles  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        count, height, width = arr.shape
        src_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": str(arr.dtype),
        }
        if profile:
            src_profile.update(
                {k: v for k, v in profile.items() if k in {"crs", "transform", "nodata"}}
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        dst_profile = cog_profiles.get("deflate")
        dst_profile.update({"blockxsize": blocksize, "blockysize": blocksize})
        with MemoryFile() as mem:
            with mem.open(**src_profile) as src:
                src.write(arr)
                cog_translate(
                    src,
                    str(out),
                    dst_profile,
                    overview_level=5 if overviews else 0,
                    in_memory=True,
                    quiet=True,
                )
        return str(out)
    except Exception as exc:  # noqa: BLE001
        _log.warning("rio-cogeo COG write failed (%s); trying rasterio", exc)
        return None


def _create_cog_rasterio(
    arr: np.ndarray,
    out: Path,
    profile: dict[str, Any] | None,
    blocksize: int,
    overviews: bool,
) -> str | None:
    """Tiled GeoTIFF with overviews via ``rasterio`` if installed, else ``None``."""
    try:
        import rasterio  # type: ignore
        from rasterio.enums import Resampling  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        count, height, width = arr.shape
        prof = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": str(arr.dtype),
            "tiled": True,
            "blockxsize": blocksize,
            "blockysize": blocksize,
            "compress": "deflate",
        }
        if profile:
            prof.update(
                {k: v for k, v in profile.items() if k in {"crs", "transform", "nodata"}}
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr)
            if overviews:
                dst.build_overviews([2, 4, 8, 16], Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
        return str(out)
    except Exception as exc:  # noqa: BLE001
        _log.warning("rasterio COG write failed (%s); falling back to .npy", exc)
        return None


def _write_profile_sidecar(profile: dict[str, Any], path: Path) -> None:
    """Write a JSON sidecar with the JSON-serialisable parts of a profile."""
    import json

    def _safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({k: _safe(v) for k, v in profile.items()}, fh, indent=2)
