"""Cloud-Optimized GeoTIFF (COG) writer with a dependency-free ``.npy`` fallback.

A COG is an ordinary GeoTIFF whose bytes are laid out so an HTTP client can fetch
**just the tile it needs** with a ranged ``GET`` — internal tiling + overviews +
a header that maps (z, x, y) to a byte range. This is the literal mechanism behind
"**O(1) tile read**": instead of downloading a 50 GB scene, a viewer pulls a few
hundred KB per visible tile, at any zoom, directly from object storage
(``research/05`` §B.6 / §C.1, https://cogeo.org/in-depth.html).

Implementation:

* When ``rasterio`` (+ optionally ``rio-cogeo``) is installed, :func:`write_cog`
  writes a real COG — internally tiled (512²), with power-of-two overviews and
  compression (ZSTD/DEFLATE, or LERC for controlled-loss reflectance) — and
  validates it with ``rio-cogeo`` if present.
* On the minimal CPU-smoke stack (no ``rasterio``), it falls back to a NumPy
  ``.npy`` plus a JSON sidecar carrying the geospatial profile, so the inference
  path still produces a loadable artifact for tests and offline runs.

Pure-import-safe: ``rasterio`` / ``rio-cogeo`` are imported lazily inside the
functions, never at module import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cloudremoval.utils.logging import get_logger

__all__ = ["write_cog", "validate_cog", "cog_profile"]

_log = get_logger(__name__)

# Default COG creation options (used when rasterio is available). 512² internal
# tiles + overviews are what make ranged tile reads O(1).
_DEFAULT_COG_PROFILE: dict[str, Any] = {
    "driver": "GTiff",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "compress": "ZSTD",
    "zstd_level": 9,
    "predictor": 2,
    "interleave": "pixel",
    "BIGTIFF": "IF_SAFER",
}
_OVERVIEW_LEVELS = (2, 4, 8, 16, 32)


def cog_profile(
    count: int,
    height: int,
    width: int,
    dtype: str,
    transform: Any | None = None,
    crs: Any | None = None,
    nodata: float | None = None,
) -> dict[str, Any]:
    """Build a rasterio profile dict for a COG with the standard tiling options.

    Args:
        count: Number of bands.
        height: Raster height.
        width: Raster width.
        dtype: NumPy dtype string (e.g. ``"float32"``).
        transform: Affine geotransform (rasterio ``Affine`` or 6-tuple) or None.
        crs: CRS (rasterio ``CRS``, EPSG string/int) or None.
        nodata: Optional nodata value.

    Returns:
        A profile mapping suitable for ``rasterio.open(..., "w", **profile)``.
    """
    profile = dict(_DEFAULT_COG_PROFILE)
    profile.update(
        {
            "count": int(count),
            "height": int(height),
            "width": int(width),
            "dtype": dtype,
        }
    )
    if transform is not None:
        profile["transform"] = transform
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata
    return profile


def write_cog(
    array: Any,
    path: str | Path,
    transform: Any | None = None,
    crs: Any | None = None,
    nodata: float | None = None,
    uncertainty: Any | None = None,
    overviews: bool = True,
    compress: str = "ZSTD",
) -> str:
    """Write ``array`` as an analysis-ready COG (lazy rasterio; ``.npy`` fallback).

    Args:
        array: Image to write, ``[C, H, W]`` channel-first (2-D is promoted to a
            single band).
        path: Output path. A ``.tif``/``.tiff`` suffix requests a real COG; any
            other suffix (or missing ``rasterio``) writes ``.npy`` + JSON sidecar.
        transform: Affine geotransform (or 6-tuple) for georeferencing.
        crs: Coordinate reference system (EPSG string/int or rasterio CRS).
        nodata: Optional nodata value baked into the COG.
        uncertainty: Optional ``[1, H, W]`` / ``[H, W]`` per-pixel uncertainty
            appended as a trailing band (the contract's "uncertainty band").
        overviews: Whether to build internal overviews (enables O(1) reads at low
            zoom). Ignored on the ``.npy`` fallback.
        compress: COG compression (``"ZSTD"``, ``"DEFLATE"``, ``"LERC"``, ...).

    Returns:
        The path actually written (the COG ``.tif`` or the ``.npy`` fallback).
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=np.float32)
        if unc.ndim == 2:
            unc = unc[None, ...]
        arr = np.concatenate([arr, unc], axis=0)

    out = Path(path)
    wants_tif = out.suffix.lower() in {".tif", ".tiff"}
    if wants_tif:
        try:
            return _write_cog_rasterio(
                arr, out, transform, crs, nodata, overviews, compress
            )
        except ImportError:
            _log.warning(
                "rasterio unavailable; writing NumPy fallback for %s "
                "(install the 'geo' extra for a real COG with O(1) range reads).",
                out.name,
            )
    return _write_npy_fallback(arr, out, transform, crs, nodata)


def _write_cog_rasterio(
    arr: np.ndarray,
    out: Path,
    transform: Any | None,
    crs: Any | None,
    nodata: float | None,
    overviews: bool,
    compress: str,
) -> str:  # pragma: no cover - requires the optional 'geo' extra
    """Write a true COG via rasterio (+ rio-cogeo validation if available)."""
    import rasterio
    from rasterio.enums import Resampling

    count, height, width = arr.shape
    profile = cog_profile(count, height, width, str(arr.dtype), transform, crs, nodata)
    profile["compress"] = compress

    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(arr)
        if overviews:
            dst.build_overviews(list(_OVERVIEW_LEVELS), Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    # Best-effort: rewrite through rio-cogeo so the byte layout is COG-valid
    # (IFDs/overviews ordered for ranged reads). Skipped silently if absent.
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        dst_profile = cog_profiles.get("zstd" if compress.upper() == "ZSTD" else "deflate")
        cog_translate(out, out, dst_profile, in_memory=True, quiet=True, overview_level=5)
        _log.debug("rio-cogeo validated/optimised COG: %s", out)
    except ImportError:
        _log.debug("rio-cogeo not installed; wrote a tiled GeoTIFF with overviews.")
    return str(out)


def _write_npy_fallback(
    arr: np.ndarray,
    out: Path,
    transform: Any | None,
    crs: Any | None,
    nodata: float | None,
) -> str:
    """Write a ``.npy`` array plus a JSON sidecar describing the (geo) profile."""
    out.parent.mkdir(parents=True, exist_ok=True)
    npy_path = out.with_suffix(".npy")
    np.save(npy_path, arr)

    sidecar = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "count": int(arr.shape[0]),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "transform": list(transform) if transform is not None else None,
        "crs": str(crs) if crs is not None else None,
        "nodata": nodata,
        "cog": False,
        "note": "rasterio unavailable; '.npy' fallback. Install extras 'geo' for a COG.",
    }
    json_path = out.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    _log.info("Wrote NumPy fallback: %s (+ %s)", npy_path.name, json_path.name)
    return str(npy_path)


def validate_cog(path: str | Path) -> bool:
    """Return ``True`` if ``path`` is a valid COG (lazy ``rio-cogeo``).

    Args:
        path: Path to a candidate COG.

    Returns:
        ``True`` if ``rio-cogeo`` validates it as a COG; ``False`` if it is the
        ``.npy`` fallback, an invalid file, or ``rio-cogeo`` is not installed.
    """
    p = Path(path)
    if p.suffix.lower() not in {".tif", ".tiff"}:
        return False
    try:  # pragma: no cover - requires the optional 'geo' extra
        from rio_cogeo.cogeo import cog_validate

        is_valid, _errors, _warnings = cog_validate(str(p))
        return bool(is_valid)
    except ImportError:
        _log.debug("rio-cogeo not installed; cannot validate %s", p)
        return False
