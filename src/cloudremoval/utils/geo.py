"""Geospatial and radiometric helpers with pure-NumPy fallbacks.

The CPU-smoke path needs band normalization / denormalization and band-statistics
**without** any geospatial stack, so those are pure NumPy. Heavier operations
(``reproject``) lazily import ``rasterio``/``pyproj`` and raise a clear error if
absent — callers on the smoke path do not invoke them.

Normalization conventions match the ``SAMPLE`` ``meta`` contract (BUILD_PLAN §2):
``norm="zscore"`` uses ``band_stats={"mean":[C],"std":[C]}``; ``norm="percentile"``
uses ``band_stats={"p2":[C],"p98":[C]}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "band_stats",
    "normalize",
    "denormalize",
    "to_reflectance_dn",
    "reproject_array",
]

if TYPE_CHECKING:
    import numpy as np

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Band statistics & (de)normalization — pure NumPy
# --------------------------------------------------------------------------- #
def band_stats(array: np.ndarray, norm: str = "zscore") -> dict[str, list[float]]:
    """Compute per-band statistics for a ``[C, H, W]`` array.

    Args:
        array: Channel-first image.
        norm: ``"zscore"`` -> returns ``{"mean":[C], "std":[C]}``;
            ``"percentile"`` -> returns ``{"p2":[C], "p98":[C]}``.

    Returns:
        A mapping suitable for ``meta["band_stats"]``.
    """
    import numpy as np

    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    flat = arr.reshape(arr.shape[0], -1)
    if norm == "percentile":
        p2 = np.percentile(flat, 2, axis=1)
        p98 = np.percentile(flat, 98, axis=1)
        return {"p2": p2.tolist(), "p98": p98.tolist()}
    return {"mean": flat.mean(axis=1).tolist(), "std": flat.std(axis=1).tolist()}


def normalize(
    array: np.ndarray,
    stats: dict[str, list[float]],
    norm: str = "zscore",
) -> np.ndarray:
    """Normalize a ``[C, H, W]`` array using per-band ``stats``.

    Args:
        array: Channel-first image (reflectance/DN).
        stats: ``{"mean","std"}`` (zscore) or ``{"p2","p98"}`` (percentile).
        norm: ``"zscore"`` or ``"percentile"``.

    Returns:
        The normalized array (same shape, ``float32``).
    """
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    c = arr.shape[0]
    if norm == "percentile":
        lo = np.asarray(stats["p2"], dtype=np.float32).reshape(c, 1, 1)
        hi = np.asarray(stats["p98"], dtype=np.float32).reshape(c, 1, 1)
        return ((arr - lo) / (hi - lo + _EPS)).astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(c, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(c, 1, 1)
    return ((arr - mean) / (std + _EPS)).astype(np.float32)


def denormalize(
    array: np.ndarray,
    stats: dict[str, list[float]],
    norm: str = "zscore",
) -> np.ndarray:
    """Invert :func:`normalize`, mapping normalized values back to reflectance/DN.

    Args:
        array: Normalized channel-first image.
        stats: The same ``stats`` used to normalize.
        norm: ``"zscore"`` or ``"percentile"``.

    Returns:
        The de-normalized array (``float32``).
    """
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    c = arr.shape[0]
    if norm == "percentile":
        lo = np.asarray(stats["p2"], dtype=np.float32).reshape(c, 1, 1)
        hi = np.asarray(stats["p98"], dtype=np.float32).reshape(c, 1, 1)
        return (arr * (hi - lo + _EPS) + lo).astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(c, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(c, 1, 1)
    return (arr * (std + _EPS) + mean).astype(np.float32)


def to_reflectance_dn(
    dn: np.ndarray,
    lmax: list[float] | np.ndarray,
    sun_elevation_deg: float,
    esun: list[float] | np.ndarray | None = None,
    qcal_max: float = 1023.0,
    earth_sun_dist: float = 1.0,
) -> np.ndarray:
    """Approximate DN -> TOA reflectance for LISS-IV (numpy; B1 may refine).

    Implements the standard radiance-then-reflectance chain:
    ``L = (Lmax / qcal_max) * DN`` then
    ``rho = pi * L * d^2 / (ESUN * sin(sun_elevation))``. If ``esun`` is omitted,
    the radiance is min-max scaled as a smoke-safe stand-in. This is a prototype
    helper (ARCHITECTURE §2.2); production calibration lives in ``data.preprocess``.

    Args:
        dn: ``[C, H, W]`` 10-bit digital numbers.
        lmax: Per-band spectral radiance at ``qcal_max`` (length C).
        sun_elevation_deg: Sun elevation in degrees.
        esun: Per-band mean exo-atmospheric solar irradiance (length C) or None.
        qcal_max: Max quantised calibrated value (1023 for 10-bit).
        earth_sun_dist: Earth-Sun distance in AU.

    Returns:
        Approximate TOA reflectance ``[C, H, W]`` (``float32``).
    """
    import numpy as np

    arr = np.asarray(dn, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    c = arr.shape[0]
    lmax_a = np.asarray(lmax, dtype=np.float64).reshape(c, 1, 1)
    radiance = (lmax_a / qcal_max) * arr

    sin_elev = max(np.sin(np.deg2rad(sun_elevation_deg)), _EPS)
    if esun is None:
        rng = radiance.reshape(c, -1)
        lo = rng.min(axis=1).reshape(c, 1, 1)
        hi = rng.max(axis=1).reshape(c, 1, 1)
        refl = (radiance - lo) / (hi - lo + _EPS)
    else:
        esun_a = np.asarray(esun, dtype=np.float64).reshape(c, 1, 1)
        refl = (np.pi * radiance * earth_sun_dist**2) / (esun_a * sin_elev)
    return np.clip(refl, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reprojection (lazy rasterio/pyproj; not on the smoke path)
# --------------------------------------------------------------------------- #
def reproject_array(
    array: np.ndarray,
    src_crs: Any,
    dst_crs: Any,
    src_transform: Any,
    dst_transform: Any | None = None,
    dst_shape: tuple[int, int] | None = None,
    resampling: str = "bilinear",
) -> tuple[np.ndarray, Any]:
    """Reproject a ``[C, H, W]`` array between CRSs (lazy ``rasterio``).

    This is a thin wrapper around ``rasterio.warp.reproject``; B1 may replace or
    extend it. It is **not** exercised on the CPU-smoke path.

    Raises:
        ImportError: If ``rasterio`` is unavailable.
    """
    import numpy as np
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, ...]
    count, height, width = arr.shape

    if dst_transform is None or dst_shape is None:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, width, height, *_bounds_from_transform(src_transform, width, height)
        )
        dst_shape = (dst_h, dst_w)

    resampling_enum = getattr(Resampling, resampling, Resampling.bilinear)
    dst = np.zeros((count, dst_shape[0], dst_shape[1]), dtype=arr.dtype)
    for band in range(count):
        reproject(
            source=arr[band],
            destination=dst[band],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_enum,
        )
    return dst, dst_transform


def _bounds_from_transform(transform: Any, width: int, height: int):
    """Compute (left, bottom, right, top) bounds from an affine transform."""
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return left, bottom, right, top
