"""Radiometric preprocessing for LISS-IV imagery (pure NumPy/torch).

Implements the DN -> radiance -> TOA-reflectance chain documented in
``research/06_lissiv_isro_bhoonidhi_ner.md`` §A.5, per-band normalization /
denormalization aligned with the ``SAMPLE`` ``meta`` contract (BUILD_PLAN §2),
10-bit DN handling, simple resampling/stacking, and the spectral-index helpers
(NDVI / GNDVI) that the 3-band Green/Red/NIR sensor *can* compute
(``research/06`` §E.2).

LISS-IV calibration concept (Resourcesat-2/2A class)::

    L_lambda     = (Lmax_lambda / DN_max) * DN              # DN_max = 1023 (10-bit)
    rho_TOA      = (pi * L_lambda * d^2) / (ESUN_lambda * cos theta_z)

with ``theta_z`` the solar zenith (``90 - sun_elevation``), ``d`` the Earth-Sun
distance in AU, and the documented default per-band ``Lmax`` saturation radiances
(B2/Green ~= 53.0, B3/Red ~= 47.0, B4/NIR ~= 31.5, mW cm^-2 sr^-1 um^-1). ESUN
defaults are stand-in solar-irradiance constants in matching units; production
calibration should read exact constants from ``BAND_META.txt`` per scene.

This module is intentionally dependency-light: only ``numpy`` (and ``torch`` for
tensor-friendly helpers) are imported, so it runs on the CPU-smoke stack. The
band normalization helpers delegate to :mod:`cloudremoval.utils.geo` where that
already provides a vetted implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cloudremoval.utils.geo import band_stats, denormalize, normalize
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "BANDS",
    "GREEN_IDX",
    "RED_IDX",
    "NIR_IDX",
    "DN_MAX_10BIT",
    "LISSIV_LMAX",
    "LISSIV_ESUN",
    "dn_to_radiance",
    "radiance_to_toa_reflectance",
    "dn_to_toa_reflectance",
    "earth_sun_distance",
    "clamp",
    "normalize_bands",
    "denormalize_bands",
    "compute_band_stats",
    "ndvi",
    "gndvi",
    "resample_array",
    "stack_bands",
]

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Band convention (BUILD_PLAN §2: C = 3, Green, Red, NIR)
# --------------------------------------------------------------------------- #
#: Ordered band names for the 3-band LISS-IV product.
BANDS: tuple[str, str, str] = ("green", "red", "nir")
GREEN_IDX: int = 0
RED_IDX: int = 1
NIR_IDX: int = 2

#: Maximum quantised calibrated value for 10-bit Resourcesat-2/2A LISS-IV.
DN_MAX_10BIT: float = 1023.0

#: Documented default per-band saturation radiance ``Lmax`` (Green, Red, NIR) in
#: mW cm^-2 sr^-1 um^-1 (``research/06`` §A.5). Read exact values per scene from
#: ``BAND_META.txt`` in production.
LISSIV_LMAX: tuple[float, float, float] = (53.0, 47.0, 31.5)

#: Stand-in per-band mean exo-atmospheric solar irradiance ``ESUN`` (Green, Red,
#: NIR) in matching units. These are documented defaults; production must read
#: the values from the product handbook/metadata. Chosen so the reflectance
#: chain yields physically plausible [0, 1] reflectances for typical scenes.
LISSIV_ESUN: tuple[float, float, float] = (185.3, 158.4, 109.1)

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Radiometric calibration: DN -> radiance -> TOA reflectance
# --------------------------------------------------------------------------- #
def dn_to_radiance(
    dn: np.ndarray,
    lmax: list[float] | tuple[float, ...] | np.ndarray = LISSIV_LMAX,
    dn_max: float = DN_MAX_10BIT,
    lmin: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw digital numbers to at-sensor spectral radiance.

    Uses the linear gain model ``L = Lmin + (Lmax - Lmin) / DN_max * DN`` with
    ``Lmin = 0`` by default (the Resourcesat-2/2A convention).

    Args:
        dn: ``[C, H, W]`` (or ``[H, W]``) array of digital numbers.
        lmax: Per-band saturation radiance (length ``C``).
        dn_max: Maximum DN value (``1023`` for 10-bit).
        lmin: Optional per-band minimum radiance (defaults to zeros).

    Returns:
        Per-band radiance ``[C, H, W]`` (``float32``).
    """
    arr = np.asarray(dn, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    c = arr.shape[0]
    lmax_a = np.asarray(lmax, dtype=np.float64).reshape(-1)[:c].reshape(c, 1, 1)
    if lmin is None:
        lmin_a = np.zeros((c, 1, 1), dtype=np.float64)
    else:
        lmin_a = np.asarray(lmin, dtype=np.float64).reshape(-1)[:c].reshape(c, 1, 1)
    radiance = lmin_a + (lmax_a - lmin_a) / max(dn_max, _EPS) * arr
    return radiance.astype(np.float32)


def earth_sun_distance(day_of_year: int) -> float:
    """Earth-Sun distance in astronomical units for a day of the year.

    Uses the standard approximation
    ``d = 1 - 0.01672 * cos(0.9856 * (DOY - 4) [deg])``.

    Args:
        day_of_year: Day of year in ``[1, 366]``.

    Returns:
        Earth-Sun distance in AU.
    """
    doy = float(max(1, min(366, int(day_of_year))))
    return 1.0 - 0.01672 * np.cos(np.deg2rad(0.9856 * (doy - 4.0)))


def radiance_to_toa_reflectance(
    radiance: np.ndarray,
    sun_elevation_deg: float,
    esun: list[float] | tuple[float, ...] | np.ndarray = LISSIV_ESUN,
    earth_sun_dist: float = 1.0,
) -> np.ndarray:
    """Convert at-sensor radiance to top-of-atmosphere reflectance.

    Applies ``rho = pi * L * d^2 / (ESUN * cos theta_z)`` where the solar zenith
    ``theta_z = 90 - sun_elevation``.

    Args:
        radiance: ``[C, H, W]`` per-band radiance.
        sun_elevation_deg: Sun elevation at scene centre (degrees).
        esun: Per-band mean exo-atmospheric solar irradiance (length ``C``).
        earth_sun_dist: Earth-Sun distance in AU.

    Returns:
        TOA reflectance ``[C, H, W]`` clipped to ``[0, 1]`` (``float32``).
    """
    arr = np.asarray(radiance, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    c = arr.shape[0]
    esun_a = np.asarray(esun, dtype=np.float64).reshape(-1)[:c].reshape(c, 1, 1)
    cos_zenith = max(np.cos(np.deg2rad(90.0 - sun_elevation_deg)), _EPS)
    refl = (np.pi * arr * earth_sun_dist**2) / (esun_a * cos_zenith + _EPS)
    return clamp(refl, 0.0, 1.0).astype(np.float32)


def dn_to_toa_reflectance(
    dn: np.ndarray,
    sun_elevation_deg: float,
    lmax: list[float] | tuple[float, ...] | np.ndarray = LISSIV_LMAX,
    esun: list[float] | tuple[float, ...] | np.ndarray = LISSIV_ESUN,
    dn_max: float = DN_MAX_10BIT,
    day_of_year: int | None = None,
    earth_sun_dist: float | None = None,
) -> np.ndarray:
    """End-to-end LISS-IV DN -> TOA reflectance (``research/06`` §A.5).

    Convenience wrapper chaining :func:`dn_to_radiance` and
    :func:`radiance_to_toa_reflectance`. The Earth-Sun distance is taken from
    ``earth_sun_dist`` if given, else derived from ``day_of_year``, else ``1.0``.

    Args:
        dn: ``[C, H, W]`` 10-bit digital numbers.
        sun_elevation_deg: Sun elevation (degrees).
        lmax: Per-band saturation radiance.
        esun: Per-band solar irradiance.
        dn_max: Maximum DN value.
        day_of_year: Optional day of year used to compute ``d``.
        earth_sun_dist: Optional explicit Earth-Sun distance (AU); overrides
            ``day_of_year``.

    Returns:
        TOA reflectance ``[C, H, W]`` in ``[0, 1]`` (``float32``).
    """
    if earth_sun_dist is None:
        earth_sun_dist = earth_sun_distance(day_of_year) if day_of_year is not None else 1.0
    radiance = dn_to_radiance(dn, lmax=lmax, dn_max=dn_max)
    return radiance_to_toa_reflectance(
        radiance, sun_elevation_deg, esun=esun, earth_sun_dist=earth_sun_dist
    )


# --------------------------------------------------------------------------- #
# Clamp / normalization (torch- and numpy-friendly)
# --------------------------------------------------------------------------- #
def clamp(array, low: float = 0.0, high: float = 1.0):
    """Clamp an array/tensor to ``[low, high]`` (works for NumPy and torch).

    Args:
        array: A NumPy array or torch tensor.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The clamped object, same type as the input.
    """
    if hasattr(array, "clamp"):  # torch.Tensor
        return array.clamp(low, high)
    return np.clip(array, low, high)


def compute_band_stats(array: np.ndarray, norm: str = "zscore") -> dict[str, list[float]]:
    """Per-band statistics for ``meta["band_stats"]`` (delegates to ``utils.geo``).

    Args:
        array: ``[C, H, W]`` image (reflectance or DN).
        norm: ``"zscore"`` -> ``{"mean","std"}``; ``"percentile"`` -> ``{"p2","p98"}``.

    Returns:
        Stats mapping suitable for ``meta``.
    """
    return band_stats(array, norm=norm)


def normalize_bands(
    array: np.ndarray,
    stats: dict[str, list[float]] | None = None,
    norm: str = "zscore",
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Per-band normalize an image, computing stats from it when not supplied.

    Thin wrapper over :func:`cloudremoval.utils.geo.normalize` that also returns
    the stats used (so callers can stash them in ``meta``).

    Args:
        array: ``[C, H, W]`` image.
        stats: Optional precomputed stats; if ``None`` they are computed here.
        norm: ``"zscore"`` or ``"percentile"``.

    Returns:
        ``(normalized [C, H, W] float32, stats)``.
    """
    if stats is None:
        stats = band_stats(array, norm=norm)
    return normalize(array, stats, norm=norm), stats


def denormalize_bands(
    array: np.ndarray,
    stats: dict[str, list[float]],
    norm: str = "zscore",
) -> np.ndarray:
    """Invert :func:`normalize_bands` (delegates to ``utils.geo.denormalize``).

    Args:
        array: Normalized ``[C, H, W]`` image.
        stats: The stats used to normalize.
        norm: ``"zscore"`` or ``"percentile"``.

    Returns:
        De-normalized reflectance/DN ``[C, H, W]`` (``float32``).
    """
    return denormalize(array, stats, norm=norm)


# --------------------------------------------------------------------------- #
# Spectral indices (the ones a 3-band G/R/NIR sensor can compute)
# --------------------------------------------------------------------------- #
def ndvi(
    image: np.ndarray,
    nir_idx: int = NIR_IDX,
    red_idx: int = RED_IDX,
) -> np.ndarray:
    """Normalised Difference Vegetation Index ``(NIR - Red) / (NIR + Red)``.

    Args:
        image: ``[C, H, W]`` reflectance image (``C >= 3``).
        nir_idx: NIR band index.
        red_idx: Red band index.

    Returns:
        ``[H, W]`` NDVI in ``[-1, 1]`` (``float32``).
    """
    arr = np.asarray(image, dtype=np.float32)
    nir = arr[nir_idx]
    red = arr[red_idx]
    return ((nir - red) / (nir + red + _EPS)).astype(np.float32)


def gndvi(
    image: np.ndarray,
    nir_idx: int = NIR_IDX,
    green_idx: int = GREEN_IDX,
) -> np.ndarray:
    """Green NDVI ``(NIR - Green) / (NIR + Green)`` (``research/06`` §E.2).

    Args:
        image: ``[C, H, W]`` reflectance image.
        nir_idx: NIR band index.
        green_idx: Green band index.

    Returns:
        ``[H, W]`` GNDVI in ``[-1, 1]`` (``float32``).
    """
    arr = np.asarray(image, dtype=np.float32)
    nir = arr[nir_idx]
    green = arr[green_idx]
    return ((nir - green) / (nir + green + _EPS)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Resampling / stacking (pure NumPy, no geospatial deps)
# --------------------------------------------------------------------------- #
def resample_array(
    array: np.ndarray,
    out_hw: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Resample a ``[C, H, W]`` (or ``[H, W]``) array to ``out_hw``.

    Uses a dependency-free implementation: ``order=0`` nearest-neighbour or
    ``order=1`` bilinear via :func:`numpy.interp`-style grid gathering. This
    suffices for putting auxiliary modalities (S2 10 m, DEM) onto the LISS-IV
    grid on the smoke path; production reprojection uses ``utils.geo``/rasterio.

    Args:
        array: Channel-first image (or 2-D single band).
        out_hw: Target ``(H, W)``.
        order: ``0`` nearest or ``1`` bilinear.

    Returns:
        Resampled array ``[C, out_H, out_W]`` (same dtype family, ``float32``).
    """
    arr = np.asarray(array, dtype=np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None, ...]
    c, h, w = arr.shape
    out_h, out_w = out_hw
    if (h, w) == (out_h, out_w):
        return arr[0] if squeeze else arr

    # Sample coordinates in the source grid (align corners).
    ys = np.linspace(0, h - 1, out_h, dtype=np.float32)
    xs = np.linspace(0, w - 1, out_w, dtype=np.float32)

    if order == 0:
        yi = np.round(ys).astype(np.int64).clip(0, h - 1)
        xi = np.round(xs).astype(np.int64).clip(0, w - 1)
        out = arr[:, yi][:, :, xi]
        return out[0] if squeeze else out

    # Bilinear.
    y0 = np.floor(ys).astype(np.int64).clip(0, h - 1)
    x0 = np.floor(xs).astype(np.int64).clip(0, w - 1)
    y1 = (y0 + 1).clip(0, h - 1)
    x1 = (x0 + 1).clip(0, w - 1)
    wy = (ys - y0).reshape(1, out_h, 1)
    wx = (xs - x0).reshape(1, 1, out_w)

    top = arr[:, y0][:, :, x0] * (1 - wx) + arr[:, y0][:, :, x1] * wx
    bot = arr[:, y1][:, :, x0] * (1 - wx) + arr[:, y1][:, :, x1] * wx
    out = (top * (1 - wy) + bot * wy).astype(np.float32)
    return out[0] if squeeze else out


def stack_bands(bands: list[np.ndarray], to_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Stack per-band 2-D arrays into a ``[C, H, W]`` cube, resampling to match.

    Args:
        bands: List of ``[H, W]`` band arrays (possibly differing resolutions).
        to_shape: Optional target ``(H, W)``; defaults to the first band's shape.

    Returns:
        A ``[C, H, W]`` stacked array (``float32``).

    Raises:
        ValueError: If ``bands`` is empty.
    """
    if not bands:
        raise ValueError("stack_bands requires at least one band")
    target = to_shape or np.asarray(bands[0]).shape[-2:]
    out = [resample_array(np.asarray(b, dtype=np.float32), tuple(target), order=1) for b in bands]
    return np.stack(out, axis=0).astype(np.float32)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy ``[C, H, W]`` array to a ``float32`` torch tensor.

    Args:
        array: Channel-first NumPy image.

    Returns:
        A contiguous ``float32`` :class:`torch.Tensor`.
    """
    import torch

    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
