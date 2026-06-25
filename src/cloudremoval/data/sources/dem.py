"""DEM access (Copernicus GLO-30) + slope / aspect / hillshade for shadow geometry.

A DEM lets the pipeline distinguish cloud shadow from terrain shadow and condition
reconstruction on slope/aspect over NER relief (``research/04`` §A.5,
``research/06`` §E.3). This module provides terrain-derivative helpers used by
:mod:`cloudremoval.data.masking` and a documented fetch path.

* :func:`slope`, :func:`aspect`, :func:`hillshade` -- pure-NumPy ``numpy.gradient``
  implementations (no dependency), the default on the CPU-smoke stack.
* :func:`fetch_dem` -- documented Copernicus GLO-30 access via lazy
  ``planetary-computer`` + STAC (delegates to :mod:`cloudremoval.data.sources.stac`)
  or ``rasterio`` on a local DEM file; raises a clear error if no backend is
  available.

GEE/STAC identifiers (``research/04`` §A.5):
``COPERNICUS/DEM/GLO30`` (preferred), ``USGS/SRTMGL1_003``, ``NASA/NASADEM_HGT/001``;
Planetary Computer collection ``cop-dem-glo-30``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudremoval.utils.logging import get_logger

__all__ = [
    "GLO30_GEE_ID",
    "SRTM_GEE_ID",
    "NASADEM_GEE_ID",
    "GLO30_STAC_COLLECTION",
    "slope",
    "aspect",
    "hillshade",
    "fetch_dem",
    "read_local_dem",
]

_log = get_logger(__name__)
_EPS = 1e-6

#: Verified GEE / STAC identifiers for global DEMs (``research/04`` §A.5).
GLO30_GEE_ID = "COPERNICUS/DEM/GLO30"
SRTM_GEE_ID = "USGS/SRTMGL1_003"
NASADEM_GEE_ID = "NASA/NASADEM_HGT/001"
GLO30_STAC_COLLECTION = "cop-dem-glo-30"


# --------------------------------------------------------------------------- #
# Terrain derivatives (pure NumPy)
# --------------------------------------------------------------------------- #
def _gradients(dem: np.ndarray, pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dz/dx, dz/dy)`` of a ``[H, W]`` DEM (``numpy.gradient``)."""
    arr = np.asarray(dem, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    dzdy, dzdx = np.gradient(arr, max(pixel_size, _EPS))
    return dzdx, dzdy


def slope(dem: np.ndarray, pixel_size: float = 30.0, degrees: bool = True) -> np.ndarray:
    """Terrain slope of a ``[H, W]`` DEM.

    Args:
        dem: Elevation array ``[H, W]`` or ``[1, H, W]`` (metres).
        pixel_size: Ground sample distance in metres.
        degrees: Return degrees (else radians).

    Returns:
        ``[H, W]`` slope (``float32``).
    """
    dzdx, dzdy = _gradients(dem, pixel_size)
    rise = np.sqrt(dzdx**2 + dzdy**2)
    slp = np.arctan(rise)
    if degrees:
        slp = np.rad2deg(slp)
    return slp.astype(np.float32)


def aspect(dem: np.ndarray, pixel_size: float = 30.0, degrees: bool = True) -> np.ndarray:
    """Terrain aspect (downslope azimuth) of a ``[H, W]`` DEM.

    Args:
        dem: Elevation array (metres).
        pixel_size: Ground sample distance in metres.
        degrees: Return degrees in ``[0, 360)`` (else radians).

    Returns:
        ``[H, W]`` aspect (``float32``).
    """
    dzdx, dzdy = _gradients(dem, pixel_size)
    asp = np.arctan2(dzdy, -dzdx)
    if degrees:
        asp = (np.rad2deg(asp) + 360.0) % 360.0
    return asp.astype(np.float32)


def hillshade(
    dem: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    pixel_size: float = 30.0,
) -> np.ndarray:
    """Analytical hillshade of a ``[H, W]`` DEM (standard ESRI formulation).

    Low values mark slopes turned away from the sun (terrain shadow), used by
    :func:`cloudremoval.data.masking.detect_shadows` to separate terrain shadow
    from cloud shadow.

    Args:
        dem: Elevation array (metres).
        azimuth: Solar azimuth in degrees (clockwise from north).
        altitude: Solar altitude/elevation in degrees.
        pixel_size: Ground sample distance in metres.

    Returns:
        ``[H, W]`` hillshade in ``[0, 1]`` (``float32``).
    """
    slope_rad = np.deg2rad(slope(dem, pixel_size=pixel_size, degrees=True))
    aspect_rad = np.deg2rad(aspect(dem, pixel_size=pixel_size, degrees=True))
    zenith_rad = np.deg2rad(90.0 - altitude)
    az_rad = np.deg2rad(360.0 - azimuth + 90.0)

    shaded = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
        slope_rad
    ) * np.cos(az_rad - aspect_rad)
    return np.clip(shaded, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# DEM access
# --------------------------------------------------------------------------- #
def read_local_dem(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a local DEM (GeoTIFF/``.npy``) into ``([1, H, W], profile)``.

    Uses :func:`cloudremoval.utils.io.read_array` so it works with or without
    ``rasterio`` (``.npy`` fallback).

    Args:
        path: Path to a DEM raster.

    Returns:
        ``(dem[1, H, W] float32, profile-or-empty-dict)``.
    """
    from cloudremoval.utils.io import read_array

    arr = np.asarray(read_array(path), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr[:1], {}


def fetch_dem(
    bbox: tuple[float, float, float, float],
    out_path: str | Path | None = None,
    collection: str = GLO30_STAC_COLLECTION,
) -> str:
    """Fetch a Copernicus GLO-30 DEM for ``bbox`` (documented; lazy backends).

    Delegates to STAC (:mod:`cloudremoval.data.sources.stac`) over the Planetary
    Computer ``cop-dem-glo-30`` collection. This requires network access and the
    optional STAC stack and is **not** exercised on the CPU smoke.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        out_path: Optional local path to write the merged DEM.
        collection: STAC collection id (default Copernicus GLO-30).

    Returns:
        The STAC asset href (or local path if downloaded).

    Raises:
        RuntimeError: If no STAC backend is available.
    """
    try:
        from cloudremoval.data.sources.stac import search_stac
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("STAC backend unavailable for DEM fetch") from exc
    items = search_stac(collection=collection, bbox=bbox, datetime=None)
    if not items:
        raise RuntimeError(f"no DEM items found for bbox={bbox} in {collection}")
    href = items[0]["assets"].get("data", {}).get("href") or next(
        iter(items[0]["assets"].values())
    ).get("href")
    _log.info("DEM asset for bbox=%s -> %s", bbox, href)
    if out_path is not None:
        _log.info("download to %s left to the caller (lazy rasterio/aws)", out_path)
    return str(href)
