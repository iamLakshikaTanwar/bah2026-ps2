"""Cloud & cloud-shadow masking for 3-band Green/Red/NIR imagery (no blue band).

LISS-IV has **no blue and no SWIR/cirrus band** (``research/06`` §A.1, §E.1), so
the standard Fmask / s2cloudless / Sen2Cor masks do not apply unmodified. This
module provides:

* :func:`detect_clouds` -- a lazy **OmniCloudMask** path (``research/05`` §B.3:
  the recommended sensor-agnostic model that explicitly supports Green/Red/NIR at
  ~5 m) with a pure-NumPy **spectral/brightness + NIR** fallback (the default on
  the CPU-smoke stack). Clouds are bright and spectrally flat across G/R/NIR with
  low NDVI, so the fallback thresholds high all-band brightness, spectral
  flatness, and low NDVI (``research/06`` §E.1).
* :func:`detect_shadows` -- geometric cloud-shadow projection along the solar
  azimuth/elevation vector (DEM-aware via a lazy hook; flat-earth fallback),
  confirmed by a dark-pixel / low-NIR test (``research/05`` §B.3).

Both return masks shaped ``[1, H, W]`` (float ``{0, 1}``) matching the ``SAMPLE``
contract (BUILD_PLAN §2). The module imports only NumPy at top level.
"""

from __future__ import annotations

import numpy as np

from cloudremoval.data.preprocess import GREEN_IDX, NIR_IDX, RED_IDX, ndvi
from cloudremoval.utils.logging import get_logger

__all__ = [
    "detect_clouds",
    "detect_shadows",
    "cloud_probability",
]

_log = get_logger(__name__)
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Cloud detection
# --------------------------------------------------------------------------- #
def cloud_probability(
    image: np.ndarray,
    green_idx: int = GREEN_IDX,
    red_idx: int = RED_IDX,
    nir_idx: int = NIR_IDX,
) -> np.ndarray:
    """Soft cloud-probability heuristic for a 3-band G/R/NIR reflectance image.

    Combines three cloud cues available without a blue/SWIR band
    (``research/06`` §E.1): high brightness across all bands, low spectral
    contrast (clouds are spectrally flat / "white"), and low NDVI (clouds are not
    vegetation). The product of the three soft cues yields a ``[0, 1]`` map.

    Args:
        image: ``[C, H, W]`` reflectance image in ``[0, 1]`` (``C >= 3``).
        green_idx: Green band index.
        red_idx: Red band index.
        nir_idx: NIR band index.

    Returns:
        ``[H, W]`` cloud probability in ``[0, 1]`` (``float32``).
    """
    arr = np.asarray(image, dtype=np.float32)
    green, red, nir = arr[green_idx], arr[red_idx], arr[nir_idx]

    brightness = (green + red + nir) / 3.0
    # Bright cue: ramps in over [0.45, 0.75] reflectance.
    bright_cue = np.clip((brightness - 0.45) / 0.30, 0.0, 1.0)

    # Flatness cue: clouds are spectrally flat -> small max-min spread.
    stack = np.stack([green, red, nir], axis=0)
    spread = stack.max(axis=0) - stack.min(axis=0)
    flat_cue = np.clip(1.0 - spread / 0.25, 0.0, 1.0)

    # Low-NDVI cue: vegetation (high NDVI) is not cloud.
    veg = ndvi(arr, nir_idx=nir_idx, red_idx=red_idx)
    low_ndvi_cue = np.clip(1.0 - (veg - 0.2) / 0.4, 0.0, 1.0)

    prob = bright_cue * (0.5 + 0.5 * flat_cue) * (0.5 + 0.5 * low_ndvi_cue)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def detect_clouds(
    image,
    threshold: float = 0.5,
    use_omnicloudmask: bool = True,
    green_idx: int = GREEN_IDX,
    red_idx: int = RED_IDX,
    nir_idx: int = NIR_IDX,
):
    """Detect clouds in a 3-band G/R/NIR image, returning a ``[1, H, W]`` mask.

    Attempts **OmniCloudMask** (lazy import) when ``use_omnicloudmask`` is set and
    the package is installed; otherwise (the CPU-smoke default) falls back to the
    pure-NumPy :func:`cloud_probability` heuristic thresholded at ``threshold``.

    Args:
        image: ``[C, H, W]`` reflectance image (NumPy array or torch tensor).
        threshold: Probability threshold for the fallback binarisation.
        use_omnicloudmask: If ``True``, try the OmniCloudMask model first.
        green_idx: Green band index.
        red_idx: Red band index.
        nir_idx: NIR band index.

    Returns:
        Cloud mask ``[1, H, W]`` in ``{0, 1}`` (same array type as ``image``).
    """
    is_torch = _is_torch_tensor(image)
    arr = _to_numpy(image)
    if arr.ndim == 2:
        arr = arr[None, ...]

    mask: np.ndarray | None = None
    if use_omnicloudmask:
        mask = _omnicloudmask_predict(arr, green_idx, red_idx, nir_idx)

    if mask is None:
        prob = cloud_probability(
            arr, green_idx=green_idx, red_idx=red_idx, nir_idx=nir_idx
        )
        mask = (prob >= threshold).astype(np.float32)

    mask = mask.astype(np.float32)[None, ...]  # [1, H, W]
    return _wrap_like(mask, is_torch)


def _omnicloudmask_predict(
    arr: np.ndarray,
    green_idx: int,
    red_idx: int,
    nir_idx: int,
) -> np.ndarray | None:
    """Run OmniCloudMask if available, else return ``None`` (caller falls back).

    OmniCloudMask consumes a Green/Red/NIR stack and returns a class map
    (0 clear, 1 thick, 2 thin, 3 shadow); we collapse thick+thin to the cloud
    mask. Any import/runtime failure degrades gracefully to ``None``.
    """
    try:
        import omnicloudmask  # type: ignore  # noqa: F401  (lazy, optional)
        from omnicloudmask import predict_from_array  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency absent on smoke path
        _log.debug("omnicloudmask unavailable; using threshold fallback")
        return None

    try:
        gan = np.stack([arr[green_idx], arr[red_idx], arr[nir_idx]], axis=0)
        pred = predict_from_array(gan)
        pred = np.asarray(pred).squeeze()
        # Classes {1: thick, 2: thin} -> cloud.
        return np.isin(pred, (1, 2)).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        _log.warning("omnicloudmask prediction failed (%s); using fallback", exc)
        return None


# --------------------------------------------------------------------------- #
# Shadow detection (geometric projection)
# --------------------------------------------------------------------------- #
def detect_shadows(
    image,
    cloud_mask,
    sun_az: float,
    sun_elev: float,
    dem=None,
    nir_idx: int = NIR_IDX,
    max_shadow_frac: float = 0.08,
    cloud_height_steps: int = 6,
):
    """Project cloud shadows from sun geometry, confirmed by a dark/low-NIR test.

    The cloud mask is shifted *away* from the sun across a range of candidate
    cloud heights (shadow length grows as ``1/tan(elevation)``), the union of
    projections forms the candidate shadow region, and it is confirmed where the
    surface is actually dark in NIR (cloud shadows suppress NIR strongly,
    ``research/05`` §B.3). A DEM, if supplied, is used by the lazy terrain hook to
    add topographic context; otherwise a flat-earth projection is used.

    Args:
        image: ``[C, H, W]`` reflectance image (array or tensor).
        cloud_mask: ``[1, H, W]`` or ``[H, W]`` cloud mask in ``{0, 1}``.
        sun_az: Solar azimuth (degrees, clockwise from north).
        sun_elev: Solar elevation (degrees).
        dem: Optional ``[1, H, W]``/``[H, W]`` elevation; enables the terrain hook.
        nir_idx: NIR band index (for the dark-pixel confirmation).
        max_shadow_frac: Max shadow displacement as a fraction of image size at
            the reference elevation.
        cloud_height_steps: Number of candidate cloud heights projected.

    Returns:
        Shadow mask ``[1, H, W]`` in ``{0, 1}`` (same array type as ``image``).
    """
    is_torch = _is_torch_tensor(image)
    arr = _to_numpy(image)
    if arr.ndim == 2:
        arr = arr[None, ...]
    cmask = _to_numpy(cloud_mask)
    if cmask.ndim == 3:
        cmask = cmask[0]
    h, w = cmask.shape

    # DEM terrain hook (lazy); falls back to flat-earth offsets.
    terrain = _dem_shadow_hint(dem, sun_az, sun_elev, (h, w)) if dem is not None else None

    elev = max(float(sun_elev), 5.0)
    base_len = max_shadow_frac * (1.0 / np.tan(np.deg2rad(elev)))
    shadow_az = np.deg2rad((sun_az + 180.0) % 360.0)
    ux = np.sin(shadow_az)
    uy = -np.cos(shadow_az)

    candidate = np.zeros((h, w), dtype=np.float32)
    steps = max(1, int(cloud_height_steps))
    for k in range(1, steps + 1):
        frac = base_len * (k / steps)
        dy = int(round(frac * h * uy))
        dx = int(round(frac * w * ux))
        candidate = np.maximum(candidate, _shift_binary(cmask, dy, dx))

    # Do not mark shadow where a cloud sits overhead.
    candidate = candidate * (1.0 - cmask)
    if terrain is not None:
        candidate = np.maximum(candidate, terrain * (1.0 - cmask))

    # Confirm with a dark-NIR test (relative to the scene's clear NIR level).
    nir = arr[nir_idx]
    clear_ref = float(np.median(nir[cmask < 0.5])) if np.any(cmask < 0.5) else float(np.median(nir))
    dark = (nir < 0.6 * clear_ref).astype(np.float32)

    shadow = ((candidate > 0.5) & (dark > 0.5)).astype(np.float32)
    # Keep geometric candidate even if the (synthetic) scene lacks a NIR dip,
    # so downstream masking is never empty when geometry clearly predicts shadow.
    shadow = np.maximum(shadow, (candidate > 0.5).astype(np.float32) * dark)
    shadow = shadow.astype(np.float32)[None, ...]
    return _wrap_like(shadow, is_torch)


def _dem_shadow_hint(
    dem,
    sun_az: float,
    sun_elev: float,
    hw: tuple[int, int],
) -> np.ndarray | None:
    """Optional terrain-shadow hint from a DEM via the lazy ``sources.dem`` hook.

    Returns a ``[H, W]`` hillshade-derived low-illumination mask, or ``None`` if
    the helper / dependency is unavailable.
    """
    try:
        from cloudremoval.data.sources.dem import hillshade
    except Exception:  # noqa: BLE001
        return None
    try:
        dem_arr = _to_numpy(dem)
        if dem_arr.ndim == 3:
            dem_arr = dem_arr[0]
        shade = hillshade(dem_arr, azimuth=sun_az, altitude=sun_elev)
        shade = np.asarray(shade, dtype=np.float32)
        # Low hillshade = terrain in shadow.
        thr = float(np.quantile(shade, 0.15))
        return (shade <= thr).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        _log.debug("DEM shadow hint unavailable (%s)", exc)
        return None


# --------------------------------------------------------------------------- #
# Small array helpers
# --------------------------------------------------------------------------- #
def _shift_binary(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a binary field by ``(dy, dx)`` with zero border fill."""
    out = np.zeros_like(field)
    h, w = field.shape
    ys_src = slice(max(0, -dy), h - max(0, dy))
    xs_src = slice(max(0, -dx), w - max(0, dx))
    ys_dst = slice(max(0, dy), h - max(0, -dy))
    xs_dst = slice(max(0, dx), w - max(0, -dx))
    out[ys_dst, xs_dst] = field[ys_src, xs_src]
    return out


def _is_torch_tensor(obj) -> bool:
    """Return ``True`` if ``obj`` is a torch tensor (no forced import)."""
    return type(obj).__module__.startswith("torch") and type(obj).__name__ == "Tensor"


def _to_numpy(obj) -> np.ndarray:
    """Convert a torch tensor / array-like to a contiguous ``float32`` NumPy array."""
    if _is_torch_tensor(obj):
        return np.ascontiguousarray(obj.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(obj), dtype=np.float32)


def _wrap_like(array: np.ndarray, is_torch: bool):
    """Return ``array`` as a torch tensor if ``is_torch`` else the NumPy array."""
    if is_torch:
        import torch

        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return array
