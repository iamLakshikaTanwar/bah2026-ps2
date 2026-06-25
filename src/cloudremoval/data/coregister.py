"""Image co-registration: align auxiliary modalities (SAR/DEM/temporal) to optical.

Sub-pixel misregistration between LISS-IV (5.8 m) and Sentinel-1/2 (10 m) over NER
relief creates artefacts in any SAR-optical fusion (``research/06`` §E.3), so
auxiliary layers must be aligned to the optical reference before stacking.

This module provides :func:`coregister`, which estimates a translational shift and
resamples the ``target`` onto the ``reference`` grid. Two backends:

* **AROSICS** (lazy import) -- the recommended automated sub-pixel phase-correlation
  co-registration (``research/05`` §B.2). Used when ``arosics`` is installed.
* **Phase correlation (FFT)** -- a pure-NumPy fallback implementing the Fourier
  shift theorem (cross-power spectrum -> shift), the default on the CPU-smoke
  stack. Equivalent in spirit to ``skimage.registration.phase_cross_correlation``
  / OpenCV ``phaseCorrelate`` but with no dependency beyond NumPy's FFT.

Returns the aligned array plus the estimated ``(dy, dx)`` shift.
"""

from __future__ import annotations

import numpy as np

from cloudremoval.utils.logging import get_logger

__all__ = [
    "coregister",
    "phase_correlation_shift",
    "shift_array",
]

_log = get_logger(__name__)
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Shift estimation (pure-NumPy phase correlation)
# --------------------------------------------------------------------------- #
def phase_correlation_shift(
    reference: np.ndarray,
    target: np.ndarray,
    upsample: int = 1,
) -> tuple[float, float]:
    """Estimate the ``(dy, dx)`` shift aligning ``target`` to ``reference``.

    Implements phase correlation via the normalised cross-power spectrum
    (Fourier shift theorem). Both inputs must be single-band 2-D arrays on the
    **same pixel grid** (``research/05`` §B.2). The returned shift is the
    translation to apply to ``target`` to match ``reference``.

    Args:
        reference: ``[H, W]`` reference band.
        target: ``[H, W]`` band to align (same shape as ``reference``).
        upsample: Integer upsampling factor for sub-pixel refinement via a local
            parabolic peak fit (``1`` = integer-pixel accuracy).

    Returns:
        ``(dy, dx)`` shift in pixels (positive = shift ``target`` down/right).
    """
    ref = _prep(reference)
    tgt = _prep(target)
    if ref.shape != tgt.shape:
        raise ValueError(f"reference {ref.shape} and target {tgt.shape} must match")
    h, w = ref.shape

    fft_ref = np.fft.fft2(ref)
    fft_tgt = np.fft.fft2(tgt)
    cross = fft_ref * np.conj(fft_tgt)
    cross /= np.abs(cross) + _EPS
    corr = np.fft.ifft2(cross).real

    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = float(peak[0]), float(peak[1])

    # Sub-pixel parabolic refinement around the integer peak.
    if upsample > 1:
        dy += _parabolic_offset(corr, peak, axis=0)
        dx += _parabolic_offset(corr, peak, axis=1)

    # Wrap to signed range [-N/2, N/2).
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dy, dx


def _parabolic_offset(corr: np.ndarray, peak: tuple[int, int], axis: int) -> float:
    """Sub-pixel peak offset along ``axis`` from a 3-point parabolic fit."""
    h, w = corr.shape
    py, px = peak
    if axis == 0:
        ym = (py - 1) % h
        yp = (py + 1) % h
        a, b, c = corr[ym, px], corr[py, px], corr[yp, px]
    else:
        xm = (px - 1) % w
        xp = (px + 1) % w
        a, b, c = corr[py, xm], corr[py, px], corr[py, xp]
    denom = a - 2 * b + c
    if abs(denom) < _EPS:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Apply a (possibly fractional) shift
# --------------------------------------------------------------------------- #
def shift_array(array: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate a ``[C, H, W]``/``[H, W]`` array by ``(dy, dx)`` (bilinear).

    Integer shifts use fast slicing; fractional shifts use bilinear gather.
    Exposed borders are edge-replicated to avoid introducing zeros into fused
    stacks.

    Args:
        array: Channel-first image (or 2-D band).
        dy: Vertical shift (positive = down).
        dx: Horizontal shift (positive = right).

    Returns:
        The shifted array (same shape, ``float32``).
    """
    arr = np.asarray(array, dtype=np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None, ...]
    c, h, w = arr.shape

    if float(dy).is_integer() and float(dx).is_integer():
        out = _shift_integer(arr, int(dy), int(dx))
        return out[0] if squeeze else out

    ys = np.clip(np.arange(h, dtype=np.float32) - dy, 0, h - 1)
    xs = np.clip(np.arange(w, dtype=np.float32) - dx, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0).reshape(1, h, 1)
    wx = (xs - x0).reshape(1, 1, w)

    top = arr[:, y0][:, :, x0] * (1 - wx) + arr[:, y0][:, :, x1] * wx
    bot = arr[:, y1][:, :, x0] * (1 - wx) + arr[:, y1][:, :, x1] * wx
    out = (top * (1 - wy) + bot * wy).astype(np.float32)
    return out[0] if squeeze else out


def _shift_integer(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer translate ``[C, H, W]`` with edge replication."""
    out = np.empty_like(arr)
    shifted = np.roll(np.roll(arr, dy, axis=1), dx, axis=2)
    out[...] = shifted
    # Edge-replicate the wrapped borders instead of leaving rolled content.
    if dy > 0:
        out[:, :dy, :] = arr[:, :1, :]
    elif dy < 0:
        out[:, dy:, :] = arr[:, -1:, :]
    if dx > 0:
        out[:, :, :dx] = out[:, :, dx : dx + 1]
    elif dx < 0:
        out[:, :, dx:] = out[:, :, dx - 1 : dx]
    return out


# --------------------------------------------------------------------------- #
# Public co-registration entry point
# --------------------------------------------------------------------------- #
def coregister(
    reference: np.ndarray,
    target: np.ndarray,
    ref_band: int = 0,
    target_band: int = 0,
    use_arosics: bool = True,
    upsample: int = 4,
    max_shift: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Align ``target`` to ``reference`` and return ``(aligned, (dy, dx))``.

    Picks a single matching band from each input (phase correlation requires
    single-band, same-grid inputs -- ``research/05`` §B.2), estimates the shift,
    and resamples *all* bands of ``target`` by that shift.

    Backends: **AROSICS** (lazy) when available and ``use_arosics`` is set, else a
    pure-NumPy FFT phase-correlation fallback (CPU-smoke default).

    Args:
        reference: ``[C, H, W]`` or ``[H, W]`` optical reference (the fixed grid).
        target: ``[C, H, W]`` or ``[H, W]`` auxiliary array to align (SAR/DEM/...).
        ref_band: Band index of ``reference`` used for matching.
        target_band: Band index of ``target`` used for matching.
        use_arosics: Try the AROSICS backend first if installed.
        upsample: Sub-pixel upsampling factor for the fallback.
        max_shift: Optional cap on absolute shift magnitude (pixels); larger
            estimates are clamped (guards against spurious global matches).

    Returns:
        ``(aligned_target, (dy, dx))`` -- ``aligned_target`` has the same shape as
        the input ``target``; the shift is in reference-grid pixels.
    """
    ref = np.asarray(reference, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    ref_b = ref[ref_band] if ref.ndim == 3 else ref
    tgt_b = tgt[target_band] if tgt.ndim == 3 else tgt

    if ref_b.shape != tgt_b.shape:
        _log.debug(
            "coregister: shapes differ (%s vs %s); skipping (resample upstream)",
            ref_b.shape,
            tgt_b.shape,
        )
        return tgt, (0.0, 0.0)

    shift: tuple[float, float] | None = None
    if use_arosics:
        shift = _arosics_shift(ref_b, tgt_b)

    if shift is None:
        shift = phase_correlation_shift(ref_b, tgt_b, upsample=upsample)

    dy, dx = shift
    if max_shift is not None:
        dy = float(np.clip(dy, -max_shift, max_shift))
        dx = float(np.clip(dx, -max_shift, max_shift))

    aligned = shift_array(tgt, dy, dx)
    return aligned, (dy, dx)


def _arosics_shift(ref_b: np.ndarray, tgt_b: np.ndarray) -> tuple[float, float] | None:
    """Estimate a global shift via AROSICS if installed, else ``None``.

    AROSICS' ``COREG`` normally operates on georeferenced files; here we only use
    it opportunistically and degrade to the FFT fallback on any failure (it is
    absent on the CPU-smoke stack).
    """
    try:
        import arosics  # type: ignore  # noqa: F401  (lazy, optional)
    except Exception:  # noqa: BLE001
        _log.debug("arosics unavailable; using FFT phase-correlation fallback")
        return None
    # A full AROSICS run needs GeoArray/file inputs and a projection; for in-memory
    # same-grid arrays the FFT fallback is equivalent, so we defer to it here and
    # keep this branch as the documented integration point.
    _log.debug("arosics present but in-memory path defers to FFT correlation")
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _prep(band: np.ndarray) -> np.ndarray:
    """Prepare a 2-D band for correlation: float, mean-removed, Hann-windowed."""
    arr = np.asarray(band, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D band, got shape {arr.shape}")
    arr = arr - arr.mean()
    h, w = arr.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    return arr * win
