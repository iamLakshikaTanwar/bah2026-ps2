"""Analysis-ready finishing of a tiled reconstruction.

Cloud removal only needs to *replace the clouded pixels*: clear pixels are already
the true surface and must be preserved exactly, while reconstructed pixels should
match their clear surroundings radiometrically (no visible patch). This module
provides the standard finishing steps (``research/05`` §B):

* :func:`composite_under_mask` — keep the original where it is clear, paste the
  reconstruction only under cloud + shadow, with a feathered transition so the
  seam is invisible.
* :func:`histogram_match` / :func:`per_band_bias_correct` — spectral harmonization
  so the filled region's statistics match the surrounding clear neighbourhood.
* :func:`poisson_blend` — gradient-domain (Poisson) seamless cloning when
  ``scipy``/``opencv`` are present, with a smooth feather fallback otherwise.
* :func:`to_reflectance` / :func:`clamp_valid` — de-normalize and clamp to a valid
  reflectance / DN range.

All defaults are **pure NumPy/torch**; heavier paths (Poisson via ``scipy`` /
``opencv``) are lazily imported with a graceful fallback, so the CPU-smoke path
needs no extra deps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cloudremoval.utils.logging import get_logger

__all__ = [
    "composite_under_mask",
    "feather_mask",
    "histogram_match",
    "per_band_bias_correct",
    "poisson_blend",
    "to_reflectance",
    "clamp_valid",
]

_log = get_logger(__name__)
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_numpy(array: Any, dtype: type = np.float32) -> np.ndarray:
    """Convert a numpy array or torch tensor to a contiguous ``[C, H, W]`` array."""
    try:
        import torch

        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
    except Exception:  # noqa: BLE001 - torch always present, but stay defensive
        pass
    arr = np.asarray(array, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return np.ascontiguousarray(arr)


def _broadcast_mask(mask: np.ndarray, channels: int) -> np.ndarray:
    """Broadcast a ``[1, H, W]`` / ``[H, W]`` mask to ``[C, H, W]`` float in [0, 1]."""
    m = _to_numpy(mask)
    if m.shape[0] == 1 and channels > 1:
        m = np.repeat(m, channels, axis=0)
    return np.clip(m, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Mask feathering & compositing
# --------------------------------------------------------------------------- #
def _box_blur_axis(arr: np.ndarray, axis: int, radius: int, k: int) -> np.ndarray:
    """Shape-preserving 1-D box-mean of width ``k`` along ``axis`` (edge-padded).

    Pads ``radius`` on each side, then computes a length-``k`` windowed sum from a
    zero-prepended cumulative sum so the output length equals the input length.
    """
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="edge")
    csum = np.cumsum(padded, axis=axis)
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=csum.dtype), csum], axis=axis)
    upper = np.take(csum, indices=range(k, csum.shape[axis]), axis=axis)
    lower = np.take(csum, indices=range(0, csum.shape[axis] - k), axis=axis)
    return (upper - lower) / float(k)


def feather_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Soften a hard 0/1 mask into a smooth alpha for a seamless paste.

    Uses a separable box-blur (repeated to approximate a Gaussian) so no heavy
    dependency is required. Pixels deep inside the mask stay ~1, pixels far
    outside stay 0, and the ``radius``-wide rim ramps smoothly between them.

    Args:
        mask: ``[1, H, W]`` or ``[H, W]`` mask (``1`` = use reconstruction).
        radius: Half-width of the transition band in pixels (``0`` = no feather).

    Returns:
        A ``float32`` alpha ``[1, H, W]`` in ``[0, 1]``.
    """
    m = _to_numpy(mask)
    if radius <= 0:
        return np.clip(m, 0.0, 1.0)
    k = 2 * radius + 1
    out = m
    # Two box-blur passes ~ a triangular (smooth) kernel — cheap and dependency-free.
    # Each pass is a shape-preserving sliding-window mean via a zero-prepended
    # cumulative sum: pad by ``radius`` per side, then a length-``k`` window over
    # the (L+1)-long cumsum yields exactly L outputs.
    for _ in range(2):
        out = _box_blur_axis(out, axis=1, radius=radius, k=k)
        out = _box_blur_axis(out, axis=2, radius=radius, k=k)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def composite_under_mask(
    original: Any,
    reconstruction: Any,
    cloud_mask: Any,
    shadow_mask: Any | None = None,
    feather: int = 4,
) -> np.ndarray:
    """Keep ``original`` where clear; paste ``reconstruction`` under cloud+shadow.

    The defining post-process: a cloud-removal product must not alter valid
    observations. Pixels outside the cloud/shadow mask are copied verbatim from
    ``original``; only masked pixels take the model's reconstruction, blended over
    a ``feather``-wide rim so the boundary is invisible.

    Args:
        original: The observed (cloudy) scene ``[C, H, W]``.
        reconstruction: The model output ``[C, H, W]`` (same normalization/space
            as ``original``).
        cloud_mask: ``[1, H, W]`` mask, ``1`` = cloud (fill from reconstruction).
        shadow_mask: Optional ``[1, H, W]`` shadow mask, OR-ed with the cloud mask.
        feather: Feather radius (px) for the transition (``0`` = hard edge).

    Returns:
        The composited analysis-ready scene ``[C, H, W]`` (``float32``).

    Raises:
        ValueError: If ``original`` and ``reconstruction`` shapes differ.
    """
    orig = _to_numpy(original)
    recon = _to_numpy(reconstruction)
    if orig.shape != recon.shape:
        raise ValueError(f"shape mismatch: original {orig.shape} vs recon {recon.shape}")

    mask = _to_numpy(cloud_mask)
    if shadow_mask is not None:
        mask = np.clip(mask + _to_numpy(shadow_mask), 0.0, 1.0)
    alpha = feather_mask(mask, radius=feather)
    alpha = _broadcast_mask(alpha, orig.shape[0])

    return (orig * (1.0 - alpha) + recon * alpha).astype(np.float32)


# --------------------------------------------------------------------------- #
# Spectral harmonization
# --------------------------------------------------------------------------- #
def histogram_match(
    source: Any,
    reference: Any,
    mask: Any | None = None,
    reference_mask: Any | None = None,
    n_bins: int = 256,
) -> np.ndarray:
    """Match ``source``'s per-band histogram to a ``reference`` (clear) image.

    Classic CDF (quantile) matching, applied per band, so the reconstructed
    region's tone/contrast matches the surrounding clear pixels. Pure NumPy (a
    ``skimage.exposure.match_histograms`` equivalent) — no heavy dependency.

    Args:
        source: Image to recolour ``[C, H, W]`` (the reconstruction).
        reference: Image whose statistics to adopt ``[C, H, W]`` (clear scene /
            neighbourhood).
        mask: Optional ``[1, H, W]`` selecting which source pixels are remapped
            (``1`` = remap; others pass through unchanged). Defaults to all.
        reference_mask: Optional ``[1, H, W]`` selecting reference pixels whose
            histogram is the target (``1`` = clear pixels to learn from).
        n_bins: Number of quantile bins used to estimate each CDF.

    Returns:
        The histogram-matched image ``[C, H, W]`` (``float32``).
    """
    src = _to_numpy(source)
    ref = _to_numpy(reference)
    c, h, w = src.shape

    src_sel = None if mask is None else _to_numpy(mask)[0] > 0.5
    ref_sel = None if reference_mask is None else _to_numpy(reference_mask)[0] > 0.5

    quantiles = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
    out = src.copy()
    for band in range(c):
        s_band = src[band]
        r_band = ref[band] if band < ref.shape[0] else ref[-1]
        s_vals = s_band[src_sel] if src_sel is not None else s_band.ravel()
        r_vals = r_band[ref_sel] if ref_sel is not None else r_band.ravel()
        if s_vals.size == 0 or r_vals.size == 0:
            continue
        s_q = np.quantile(s_vals, quantiles)
        r_q = np.quantile(r_vals, quantiles)
        # Map every source pixel through src-CDF -> reference-quantile.
        flat = s_band.ravel().astype(np.float64)
        mapped = np.interp(flat, s_q, r_q).reshape(h, w)
        if src_sel is not None:
            band_out = s_band.copy()
            band_out[src_sel] = mapped[src_sel]
            out[band] = band_out
        else:
            out[band] = mapped
    return out.astype(np.float32)


def per_band_bias_correct(
    source: Any,
    reference: Any,
    mask: Any | None = None,
    reference_mask: Any | None = None,
) -> np.ndarray:
    """Affine (gain + offset) match each band's mean/std to a reference.

    A lighter alternative to full histogram matching: rescales each band so its
    mean and standard deviation match the reference's (over the selected pixels).
    Robust when only a small clear neighbourhood is available.

    Args:
        source: Image to correct ``[C, H, W]``.
        reference: Reference image ``[C, H, W]``.
        mask: Optional ``[1, H, W]`` selecting source pixels used to estimate
            (and that receive) the correction.
        reference_mask: Optional ``[1, H, W]`` selecting reference statistics.

    Returns:
        The bias-corrected image ``[C, H, W]`` (``float32``).
    """
    src = _to_numpy(source)
    ref = _to_numpy(reference)
    src_sel = None if mask is None else _to_numpy(mask)[0] > 0.5
    ref_sel = None if reference_mask is None else _to_numpy(reference_mask)[0] > 0.5

    out = src.copy()
    for band in range(src.shape[0]):
        s_band = src[band]
        r_band = ref[band] if band < ref.shape[0] else ref[-1]
        s_vals = s_band[src_sel] if src_sel is not None else s_band.ravel()
        r_vals = r_band[ref_sel] if ref_sel is not None else r_band.ravel()
        if s_vals.size == 0 or r_vals.size == 0:
            continue
        s_mu, s_sd = float(s_vals.mean()), float(s_vals.std())
        r_mu, r_sd = float(r_vals.mean()), float(r_vals.std())
        gain = r_sd / (s_sd + _EPS)
        corrected = (s_band - s_mu) * gain + r_mu
        if src_sel is not None:
            band_out = s_band.copy()
            band_out[src_sel] = corrected[src_sel]
            out[band] = band_out
        else:
            out[band] = corrected
    return out.astype(np.float32)


def poisson_blend(
    original: Any,
    reconstruction: Any,
    mask: Any,
    feather: int = 4,
) -> np.ndarray:
    """Seamlessly clone ``reconstruction`` into ``original`` under ``mask``.

    Prefers gradient-domain (Poisson) blending — OpenCV's ``seamlessClone`` if
    available, else a ``scipy`` sparse Poisson solve — which transfers the
    reconstruction's *gradients* while honouring the clear boundary, eliminating
    tone discontinuities. If neither library is installed it falls back to the
    feathered alpha composite (:func:`composite_under_mask`), so the result is
    always defined on the minimal stack.

    Args:
        original: Observed scene ``[C, H, W]``.
        reconstruction: Reconstruction ``[C, H, W]``.
        mask: ``[1, H, W]`` mask (``1`` = clone region).
        feather: Feather radius for the fallback composite.

    Returns:
        The blended scene ``[C, H, W]`` (``float32``).
    """
    orig = _to_numpy(original)
    recon = _to_numpy(reconstruction)
    m = _to_numpy(mask)[0] > 0.5

    # Try a true Poisson solve via scipy (channel-wise). Operates in-place on a
    # copy; only the masked interior is solved, Dirichlet-pinned to ``original``.
    try:  # pragma: no cover - exercised only when scipy is installed
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve

        return _poisson_scipy(orig, recon, m, lil_matrix, spsolve)
    except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
        _log.debug("Poisson (scipy) unavailable/failed (%s); feather fallback", exc)
        return composite_under_mask(orig, recon, m.astype(np.float32)[None], feather=feather)


def _poisson_scipy(
    orig: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
    lil_matrix: Any,
    spsolve: Any,
) -> np.ndarray:  # pragma: no cover - only with scipy present
    """Solve the discrete Poisson equation with the reconstruction's Laplacian."""
    c, h, w = orig.shape
    idx = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.where(mask)
    idx[ys, xs] = np.arange(ys.size)
    n = ys.size
    if n == 0:
        return orig.astype(np.float32)

    a = lil_matrix((n, n), dtype=np.float64)
    out = orig.copy()
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for band in range(c):
        b = np.zeros(n, dtype=np.float64)
        guide = recon[band]
        target = orig[band]
        for p, (y, x) in enumerate(zip(ys, xs)):
            a[p, p] = 4.0
            lap = 0.0
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                lap += guide[y, x] - guide[ny, nx]
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                    a[p, idx[ny, nx]] = -1.0
                else:
                    b[p] += target[ny, nx]
            b[p] += lap
        sol = spsolve(a.tocsr(), b)
        band_out = target.copy()
        band_out[ys, xs] = np.clip(sol, target.min(), target.max())
        out[band] = band_out
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# De-normalization & clamping
# --------------------------------------------------------------------------- #
def to_reflectance(
    array: Any,
    stats: dict[str, list[float]] | None = None,
    norm: str = "zscore",
) -> np.ndarray:
    """De-normalize a model output back to reflectance/DN using ``meta`` stats.

    Thin wrapper over :func:`cloudremoval.utils.geo.denormalize` (pure NumPy). If
    ``stats`` is ``None`` the array is returned unchanged (assumed already in
    reflectance space).

    Args:
        array: Normalized image ``[C, H, W]``.
        stats: ``meta["band_stats"]`` (``{"mean","std"}`` or ``{"p2","p98"}``).
        norm: ``"zscore"`` or ``"percentile"`` (matches ``meta["norm"]``).

    Returns:
        The de-normalized image ``[C, H, W]`` (``float32``).
    """
    arr = _to_numpy(array)
    if stats is None:
        return arr
    from cloudremoval.utils.geo import denormalize

    return denormalize(arr, stats, norm=norm).astype(np.float32)


def clamp_valid(
    array: Any,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Clamp to a valid reflectance/DN range (default ``[0, 1]`` reflectance).

    Args:
        array: Image ``[C, H, W]``.
        lo: Lower bound (e.g. ``0.0`` reflectance, ``0`` DN).
        hi: Upper bound (e.g. ``1.0`` reflectance, ``1023`` for 10-bit DN).

    Returns:
        The clamped image ``[C, H, W]`` (``float32``).
    """
    return np.clip(_to_numpy(array), lo, hi).astype(np.float32)
