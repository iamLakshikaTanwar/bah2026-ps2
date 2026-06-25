"""Pure-torch, mask-aware MVES metrics for cloud-removal evaluation.

Owns the **frozen metric contract** (``docs/BUILD_PLAN.md`` §3.5) and the metric
definitions in ``research/05_metrics_preprocessing_deployment.md`` (MVES Tier 0/1):
masked + whole-image PSNR / SSIM / RMSE / MAE, the spectral-fidelity gate
SAM / ERGAS / SID, NDVI MAE, per-band correlation / bias, and a lazily-imported
LPIPS perceptual metric.

Design rules
------------
* **Tier-0 metrics are pure ``torch``** (with a tiny NumPy-free core) so they run
  on the CPU-smoke stack with *zero* heavy dependencies.
* **Mask-aware:** every metric accepts an optional ``mask`` ``[*, 1, H, W]`` (or
  broadcastable) selecting the pixels to score — typically the cloud (or
  cloud+shadow) region. ``mask=None`` scores the whole image. The cloud-region
  column is the real score (whole-image is inflated by the already-clear pixels).
* **Numerically safe:** divisions are ``eps``-guarded; an all-empty mask yields a
  graceful value (``nan`` for ratio metrics, a high but finite PSNR), never a
  crash.
* **Return types:** every scalar metric returns a Python ``float``. The two
  list-valued metrics (:func:`per_band_correlation`, :func:`per_band_bias`)
  return ``list[float]``. :func:`compute_all` returns a nested
  ``dict[str, dict[str, float]]``. This is documented per-function and is stable.

Input convention
----------------
``pred`` and ``target`` are ``[B, C, H, W]`` or ``[C, H, W]`` float tensors in the
*same* space (the caller is consistent — normalized or reflectance). For LISS-IV,
the channel order is Green=0, Red=1, NIR=2 (so NDVI uses ``red_idx=1, nir_idx=2``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import EvalConfig

__all__ = [
    "psnr",
    "ssim",
    "ms_ssim",
    "sam",
    "ergas",
    "sid",
    "rmse",
    "mae",
    "ndvi_mae",
    "spectral_correlation",
    "per_band_correlation",
    "per_band_bias",
    "lpips",
    "compute_all",
    "compute_stratified",
]

_log = get_logger(__name__)

_EPS = 1e-8
# A finite stand-in for an "infinite" PSNR (identical images) so callers never
# have to special-case ``inf`` when averaging.
_PSNR_MAX_DB = 100.0


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #
def _as_bchw(x: Tensor) -> Tensor:
    """Return ``x`` as a 4-D ``[B, C, H, W]`` tensor (adds a batch dim if 3-D)."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"expected a [C,H,W] or [B,C,H,W] tensor, got shape {tuple(x.shape)}")


def _prep_pair(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Coerce both tensors to 4-D float and validate matching shapes."""
    p = _as_bchw(pred).float()
    t = _as_bchw(target).float()
    if p.shape != t.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    return p, t


def _prep_mask(mask: Tensor | None, ref: Tensor) -> Tensor | None:
    """Coerce a mask to a float ``[B, 1, H, W]`` broadcastable against ``ref``.

    Accepts boolean or float masks, with or without a batch/channel dim. Returns
    ``None`` unchanged. The returned mask broadcasts over the channel dimension.
    """
    if mask is None:
        return None
    m = mask
    if m.dim() == 2:  # [H, W]
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:  # [1,H,W] or [C,H,W] -> collapse channels, add batch
        m = m[:1].unsqueeze(0)
    elif m.dim() == 4:  # [B,1,H,W] or [B,C,H,W]
        m = m[:, :1]
    else:
        raise ValueError(f"unsupported mask shape {tuple(mask.shape)}")
    m = m.to(ref.dtype)
    # Spatial size must match ``ref`` so masked selection is well-defined.
    if m.shape[-2:] != ref.shape[-2:]:
        raise ValueError(
            f"mask spatial size {tuple(m.shape[-2:])} != image {tuple(ref.shape[-2:])}"
        )
    return m


def _masked_reduce(per_pixel: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of an elementwise error map over (optionally) ``mask`` pixels.

    ``mask`` broadcasts over channels. An empty mask returns ``nan`` (the caller
    decides how to surface "no pixels to score").
    """
    if mask is None:
        return per_pixel.mean()
    weighted = (per_pixel * mask).sum()
    denom = mask.expand_as(per_pixel).sum()
    if denom <= _EPS:
        return per_pixel.new_tensor(float("nan"))
    return weighted / denom


def _to_float(value: Tensor | float) -> float:
    """Convert a scalar tensor/number to a Python float (passes through nan/inf)."""
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)


# --------------------------------------------------------------------------- #
# Pixel-fidelity metrics
# --------------------------------------------------------------------------- #
def mae(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """Mean absolute error over (optionally masked) pixels. Returns ``float``."""
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    return _to_float(_masked_reduce((p - t).abs(), m))


def rmse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """Root-mean-square error over (optionally masked) pixels. Returns ``float``."""
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    mse = _masked_reduce((p - t) ** 2, m)
    return _to_float(torch.sqrt(mse.clamp_min(0.0)))


def psnr(
    pred: Tensor, target: Tensor, mask: Tensor | None = None, data_range: float = 1.0
) -> float:
    """Peak signal-to-noise ratio in dB (higher is better). Returns ``float``.

    Computed over (optionally) ``mask``-selected pixels. Identical images give a
    finite ``_PSNR_MAX_DB`` (not ``inf``) so averaging over a batch is safe; an
    empty mask yields ``nan``.

    Args:
        pred: Predicted image ``[B,C,H,W]`` / ``[C,H,W]``.
        target: Reference image, same shape.
        mask: Optional region selector ``[*,1,H,W]``.
        data_range: Maximum signal value (``1.0`` for normalized reflectance).
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    mse = _masked_reduce((p - t) ** 2, m)
    if torch.isnan(mse):
        return float("nan")
    if mse <= _EPS:
        return _PSNR_MAX_DB
    value = 10.0 * math.log10((data_range**2) / float(mse))
    return min(value, _PSNR_MAX_DB)


# --------------------------------------------------------------------------- #
# SSIM / MS-SSIM (pure torch, per-band then averaged)
# --------------------------------------------------------------------------- #
def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> Tensor:
    """1-D Gaussian kernel normalised to sum 1."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _ssim_map(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> Tensor:
    """Per-pixel, per-channel SSIM map for ``[B,C,H,W]`` inputs (depthwise conv)."""
    channels = pred.shape[1]
    win = min(window_size, pred.shape[-1], pred.shape[-2])
    if win % 2 == 0:  # keep the window odd for symmetric padding
        win -= 1
    win = max(win, 3)
    k1d = _gaussian_kernel(win, sigma, pred.device, pred.dtype)
    k2d = (k1d[:, None] @ k1d[None, :]).expand(channels, 1, win, win)
    pad = win // 2

    def filt(x: Tensor) -> Tensor:
        return F.conv2d(x, k2d, padding=pad, groups=channels)

    mu_x, mu_y = filt(pred), filt(target)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(target * target) - mu_y2
    sigma_xy = filt(pred * target) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return num / (den + _EPS)


def ssim(
    pred: Tensor, target: Tensor, mask: Tensor | None = None, data_range: float = 1.0
) -> float:
    """Structural similarity (per-band, then averaged), in ``[-1, 1]`` (≈1 good).

    Returns a Python ``float``. When a ``mask`` is given the SSIM map is averaged
    only over masked pixels (the windowed statistics still use the full image, so
    structure near the mask boundary is well defined). An empty mask → ``nan``.
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    smap = _ssim_map(p, t, data_range=data_range)
    return _to_float(_masked_reduce(smap, m))


def ms_ssim(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    data_range: float = 1.0,
    levels: int = 3,
) -> float:
    """Multi-scale SSIM (mean of single-scale SSIM over a Gaussian pyramid).

    A lightweight, pure-torch approximation of MS-SSIM: the SSIM is computed at
    successively half-resolution scales and averaged. Returns a Python ``float``.
    The ``mask`` (if any) is downsampled per scale. Falls back to single-scale
    SSIM when the image is too small to downsample.
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    scores: list[float] = []
    cur_p, cur_t, cur_m = p, t, m
    for level in range(max(levels, 1)):
        if min(cur_p.shape[-2:]) < 8:
            break
        smap = _ssim_map(cur_p, cur_t, data_range=data_range)
        val = _to_float(_masked_reduce(smap, cur_m))
        if not math.isnan(val):
            scores.append(val)
        if level < levels - 1:
            cur_p = F.avg_pool2d(cur_p, 2)
            cur_t = F.avg_pool2d(cur_t, 2)
            cur_m = F.avg_pool2d(cur_m, 2) if cur_m is not None else None
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Spectral-fidelity metrics (the deliverable-defining gate)
# --------------------------------------------------------------------------- #
def sam(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """Spectral Angle Mapper in **degrees** (lower is better). Returns ``float``.

    Per-pixel angle between predicted and target spectral vectors (across
    channels), averaged over (optionally masked) pixels. Invariant to per-pixel
    gain/illumination scaling — it captures spectral *shape*. With the 3 LISS-IV
    bands the vector is short, so pair it with ERGAS / per-band correlation.
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    dot = (p * t).sum(dim=1, keepdim=True)  # [B,1,H,W]
    denom = p.norm(dim=1, keepdim=True) * t.norm(dim=1, keepdim=True) + _EPS
    cos = (dot / denom).clamp(-1.0 + _EPS, 1.0 - _EPS)
    angle = torch.acos(cos)  # radians, [B,1,H,W]
    mean_rad = _masked_reduce(angle, m)
    return _to_float(mean_rad * (180.0 / math.pi))


def ergas(pred: Tensor, target: Tensor, mask: Tensor | None = None, ratio: float = 1.0) -> float:
    """ERGAS — global relative dimensionless spectral error (lower better).

    ``100 * ratio * sqrt( mean_b( RMSE_b^2 / mu_b^2 ) )`` where ``mu_b`` is the
    target band mean over the scored region. Returns a Python ``float``. ``ratio``
    is the (high/low) resolution ratio — ``1.0`` when pred/target share a grid.
    An empty mask → ``nan``.
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    channels = p.shape[1]
    per_band: list[Tensor] = []
    for c in range(channels):
        pc = p[:, c : c + 1]
        tc = t[:, c : c + 1]
        mse_b = _masked_reduce((pc - tc) ** 2, m)
        mean_b = _masked_reduce(tc, m)
        if torch.isnan(mse_b) or torch.isnan(mean_b):
            return float("nan")
        ratio_b = mse_b / (mean_b**2 + _EPS)
        per_band.append(ratio_b)
    mean_ratio = torch.stack(per_band).mean()
    return _to_float(100.0 * ratio * torch.sqrt(mean_ratio.clamp_min(0.0)))


def sid(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """Spectral Information Divergence (symmetric KL of spectra). Returns ``float``.

    Each pixel's spectrum (across channels) is turned into a probability vector by
    taking absolute values and normalising; SID is ``KL(p‖q)+KL(q‖p)`` averaged
    over (optionally masked) pixels. Lower is better. Needs non-negative spectra
    and is unstable for near-zero 3-band vectors — reported as a secondary metric.
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    pa = p.abs() + _EPS
    ta = t.abs() + _EPS
    pp = pa / pa.sum(dim=1, keepdim=True)
    qq = ta / ta.sum(dim=1, keepdim=True)
    kl_pq = (pp * (pp / qq).log()).sum(dim=1, keepdim=True)
    kl_qp = (qq * (qq / pp).log()).sum(dim=1, keepdim=True)
    per_pixel = kl_pq + kl_qp  # [B,1,H,W]
    return _to_float(_masked_reduce(per_pixel, m))


def _ndvi(img: Tensor, red_idx: int, nir_idx: int) -> Tensor:
    """Per-pixel NDVI ``(NIR-Red)/(NIR+Red)`` as ``[B,1,H,W]`` (eps-guarded)."""
    red = img[:, red_idx : red_idx + 1]
    nir = img[:, nir_idx : nir_idx + 1]
    return (nir - red) / (nir + red + _EPS)


def ndvi_mae(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    red_idx: int = 1,
    nir_idx: int = 2,
) -> float:
    """Mean absolute error of NDVI over (optionally masked) pixels. Returns ``float``.

    The most reviewer-legible "is it analysis-ready" proxy: a model can match RGB
    visually yet shift NIR and wreck NDVI. Channel order defaults to LISS-IV
    (Green=0, Red=1, NIR=2).
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    diff = (_ndvi(p, red_idx, nir_idx) - _ndvi(t, red_idx, nir_idx)).abs()
    return _to_float(_masked_reduce(diff, m))


# --------------------------------------------------------------------------- #
# Per-band agreement
# --------------------------------------------------------------------------- #
def _masked_select_band(x: Tensor, mask: Tensor | None) -> Tensor:
    """Flatten a single-band tensor to the masked (or all) pixel values."""
    if mask is None:
        return x.reshape(-1)
    sel = mask.expand_as(x) > 0.5
    return x[sel]


def spectral_correlation(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """Mean per-band Pearson correlation over (optionally masked) pixels.

    Returns a single Python ``float`` (the average of the per-band correlations).
    Insensitive to per-band bias/gain — pair with :func:`per_band_bias`. Returns
    ``nan`` when there are too few pixels or a band has zero variance.
    """
    bands = per_band_correlation(pred, target, mask)
    valid = [b for b in bands if not math.isnan(b)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def per_band_correlation(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> list[float]:
    """Per-band Pearson correlation coefficients. Returns ``list[float]`` (len=C).

    Each entry is ``corr(pred_b, target_b)`` over the masked pixels; ``nan`` if a
    band is constant or fewer than two pixels are selected.
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    out: list[float] = []
    for c in range(p.shape[1]):
        pv = _masked_select_band(p[:, c : c + 1], m)
        tv = _masked_select_band(t[:, c : c + 1], m)
        if pv.numel() < 2:
            out.append(float("nan"))
            continue
        pv = pv - pv.mean()
        tv = tv - tv.mean()
        denom = pv.norm() * tv.norm()
        if denom <= _EPS:
            out.append(float("nan"))
        else:
            out.append(_to_float((pv * tv).sum() / denom))
    return out


def per_band_bias(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> list[float]:
    """Per-band mean signed bias ``mean(pred_b - target_b)``. Returns ``list[float]``.

    SAM and correlation both hide systematic bias that wrecks reflectance-based
    products, so this is reported separately (one value per channel).
    """
    p, t = _prep_pair(pred, target)
    m = _prep_mask(mask, p)
    out: list[float] = []
    for c in range(p.shape[1]):
        diff = p[:, c : c + 1] - t[:, c : c + 1]
        out.append(_to_float(_masked_reduce(diff, m)))
    return out


# --------------------------------------------------------------------------- #
# LPIPS (perceptual, lazy + graceful)
# --------------------------------------------------------------------------- #
def lpips(pred: Tensor, target: Tensor, mask: Tensor | None = None, net: str = "alex") -> float:
    """LPIPS perceptual distance on an RGB render (lower better). Returns ``float``.

    Lazily imports the optional ``lpips`` package. The 3 LISS-IV bands are mapped
    to RGB and scaled to ``[-1, 1]`` (the network's expected range). If ``lpips``
    is unavailable (the CPU-smoke minimal stack), this returns ``float('nan')``
    after a one-line warning — it never crashes. The ``mask`` argument is accepted
    for signature uniformity but LPIPS is a whole-image perceptual metric and the
    mask is not applied.
    """
    try:
        import lpips as _lpips_pkg  # type: ignore  # optional heavy dep
    except Exception:  # noqa: BLE001 - any import failure → graceful nan
        _log.warning("lpips unavailable; returning nan for the LPIPS metric.")
        return float("nan")

    p, t = _prep_pair(pred, target)

    def to_rgb(x: Tensor) -> Tensor:
        x3 = x[:, :3] if x.shape[1] >= 3 else x[:, :1].repeat(1, 3, 1, 1)
        return x3.clamp(0.0, 1.0) * 2.0 - 1.0

    try:
        model = _LPIPS_CACHE.get(net)
        if model is None:
            model = _lpips_pkg.LPIPS(net=net)
            model.eval()
            _LPIPS_CACHE[net] = model
        with torch.no_grad():
            dist = model(to_rgb(p), to_rgb(t))
        return _to_float(dist.mean())
    except Exception as exc:  # noqa: BLE001 - missing weights, etc.
        _log.warning("lpips failed (%s); returning nan.", exc)
        return float("nan")


_LPIPS_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Aggregators
# --------------------------------------------------------------------------- #
def compute_all(
    pred: Tensor,
    target: Tensor,
    cloud_mask: Tensor | None = None,
    data_range: float = 1.0,
    include_lpips: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute the full Tier-0/1 metric panel, whole-image **and** cloud-region.

    Args:
        pred: Predicted image ``[B,C,H,W]`` / ``[C,H,W]``.
        target: Reference image, same shape.
        cloud_mask: Optional cloud (or cloud+shadow) region ``[*,1,H,W]``. When
            given, a second "cloud"-keyed column is produced over those pixels.
        data_range: Signal range for PSNR/SSIM.
        include_lpips: If ``True`` also compute LPIPS (lazy; ``nan`` if absent).

    Returns:
        A nested mapping ``{region: {metric: value}}`` where ``region`` is
        ``"whole"`` always and ``"cloud"`` when ``cloud_mask`` is provided. Scalar
        metrics map to ``float``; the per-band metrics map to ``list[float]``
        under keys ``"per_band_correlation"`` / ``"per_band_bias"``. The return
        type is therefore ``dict[str, dict[str, float | list[float]]]`` (typed
        loosely as ``dict[str, dict[str, float]]``).
    """

    def panel(mask: Tensor | None) -> dict[str, float]:
        out: dict[str, float] = {
            "psnr": psnr(pred, target, mask, data_range=data_range),
            "ssim": ssim(pred, target, mask, data_range=data_range),
            "ms_ssim": ms_ssim(pred, target, mask, data_range=data_range),
            "sam": sam(pred, target, mask),
            "ergas": ergas(pred, target, mask),
            "sid": sid(pred, target, mask),
            "rmse": rmse(pred, target, mask),
            "mae": mae(pred, target, mask),
            "ndvi_mae": ndvi_mae(pred, target, mask),
            "spectral_correlation": spectral_correlation(pred, target, mask),
        }
        # List-valued extras (typed loosely; documented above).
        out["per_band_correlation"] = per_band_correlation(pred, target, mask)  # type: ignore[assignment]
        out["per_band_bias"] = per_band_bias(pred, target, mask)  # type: ignore[assignment]
        if include_lpips:
            out["lpips"] = lpips(pred, target, mask)
        return out

    results: dict[str, dict[str, float]] = {"whole": panel(None)}
    if cloud_mask is not None:
        results["cloud"] = panel(cloud_mask)
    return results


def compute_stratified(
    pred: Tensor,
    target: Tensor,
    masks: dict[str, Tensor | None],
    cfg: EvalConfig | None = None,
    data_range: float = 1.0,
    include_lpips: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute every scalar metric across a set of named region masks (strata).

    This is the BUILD_PLAN §3.5 stratified view used by the evaluator: pass a
    mapping such as ``{"whole": None, "cloud": cloud_mask, "shadow": shadow_mask,
    "thin": thin_mask, "thick": thick_mask}`` and get back
    ``{metric: {stratum: value}}``. Strata are filtered to ``cfg.strata`` when a
    config is supplied; otherwise all provided masks are used.

    Args:
        pred: Predicted image.
        target: Reference image.
        masks: Mapping ``stratum -> mask`` (``None`` = whole image).
        cfg: Optional :class:`EvalConfig` restricting the strata.
        data_range: Signal range for PSNR/SSIM.
        include_lpips: Whether to include LPIPS per stratum.

    Returns:
        ``{metric: {stratum: float}}`` — the transpose of :func:`compute_all`,
        with one entry per scalar metric.
    """
    wanted = list(cfg.strata) if cfg is not None else list(masks.keys())
    metric_names = [
        "psnr",
        "ssim",
        "ms_ssim",
        "sam",
        "ergas",
        "sid",
        "rmse",
        "mae",
        "ndvi_mae",
        "spectral_correlation",
    ]
    if include_lpips:
        metric_names.append("lpips")

    table: dict[str, dict[str, float]] = {name: {} for name in metric_names}
    for stratum in wanted:
        if stratum not in masks:
            continue
        mask = masks[stratum]
        panel = {
            "psnr": psnr(pred, target, mask, data_range=data_range),
            "ssim": ssim(pred, target, mask, data_range=data_range),
            "ms_ssim": ms_ssim(pred, target, mask, data_range=data_range),
            "sam": sam(pred, target, mask),
            "ergas": ergas(pred, target, mask),
            "sid": sid(pred, target, mask),
            "rmse": rmse(pred, target, mask),
            "mae": mae(pred, target, mask),
            "ndvi_mae": ndvi_mae(pred, target, mask),
            "spectral_correlation": spectral_correlation(pred, target, mask),
        }
        if include_lpips:
            panel["lpips"] = lpips(pred, target, mask)
        for name, value in panel.items():
            table[name][stratum] = value
    return table
