"""Tests for the pure-torch MVES metrics (BUILD_PLAN §3.5).

Validates the metric panel's identities and mask-aware behaviour on the minimal
stack: identical images give high PSNR / ~0 SAM / ~1 SSIM / 0 RMSE/MAE; NDVI MAE
is sane; the masked path scores only selected pixels; an empty mask degrades to
``nan`` for ratio metrics; ``compute_all`` returns both whole and cloud columns;
and LPIPS degrades gracefully (``nan``) when the optional package is absent.
"""

from __future__ import annotations

import math

import pytest
import torch

from cloudremoval.evaluation import metrics as M


def _img(seed: int = 0, c: int = 3, h: int = 32, w: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, w, generator=g, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Identity behaviour
# --------------------------------------------------------------------------- #
def test_identity_metrics() -> None:
    """psnr high, sam ~0, ssim ~1, rmse/mae = 0 for identical images."""
    x = _img(0)
    assert M.psnr(x, x) >= 99.0  # finite stand-in for "infinite"
    # SAM is eps-guarded (``acos`` clamps to ``1 - eps``), so identical images
    # give a tiny-but-nonzero angle in degrees rather than exactly 0.
    assert M.sam(x, x) == pytest.approx(0.0, abs=0.05)
    assert M.ssim(x, x) == pytest.approx(1.0, abs=1e-3)
    assert M.rmse(x, x) == pytest.approx(0.0, abs=1e-6)
    assert M.mae(x, x) == pytest.approx(0.0, abs=1e-6)
    assert M.ndvi_mae(x, x) == pytest.approx(0.0, abs=1e-6)


def test_errors_increase_with_noise() -> None:
    """Adding noise lowers PSNR/SSIM and raises RMSE versus the identical case."""
    x = _img(1)
    y = (x + 0.2 * _img(2)).clamp(0, 1)
    assert M.psnr(x, y) < M.psnr(x, x)
    assert M.ssim(x, y) < M.ssim(x, x)
    assert M.rmse(x, y) > 0.0


def test_ndvi_mae_sane_range() -> None:
    """NDVI MAE is a finite non-negative scalar within the plausible [0, 2] range."""
    x = _img(3)
    y = (x + 0.1 * _img(4)).clamp(0, 1)
    val = M.ndvi_mae(x, y)
    assert isinstance(val, float)
    assert 0.0 <= val <= 2.0


# --------------------------------------------------------------------------- #
# Mask-aware paths
# --------------------------------------------------------------------------- #
def test_mask_aware_scores_only_selected_pixels() -> None:
    """A metric with a mask reflects only the masked region's error.

    Construct an image that is exact inside the mask and wrong outside: the masked
    RMSE should be ~0 while the whole-image RMSE is clearly positive.
    """
    x = _img(5)
    y = x.clone()
    mask = torch.zeros(1, x.shape[-2], x.shape[-1])
    mask[:, 8:16, 8:16] = 1.0
    # Corrupt only the *unmasked* region.
    outside = mask[0] < 0.5
    y[:, outside] = (y[:, outside] + 0.5).clamp(0, 1)

    assert M.rmse(x, y, mask) == pytest.approx(0.0, abs=1e-6)
    assert M.rmse(x, y) > 0.05  # whole-image sees the corruption


def test_empty_mask_returns_nan_for_ratio_metrics() -> None:
    """An all-zero mask yields ``nan`` (no pixels to score) for ratio metrics."""
    x = _img(6)
    y = _img(7)
    empty = torch.zeros(1, x.shape[-2], x.shape[-1])
    assert math.isnan(M.rmse(x, y, empty))
    assert math.isnan(M.mae(x, y, empty))
    assert math.isnan(M.sam(x, y, empty))
    assert math.isnan(M.psnr(x, y, empty))
    assert math.isnan(M.ergas(x, y, empty))


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
def test_compute_all_returns_whole_and_cloud() -> None:
    """``compute_all`` produces a whole panel always and a cloud panel with a mask."""
    x = _img(8)
    y = (x + 0.1 * _img(9)).clamp(0, 1)
    mask = (_img(10, c=1) > 0.5).float()

    result = M.compute_all(x, y, cloud_mask=mask)
    assert "whole" in result and "cloud" in result
    for region in ("whole", "cloud"):
        panel = result[region]
        for metric in ("psnr", "ssim", "sam", "ergas", "ndvi_mae", "rmse", "mae"):
            assert metric in panel
        # Per-band extras are list-valued.
        assert isinstance(panel["per_band_correlation"], list)
        assert isinstance(panel["per_band_bias"], list)


def test_compute_all_no_mask_has_no_cloud_column() -> None:
    """Without a mask, only the whole-image column is returned."""
    x = _img(11)
    y = _img(12)
    result = M.compute_all(x, y)
    assert set(result.keys()) == {"whole"}


def test_compute_stratified_transposes_panel() -> None:
    """``compute_stratified`` returns ``{metric: {stratum: value}}``."""
    x = _img(13)
    y = (x + 0.1 * _img(14)).clamp(0, 1)
    masks = {
        "whole": None,
        "cloud": (_img(15, c=1) > 0.5).float(),
    }
    table = M.compute_stratified(x, y, masks)
    assert "psnr" in table
    assert set(table["psnr"].keys()) == {"whole", "cloud"}


# --------------------------------------------------------------------------- #
# LPIPS graceful degradation
# --------------------------------------------------------------------------- #
def test_lpips_graceful_when_absent() -> None:
    """LPIPS returns ``nan`` (never raises) when the optional package is missing."""
    import importlib.util

    x = _img(16)
    y = _img(17)
    value = M.lpips(x, y)
    if importlib.util.find_spec("lpips") is None:
        assert math.isnan(value)
    else:  # pragma: no cover - lpips not on the minimal stack
        assert isinstance(value, float)


def test_batched_and_unbatched_inputs_agree() -> None:
    """A ``[C,H,W]`` input is treated as a single-item batch (shape coercion)."""
    x = _img(18)
    xb = x.unsqueeze(0)
    assert M.psnr(x, x) == pytest.approx(M.psnr(xb, xb), abs=1e-4)
    assert M.sam(x, x) == pytest.approx(M.sam(xb, xb), abs=1e-4)
