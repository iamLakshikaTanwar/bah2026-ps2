"""Tests for the tiled-inference + finishing layer (BUILD_PLAN §3.8).

Pure NumPy/torch, no geospatial deps:

* ``tiled_inference`` with an identity model reproduces the input (seam-free);
* reflect-padding handles scenes not divisible by the tile/stride grid;
* ``composite_under_mask`` keeps the clear region exact and pastes the
  reconstruction under the mask;
* the COG writer falls back to ``.npy`` + JSON sidecar when rasterio is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput


class _IdentityModel(BaseCloudRemovalModel):
    """A model that returns the cloudy input unchanged (for seam/round-trip tests)."""

    def __init__(self) -> None:
        import torch.nn as nn

        nn.Module.__init__(self)
        # One throwaway parameter so the module is well-formed.
        self._noop = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, sample: dict) -> ModelOutput:  # type: ignore[override]
        return ModelOutput(reconstruction=sample["optical_cloudy"])

    def loss(self, sample: dict, output: ModelOutput) -> dict:  # type: ignore[override]
        return {"total": torch.zeros(1)}


# --------------------------------------------------------------------------- #
# Tiled inference
# --------------------------------------------------------------------------- #
def test_tiled_inference_identity_is_seamfree() -> None:
    """An identity model tiled over a scene reproduces the input within tolerance."""
    from cloudremoval.inference.tiled import tiled_inference

    rng = np.random.default_rng(0)
    scene = rng.random((3, 100, 100)).astype(np.float32)
    out = tiled_inference(_IdentityModel(), scene, tile=64, overlap=16, batch_size=2, blend="hann")
    assert out.shape == scene.shape
    assert np.isfinite(out).all()
    # Feathered overlap-add of identical tiles == the input (no seams).
    assert float(np.abs(out - scene).max()) < 1e-4


def test_tiled_inference_reflect_pad_small_scene() -> None:
    """A scene smaller than the tile is reflect-padded and reconstructed to size."""
    from cloudremoval.inference.tiled import tiled_inference

    rng = np.random.default_rng(1)
    scene = rng.random((3, 40, 50)).astype(np.float32)  # < tile, non-square
    out = tiled_inference(_IdentityModel(), scene, tile=64, overlap=8, blend="hann")
    assert out.shape == scene.shape
    assert float(np.abs(out - scene).max()) < 1e-4


def test_tiled_inference_rejects_bad_overlap() -> None:
    """Overlap must lie in ``[0, tile)`` — anything else raises."""
    from cloudremoval.inference.tiled import tiled_inference

    scene = np.zeros((3, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        tiled_inference(_IdentityModel(), scene, tile=16, overlap=16)


def test_compute_tile_positions_covers_extent() -> None:
    """Tile start offsets always cover the final edge exactly."""
    from cloudremoval.inference.tiled import compute_tile_positions

    positions = compute_tile_positions(extent=100, tile=64, stride=48)
    assert positions[0] == 0
    assert positions[-1] == 100 - 64  # last tile ends exactly at the edge


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #
def test_composite_keeps_clear_region_exact() -> None:
    """Outside the mask the original is preserved verbatim (feather=0)."""
    from cloudremoval.inference.postprocess import composite_under_mask

    rng = np.random.default_rng(2)
    original = rng.random((3, 64, 64)).astype(np.float32)
    recon = rng.random((3, 64, 64)).astype(np.float32)
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 20:40, 20:40] = 1.0

    out = composite_under_mask(original, recon, mask, feather=0)
    clear = mask[0] < 0.5
    cloud = mask[0] > 0.5
    # Clear pixels are exactly the original; cloud pixels are exactly the recon.
    assert np.array_equal(out[:, clear], original[:, clear])
    assert np.array_equal(out[:, cloud], recon[:, cloud])


def test_composite_feather_blends_boundary() -> None:
    """With feathering, the masked interior still equals the reconstruction."""
    from cloudremoval.inference.postprocess import composite_under_mask

    rng = np.random.default_rng(3)
    original = rng.random((3, 64, 64)).astype(np.float32)
    recon = rng.random((3, 64, 64)).astype(np.float32)
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 16:48, 16:48] = 1.0

    out = composite_under_mask(original, recon, mask, feather=4)
    # Deep inside the mask (away from the feathered rim) it is the reconstruction.
    interior = out[:, 28:36, 28:36]
    assert np.allclose(interior, recon[:, 28:36, 28:36], atol=1e-4)
    assert out.shape == original.shape


def test_clamp_valid_bounds() -> None:
    """``clamp_valid`` clips out-of-range values to the valid reflectance band."""
    from cloudremoval.inference.postprocess import clamp_valid

    arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    out = clamp_valid(arr, 0.0, 1.0)
    assert float(out.min()) == 0.0 and float(out.max()) == 1.0


# --------------------------------------------------------------------------- #
# COG writer fallback
# --------------------------------------------------------------------------- #
def test_cog_writer_npy_fallback(tmp_path: Path) -> None:
    """Without rasterio, ``write_cog`` writes ``.npy`` + a JSON sidecar."""
    from cloudremoval.inference.cog_writer import write_cog

    rng = np.random.default_rng(4)
    arr = rng.random((3, 32, 32)).astype(np.float32)
    written = write_cog(arr, tmp_path / "recon.tif", crs="EPSG:32646")

    out = Path(written)
    assert out.suffix == ".npy"  # fell back (no rasterio on the smoke stack)
    assert out.exists()
    reloaded = np.load(out)
    assert reloaded.shape == arr.shape
    assert np.allclose(reloaded, arr)

    sidecar = tmp_path / "recon.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["count"] == 3 and meta["height"] == 32 and meta["width"] == 32
    assert meta["cog"] is False


def test_cog_writer_appends_uncertainty_band(tmp_path: Path) -> None:
    """An uncertainty band is concatenated as a trailing channel."""
    from cloudremoval.inference.cog_writer import write_cog

    rng = np.random.default_rng(5)
    arr = rng.random((3, 16, 16)).astype(np.float32)
    unc = rng.random((1, 16, 16)).astype(np.float32)
    written = write_cog(arr, tmp_path / "recon.tif", uncertainty=unc)
    reloaded = np.load(written)
    assert reloaded.shape[0] == 4  # 3 reflectance + 1 uncertainty
