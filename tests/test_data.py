"""Data-layer tests: SAMPLE schema, synthetic clouds, tiling, transforms, coreg.

All pure NumPy/torch — no geospatial dependency. Validates the data contracts the
trainer / evaluator / inference engine rely on (BUILD_PLAN §2, §3.4):

* ``build_dataset("synthetic")`` yields a schema-conformant ``SAMPLE`` dict;
* the synthetic-cloud generator changes pixels under the cloud mask and emits
  binary masks;
* ``tile_array`` / ``untile_array`` round-trip exactly with no overlap;
* a sample transform is applied identically to every spatial modality;
* ``coregister`` recovers a known translational shift.
"""

from __future__ import annotations

import numpy as np
import torch

# The full set of keys the SAMPLE contract guarantees (BUILD_PLAN §2).
_SAMPLE_KEYS = {
    "optical_cloudy",
    "optical_clear",
    "cloud_mask",
    "shadow_mask",
    "sar",
    "dem",
    "temporal_refs",
    "meta",
}


def _data_cfg(**over):
    from cloudremoval.config import DataConfig

    base = {
        "name": "synthetic",
        "tile_size": 32,
        "halo": 4,
        "batch_size": 2,
        "num_workers": 0,
        "synthetic_n": 4,
        "use_sar": True,
        "use_dem": True,
        "use_temporal": False,
    }
    base.update(over)
    return DataConfig.model_validate(base)


# --------------------------------------------------------------------------- #
# SAMPLE schema conformance
# --------------------------------------------------------------------------- #
def test_build_dataset_sample_schema() -> None:
    """``build_dataset(synthetic)[0]`` conforms to the SAMPLE contract (§2)."""
    from cloudremoval.data.datasets import build_dataset

    ds = build_dataset(_data_cfg(), split="train")
    assert len(ds) == 4
    sample = ds[0]

    assert set(sample.keys()) == _SAMPLE_KEYS
    size = 32
    # Channel-first, float32, correct shapes.
    assert sample["optical_cloudy"].shape == (3, size, size)
    assert sample["optical_clear"].shape == (3, size, size)
    assert sample["cloud_mask"].shape == (1, size, size)
    assert sample["shadow_mask"].shape == (1, size, size)
    assert sample["sar"].shape == (2, size, size)
    assert sample["dem"].shape == (1, size, size)
    assert sample["temporal_refs"].shape == (0, 3, size, size)  # use_temporal=False
    for key in ("optical_cloudy", "optical_clear", "sar", "dem"):
        assert sample[key].dtype == torch.float32
        assert torch.isfinite(sample[key]).all()

    meta = sample["meta"]
    for field in ("scene_id", "norm", "band_stats", "has_target", "has_sar", "has_dem"):
        assert field in meta
    assert meta["has_target"] is True


def test_dataset_is_deterministic() -> None:
    """Indexing the same item twice returns identical tensors (seeded by idx)."""
    from cloudremoval.data.datasets import build_dataset

    ds = build_dataset(_data_cfg(), split="train")
    a = ds[1]["optical_cloudy"]
    b = ds[1]["optical_cloudy"]
    assert torch.equal(a, b)


def test_collate_pads_temporal_and_lists_meta() -> None:
    """``collate_sample`` stacks tensors on dim 0 and makes ``meta`` a list."""
    from cloudremoval.data.datasets import build_dataset, collate_sample

    ds = build_dataset(_data_cfg(), split="train")
    batch = collate_sample([ds[0], ds[1]])
    assert batch["optical_cloudy"].shape == (2, 3, 32, 32)
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2
    # Variable-T temporal refs become a padded [B, T, C, H, W] block + a mask.
    assert batch["temporal_refs"].shape[0] == 2
    assert "temporal_mask" in batch


# --------------------------------------------------------------------------- #
# Synthetic clouds
# --------------------------------------------------------------------------- #
def test_synthetic_clouds_change_pixels_under_mask() -> None:
    """The cloudy image differs from clear under the cloud mask; masks are binary."""
    from cloudremoval.data.synthetic_clouds import add_synthetic_clouds

    rng = np.random.default_rng(0)
    clear = rng.random((3, 64, 64)).astype(np.float32)
    cloudy, cloud_mask, shadow_mask = add_synthetic_clouds(
        clear, cloud_mode="thick", coverage=0.5, seed=7
    )

    assert cloudy.shape == clear.shape
    assert cloud_mask.shape == (1, 64, 64)
    assert shadow_mask.shape == (1, 64, 64)
    # Masks are strictly binary.
    assert set(np.unique(cloud_mask)).issubset({0.0, 1.0})
    assert set(np.unique(shadow_mask)).issubset({0.0, 1.0})
    # Some pixels are clouded, and there the cloudy image differs from clear.
    sel = np.broadcast_to(cloud_mask, clear.shape) > 0.5
    assert sel.any(), "expected a non-empty cloud region"
    diff = np.abs(cloudy - clear)[sel]
    assert float(diff.mean()) > 1e-3
    # Output stays in valid reflectance range.
    assert float(cloudy.min()) >= 0.0 and float(cloudy.max()) <= 1.0


def test_synthetic_clouds_torch_in_torch_out() -> None:
    """A torch tensor in yields torch tensors out (dtype mirroring)."""
    from cloudremoval.data.synthetic_clouds import add_synthetic_clouds

    clear = torch.rand(3, 48, 48)
    cloudy, cmask, smask = add_synthetic_clouds(clear, seed=1)
    assert torch.is_tensor(cloudy) and torch.is_tensor(cmask) and torch.is_tensor(smask)


# --------------------------------------------------------------------------- #
# Tiling round-trip
# --------------------------------------------------------------------------- #
def test_tile_untile_exact_roundtrip_no_overlap() -> None:
    """``tile_array`` then ``untile_array`` reconstructs exactly with halo=0."""
    from cloudremoval.data.cog_tiling import tile_array, untile_array

    rng = np.random.default_rng(3)
    img = rng.random((3, 100, 100)).astype(np.float32)
    tiles, index = tile_array(img, tile_size=50, halo=0)
    assert len(tiles) == 4  # 2x2 non-overlapping
    recon = untile_array(tiles, index, blend="none")
    assert recon.shape == img.shape
    assert np.allclose(recon, img, atol=1e-5)


def test_tile_untile_overlap_is_seamfree() -> None:
    """With a halo + Hann blend the reconstruction matches the source closely."""
    from cloudremoval.data.cog_tiling import tile_array, untile_array

    rng = np.random.default_rng(4)
    img = rng.random((3, 96, 96)).astype(np.float32)
    tiles, index = tile_array(img, tile_size=32, halo=8)
    recon = untile_array(tiles, index, blend="hann")
    assert recon.shape == img.shape
    # Overlap-add of identical content reproduces the input up to blend rounding.
    assert float(np.abs(recon - img).max()) < 1e-4


# --------------------------------------------------------------------------- #
# Transforms applied consistently
# --------------------------------------------------------------------------- #
def test_transform_applied_consistently_across_modalities() -> None:
    """A geometric transform hits every spatial modality identically.

    A horizontal flip of ``optical_cloudy`` must equal the flip we expect, and the
    same flip must have been applied to ``cloud_mask`` (pixel correspondence is
    preserved across modalities).
    """
    from cloudremoval.data.transforms import RandomFlip

    rng = np.random.default_rng(5)
    sample = {
        "optical_cloudy": torch.from_numpy(rng.random((3, 16, 16)).astype(np.float32)),
        "optical_clear": torch.from_numpy(rng.random((3, 16, 16)).astype(np.float32)),
        "cloud_mask": torch.from_numpy((rng.random((1, 16, 16)) > 0.5).astype(np.float32)),
        "shadow_mask": torch.zeros(1, 16, 16),
        "sar": torch.from_numpy(rng.random((2, 16, 16)).astype(np.float32)),
        "dem": torch.from_numpy(rng.random((1, 16, 16)).astype(np.float32)),
        "temporal_refs": torch.zeros(0, 3, 16, 16),
        "meta": {"norm": "zscore"},
    }
    original = {k: v.clone() for k, v in sample.items() if torch.is_tensor(v)}

    # Force a horizontal-only flip deterministically.
    flip = RandomFlip(p_horizontal=1.0, p_vertical=0.0)
    out = flip(sample)

    expected_cloudy = torch.flip(original["optical_cloudy"], dims=[-1])
    assert torch.equal(out["optical_cloudy"], expected_cloudy)
    # Same flip applied to every other spatial modality.
    for key in ("optical_clear", "cloud_mask", "sar", "dem"):
        assert torch.equal(out[key], torch.flip(original[key], dims=[-1]))


def test_eval_transform_is_identity() -> None:
    """The non-train split returns an identity transform (deterministic eval)."""
    from cloudremoval.data.transforms import Identity, build_transforms

    tr = build_transforms(_data_cfg(), split="val")
    assert isinstance(tr, Identity)
    sample = {"optical_cloudy": torch.rand(3, 8, 8), "meta": {}}
    assert tr(sample) is sample


# --------------------------------------------------------------------------- #
# Co-registration
# --------------------------------------------------------------------------- #
def test_coregister_recovers_known_shift() -> None:
    """``coregister`` estimates the (inverse) shift that re-aligns a shifted band.

    We shift a smooth field by a known ``(dy, dx)`` to make the target, then ask
    ``coregister`` to align it back to the reference. The estimate must be the
    inverse of the applied shift and the aligned residual must shrink.
    """
    from cloudremoval.data.coregister import coregister, shift_array
    from cloudremoval.data.synthetic_clouds import fractal_noise_2d

    rng = np.random.default_rng(11)
    reference = fractal_noise_2d((96, 96), rng, octaves=4, base_cells=6).astype(np.float32)
    applied_dy, applied_dx = 3, -2
    target = shift_array(reference, applied_dy, applied_dx)

    aligned, (dy, dx) = coregister(reference, target, use_arosics=False, upsample=4)

    # The estimated shift inverts the applied one (integer-pixel here).
    assert abs(dy - (-applied_dy)) <= 1.0
    assert abs(dx - (-applied_dx)) <= 1.0
    # Aligning reduces the interior residual substantially.
    interior = (slice(8, -8), slice(8, -8))
    before = float(np.abs(target[interior] - reference[interior]).mean())
    after = float(np.abs(aligned[interior] - reference[interior]).mean())
    assert after < before * 0.25


def test_phase_correlation_zero_shift_on_identical() -> None:
    """Identical inputs yield ~zero estimated shift."""
    from cloudremoval.data.coregister import phase_correlation_shift
    from cloudremoval.data.synthetic_clouds import fractal_noise_2d

    rng = np.random.default_rng(12)
    band = fractal_noise_2d((64, 64), rng, octaves=3, base_cells=4).astype(np.float32)
    dy, dx = phase_correlation_shift(band, band)
    assert abs(dy) < 1.0 and abs(dx) < 1.0
