"""Pytest fixtures and ``sys.path`` bootstrap for the ``cloudremoval`` test suite.

This file is owned by B0 and is the only test file B0 creates; B1-B5 add the
actual test modules. It provides:

* a ``sys.path`` insertion of ``src/`` so ``import cloudremoval`` works **before**
  ``pip install -e .`` (BUILD_PLAN: importable without install);
* :func:`tiny_config` — a :class:`Config` matching ``configs/cpu_smoke.yaml``
  (small tiles, ``max_steps`` capped, CPU);
* :func:`synthetic_sample` — a single ``SAMPLE`` dict (§2) with small tensors;
* :func:`synthetic_batch` — a batched ``SAMPLE`` dict (tensors stacked on dim 0,
  ``meta`` a ``list[dict]``) as produced by the default collate.

The synthetic builders here are a deterministic stand-in so B5's tests have data
even before B1's ``make_synthetic_sample`` lands; B1's dataset is the production
source. Shapes/keys are kept identical to the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# Make ``src/`` importable without an editable install (must run at import time).
# The repo root is also added so the top-level ``scripts`` package (which the CLI
# delegates to and the smoke test imports as ``scripts.<name>``) resolves even
# under pytest's ``--import-mode=importlib``, where the rootdir is *not* added to
# ``sys.path`` and ``scripts`` is not part of the installed ``cloudremoval`` wheel.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_sample(
    c: int = 3,
    h: int = 64,
    w: int = 64,
    t: int = 0,
    seed: int = 0,
    has_target: bool = True,
) -> dict[str, Any]:
    """Build one ``SAMPLE`` dict (§2) with deterministic small tensors.

    Image tensors are ``float32`` channel-first, roughly normalized. Optional
    modalities are zero-filled (never missing) per the contract, with presence
    flags in ``meta``.
    """
    import torch

    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, dtype=torch.float32)

    optical_clear = rand(c, h, w)
    cloud_mask = (rand(1, h, w) > 0.6).to(torch.float32)  # ~40% cloud
    shadow_mask = (rand(1, h, w) > 0.85).to(torch.float32)
    # Cloudy = clear brightened where cloud, plus light haze elsewhere.
    optical_cloudy = (optical_clear * (1 - cloud_mask) + cloud_mask * 0.9).clamp(0, 1)

    sample: dict[str, Any] = {
        "optical_cloudy": optical_cloudy,
        "optical_clear": optical_clear if has_target else torch.zeros(c, h, w),
        "cloud_mask": cloud_mask,
        "shadow_mask": shadow_mask,
        "sar": rand(2, h, w),
        "dem": rand(1, h, w),
        "temporal_refs": rand(t, c, h, w) if t > 0 else torch.zeros(0, c, h, w),
        "meta": {
            "scene_id": f"synthetic_{seed:04d}",
            "date": "2024-01-01",
            "sun_elevation": 45.0,
            "crs": "EPSG:32646",
            "transform": (5.8, 0.0, 0.0, 0.0, -5.8, 0.0),
            "band_stats": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
            "norm": "zscore",
            "cloud_type": "mixed",
            "has_target": has_target,
            "has_sar": True,
            "has_dem": True,
            "has_temporal": t > 0,
        },
    }
    return sample


@pytest.fixture
def tiny_config():
    """A small CPU :class:`Config` mirroring ``configs/cpu_smoke.yaml``."""
    from cloudremoval.config import Config

    return Config.model_validate(
        {
            "data": {
                "name": "synthetic",
                "tile_size": 64,
                "halo": 8,
                "batch_size": 2,
                "num_workers": 0,
                "synthetic_n": 4,
                "use_sar": True,
                "use_dem": True,
                "use_temporal": False,
            },
            "model": {"name": "unet", "base_channels": 8, "depth": 2},
            "train": {
                "epochs": 1,
                "max_steps": 2,
                "device": "cpu",
                "precision": "32",
                "seed": 1337,
            },
            "infer": {"tile_size": 64, "halo": 8, "device": "cpu"},
        }
    )


@pytest.fixture
def synthetic_sample() -> dict[str, Any]:
    """A single ``SAMPLE`` dict (§2) with 3x64x64 tensors and no temporal refs."""
    return _make_sample(c=3, h=64, w=64, t=0, seed=0)


@pytest.fixture
def synthetic_batch() -> dict[str, Any]:
    """A batched ``SAMPLE`` dict (B=2) as the default collate would produce.

    Tensors are stacked on dim 0 (``optical_cloudy:[B,C,H,W]``,
    ``temporal_refs:[B,T,C,H,W]`` with T=0) and ``meta`` is a ``list[dict]``.
    """
    import torch

    samples = [_make_sample(seed=i) for i in range(2)]
    keys = [k for k in samples[0] if k != "meta"]
    batch: dict[str, Any] = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    batch["meta"] = [s["meta"] for s in samples]
    return batch
