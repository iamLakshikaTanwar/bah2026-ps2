"""End-to-end integration smoke: simulate -> train -> eval -> infer.

This is the contract test (BUILD_PLAN §5): on the minimal CPU stack the whole
pipeline must run on tiny synthetic data and produce finite metrics + a written
reconstruction. It exercises the *real* scripts (``scripts.simulate`` /
``scripts.train`` / ``scripts.eval`` / ``scripts.infer``) through their
``main(cfg, **kwargs)`` entry points — the same code paths the ``cloudremoval``
CLI drives — with all outputs redirected under ``tmp_path`` so no artifacts leak.

Kept fast (<~60s on CPU): 32x32 tiles, 4 synthetic samples, ``max_steps=2``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def _tiny_cfg(tmp_path: Path):
    """A minimal CPU :class:`Config` with every output redirected under ``tmp_path``."""
    from cloudremoval.config import Config

    return Config.model_validate(
        {
            "data": {
                "name": "synthetic",
                "tile_size": 32,
                "halo": 4,
                "batch_size": 2,
                "num_workers": 0,
                "synthetic_n": 4,
                "use_sar": True,
                "use_dem": True,
                "use_temporal": False,
                "cloud_mode": "mixed",
            },
            "model": {"name": "unet", "base_channels": 8, "depth": 2},
            "train": {
                "epochs": 1,
                "max_steps": 2,
                "device": "cpu",
                "precision": "32",
                "seed": 1337,
                "ckpt_dir": str(tmp_path / "ckpt"),
            },
            "eval": {"masked": True, "out_dir": str(tmp_path / "eval")},
            "infer": {
                "tile_size": 32,
                "halo": 4,
                "device": "cpu",
                "out_cog": str(tmp_path / "recon.tif"),
                "write_uncertainty": False,
            },
            "serve": {"cog_dir": str(tmp_path)},
        }
    )


def _script(name: str):
    """Import a ``scripts.<name>`` module (CLI delegates to these)."""
    return importlib.import_module(f"scripts.{name}")


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    """simulate -> train -> eval -> infer on a tiny synthetic config.

    Asserts: paired tiles + manifest are written, a checkpoint is produced,
    evaluation returns finite whole + cloud metrics, and inference writes a
    finite, correctly-shaped reconstruction.
    """
    cfg = _tiny_cfg(tmp_path)

    # ----- simulate: paired tiles + manifest ------------------------------- #
    sim_dir = tmp_path / "simulated"
    _script("simulate").main(cfg, out_dir=sim_dir, n=3)
    assert (sim_dir / "manifest.json").exists()
    tile_dirs = [p for p in sim_dir.iterdir() if p.is_dir()]
    assert len(tile_dirs) == 3
    # Each tile has cloudy/clear/masks materialised as .npy.
    for key in ("optical_cloudy", "optical_clear", "cloud_mask", "shadow_mask"):
        assert (tile_dirs[0] / f"{key}.npy").exists()

    # ----- train: a checkpoint is saved ------------------------------------ #
    summary = _script("train").main(cfg)
    assert {"history", "best_ckpt", "final_ckpt"} <= set(summary)
    best = Path(summary["best_ckpt"])
    assert best.exists()

    # ----- eval: finite whole + cloud metrics ------------------------------ #
    results = _script("eval").main(cfg, ckpt=str(best))
    assert results["model"] == "unet"
    metrics = results["metrics"]
    assert "whole" in metrics and "cloud" in metrics
    # PSNR/SSIM are computed for both regions and are finite numbers.
    for region in ("whole", "cloud"):
        for metric in ("psnr", "ssim", "sam"):
            value = metrics[region][metric]
            assert isinstance(value, float)
            assert np.isfinite(value), f"{region}/{metric} not finite: {value}"
    assert (tmp_path / "eval" / "report.md").exists()

    # ----- infer: a finite reconstruction is written ----------------------- #
    _script("infer").main(cfg, ckpt=str(best))
    # rasterio is absent on the smoke stack -> .npy fallback (+ JSON sidecar).
    out_npy = tmp_path / "recon.npy"
    assert out_npy.exists()
    recon = np.load(out_npy)
    assert recon.ndim == 3 and recon.shape[0] == 3
    assert np.isfinite(recon).all()
    assert (tmp_path / "recon.json").exists()


def test_cli_help_lists_all_subcommands() -> None:
    """The Typer app exposes the full pipeline (the §3.10 verbs)."""
    from typer.testing import CliRunner

    from cloudremoval.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in (
        "download",
        "preprocess",
        "simulate",
        "train",
        "eval",
        "benchmark",
        "infer",
        "serve",
    ):
        assert verb in result.output


def test_benchmark_covers_all_registered_models(tmp_path: Path) -> None:
    """``cloudremoval benchmark`` ranks every registered model (all six)."""
    from cloudremoval.data.datasets import build_dataloader
    from cloudremoval.evaluation.benchmark import run_benchmark_all
    from cloudremoval.models.registry import list_models

    cfg = _tiny_cfg(tmp_path)
    loader = build_dataloader(cfg.data, split="val")
    leaderboard = run_benchmark_all(loader, cfg, out_dir=tmp_path / "bench", max_batches=1)

    ranked_names = {row["model"] for row in leaderboard}
    assert ranked_names == set(list_models())
    assert len(ranked_names) == 6
    # The leaderboard JSON/CSV are persisted.
    assert (tmp_path / "bench" / "leaderboard.json").exists()
    assert (tmp_path / "bench" / "leaderboard.csv").exists()
