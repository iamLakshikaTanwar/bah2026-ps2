"""Runnable wrapper for tiled inference → COG (``cloudremoval infer``).

Invoked by ``cli.py`` as ``scripts.infer.main(cfg, ckpt=..., input=..., out=...)``
(BUILD_PLAN §3.10), and runnable directly as ``python scripts/infer.py --config
configs/cpu_smoke.yaml``. It builds the configured model (loading ``--ckpt`` if
given), reads (or synthesizes) a cloudy scene, runs sliding-window blended
inference, composites the reconstruction under the cloud mask, and writes a COG
(``.npy`` fallback when ``rasterio`` is absent) to ``cfg.infer.out_cog`` (or ``--out``).

A ``sys.path`` bootstrap makes ``cloudremoval`` importable without an editable
install and without a ``scripts`` package (B1 owns ``scripts/__init__.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# --- sys.path bootstrap (no scripts package; run before any cloudremoval import) -
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from cloudremoval.config import Config, load_config  # noqa: E402
from cloudremoval.inference.cog_writer import write_cog  # noqa: E402
from cloudremoval.inference.postprocess import clamp_valid, composite_under_mask  # noqa: E402
from cloudremoval.inference.tiled import tiled_inference  # noqa: E402
from cloudremoval.utils.logging import get_logger  # noqa: E402
from cloudremoval.utils.seed import seed_everything  # noqa: E402

_log = get_logger(__name__)


def _load_scene(cfg: Config, input_path: str | Path | None) -> dict[str, np.ndarray]:
    """Read a cloudy scene from ``input_path`` or synthesize a tiny one.

    Returns a SAMPLE-like dict of full-scene numpy arrays (at least
    ``optical_cloudy`` and ``cloud_mask``). The synthetic branch keeps the script
    runnable on the CPU-smoke path with no data or network.
    """
    if input_path is not None:
        from cloudremoval.utils.io import read_array

        arr = np.asarray(read_array(input_path), dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return {"optical_cloudy": arr}

    # Synthetic fallback: smooth base + a circular cloud blob.
    rng = np.random.default_rng(cfg.train.seed)
    c, size = cfg.model.in_channels, max(cfg.infer.tile_size * 2, 128)
    base = rng.random((c, size, size), dtype=np.float32) * 0.2 + 0.4
    yy, xx = np.mgrid[0:size, 0:size]
    cloud = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 4) ** 2).astype(np.float32)
    cloudy = np.clip(base * (1 - cloud) + cloud * 0.9, 0.0, 1.0).astype(np.float32)
    _log.info("No --input given; using a synthetic %dx%d scene.", size, size)
    return {"optical_cloudy": cloudy, "cloud_mask": cloud[None, ...]}


def main(cfg: Config, **kwargs: Any) -> None:
    """Run tiled inference and write a reconstructed COG.

    Args:
        cfg: The loaded :class:`Config` (uses ``model`` + ``infer``).
        **kwargs: ``ckpt`` (checkpoint path), ``input`` (scene path), ``out``
            (output COG path) — all optional, as forwarded by ``cli.py``.
    """
    from cloudremoval.models.registry import build_model

    seed_everything(cfg.train.seed)
    ckpt = kwargs.get("ckpt")
    input_path = kwargs.get("input")
    out_path = kwargs.get("out") or cfg.infer.out_cog

    model = build_model(cfg)
    if ckpt:
        import torch

        state = torch.load(ckpt, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=False)
        _log.info("Loaded checkpoint: %s", ckpt)
    model.eval()

    scene = _load_scene(cfg, input_path)
    recon = tiled_inference(
        model,
        scene,
        tile=cfg.infer.tile_size,
        overlap=cfg.infer.halo,
        device=cfg.infer.device,
        blend=cfg.infer.blend,
    )
    if "cloud_mask" in scene:
        recon = composite_under_mask(scene["optical_cloudy"], recon, scene["cloud_mask"])
    recon = clamp_valid(recon, 0.0, 1.0)

    written = write_cog(recon, out_path)
    _log.info("Wrote reconstruction: %s (shape=%s)", written, tuple(recon.shape))


def _cli() -> None:
    """Minimal standalone CLI (``python scripts/infer.py --config ... [--out ...]``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Tiled cloud-removal inference -> COG.")
    parser.add_argument("--config", "-c", required=True, type=Path)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, ckpt=args.ckpt, input=args.input, out=args.out)


if __name__ == "__main__":
    _cli()
