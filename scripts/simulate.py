"""``cloudremoval simulate`` -- generate synthetic cloudy/clear paired tiles.

Builds the configured (synthetic) dataset and writes paired ``cloudy`` / ``clear``
tiles plus cloud/shadow masks (and the SAR/DEM proxies) to disk, so downstream
training/eval has materialised paired data even with **no external data** on CPU
(BUILD_PLAN §4 B1 done-criterion: ``cloudremoval simulate`` writes paired tiles).

Invoked by ``cloudremoval.cli`` as ``main(cfg, **kwargs)``; also runnable directly
(``python scripts/simulate.py --config configs/cpu_smoke.yaml``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# --- sys.path bootstrap so `import cloudremoval` works without an install ----- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cloudremoval.config import Config


def main(cfg: Config, out_dir: str | Path | None = None, n: int | None = None, **_: Any) -> None:
    """Generate and write synthetic cloudy/clear paired tiles.

    Args:
        cfg: The loaded :class:`~cloudremoval.config.Config`.
        out_dir: Output directory (defaults to ``<eval.out_dir>/../simulated`` or
            ``outputs/simulated``).
        n: Optional cap on the number of tiles to write (defaults to the dataset
            length).
        **_: Ignored extra kwargs forwarded by the CLI.
    """
    import numpy as np

    from cloudremoval.data.datasets import build_dataset
    from cloudremoval.utils.io import save_npy
    from cloudremoval.utils.logging import get_logger
    from cloudremoval.utils.seed import seed_everything

    log = get_logger("scripts.simulate")
    seed_everything(int(getattr(cfg.train, "seed", 1337)))

    dataset = build_dataset(cfg, split="train")
    count = len(dataset) if n is None else min(int(n), len(dataset))

    base = Path(out_dir) if out_dir is not None else _default_out_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)
    log.info("simulating %d paired tiles into %s", count, base)

    manifest: list[dict[str, Any]] = []
    for i in range(count):
        sample = dataset[i]
        sid = str(sample["meta"].get("scene_id", f"tile_{i:05d}"))
        tile_dir = base / sid
        tile_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {"id": sid}
        for key in (
            "optical_cloudy",
            "optical_clear",
            "cloud_mask",
            "shadow_mask",
            "sar",
            "dem",
        ):
            arr = sample[key].detach().cpu().numpy().astype(np.float32)
            path = tile_dir / f"{key}.npy"
            save_npy(arr, path)
            record[key] = str(path)
        refs = sample["temporal_refs"]
        if refs.shape[0] > 0:
            path = tile_dir / "temporal_refs.npy"
            save_npy(refs.detach().cpu().numpy().astype(np.float32), path)
            record["temporal_refs"] = str(path)
        record["meta"] = _jsonable_meta(sample["meta"])
        manifest.append(record)

    _write_manifest(base / "manifest.json", manifest)
    log.info("wrote %d tiles + manifest.json to %s", count, base)


def _default_out_dir(cfg: Config) -> Path:
    """Choose a default simulated-tile directory near the eval outputs."""
    out = getattr(getattr(cfg, "eval", None), "out_dir", None)
    if out is not None:
        return Path(out).parent / "simulated"
    return Path("outputs/simulated")


def _jsonable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce a ``meta`` dict to JSON-serialisable primitives."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: _scalar(vv) for kk, vv in v.items()}
        elif isinstance(v, (list, tuple)):
            out[k] = [_scalar(x) for x in v]
        else:
            out[k] = str(v)
    return out


def _scalar(v: Any) -> Any:
    """Best-effort scalar/list coercion for JSON serialisation."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    return float(v) if hasattr(v, "__float__") else str(v)


def _write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    """Write the simulation manifest as JSON."""
    import json

    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    from cloudremoval.config import load_config

    parser = argparse.ArgumentParser(description="Generate synthetic cloudy/clear pairs.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--n", type=int, default=None, help="Number of tiles.")
    args = parser.parse_args()
    main(load_config(args.config), out_dir=args.out, n=args.n)
