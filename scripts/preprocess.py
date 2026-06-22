"""``cloudremoval preprocess`` -- radiometric/geometric preprocessing + masking.

For real LISS-IV scenes this converts DN -> TOA reflectance, co-registers auxiliary
modalities, runs cloud/shadow masking, and tiles to patches with per-band
normalization (``research/05`` §B). On the synthetic CPU path there is nothing to
calibrate, so it demonstrates the pipeline on one built dataset sample (DN->TOA
sanity, cloud/shadow detection, tiling round-trip) and reports shapes -- keeping
the command runnable on the minimal stack.

Invoked by ``cloudremoval.cli`` as ``main(cfg, **kwargs)``; also runnable directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cloudremoval.config import Config


def main(cfg: Config, **_: Any) -> None:
    """Run (or demonstrate) the preprocessing pipeline for the configured data.

    For ``data.name == "synthetic"`` (the smoke path) this builds one sample and
    exercises masking + tiling, logging shapes. For real datasets it documents the
    per-scene calibration/co-registration/masking/tiling steps (which require the
    geospatial stack).

    Args:
        cfg: The loaded :class:`~cloudremoval.config.Config`.
        **_: Ignored extra kwargs forwarded by the CLI.
    """
    from cloudremoval.data.cog_tiling import tile_array, untile_array
    from cloudremoval.data.datasets import build_dataset
    from cloudremoval.data.masking import detect_clouds, detect_shadows
    from cloudremoval.data.preprocess import denormalize_bands, ndvi
    from cloudremoval.utils.logging import get_logger
    from cloudremoval.utils.seed import seed_everything

    log = get_logger("scripts.preprocess")
    seed_everything(int(getattr(cfg.train, "seed", 1337)))

    name = str(getattr(cfg.data, "name", "synthetic"))
    log.info("preprocess: dataset=%s tile_size=%s norm=%s", name, cfg.data.tile_size, cfg.data.norm)

    dataset = build_dataset(cfg, split="train")
    sample = dataset[0]
    meta = sample["meta"]

    # Recover reflectance from the normalized cloudy optical for masking.
    cloudy_n = sample["optical_cloudy"].numpy()
    refl = denormalize_bands(cloudy_n, meta["band_stats"], norm=meta.get("norm", "zscore"))

    cloud_mask = detect_clouds(refl)
    shadow_mask = detect_shadows(
        refl,
        cloud_mask,
        sun_az=float(meta.get("sun_azimuth", 135.0)),
        sun_elev=float(meta.get("sun_elevation", 45.0)),
    )
    veg = ndvi(refl)

    log.info(
        "sample %s: optical=%s cloud_mask=%s shadow_mask=%s ndvi[min=%.3f,max=%.3f]",
        meta.get("scene_id"),
        tuple(refl.shape),
        tuple(cloud_mask.shape),
        tuple(shadow_mask.shape),
        float(veg.min()),
        float(veg.max()),
    )

    # Tiling round-trip sanity (the sliding-window+halo machinery reused at infer).
    tiles, index = tile_array(refl, tile_size=max(8, cfg.data.tile_size // 2), halo=cfg.data.halo)
    recon = untile_array(tiles, index, blend="hann")
    log.info("tiling: %d tiles -> reconstruction %s", len(tiles), tuple(recon.shape))
    log.info("preprocess complete (synthetic demonstration path)")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    from cloudremoval.config import load_config

    parser = argparse.ArgumentParser(description="Preprocess / mask / tile imagery.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    args = parser.parse_args()
    main(load_config(args.config))
