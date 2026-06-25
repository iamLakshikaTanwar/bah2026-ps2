"""``cloudremoval download`` -- fetch source imagery for an AOI (documented).

Dispatches to the requested data source (``bhoonidhi`` | ``stac`` | ``gee`` |
``sentinel`` | ``dem``). All real fetches require optional heavy deps + network +
credentials and are imported **lazily**; with no source selected (or on the smoke
path) the command prints the documented Bhoonidhi LISS-IV ordering recipe rather
than failing, so it is always runnable.

Invoked by ``cloudremoval.cli`` as ``main(cfg, aoi=..., source=...)``; also runnable
directly (``python scripts/download.py --config ... --source bhoonidhi``).
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


def main(
    cfg: Config,
    aoi: str | Path | None = None,
    source: str | None = None,
    date_range: tuple[str, str] | None = None,
    **_: Any,
) -> None:
    """Fetch (or document fetching) source imagery for an AOI.

    Args:
        cfg: The loaded :class:`~cloudremoval.config.Config`.
        aoi: AOI as a KML/GeoJSON path or a ``"min_lon,min_lat,max_lon,max_lat"``
            bbox string.
        source: ``bhoonidhi`` | ``stac`` | ``gee`` | ``sentinel`` | ``dem``.
            Defaults to ``bhoonidhi`` (the LISS-IV source).
        date_range: Optional ``(start, end)`` ISO dates for catalogue search.
        **_: Ignored extra kwargs forwarded by the CLI.
    """
    from cloudremoval.utils.logging import get_logger

    log = get_logger("scripts.download")
    src = (source or "bhoonidhi").lower()
    bbox = _parse_bbox(aoi)
    log.info("download: source=%s aoi=%s bbox=%s", src, aoi, bbox)

    if src == "bhoonidhi":
        _download_bhoonidhi(log, aoi, date_range)
    elif src in {"stac", "sentinel", "dem"}:
        _download_via_stac(log, src, bbox, date_range)
    elif src == "gee":
        _download_via_gee(log, bbox, date_range)
    else:
        log.error("unknown source '%s' (bhoonidhi|stac|gee|sentinel|dem)", src)


def _download_bhoonidhi(log, aoi: str | Path | None, date_range) -> None:
    """Print the documented Bhoonidhi LISS-IV ordering workflow (no live API)."""
    from cloudremoval.data.sources.bhoonidhi import order_workflow

    recipe = order_workflow(aoi=str(aoi) if aoi else None, date_range=date_range)
    log.info("Bhoonidhi LISS-IV ordering recipe:")
    log.info("  portal: %s", recipe["portal"])
    log.info("  open-data: %s", recipe["open_data"])
    for step in recipe["steps"]:
        log.info("  - %s", step)
    log.info("  delivery: %s", recipe["delivery"])


def _download_via_stac(log, src: str, bbox, date_range) -> None:
    """Search a STAC source (Sentinel-1/2 or DEM); requires the STAC stack."""
    if bbox is None:
        log.error("--aoi bbox is required for source=%s", src)
        return
    dt = f"{date_range[0]}/{date_range[1]}" if date_range else None
    try:
        if src == "sentinel":
            from cloudremoval.data.sources.sentinel import fetch_sentinel2

            items = fetch_sentinel2(bbox, dt or "2024-01/2024-12")
        elif src == "dem":
            from cloudremoval.data.sources.dem import fetch_dem

            href = fetch_dem(bbox)
            log.info("DEM asset: %s", href)
            return
        else:  # generic stac -> sentinel-2-l2a
            from cloudremoval.data.sources.stac import search_stac

            items = search_stac("sentinel-2-l2a", bbox=bbox, datetime=dt)
        log.info("STAC returned %d items for source=%s", len(items), src)
    except RuntimeError as exc:
        log.error("STAC backend unavailable: %s", exc)


def _download_via_gee(log, bbox, date_range) -> None:
    """Build a Sentinel-2 LISS-IV composite via GEE; requires authenticated ``ee``."""
    if bbox is None:
        log.error("--aoi bbox is required for source=gee")
        return
    try:
        from cloudremoval.data.sources.gee import build_s2_lissiv_composite, initialize

        initialize()
        import ee  # type: ignore

        geom = ee.Geometry.Rectangle(list(bbox))
        start, end = date_range or ("2024-01-01", "2024-12-31")
        _ = build_s2_lissiv_composite(geom, start, end)
        log.info("built S2->LISS-IV composite over bbox=%s (%s..%s)", bbox, start, end)
    except RuntimeError as exc:
        log.error("GEE unavailable: %s", exc)


def _parse_bbox(aoi: str | Path | None) -> tuple[float, float, float, float] | None:
    """Parse a ``min_lon,min_lat,max_lon,max_lat`` bbox string, else ``None``."""
    if aoi is None:
        return None
    text = str(aoi)
    parts = text.replace(" ", "").split(",")
    if len(parts) == 4:
        try:
            return tuple(float(p) for p in parts)  # type: ignore[return-value]
        except ValueError:
            return None
    return None  # a KML/GeoJSON path; geometry parsing left to the source backend


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    from cloudremoval.config import load_config

    parser = argparse.ArgumentParser(description="Download / document source imagery.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    parser.add_argument("--aoi", default=None, help="AOI KML/GeoJSON path or bbox.")
    parser.add_argument("--source", default=None, help="bhoonidhi|stac|gee|sentinel|dem.")
    args = parser.parse_args()
    main(load_config(args.config), aoi=args.aoi, source=args.source)
