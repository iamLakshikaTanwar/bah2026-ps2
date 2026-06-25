"""STAC search / read returning COG hrefs (lazy ``pystac-client`` + ``planetary-computer``).

STAC + COG enables O(1)-per-tile range reads over petabyte archives without bulk
download (``research/05`` §C.1). This module searches a STAC API (Microsoft
Planetary Computer by default, or Element-84 Earth Search) and returns signed COG
asset hrefs that the windowed reader (:func:`cloudremoval.data.cog_tiling.windowed_read`)
can stream. ``pystac-client`` and ``planetary-computer`` are imported **lazily**;
a clear error is raised if they (and network access) are unavailable -- nothing
here runs on the CPU smoke.
"""

from __future__ import annotations

from typing import Any

from cloudremoval.utils.logging import get_logger

__all__ = [
    "PC_STAC_URL",
    "EARTH_SEARCH_URL",
    "COLLECTIONS",
    "search_stac",
    "sign_href",
    "read_cog_window",
]

_log = get_logger(__name__)

#: Microsoft Planetary Computer STAC endpoint (``research/04`` §C.5).
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
#: Element-84 Earth Search (AWS Open Data) STAC endpoint (``research/04`` §C.6).
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

#: Common Planetary Computer collection ids (``research/04`` §C.5).
COLLECTIONS = {
    "sentinel-2-l2a": "sentinel-2-l2a",
    "sentinel-1-grd": "sentinel-1-grd",
    "landsat-c2-l2": "landsat-c2-l2",
    "cop-dem-glo-30": "cop-dem-glo-30",
    "hls2-l30": "hls2-l30",
    "hls2-s30": "hls2-s30",
}


def _require_pystac():
    """Import ``pystac_client`` (lazy), raising a clear error if unavailable."""
    try:
        from pystac_client import Client  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pystac-client is not installed. Install `pystac-client` (and "
            "`planetary-computer` for signed Planetary Computer assets) to use "
            "cloudremoval.data.sources.stac."
        ) from exc
    return Client


def sign_href(href: str) -> str:
    """Sign a Planetary Computer asset href if ``planetary-computer`` is installed.

    Args:
        href: A possibly-unsigned COG asset href.

    Returns:
        The signed href (or the original href if signing is unavailable).
    """
    try:
        import planetary_computer as pc  # type: ignore
    except Exception:  # noqa: BLE001
        return href
    try:
        return pc.sign(href)
    except Exception as exc:  # noqa: BLE001
        _log.debug("planetary-computer signing failed (%s); returning raw href", exc)
        return href


def search_stac(
    collection: str,
    bbox: tuple[float, float, float, float] | None = None,
    datetime: str | None = None,
    query: dict[str, Any] | None = None,
    url: str = PC_STAC_URL,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search a STAC API and return matching item dicts with signed asset hrefs.

    Args:
        collection: STAC collection id (e.g. ``"sentinel-2-l2a"``,
            ``"cop-dem-glo-30"``).
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        datetime: STAC datetime / interval string (e.g. ``"2024-01/2024-03"``).
        query: Optional STAC query (e.g. ``{"eo:cloud_cover": {"lt": 20}}``).
        url: STAC API URL (Planetary Computer by default).
        limit: Maximum items to return.

    Returns:
        A list of ``{id, properties, assets}`` dicts (asset hrefs signed for PC).

    Raises:
        RuntimeError: If ``pystac-client`` (or network) is unavailable.
    """
    Client = _require_pystac()
    catalog = Client.open(url)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime,
        query=query,
        max_items=limit,
    )
    items: list[dict[str, Any]] = []
    for item in search.items():
        assets = {
            name: {"href": sign_href(asset.href), "type": asset.media_type}
            for name, asset in item.assets.items()
        }
        items.append({"id": item.id, "properties": dict(item.properties), "assets": assets})
    _log.info("STAC search '%s' returned %d items", collection, len(items))
    return items


def read_cog_window(
    href: str,
    row: int,
    col: int,
    height: int,
    width: int,
):
    """Stream a window from a COG href (delegates to the lazy windowed reader).

    Args:
        href: A (signed) COG asset href.
        row: Top row of the window.
        col: Left column.
        height: Window height.
        width: Window width.

    Returns:
        ``[C, height, width]`` array (``float32``).
    """
    from cloudremoval.data.cog_tiling import windowed_read

    return windowed_read(href, row, col, height, width)
