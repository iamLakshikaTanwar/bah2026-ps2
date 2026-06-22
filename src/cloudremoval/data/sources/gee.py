"""Google Earth Engine helpers with verified collection IDs (lazy ``ee``).

GEE provides server-side, on-the-fly access to the Sentinel-2/Sentinel-1/Landsat/
DEM/cloud-mask collections used to build LISS-IV-aligned training tiles
(``research/04`` §C.4). The Earth Engine client (``ee``) is imported **lazily** and
every fetch raises a clear, actionable error if EE is not installed or the user is
not authenticated -- nothing here runs on the CPU smoke.

All collection IDs below are the exact, verified identifiers from
``research/04`` §C.4 / §A.5.
"""

from __future__ import annotations

from typing import Any

from cloudremoval.utils.logging import get_logger

__all__ = [
    "COLLECTIONS",
    "S2_SR",
    "S2_TOA",
    "S2_CLOUD_PROB",
    "CLOUD_SCORE_PLUS",
    "S1_GRD",
    "L8_L2",
    "L9_L2",
    "DEM_GLO30",
    "DEM_SRTM",
    "DEM_NASADEM",
    "S2_TO_LISSIV_BANDS",
    "initialize",
    "get_collection",
    "build_s2_lissiv_composite",
    "export_image",
]

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Verified GEE collection IDs (research/04 §C.4, §A.5, §A.6)
# --------------------------------------------------------------------------- #
S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
S2_TOA = "COPERNICUS/S2_HARMONIZED"
S2_CLOUD_PROB = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S1_GRD = "COPERNICUS/S1_GRD"
L8_L2 = "LANDSAT/LC08/C02/T1_L2"
L9_L2 = "LANDSAT/LC09/C02/T1_L2"
DEM_GLO30 = "COPERNICUS/DEM/GLO30"
DEM_SRTM = "USGS/SRTMGL1_003"
DEM_NASADEM = "NASA/NASADEM_HGT/001"

#: All verified collections keyed by a short alias.
COLLECTIONS: dict[str, str] = {
    "s2_sr": S2_SR,
    "s2_toa": S2_TOA,
    "s2_cloud_prob": S2_CLOUD_PROB,
    "cloud_score_plus": CLOUD_SCORE_PLUS,
    "s1_grd": S1_GRD,
    "landsat8_l2": L8_L2,
    "landsat9_l2": L9_L2,
    "dem_glo30": DEM_GLO30,
    "dem_srtm": DEM_SRTM,
    "dem_nasadem": DEM_NASADEM,
}

#: Sentinel-2 -> LISS-IV band mapping (``research/06`` §B.3): B3->Green, B4->Red,
#: B8->NIR (B8 is the closer match by bandwidth; apply SBAF before transfer).
S2_TO_LISSIV_BANDS: dict[str, str] = {"B3": "green", "B4": "red", "B8": "nir"}


# --------------------------------------------------------------------------- #
# Lazy EE access
# --------------------------------------------------------------------------- #
def _require_ee():
    """Import and return the ``ee`` module, raising a clear error if unavailable."""
    try:
        import ee  # type: ignore  (lazy, optional)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "earthengine-api is not installed. Install `earthengine-api` and run "
            "`earthengine authenticate` to use cloudremoval.data.sources.gee."
        ) from exc
    return ee


def initialize(project: str | None = None) -> None:
    """Initialise Earth Engine (must be authenticated first).

    Args:
        project: Optional Google Cloud project id for the EE session.

    Raises:
        RuntimeError: If ``ee`` is missing or initialisation/authentication fails.
    """
    ee = _require_ee()
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        _log.info("Earth Engine initialised (project=%s)", project or "<default>")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Earth Engine initialisation failed. Run `earthengine authenticate` "
            "and ensure your account/project is registered for EE."
        ) from exc


def get_collection(alias_or_id: str):
    """Return an ``ee.ImageCollection`` for a known alias or a raw collection id.

    Args:
        alias_or_id: A key of :data:`COLLECTIONS` (e.g. ``"s2_sr"``) or a full id.

    Returns:
        An ``ee.ImageCollection`` (requires EE initialised).
    """
    ee = _require_ee()
    coll_id = COLLECTIONS.get(alias_or_id, alias_or_id)
    return ee.ImageCollection(coll_id)


# --------------------------------------------------------------------------- #
# Composite + export (documented; require EE)
# --------------------------------------------------------------------------- #
def build_s2_lissiv_composite(
    aoi: Any,
    start: str,
    end: str,
    max_cloud_pct: float = 20.0,
    use_cloud_score_plus: bool = True,
):
    """Build a cloud-masked Sentinel-2 {B3,B4,B8} composite as a LISS-IV surrogate.

    Filters ``COPERNICUS/S2_SR_HARMONIZED`` by AOI/date/cloud%, masks clouds with
    Cloud Score+ (``GOOGLE/CLOUD_SCORE_PLUS``) or the s2cloudless probability, and
    median-composites the three LISS-IV-aligned bands (``research/06`` §B.3). Apply
    SBAF/histogram-matching downstream before training on LISS-IV.

    Args:
        aoi: An ``ee.Geometry`` AOI.
        start: ISO start date.
        end: ISO end date.
        max_cloud_pct: Scene-level cloud-cover filter.
        use_cloud_score_plus: Use Cloud Score+ (else s2cloudless probability).

    Returns:
        An ``ee.Image`` with bands ``[green, red, nir]`` over the AOI.
    """
    ee = _require_ee()
    coll = (
        get_collection("s2_sr")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )
    if use_cloud_score_plus:
        cs = get_collection("cloud_score_plus")
        coll = coll.linkCollection(cs, ["cs_cdf"])
        coll = coll.map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.6)))
    composite = coll.select(["B3", "B4", "B8"]).median()
    return composite.rename(["green", "red", "nir"]).clip(aoi)


def export_image(
    image: Any,
    description: str,
    region: Any,
    scale: float = 5.8,
    folder: str = "cloudremoval",
    crs: str = "EPSG:4326",
):
    """Start an Earth Engine export of ``image`` to Google Drive (documented).

    Exports at the LISS-IV working scale (5.8 m) over ``region``; the resulting
    GeoTIFF is then ingested by the local pipeline. Requires EE + an authenticated,
    Drive-enabled account.

    Args:
        image: An ``ee.Image`` to export.
        description: Task / output filename description.
        region: An ``ee.Geometry`` export region.
        scale: Output pixel size in metres (5.8 for LISS-IV).
        folder: Google Drive folder.
        crs: Output CRS.

    Returns:
        The started ``ee.batch.Task``.
    """
    ee = _require_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
    )
    task.start()
    _log.info("started EE export task '%s' (scale=%.1fm)", description, scale)
    return task
