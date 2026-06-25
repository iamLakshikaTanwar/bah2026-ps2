"""Bhoonidhi / NRSC client for LISS-IV scenes (documented stub + local reader).

`Bhoonidhi <https://bhoonidhi.nrsc.gov.in>`_ is ISRO/NRSC's EO data hub and the
canonical source for LISS-IV (``research/06`` §C). Under the Indian Space Policy
2023, 5 m Resourcesat-2/2A LISS-IV is open data (free after registration). There
is no public bulk REST API; ordering is an interactive portal workflow, so this
module is **mostly a documented ordering stub plus a local-directory reader** for
already-downloaded scenes -- no live API call is made on the smoke path.

Delivery format (``research/06`` §A.4): a ZIP per scene with one GeoTIFF per band
(``BAND2/3/4.tif``) plus a ``BAND_META.txt`` (and/or XML). This module provides:

* :class:`BandMetaParser` -- parse ``BAND_META.txt`` for sun elevation, date,
  projection, scene id and (when present) per-band Lmax.
* :data:`LISSIV_LMAX` / :data:`LISSIV_ESUN` -- DN -> reflectance coefficients
  (defaults; production reads them per scene from metadata).
* :func:`read_lissiv_scene` -- local reader: per-band GeoTIFFs (or ``.npy``) ->
  ``[3, H, W]`` TOA reflectance + parsed meta.
* :func:`order_workflow` -- the documented search/EULA/order/download recipe.

GeoTIFF reads are via lazy ``rasterio`` (``utils.io``); a ``.npy`` fallback keeps
the reader importable and testable on the minimal stack.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from cloudremoval.data.preprocess import (
    LISSIV_ESUN,
    LISSIV_LMAX,
    dn_to_toa_reflectance,
)
from cloudremoval.utils.logging import get_logger

__all__ = [
    "BHOONIDHI_PORTAL",
    "LISSIV_LMAX",
    "LISSIV_ESUN",
    "BandMetaParser",
    "read_lissiv_scene",
    "order_workflow",
]

_log = get_logger(__name__)
_EPS = 1e-6

#: NRSC Bhoonidhi portal URL (``research/06`` §C.1).
BHOONIDHI_PORTAL = "https://bhoonidhi.nrsc.gov.in"

#: Default LISS-IV band filenames within a Bhoonidhi scene (Green, Red, NIR).
_DEFAULT_BAND_FILES = ("BAND2.tif", "BAND3.tif", "BAND4.tif")


# --------------------------------------------------------------------------- #
# BAND_META.txt parser
# --------------------------------------------------------------------------- #
class BandMetaParser:
    """Parse a Bhoonidhi ``BAND_META.txt`` into a normalised metadata dict.

    The metadata is a flat ``KEY = VALUE`` (or ``KEY: VALUE``) text file
    (``research/06`` §A.4). Fields of interest: ``OTSProductID``, ``DateOfPass``,
    ``SunElevationAtCenter`` (and sun azimuth), corner coordinates, satellite /
    sensor, path/row, projection/datum, and -- when present -- per-band saturation
    radiance (``Lmax``/``Bx_Lmax``).

    Example::

        meta = BandMetaParser.from_file("scene/BAND_META.txt").as_dict()
    """

    #: Recognised aliases (lower-cased, stripped) -> canonical key.
    _ALIASES = {
        "otsproductid": "scene_id",
        "productid": "scene_id",
        "dateofpass": "date",
        "dateofdump": "date",
        "sunelevationatcenter": "sun_elevation",
        "sunelevation": "sun_elevation",
        "sunazimuthatcenter": "sun_azimuth",
        "sunazimuth": "sun_azimuth",
        "satellite": "satellite",
        "sensor": "sensor",
        "path": "path",
        "row": "row",
        "datumname": "datum",
        "mapprojection": "projection",
    }

    def __init__(self, raw: dict[str, str]) -> None:
        """Args: ``raw`` mapping of original metadata keys to string values."""
        self.raw = raw

    @classmethod
    def from_file(cls, path: str | Path) -> BandMetaParser:
        """Parse a ``BAND_META.txt`` file.

        Args:
            path: Path to the metadata text file.

        Returns:
            A :class:`BandMetaParser`. A missing file yields an empty parser
            (the local reader then uses documented defaults).
        """
        p = Path(path)
        raw: dict[str, str] = {}
        if not p.exists():
            _log.debug("BAND_META.txt not found at %s; using defaults", p)
            return cls(raw)
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([^=:]+)[=:]\s*(.*)$", line)
            if m:
                raw[m.group(1).strip()] = m.group(2).strip()
        return cls(raw)

    def _lookup(self, canonical: str) -> str | None:
        for key, value in self.raw.items():
            if self._ALIASES.get(key.strip().lower()) == canonical:
                return value
        return None

    def band_lmax(self) -> list[float] | None:
        """Return per-band ``Lmax`` (Green, Red, NIR) if present in metadata.

        Looks for keys like ``B2_Lmax`` / ``Band2_Lmax`` / ``LMAX_BAND2``. Returns
        ``None`` if not all three are found (caller uses :data:`LISSIV_LMAX`).
        """
        out: list[float] = []
        for band in (2, 3, 4):
            val = None
            for key, value in self.raw.items():
                k = key.strip().lower().replace(" ", "")
                if f"b{band}" in k and "lmax" in k or f"band{band}" in k and "lmax" in k:
                    val = value
                    break
            if val is None:
                return None
            try:
                out.append(float(re.findall(r"[-+]?\d*\.?\d+", val)[0]))
            except (IndexError, ValueError):
                return None
        return out

    def as_dict(self) -> dict[str, Any]:
        """Return the normalised metadata dict (canonical keys, typed values)."""
        out: dict[str, Any] = {}
        sid = self._lookup("scene_id")
        if sid:
            out["scene_id"] = sid
        date = self._lookup("date")
        if date:
            out["date"] = date
        for key in ("sun_elevation", "sun_azimuth"):
            val = self._lookup(key)
            if val is not None:
                with contextlib.suppress(IndexError, ValueError):
                    out[key] = float(re.findall(r"[-+]?\d*\.?\d+", val)[0])
        for key in ("satellite", "sensor", "projection", "datum"):
            val = self._lookup(key)
            if val:
                out[key] = val
        proj = out.get("projection")
        datum = out.get("datum")
        if proj or datum:
            out["crs"] = f"{proj or ''}/{datum or ''}".strip("/")
        lmax = self.band_lmax()
        if lmax:
            out["lmax"] = lmax
        return out


# --------------------------------------------------------------------------- #
# Local reader
# --------------------------------------------------------------------------- #
def read_lissiv_scene(
    scene_dir: str | Path,
    band_files: tuple[str, str, str] | None = None,
    to_reflectance: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a local LISS-IV scene into ``([3, H, W] reflectance, meta)``.

    Reads the three per-band rasters (Green/Red/NIR), parses ``BAND_META.txt``, and
    -- if ``to_reflectance`` -- converts 10-bit DN to TOA reflectance using the
    metadata sun-elevation and per-band ``Lmax`` (falling back to documented
    defaults, ``research/06`` §A.5). Band rasters are read via lazy ``rasterio``
    with a ``.npy`` fallback, so the function is importable on the minimal stack.

    Args:
        scene_dir: Directory containing ``BAND2/3/4.tif`` and ``BAND_META.txt``.
        band_files: Optional explicit ``(green, red, nir)`` filenames.
        to_reflectance: Convert DN -> TOA reflectance (else return scaled DN/1023).

    Returns:
        ``(image[3, H, W] float32, meta)`` -- ``meta`` has ``scene_id``,
        ``sun_elevation``, ``crs``, etc.

    Raises:
        FileNotFoundError: If none of the expected band rasters can be located.
    """
    from cloudremoval.utils.io import read_array

    sdir = Path(scene_dir)
    meta = BandMetaParser.from_file(sdir / "BAND_META.txt").as_dict()
    files = band_files or _resolve_band_files(sdir)

    bands: list[np.ndarray] = []
    for fname in files:
        fpath = sdir / fname
        if not fpath.exists():
            alt = fpath.with_suffix(".npy")
            if alt.exists():
                fpath = alt
            else:
                raise FileNotFoundError(f"band raster not found: {fpath}")
        arr = np.asarray(read_array(fpath), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        bands.append(arr)

    dn = np.stack(bands, axis=0).astype(np.float32)
    sun_elev = float(meta.get("sun_elevation", 45.0))
    lmax = meta.get("lmax", list(LISSIV_LMAX))

    if to_reflectance:
        image = dn_to_toa_reflectance(dn, sun_elev, lmax=lmax, esun=list(LISSIV_ESUN))
    else:
        image = np.clip(dn / 1023.0, 0.0, 1.0).astype(np.float32)

    meta.setdefault("scene_id", sdir.name)
    meta.setdefault("sun_elevation", sun_elev)
    meta.setdefault("sun_azimuth", 135.0)
    return image, meta


def _resolve_band_files(scene_dir: Path) -> tuple[str, str, str]:
    """Locate (Green, Red, NIR) band filenames within a scene directory.

    Prefers the canonical ``BAND2/3/4`` names; otherwise sorts ``*BAND*`` rasters
    and assumes ascending order is G/R/NIR.
    """
    for triple in (_DEFAULT_BAND_FILES,):
        if all(
            (scene_dir / f).exists() or (scene_dir / f).with_suffix(".npy").exists() for f in triple
        ):
            return triple
    candidates = sorted(
        p.name
        for p in scene_dir.iterdir()
        if "band" in p.name.lower() and p.suffix.lower() in {".tif", ".tiff", ".npy"}
    )
    if len(candidates) >= 3:
        return candidates[0], candidates[1], candidates[2]
    return _DEFAULT_BAND_FILES


# --------------------------------------------------------------------------- #
# Documented ordering workflow (no live call)
# --------------------------------------------------------------------------- #
def order_workflow(
    aoi: str | None = None,
    date_range: tuple[str, str] | None = None,
    max_cloud_pct: float | None = None,
) -> dict[str, Any]:
    """Return the documented Bhoonidhi search/order recipe (no network call).

    Mirrors the portal workflow (``research/06`` §C.3): register + accept EULA,
    define an AOI (KML/GeoJSON or bbox), filter Satellite=Resourcesat-2/2A,
    Sensor=LISS-IV MX, date range and cloud%, preview quick-looks, then order/
    download a ZIP of per-band GeoTIFFs + ``BAND_META.txt``.

    Args:
        aoi: AOI description (KML/GeoJSON path or bbox string) for the request.
        date_range: ``(start, end)`` ISO dates.
        max_cloud_pct: Maximum acceptable cloud cover percentage.

    Returns:
        A descriptive recipe dict (portal URL + ordered steps + the chosen
        filters) -- useful for logging and for a human to action.
    """
    return {
        "portal": BHOONIDHI_PORTAL,
        "open_data": "Resourcesat-2/2A LISS-IV @5.8 m is open data (Space Policy 2023)",
        "filters": {
            "satellite": ["Resourcesat-2", "Resourcesat-2A"],
            "sensor": "LISS-IV MX",
            "aoi": aoi,
            "date_range": date_range,
            "max_cloud_pct": max_cloud_pct,
        },
        "steps": [
            "Register at bhoonidhi.nrsc.gov.in and accept the EULA.",
            "Open Data Discovery & Download.",
            "Define AOI: draw a box, upload KML/GeoJSON, or enter coordinates.",
            "Filter Satellite=Resourcesat-2/2A, Sensor=LISS-IV MX, date, cloud%.",
            "Browse quick-looks, add to cart, place order.",
            "Download ZIP (BAND2/3/4.tif + BAND_META.txt), then read_lissiv_scene().",
        ],
        "delivery": "ZIP per scene: one GeoTIFF per band + BAND_META.txt (UTM/WGS84).",
    }
