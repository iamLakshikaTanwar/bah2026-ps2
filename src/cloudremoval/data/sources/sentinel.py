"""Sentinel-1 (GRD VV/VH) & Sentinel-2 fetch + prep with S2->LISS-IV band mapping.

Sentinel-1 C-band SAR (cloud-penetrating) is the workhorse conditioning signal over
monsoon NER, and Sentinel-2 {B3,B4,B8} is the closest free optical proxy for
LISS-IV {Green,Red,NIR} (``research/06`` §B.3, §E.4). This module provides:

* :func:`map_s2_to_lissiv` -- select/reorder S2 ``B3/B4/B8`` into the LISS-IV
  Green/Red/NIR stack (with an SBAF note: passbands are close but not equal,
  especially NIR -- apply a small linear adjustment before transfer).
* :func:`despeckle_sar` -- a pure-NumPy Lee-style despeckle for VV/VH (lazy
  alternatives via SNAP/pyroSAR are documented but not required).
* :func:`fetch_sentinel2` / :func:`fetch_sentinel1` -- documented fetch through
  STAC (:mod:`cloudremoval.data.sources.stac`) or ``sentinelhub`` (lazy); raise a
  clear error if no backend / network is available.

Only NumPy is imported at module load; ``sentinelhub`` / SNAP are lazy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cloudremoval.utils.logging import get_logger

__all__ = [
    "S2_LISSIV_BANDS",
    "SBAF_NIR_NOTE",
    "map_s2_to_lissiv",
    "despeckle_sar",
    "prepare_sar",
    "fetch_sentinel2",
    "fetch_sentinel1",
]

_log = get_logger(__name__)
_EPS = 1e-6

#: Sentinel-2 band ids mapping to LISS-IV Green/Red/NIR (``research/06`` §B.3).
S2_LISSIV_BANDS = ("B3", "B4", "B8")

#: SBAF caveat: LISS-IV NIR (0.77-0.86 um) is wider than S2-B8A and offset from
#: narrow bands; S2-B8 is the closest by bandwidth. Apply a linear spectral band
#: adjustment factor / histogram match before domain transfer (``research/06`` §B.3).
SBAF_NIR_NOTE = (
    "LISS-IV NIR (0.77-0.86 um) differs from S2 B8/B8A; apply SBAF / histogram "
    "matching before training transfer."
)


def map_s2_to_lissiv(
    s2_stack: np.ndarray,
    band_order: tuple[str, ...] | None = None,
    sbaf: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Select S2 {B3,B4,B8} from a stack and return a LISS-IV Green/Red/NIR cube.

    Args:
        s2_stack: ``[N, H, W]`` Sentinel-2 stack.
        band_order: Names of the ``N`` input bands (e.g. the 13-band L2A order);
            defaults to the standard 13-band order so B3/B4/B8 map to indices
            2/3/7.
        sbaf: Optional per-band linear gains ``(g_green, g_red, g_nir)`` applied as
            a first-order spectral band adjustment.

    Returns:
        ``[3, H, W]`` LISS-IV-aligned reflectance stack (``float32``).
    """
    arr = np.asarray(s2_stack, dtype=np.float32)
    default_order = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    )
    order = band_order or default_order[: arr.shape[0]]
    idx = [order.index(b) for b in S2_LISSIV_BANDS]
    out = arr[idx].astype(np.float32)
    if sbaf is not None:
        out = out * np.asarray(sbaf, dtype=np.float32).reshape(3, 1, 1)
    return out


def despeckle_sar(sar: np.ndarray, window: int = 5) -> np.ndarray:
    """Reduce SAR speckle with a Lee-style adaptive filter (pure NumPy).

    Implements the classic Lee filter per channel: blends the local mean and the
    observed value by the local-variance ratio, preserving edges better than a box
    blur (``research/05`` §B.2 SAR despeckle). SNAP/pyroSAR provide higher-fidelity
    multilooking but are not required here.

    Args:
        sar: ``[2, H, W]`` (VV, VH) backscatter.
        window: Odd filter window size.

    Returns:
        Despeckled ``[2, H, W]`` SAR (``float32``).
    """
    arr = np.asarray(sar, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    k = max(3, window | 1)  # force odd
    pad = k // 2
    out = np.empty_like(arr)
    overall_var = arr.var()
    for c in range(arr.shape[0]):
        ch = arr[c]
        padded = np.pad(ch, pad, mode="reflect")
        local_mean = _box_mean(padded, k)
        local_sq = _box_mean(padded**2, k)
        local_var = np.clip(local_sq - local_mean**2, 0.0, None)
        weight = local_var / (local_var + overall_var + _EPS)
        out[c] = local_mean + weight * (ch - local_mean)
    return out.astype(np.float32)


def _box_mean(padded: np.ndarray, k: int) -> np.ndarray:
    """Sliding-window box mean of a padded 2-D array via an integral image."""
    integral = padded.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h = padded.shape[0] - (k - 1)
    w = padded.shape[1] - (k - 1)
    total = (
        integral[k : k + h, k : k + w]
        - integral[0:h, k : k + w]
        - integral[k : k + h, 0:w]
        + integral[0:h, 0:w]
    )
    return (total / (k * k)).astype(np.float32)


def prepare_sar(sar: np.ndarray, despeckle: bool = True, normalize: bool = True) -> np.ndarray:
    """Despeckle and per-channel normalise a ``[2, H, W]`` VV/VH SAR pair.

    Args:
        sar: ``[2, H, W]`` SAR backscatter.
        despeckle: Apply :func:`despeckle_sar` first.
        normalize: Min-max scale each channel to ``[0, 1]``.

    Returns:
        Prepared ``[2, H, W]`` SAR (``float32``).
    """
    arr = np.asarray(sar, dtype=np.float32)
    if despeckle:
        arr = despeckle_sar(arr)
    if normalize:
        for c in range(arr.shape[0]):
            ch = arr[c]
            arr[c] = (ch - ch.min()) / (ch.max() - ch.min() + _EPS)
    return arr.astype(np.float32)


# --------------------------------------------------------------------------- #
# Documented fetch (lazy STAC / sentinelhub)
# --------------------------------------------------------------------------- #
def fetch_sentinel2(
    bbox: tuple[float, float, float, float],
    datetime: str,
    max_cloud: float = 20.0,
) -> list[dict[str, Any]]:
    """Search Sentinel-2 L2A items over ``bbox``/``datetime`` (documented; lazy STAC).

    Delegates to :func:`cloudremoval.data.sources.stac.search_stac` against the
    Planetary Computer ``sentinel-2-l2a`` collection. Requires the STAC stack +
    network; **not** run on the smoke path.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        datetime: STAC datetime/interval.
        max_cloud: Maximum scene cloud cover percentage.

    Returns:
        STAC item dicts with signed COG hrefs (band assets B03/B04/B08).
    """
    from cloudremoval.data.sources.stac import search_stac

    return search_stac(
        "sentinel-2-l2a",
        bbox=bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )


def fetch_sentinel1(
    bbox: tuple[float, float, float, float],
    datetime: str,
) -> list[dict[str, Any]]:
    """Search Sentinel-1 GRD (VV/VH) items over ``bbox``/``datetime`` (documented).

    Delegates to STAC against the Planetary Computer ``sentinel-1-grd`` collection.
    Requires the STAC stack + network; **not** run on the smoke path.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        datetime: STAC datetime/interval.

    Returns:
        STAC item dicts with signed COG hrefs (vv/vh assets).
    """
    from cloudremoval.data.sources.stac import search_stac

    return search_stac("sentinel-1-grd", bbox=bbox, datetime=datetime)
