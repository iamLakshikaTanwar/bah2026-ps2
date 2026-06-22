"""Array and raster I/O with a lazy-``rasterio`` GeoTIFF path and a ``.npy`` fallback.

The CPU-smoke pipeline must read/write arrays with **no geospatial deps**, so the
GeoTIFF helpers import ``rasterio`` lazily and fall back to NumPy ``.npy`` (with a
JSON sidecar for the profile) when ``rasterio`` is absent or a non-``.tif`` path
is given. Also hosts :func:`move_to_device`, the shared tensor-mover the trainer
contract (BUILD_PLAN §3.7) expects from B0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

__all__ = [
    "move_to_device",
    "read_array",
    "write_array",
    "read_geotiff",
    "write_geotiff",
    "save_npy",
    "load_npy",
]

if TYPE_CHECKING:
    import numpy as np
    import torch


# --------------------------------------------------------------------------- #
# Device movement (trainer + inference share this)
# --------------------------------------------------------------------------- #
def move_to_device(obj: Any, device: str | torch.device) -> Any:
    """Recursively move tensors in a (possibly nested) container to ``device``.

    Handles the ``SAMPLE`` dict directly: tensors are moved, ``meta`` (and any
    other non-tensor value) is passed through unchanged. Lists/tuples/dicts are
    traversed.

    Args:
        obj: A tensor, or a dict/list/tuple possibly containing tensors.
        device: Target device (``"cpu"``, ``"cuda"``, or a ``torch.device``).

    Returns:
        The same structure with tensors relocated to ``device``.
    """
    import torch  # local: keeps this module import-light, torch is a core dep

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


# --------------------------------------------------------------------------- #
# NumPy fallbacks
# --------------------------------------------------------------------------- #
def save_npy(array: np.ndarray, path: str | Path) -> str:
    """Save a NumPy array to ``.npy`` (creating parent dirs). Returns the path."""
    import numpy as np

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    return str(out)


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from ``.npy``."""
    import numpy as np

    return np.load(Path(path))


# --------------------------------------------------------------------------- #
# Generic array read/write (dispatch on extension / rasterio availability)
# --------------------------------------------------------------------------- #
def write_array(
    array: np.ndarray,
    path: str | Path,
    profile: dict[str, Any] | None = None,
) -> str:
    """Write a ``[C, H, W]`` (or ``[H, W]``) array, choosing GeoTIFF or ``.npy``.

    ``.tif``/``.tiff`` paths attempt a GeoTIFF write via lazy ``rasterio`` and
    fall back to a ``.npy`` (plus ``.json`` profile sidecar) if ``rasterio`` is
    unavailable. Any other extension is written as ``.npy``.

    Args:
        array: Image array, channel-first if 3-D.
        path: Output path.
        profile: Optional rasterio-style profile (crs, transform, ...).

    Returns:
        The path actually written.
    """
    p = Path(path)
    if p.suffix.lower() in {".tif", ".tiff"}:
        try:
            return write_geotiff(array, p, profile=profile)
        except ImportError:
            sidecar = p.with_suffix(".npy")
            save_npy(array, sidecar)
            if profile is not None:
                _write_profile_sidecar(profile, p.with_suffix(".json"))
            return str(sidecar)
    return save_npy(array, p)


def read_array(path: str | Path) -> np.ndarray:
    """Read an array written by :func:`write_array` (GeoTIFF or ``.npy``)."""
    p = Path(path)
    if p.suffix.lower() in {".tif", ".tiff"}:
        try:
            data, _ = read_geotiff(p)
            return data
        except ImportError:
            return load_npy(p.with_suffix(".npy"))
    return load_npy(p)


# --------------------------------------------------------------------------- #
# GeoTIFF (lazy rasterio)
# --------------------------------------------------------------------------- #
def write_geotiff(
    array: np.ndarray,
    path: str | Path,
    profile: dict[str, Any] | None = None,
) -> str:
    """Write a ``[C, H, W]``/``[H, W]`` array as a GeoTIFF (lazy ``rasterio``).

    Args:
        array: Channel-first image (or 2-D single band).
        path: Output ``.tif`` path.
        profile: Optional rasterio profile; sensible defaults fill the rest.

    Returns:
        The output path.

    Raises:
        ImportError: If ``rasterio`` is not installed (callers may fall back).
    """
    import numpy as np
    import rasterio  # lazy, optional

    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, ...]
    count, height, width = arr.shape

    prof: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
    }
    if profile:
        prof.update(profile)
    prof["height"], prof["width"], prof["count"] = height, width, count
    prof["dtype"] = str(arr.dtype)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr)
    return str(out)


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a GeoTIFF into ``([C, H, W] array, profile)`` (lazy ``rasterio``).

    Raises:
        ImportError: If ``rasterio`` is not installed.
    """
    import rasterio  # lazy, optional

    with rasterio.open(Path(path)) as src:
        data = src.read()
        profile = dict(src.profile)
    return data, profile


def _write_profile_sidecar(profile: dict[str, Any], path: Path) -> None:
    """Write a JSON sidecar with the JSON-serialisable parts of a profile."""

    def _safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({k: _safe(v) for k, v in profile.items()}, fh, indent=2)
