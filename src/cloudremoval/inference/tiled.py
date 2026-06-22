"""Sliding-window (halo) tiled inference for whole-scene reconstruction.

A LISS-IV scene is far larger than any model's receptive field / memory budget,
so reconstruction is done **tile-by-tile**: slide a ``tile x tile`` window with
``overlap`` (halo) across the scene, run ``model.predict`` on each window (batched),
and stitch the outputs back with a feathered window (``inference/blending.py``) to
avoid seams. Scene edges are handled by **reflect padding** so border tiles are
full-sized. This is the inference-time twin of the training-time patchification
(``research/05`` §C.3) and the basis of "O(1)-per-tile" serving (one tile in, one
tile out, cacheable by content hash).

The engine is **pure torch/numpy** (the ``torch`` backend); ONNX/TensorRT backends
are lazy-import hooks that fall back to torch when their runtimes are absent, so the
CPU-smoke path always works. ``model`` is used **only** through the frozen
:meth:`BaseCloudRemovalModel.predict` API — no concrete model is imported.

Public API
----------
:func:`tiled_inference`
    Functional entry point: ``(model, image, ...) -> reconstruction [C, H, W]``.
:class:`TiledInferenceEngine`
    Class wrapper matching the BUILD_PLAN §3.8 contract
    (``run_array`` / ``run_cog``) built on the same machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cloudremoval.inference.blending import BlendAccumulator, make_window
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:  # avoid import cycles / heavy imports on the smoke path
    from cloudremoval.config import InferConfig
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = [
    "tiled_inference",
    "compute_tile_positions",
    "default_build_sample",
    "TiledInferenceEngine",
]

_log = get_logger(__name__)

# A ``build_sample`` callback maps a dict of per-tile numpy arrays (keys are a
# subset of the SAMPLE keys, each ``[C, h, w]``) to a full SAMPLE dict ready for
# ``model.predict`` (batched, on the right device).
BuildSample = Callable[[dict[str, np.ndarray], str], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #
def compute_tile_positions(extent: int, tile: int, stride: int) -> list[int]:
    """Return the top/left start offsets covering ``[0, extent)``.

    Offsets step by ``stride`` and the final offset is clamped so the last tile
    ends exactly at ``extent`` (no out-of-range read, full overlap at the edge).

    Args:
        extent: Size of the axis to cover (after padding).
        tile: Tile size along this axis.
        stride: Step between consecutive tile starts (``tile - overlap``).

    Returns:
        Sorted unique start offsets.
    """
    if tile >= extent:
        return [0]
    positions = list(range(0, extent - tile + 1, stride))
    if positions[-1] != extent - tile:
        positions.append(extent - tile)
    return positions


def _reflect_pad(array: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Reflect-pad the trailing (H, W) of a ``[C, H, W]`` array by ``pad_h/pad_w``."""
    if pad_h == 0 and pad_w == 0:
        return array
    # ``reflect`` requires pad < dim; fall back to ``edge`` for tiny arrays.
    mode = "reflect" if (array.shape[1] > 1 and array.shape[2] > 1) else "edge"
    return np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)


# --------------------------------------------------------------------------- #
# Default SAMPLE builder
# --------------------------------------------------------------------------- #
def default_build_sample(tile_arrays: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    """Wrap per-tile numpy arrays into a batched SAMPLE dict (§2) for ``predict``.

    The cloudy optical tile is required; ``sar``/``dem``/``cloud_mask``/
    ``shadow_mask`` are forwarded when present and zero-filled (correct shape)
    otherwise, so models can unconditionally index every key. A zero
    ``optical_clear`` target and ``meta["has_target"]=False`` mark inference.

    Args:
        tile_arrays: Mapping with at least ``"optical_cloudy"`` ``[C, h, w]`` and
            optionally ``"sar"`` ``[2, h, w]``, ``"dem"`` ``[1, h, w]``,
            ``"cloud_mask"``/``"shadow_mask"`` ``[1, h, w]``.
        device: Torch device string for the produced tensors.

    Returns:
        A SAMPLE dict with a leading batch dim of 1 on every image tensor.
    """
    cloudy = np.asarray(tile_arrays["optical_cloudy"], dtype=np.float32)
    if cloudy.ndim == 2:
        cloudy = cloudy[None, ...]
    c, h, w = cloudy.shape

    def _opt(key: str, channels: int) -> torch.Tensor:
        if key in tile_arrays and tile_arrays[key] is not None:
            arr = np.asarray(tile_arrays[key], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, ...]
        else:
            arr = np.zeros((channels, h, w), dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr))[None, ...].to(device)

    has = {
        "has_sar": "sar" in tile_arrays and tile_arrays["sar"] is not None,
        "has_dem": "dem" in tile_arrays and tile_arrays["dem"] is not None,
        "has_temporal": False,
        "has_target": False,
    }
    sample: dict[str, Any] = {
        "optical_cloudy": torch.from_numpy(np.ascontiguousarray(cloudy))[None, ...].to(device),
        "optical_clear": torch.zeros(1, c, h, w, device=device),
        "cloud_mask": _opt("cloud_mask", 1),
        "shadow_mask": _opt("shadow_mask", 1),
        "sar": _opt("sar", 2),
        "dem": _opt("dem", 1),
        "temporal_refs": torch.zeros(1, 0, c, h, w, device=device),
        "meta": [{"norm": "zscore", **has}],
    }
    return sample


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #
def _normalize_image_arg(
    image: np.ndarray | torch.Tensor | dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Split the ``image`` argument into the primary cloudy array + aux arrays.

    ``image`` may be a bare ``[C, H, W]`` array/tensor (the cloudy optical), or a
    dict of SAMPLE-like full-scene arrays (with ``"optical_cloudy"`` required).
    Returns the cloudy array and a dict of the remaining aux full-scene arrays.
    """
    aux: dict[str, np.ndarray] = {}
    if isinstance(image, dict):
        if "optical_cloudy" not in image:
            raise KeyError("image dict must contain 'optical_cloudy'")
        primary = image["optical_cloudy"]
        aux = {k: v for k, v in image.items() if k not in ("optical_cloudy", "meta")}
    else:
        primary = image

    if isinstance(primary, torch.Tensor):
        primary = primary.detach().cpu().numpy()
    primary = np.asarray(primary, dtype=np.float32)
    if primary.ndim == 2:
        primary = primary[None, ...]

    clean_aux: dict[str, np.ndarray] = {}
    for key, val in aux.items():
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        val = np.asarray(val, dtype=np.float32)
        if val.ndim == 2:
            val = val[None, ...]
        clean_aux[key] = val
    return primary, clean_aux


def tiled_inference(
    model: BaseCloudRemovalModel,
    image: np.ndarray | torch.Tensor | dict[str, Any],
    tile: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    device: str = "cpu",
    build_sample: BuildSample | None = None,
    blend: str = "hann",
    out_channels: int | None = None,
) -> np.ndarray:
    """Run ``model.predict`` over a large scene in overlapping tiles, then stitch.

    Slides a ``tile x tile`` window with ``overlap`` halo across the (reflect-
    padded) scene, builds a SAMPLE per tile (via ``build_sample`` or
    :func:`default_build_sample`), batches up to ``batch_size`` tiles through
    ``model.predict``, and blends the per-tile reconstructions with a feathered
    window (:func:`~cloudremoval.inference.blending.make_window`) to remove seams.

    Args:
        model: Any :class:`BaseCloudRemovalModel`; used only via ``.predict``.
        image: The cloudy optical scene ``[C, H, W]`` (numpy or torch), **or** a
            dict of full-scene SAMPLE arrays (``optical_cloudy`` required;
            ``sar``/``dem``/``cloud_mask``/``shadow_mask`` optional, each tiled in
            lockstep).
        tile: Square tile size in pixels.
        overlap: Halo width (pixels of overlap between neighbouring tiles).
        batch_size: Max tiles per ``predict`` call (memory/throughput knob).
        device: Torch device for tile tensors and accumulation.
        build_sample: Optional callback ``(tile_arrays, device) -> SAMPLE`` to
            customise modality wiring; defaults to :func:`default_build_sample`.
        blend: Blend-window kind (``"hann"`` / ``"gaussian"`` / ``"linear"`` /
            ``"none"``).
        out_channels: Output channel count; inferred from the first tile's
            reconstruction when ``None``.

    Returns:
        The reconstructed scene ``[C_out, H, W]`` (``float32``), cropped back to
        the original (pre-pad) size.

    Raises:
        ValueError: If ``tile <= 0`` or ``overlap`` is not in ``[0, tile)``.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must be in [0, tile), got {overlap} (tile={tile})")
    build = build_sample or default_build_sample

    primary, aux = _normalize_image_arg(image)
    c_in, h0, w0 = primary.shape

    # Pad so that (extent - tile) is a non-negative multiple-friendly span; we
    # pad up to at least one tile and to cover the stride grid cleanly.
    stride = tile - overlap
    pad_h = max(0, tile - h0) + ((-(h0 - tile) % stride) if h0 > tile else 0)
    pad_w = max(0, tile - w0) + ((-(w0 - tile) % stride) if w0 > tile else 0)
    primary_p = _reflect_pad(primary, pad_h, pad_w)
    aux_p = {k: _reflect_pad(v, pad_h, pad_w) for k, v in aux.items()}
    _, hp, wp = primary_p.shape

    tops = compute_tile_positions(hp, tile, stride)
    lefts = compute_tile_positions(wp, tile, stride)
    coords = [(t, ln) for t in tops for ln in lefts]
    _log.debug(
        "tiled_inference: scene=%dx%d pad=%dx%d tiles=%d (%dx%d, overlap=%d)",
        h0,
        w0,
        hp,
        wp,
        len(coords),
        tile,
        tile,
        overlap,
    )

    acc: BlendAccumulator | None = None
    window = make_window(tile, tile, kind=blend)

    # Process tiles in batches to keep memory bounded and throughput high.
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        recon_tiles = _predict_batch(model, primary_p, aux_p, batch_coords, tile, device, build)
        if acc is None:
            c_out = out_channels if out_channels is not None else recon_tiles[0].shape[0]
            acc = BlendAccumulator(c_out, hp, wp)
        for (top, left), recon in zip(batch_coords, recon_tiles, strict=True):
            acc.add(recon, top, left, window)

    assert acc is not None  # at least one tile always exists
    full = acc.result()
    return full[:, :h0, :w0].astype(np.float32)


def _predict_batch(
    model: BaseCloudRemovalModel,
    primary_p: np.ndarray,
    aux_p: dict[str, np.ndarray],
    coords: list[tuple[int, int]],
    tile: int,
    device: str,
    build: BuildSample,
) -> list[np.ndarray]:
    """Predict one batch of tiles, returning a list of ``[C_out, tile, tile]`` arrays.

    Builds a per-tile SAMPLE for each coordinate, concatenates them along the
    batch dim (a single ``predict`` call), and splits the reconstruction back.
    Tiles are cut from the *padded* scene so every tile is exactly ``tile``-sized.
    """
    samples = []
    for top, left in coords:
        tile_arrays: dict[str, np.ndarray] = {
            "optical_cloudy": primary_p[:, top : top + tile, left : left + tile]
        }
        for key, arr in aux_p.items():
            tile_arrays[key] = arr[:, top : top + tile, left : left + tile]
        samples.append(build(tile_arrays, device))

    batched = _collate_samples(samples)
    output = model.predict(batched)
    recon = output.reconstruction
    if recon.ndim == 3:  # model returned [C, H, W] for a single tile
        recon = recon[None, ...]
    recon_np = recon.detach().to("cpu").numpy().astype(np.float32)
    return [recon_np[i] for i in range(recon_np.shape[0])]


def _collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate a list of single-item SAMPLE dicts along the batch dim.

    Each input already carries a leading batch dim of 1 (from the builder), so we
    ``cat`` on dim 0 for tensors and concatenate the ``meta`` lists. Avoids a
    dependency on B1's ``collate_sample`` to keep B4 independent.
    """
    if len(samples) == 1:
        return samples[0]
    out: dict[str, Any] = {}
    for key in samples[0]:
        if key == "meta":
            meta: list[Any] = []
            for s in samples:
                m = s["meta"]
                meta.extend(m if isinstance(m, list) else [m])
            out["meta"] = meta
        else:
            out[key] = torch.cat([s[key] for s in samples], dim=0)
    return out


# --------------------------------------------------------------------------- #
# Contract class (BUILD_PLAN §3.8)
# --------------------------------------------------------------------------- #
class TiledInferenceEngine:
    """Whole-scene tiled inference engine (BUILD_PLAN §3.8 surface).

    Thin object wrapper over :func:`tiled_inference` that reads tile/halo/blend
    from an :class:`~cloudremoval.config.InferConfig` and adds full-scene array /
    COG entry points. The ``backend`` field selects ``torch`` (always available)
    or an accelerated runtime; non-torch backends lazily import their runtime and
    **fall back to torch** with a warning if it is missing, so the smoke path is
    unaffected.

    Args:
        model: The model to drive (used only via ``predict``).
        cfg: The ``infer`` config section.
    """

    def __init__(self, model: BaseCloudRemovalModel, cfg: InferConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.backend = self._resolve_backend(getattr(cfg, "backend", "torch"))

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        """Return a usable backend, degrading to ``torch`` if a runtime is absent."""
        if requested == "torch":
            return "torch"
        if requested == "onnx":
            try:  # pragma: no cover - optional accel dep
                import onnxruntime  # noqa: F401
            except ImportError:
                _log.warning("onnxruntime unavailable; falling back to torch backend")
                return "torch"
            return "onnx"
        if requested == "tensorrt":
            try:  # pragma: no cover - optional accel dep
                import tensorrt  # noqa: F401
            except ImportError:
                _log.warning("tensorrt unavailable; falling back to torch backend")
                return "torch"
            return "tensorrt"
        _log.warning("unknown backend '%s'; using torch", requested)
        return "torch"

    def run_array(self, stack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Reconstruct a full-scene SAMPLE-like numpy ``stack``.

        Args:
            stack: Mapping with full-scene arrays; ``"optical_cloudy"`` ``[C,H,W]``
                is required, ``sar``/``dem``/``cloud_mask``/``shadow_mask``
                optional.

        Returns:
            ``{"reconstruction": [C, H, W]}`` (and ``"uncertainty"`` if the model
            produced one — computed via a second masked pass is *not* done here;
            only the reconstruction is tiled to keep the contract simple).
        """
        recon = tiled_inference(
            self.model,
            stack,
            tile=self.cfg.tile_size,
            overlap=self.cfg.halo,
            device=self.cfg.device,
            blend=self.cfg.blend,
        )
        return {"reconstruction": recon}

    def run_cog(self, input_paths: dict[str, str], out_path: str) -> str:
        """Read scene rasters, reconstruct, and write a COG (lazy geo deps).

        Args:
            input_paths: Mapping of SAMPLE key -> raster path (at least
                ``"optical_cloudy"``). Read via :func:`utils.io.read_array`
                (rasterio when available, ``.npy`` fallback otherwise).
            out_path: Output COG path.

        Returns:
            The path actually written (``.tif`` COG or ``.npy`` fallback).
        """
        from cloudremoval.inference.cog_writer import write_cog
        from cloudremoval.utils.io import read_array

        stack = {key: read_array(path) for key, path in input_paths.items()}
        recon = self.run_array(stack)["reconstruction"]
        return write_cog(recon, out_path)
