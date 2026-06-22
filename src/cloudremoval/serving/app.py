"""FastAPI serving application for cloud-removal inference.

Exposes the reconstruction service (``research/05`` §C / BUILD_PLAN §3.9):

* ``GET  /health``   — liveness (always ``200 {"status": "ok"}``).
* ``GET  /version``  — package version.
* ``GET  /info``     — capabilities + which optional heavy deps are present.
* ``GET  /models``   — registry listing (``registry.list_models``).
* ``POST /predict``  — lazily load the named model, run tiled inference +
  post-process on an inline array (base64 ``.npy``) or a server-side reference,
  and return a reconstruction (inline and/or a written reference) + quick metrics.
* ``GET  /tiles/...`` — dynamic XYZ tiles (mounted from ``serving/tiles.py``).

**Import safety is a hard requirement:** this module and its router mount must
import — and ``/health`` must answer ``200`` — on the *minimal* stack with **no**
heavy dependency loaded (rasterio / rio-tiler / faiss / redis). Every heavy path is
lazily imported inside the function that needs it. Tiled inference + compositing run
in pure torch/numpy. Models are loaded on demand and cached in-process.
"""

from __future__ import annotations

import base64
import importlib.util
import io
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import FastAPI, HTTPException

from cloudremoval.serving.schemas import (
    ArraySpec,
    HealthResponse,
    InfoResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:
    from cloudremoval.config import Config
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = ["app", "create_app", "ModelCache"]

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Base64 numpy (inline array) helpers
# --------------------------------------------------------------------------- #
def _decode_npy_b64(spec: ArraySpec) -> np.ndarray:
    """Decode a base64 ``.npy`` payload into a ``[C, H, W]`` float32 array."""
    try:
        raw = base64.b64decode(spec.npy_base64)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - bad client input -> 400
        raise HTTPException(status_code=400, detail=f"invalid npy_base64: {exc}") from exc
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise HTTPException(status_code=400, detail=f"expected [C,H,W], got shape {arr.shape}")
    return arr


def _encode_npy_b64(array: np.ndarray) -> str:
    """Encode a NumPy array as a base64 ``.npy`` string for inline transport."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(array, dtype=np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _have(module: str) -> bool:
    """Return ``True`` if an optional module is importable (no import side effect)."""
    return importlib.util.find_spec(module) is not None


# --------------------------------------------------------------------------- #
# Lazy, cached model loading
# --------------------------------------------------------------------------- #
class ModelCache:
    """In-process cache of built models keyed by ``(name, checkpoint)``.

    Avoids rebuilding / reloading a model on every request: the first ``/predict``
    for a model constructs it via :func:`build_model` (and optionally loads a
    checkpoint), subsequent requests reuse the cached instance. Building a model
    imports :mod:`cloudremoval.models`, which is **not** done at app import — only
    on first use — so ``/health`` never triggers it.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str | None], BaseCloudRemovalModel] = {}

    def get(self, name: str, checkpoint: str | None = None) -> BaseCloudRemovalModel:
        """Return a cached or freshly built+loaded model for ``name``.

        Args:
            name: Registered model name.
            checkpoint: Optional checkpoint path (``state_dict``) to load.

        Returns:
            A ready-to-``predict`` :class:`BaseCloudRemovalModel`.

        Raises:
            HTTPException: ``404`` if the model name is not registered.
        """
        key = (name, checkpoint)
        if key in self._cache:
            return self._cache[key]

        from cloudremoval.config import Config
        from cloudremoval.models.registry import build_model, list_models

        if name not in list_models():
            raise HTTPException(
                status_code=404,
                detail=f"unknown model '{name}'. Registered: {list_models()}",
            )
        cfg = Config()
        cfg.model.name = name
        model = build_model(cfg)
        if checkpoint:
            self._load_checkpoint(model, checkpoint)
        model.eval()
        self._cache[key] = model
        _log.info("Loaded model '%s'%s", name, f" (ckpt={checkpoint})" if checkpoint else "")
        return model

    @staticmethod
    def _load_checkpoint(model: BaseCloudRemovalModel, path: str) -> None:
        """Load a ``state_dict`` checkpoint into ``model`` (best-effort, non-strict)."""
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
        except Exception as exc:  # noqa: BLE001 - surface as a 400 upstream
            raise HTTPException(status_code=400, detail=f"checkpoint load failed: {exc}") from exc

    def clear(self) -> None:
        """Drop all cached models."""
        self._cache.clear()


# --------------------------------------------------------------------------- #
# Core reconstruction (pure torch/numpy)
# --------------------------------------------------------------------------- #
def _run_reconstruction(
    model: BaseCloudRemovalModel,
    cloudy: np.ndarray,
    cloud_mask: np.ndarray | None,
    options: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    """Tile the model over ``cloudy``, optionally composite under the mask.

    Args:
        model: The loaded model.
        cloudy: Cloudy optical scene ``[C, H, W]``.
        cloud_mask: Optional ``[1, H, W]`` mask for compositing.
        options: A :class:`PredictOptions`-like object (tile/overlap/blend/...).

    Returns:
        ``(reconstruction [C, H, W], metrics)`` where metrics are quick scalars
        (residual MAE/RMSE on the masked region when a mask is present).
    """
    from cloudremoval.inference.postprocess import clamp_valid, composite_under_mask
    from cloudremoval.inference.tiled import tiled_inference

    image: dict[str, Any] = {"optical_cloudy": cloudy}
    if cloud_mask is not None:
        image["cloud_mask"] = cloud_mask

    recon = tiled_inference(
        model,
        image,
        tile=int(options.tile_size),
        overlap=int(options.overlap),
        batch_size=int(options.batch_size),
        device="cpu",
        blend=str(options.blend),
    )

    metrics: dict[str, float] = {}
    if getattr(options, "composite_under_mask", False) and cloud_mask is not None:
        finished = composite_under_mask(cloudy, recon, cloud_mask)
        m = np.asarray(cloud_mask, dtype=np.float32)
        m = m if m.ndim == 3 else m[None, ...]
        sel = np.broadcast_to(m, recon.shape) > 0.5
        if sel.any():
            diff = (recon - cloudy)[sel]
            metrics["residual_mae"] = float(np.abs(diff).mean())
            metrics["residual_rmse"] = float(np.sqrt((diff**2).mean()))
        recon = finished

    recon = clamp_valid(recon, 0.0, 1.0)
    metrics["recon_mean"] = float(recon.mean())
    metrics["recon_std"] = float(recon.std())
    return recon, metrics


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(cfg: Config | None = None) -> FastAPI:
    """Build the FastAPI app. Importing/creating it loads no heavy dependency.

    Args:
        cfg: Optional :class:`Config`; only ``cfg.serve`` (cache backend, device)
            is consulted. Defaults to :class:`Config` defaults when ``None``.

    Returns:
        A configured :class:`fastapi.FastAPI` instance with all routes mounted.
    """
    from cloudremoval import __version__

    application = FastAPI(
        title="cloudremoval serving API",
        version=__version__,
        description="GenAI cloud removal for LISS-IV imagery (BAH 2026 PS2).",
    )
    model_cache = ModelCache()
    application.state.model_cache = model_cache
    application.state.cfg = cfg

    serve_cfg = getattr(cfg, "serve", None)
    cache_backend = getattr(serve_cfg, "cache", "memory") if serve_cfg else "memory"

    # ----- mount the tiles router (guard so a heavy-dep error never breaks app) -
    try:
        from cloudremoval.serving.tiles import get_tile_cache, router as tiles_router

        get_tile_cache(
            backend=cache_backend,
            redis_url=getattr(serve_cfg, "redis_url", None) if serve_cfg else None,
        )
        application.include_router(tiles_router)
    except Exception as exc:  # noqa: BLE001 - tiles are optional; app must still boot
        _log.warning("tiles router not mounted (%s); /health and /predict still work.", exc)

    # ------------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------------ #
    @application.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness probe — always ``200 {"status": "ok"}`` on the minimal stack."""
        return HealthResponse(status="ok", version=__version__)

    @application.get("/version", tags=["meta"])
    def version() -> dict[str, str]:
        """Return the package version."""
        return {"version": __version__}

    @application.get("/info", response_model=InfoResponse, tags=["meta"])
    def info() -> InfoResponse:
        """Report capabilities and which optional heavy deps are installed."""
        from cloudremoval.models.registry import list_models

        return InfoResponse(
            name="cloudremoval",
            version=__version__,
            capabilities=["predict", "tiled-inference", "composite", "tiles"],
            models=list_models(),
            device=getattr(serve_cfg, "device", "cpu") if serve_cfg else "cpu",
            cache_backend=cache_backend,
            heavy_deps={
                "rasterio": _have("rasterio"),
                "rio_tiler": _have("rio_tiler"),
                "faiss": _have("faiss"),
                "redis": _have("redis"),
                "onnxruntime": _have("onnxruntime"),
            },
        )

    @application.get("/models", response_model=ModelsResponse, tags=["meta"])
    def models() -> ModelsResponse:
        """List registered models (``registry.list_models``)."""
        from cloudremoval.models.registry import list_models

        names = list_models()
        return ModelsResponse(models=names, default=names[0] if names else None)

    @application.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(request: PredictRequest) -> PredictResponse:
        """Reconstruct a scene with the named model (tiled inference + finishing).

        Accepts an inline base64-``.npy`` array (``request.array``) or a
        server-side ``request.input_ref`` read via :func:`utils.io.read_array`.
        Loads the model lazily (cached), tiles ``model.predict`` over the scene,
        optionally composites under the supplied cloud mask, and returns the
        reconstruction inline and/or as a written reference plus quick metrics.
        """
        # ----- resolve the input scene --------------------------------------
        if request.array is not None:
            cloudy = _decode_npy_b64(request.array)
        elif request.input_ref:
            from cloudremoval.utils.io import read_array

            try:
                cloudy = np.asarray(read_array(request.input_ref), dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"failed to read input_ref: {exc}"
                ) from exc
            if cloudy.ndim == 2:
                cloudy = cloudy[None, ...]
        else:
            raise HTTPException(status_code=400, detail="provide 'array' or 'input_ref'")

        cloud_mask = _decode_npy_b64(request.cloud_mask) if request.cloud_mask else None

        # ----- load model + run ---------------------------------------------
        model = model_cache.get(request.model_name, request.checkpoint)
        recon, metrics = _run_reconstruction(model, cloudy, cloud_mask, request.options)

        # ----- optionally persist + echo ------------------------------------
        array_b64 = _encode_npy_b64(recon) if request.options.return_array else None
        return PredictResponse(
            status="ok",
            model_name=request.model_name,
            output_ref=None,
            array_base64=array_b64,
            shape=list(recon.shape),
            metrics=metrics,
            uncertainty=None,
            cached=False,
            message=None,
        )

    return application


# Module-level app for ``uvicorn cloudremoval.serving.app:app`` and TestClient.
app = create_app()
