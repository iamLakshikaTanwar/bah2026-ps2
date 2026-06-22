"""Pydantic v2 request/response schemas for the serving API.

These models are the typed contract of the FastAPI app (``serving/app.py``). They
cover the two ways a client supplies a scene — a small **inline array** (base64
``.npy`` bytes, for tests and tiny tiles) or a **server-side reference**
(path / COG URL / STAC id, for real scenes read via ranged COG GET) — and the
reconstruction / info / tile responses.

Only ``pydantic`` is imported, so this module loads on the minimal CPU-smoke stack.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HealthResponse",
    "InfoResponse",
    "ModelsResponse",
    "ArraySpec",
    "PredictOptions",
    "PredictRequest",
    "PredictResponse",
    "TileParams",
    # Contract aliases (BUILD_PLAN §3.9)
    "ReconstructRequest",
    "ReconstructResponse",
]


# --------------------------------------------------------------------------- #
# Health / info / models
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``."""

    status: Literal["ok"] = "ok"
    version: str = ""


class InfoResponse(BaseModel):
    """Service capabilities returned by ``GET /info``."""

    name: str = "cloudremoval"
    version: str = ""
    capabilities: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    device: str = "cpu"
    cache_backend: str = "memory"
    heavy_deps: dict[str, bool] = Field(
        default_factory=dict,
        description="Availability of optional heavy deps (rasterio, rio_tiler, faiss, redis).",
    )


class ModelsResponse(BaseModel):
    """Registered-model listing returned by ``GET /models``."""

    models: list[str] = Field(default_factory=list)
    default: str | None = None


# --------------------------------------------------------------------------- #
# Predict / reconstruct
# --------------------------------------------------------------------------- #
class ArraySpec(BaseModel):
    """An inline ``[C, H, W]`` array supplied as base64-encoded NumPy ``.npy`` bytes.

    Keeps the wire format self-describing (dtype/shape live in the ``.npy``
    header) and dependency-free to decode (``numpy.load`` on a buffer). Intended
    for tiny tiles / tests; real scenes use :attr:`PredictRequest.input_ref`.
    """

    model_config = ConfigDict(protected_namespaces=())

    npy_base64: str = Field(..., description="base64 of numpy.save(...) bytes, [C,H,W] float.")
    note: str | None = None


class PredictOptions(BaseModel):
    """Tunable knobs for a reconstruction request (tiling + finishing)."""

    model_config = ConfigDict(protected_namespaces=())

    tile_size: int = 256
    overlap: int = 32
    batch_size: int = 4
    blend: Literal["hann", "gaussian", "linear", "none"] = "hann"
    composite_under_mask: bool = True
    harmonize: bool = False
    return_array: bool = Field(
        True,
        description="If true, echo the reconstruction inline (base64 npy) in the response.",
    )


class PredictRequest(BaseModel):
    """Body of ``POST /predict``.

    Exactly one input source should be set: ``array`` (inline) or ``input_ref``
    (a server-side path / COG URL / STAC item id the server can read).
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field("unet", description="Registered model to run.")
    array: ArraySpec | None = Field(None, description="Inline input array (base64 npy).")
    input_ref: str | None = Field(None, description="Server-side path / COG URL / scene id.")
    cloud_mask: ArraySpec | None = Field(None, description="Optional inline cloud mask [1,H,W].")
    options: PredictOptions = Field(default_factory=PredictOptions)
    checkpoint: str | None = Field(None, description="Optional checkpoint path to load.")


class PredictResponse(BaseModel):
    """Response of ``POST /predict``."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "error"] = "ok"
    model_name: str = "identity"
    output_ref: str | None = Field(None, description="Path/URL of the written reconstruction.")
    array_base64: str | None = Field(None, description="Inline reconstruction (base64 npy).")
    shape: list[int] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict, description="Quick scalar metrics.")
    uncertainty: dict[str, float] | None = Field(
        None, description="Summary stats of the uncertainty band, if produced."
    )
    cached: bool = False
    message: str | None = None


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #
class TileParams(BaseModel):
    """Path/query parameters for an XYZ/WMTS tile request."""

    model_config = ConfigDict(protected_namespaces=())

    z: int = Field(..., ge=0, le=30)
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    model_name: str = "unet"
    scene: str | None = Field(None, description="Scene id / COG ref to tile from.")
    rescale: str | None = Field(None, description="min,max for display rescaling.")
    colormap: str | None = None


# --------------------------------------------------------------------------- #
# Contract aliases (BUILD_PLAN §3.9 naming) — same fields, friendlier names.
# --------------------------------------------------------------------------- #
class ReconstructRequest(BaseModel):
    """BUILD_PLAN §3.9 ``ReconstructRequest`` (alias over the predict schema)."""

    model_config = ConfigDict(protected_namespaces=())

    cog_url: str | None = None
    scene_id: str | None = None
    array: ArraySpec | None = None
    model: str = "unet"
    params: PredictOptions = Field(default_factory=PredictOptions)


class ReconstructResponse(BaseModel):
    """BUILD_PLAN §3.9 ``ReconstructResponse``."""

    model_config = ConfigDict(protected_namespaces=())

    cog_url: str | None = None
    stac_item: dict[str, Any] | None = None
    stats: dict[str, float] = Field(default_factory=dict)
    array_base64: str | None = None
