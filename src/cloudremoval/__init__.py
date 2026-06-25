"""``cloudremoval`` — model-agnostic GenAI cloud removal for LISS-IV imagery.

Top-level package. Importing :mod:`cloudremoval` must succeed on the minimal
CPU-smoke stack (``numpy``, ``torch``, ``pydantic``, ``pyyaml``) with **no**
geospatial or accelerator dependencies — every heavy dependency is imported
lazily inside the function that needs it (BUILD_PLAN §5, ARCHITECTURE §1.3).

The public surface re-exported here is the integration contract other builders
rely on:

* :data:`__version__`
* :class:`Config` — the typed configuration schema
* :class:`BaseCloudRemovalModel` — the model interface
* :func:`build_model`, :func:`register_model`, :func:`list_models` — the registry
"""

from __future__ import annotations

from cloudremoval.config import Config
from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.registry import (
    build_model,
    get_model_cls,
    list_models,
    register_model,
)

__all__ = [
    "__version__",
    "Config",
    "BaseCloudRemovalModel",
    "ModelOutput",
    "build_model",
    "register_model",
    "list_models",
    "get_model_cls",
]

__version__ = "0.1.0"
