"""Models package: shared contracts + self-populating registry.

Re-exports the frozen interface (:class:`BaseCloudRemovalModel`,
:class:`ModelOutput`, :data:`LossDict`) and the registry API, then imports each
concrete model module so that ``import cloudremoval.models`` self-populates the
registry (BUILD_PLAN §3.2).

During early parallel development the concrete modules (owned by B2) may not
exist yet, so each import is wrapped in ``try/except ImportError`` with a debug
log. This keeps a partial repo importable and lets B0's scaffold run before any
model is written. Once B2 lands ``unet.py`` et al., they register automatically.
"""

from __future__ import annotations

from cloudremoval.models.base import (
    BaseCloudRemovalModel,
    LossDict,
    ModelOutput,
)
from cloudremoval.models.registry import (
    build_model,
    get_model_cls,
    list_models,
    register_model,
)
from cloudremoval.utils.logging import get_logger

__all__ = [
    "BaseCloudRemovalModel",
    "ModelOutput",
    "LossDict",
    "register_model",
    "build_model",
    "list_models",
    "get_model_cls",
]

_log = get_logger(__name__)

# Concrete model modules (owned by B2). Importing each one triggers its
# ``@register_model`` side effect. Missing modules are tolerated so the scaffold
# imports cleanly before B2's work exists.
_CONCRETE_MODULES = (
    "unet",
    "dsen2cr_fusion",
    "gan_spagan",
    "transformer_restormer",
    "diffusion",
    "uncertainty",
)

for _mod in _CONCRETE_MODULES:
    try:
        __import__(f"{__name__}.{_mod}")
    except ImportError as exc:  # not yet implemented / optional dep missing
        _log.debug("model module '%s' not available yet: %s", _mod, exc)
