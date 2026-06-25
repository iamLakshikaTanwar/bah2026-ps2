"""Model registry: name -> class mapping with a decorator and a factory.

Owns the **frozen registry contract** (``docs/BUILD_PLAN.md`` §3.2). Concrete
models self-register at import time via :func:`register_model`; everything
downstream instantiates them through :func:`build_model` (never by importing the
concrete class). Importing :mod:`cloudremoval.models` triggers the imports that
populate this registry.

Pure-Python module: no heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type checkers / docs — no runtime import cycle
    from cloudremoval.config import Config
    from cloudremoval.models.base import BaseCloudRemovalModel

__all__ = [
    "register_model",
    "build_model",
    "list_models",
    "get_model_cls",
]

# The single source of truth: registry name -> model class.
_REGISTRY: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    """Class decorator that registers a model under ``name``.

    Usage::

        @register_model("unet")
        class UNetModel(BaseCloudRemovalModel): ...

    The decorator also sets ``cls.name = name`` so the class and its registry key
    agree.

    Args:
        name: Unique registry key (e.g. ``"unet"``).

    Returns:
        The class decorator.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def deco(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_model(cfg: Config) -> BaseCloudRemovalModel:
    """Instantiate the model named ``cfg.model.name`` with ``cfg.model``.

    Args:
        cfg: The global :class:`~cloudremoval.config.Config`. Only
            ``cfg.model`` is forwarded to the constructor.

    Returns:
        A constructed :class:`~cloudremoval.models.base.BaseCloudRemovalModel`.

    Raises:
        KeyError: If ``cfg.model.name`` is not registered.
    """
    name = cfg.model.name
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg.model)


def list_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY)


def get_model_cls(name: str) -> type:
    """Return the class registered under ``name``.

    Args:
        name: Registry key.

    Returns:
        The registered class.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
