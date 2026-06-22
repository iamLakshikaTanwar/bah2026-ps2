"""Training layer (owned by B3): ``Trainer``, optional Lightning module, callbacks.

Implemented by B3 against the trainer contract (BUILD_PLAN §3.7). Empty-but-valid
namespace until those modules land; the trainer uses only ``build_model`` /
``LossBundle`` / the SAMPLE dict, never a concrete model.
"""

from __future__ import annotations

__all__: list[str] = []
