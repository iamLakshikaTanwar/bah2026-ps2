"""Inference layer (owned by B4): tiled blended inference, postprocess, COG writer.

Implemented by B4 against the inference contract (BUILD_PLAN §3.8). The tiling
engine runs on CPU with the torch backend; ONNX/TensorRT and COG writing are
optional-dependency guarded. Empty-but-valid namespace until those modules land.
"""

from __future__ import annotations

__all__: list[str] = []
