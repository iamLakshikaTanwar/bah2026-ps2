"""Evaluation layer (owned by B3): MVES metrics, evaluator, benchmark, report.

Implemented by B3 against the metric contract (BUILD_PLAN §3.5). Tier-0 metrics
are pure torch/numpy; heavy metric deps (lpips/sewar) are lazy with graceful
NaN fallbacks. Empty-but-valid namespace until those modules land.
"""

from __future__ import annotations

__all__: list[str] = []
