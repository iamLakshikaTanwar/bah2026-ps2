"""Data layer (owned by B1): SAMPLE-producing datasets and the data engine.

Concrete modules (``datasets``, ``synthetic_clouds``, ``preprocess``,
``coregister``, ``masking``, ``transforms``, ``cog_tiling`` and ``sources/*``)
are implemented by B1 against the SAMPLE contract (BUILD_PLAN §2, §3.4). This
package is an empty-but-valid namespace so imports resolve before B1's work
lands; nothing is re-exported here to avoid import errors on a partial repo.
"""

from __future__ import annotations

__all__: list[str] = []
