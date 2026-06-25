"""Multi-source data loaders (owned by B1): Bhoonidhi, GEE, STAC, Sentinel, DEM.

All real-source loaders are optional-dependency guarded (lazy imports of
``rasterio``/``pystac-client``/``earthengine-api`` etc.); the synthetic CPU path
never touches this package. Empty-but-valid namespace until B1 lands the modules.
"""

from __future__ import annotations

__all__: list[str] = []
