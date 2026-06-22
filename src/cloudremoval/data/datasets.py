"""Datasets returning the ``SAMPLE`` dict (BUILD_PLAN §2, §3.4).

This is the integration surface of the data layer. Every dataset's
``__getitem__`` returns the universal ``SAMPLE`` dict so the trainer / evaluator /
inference engine never special-case a data source.

Datasets
--------
:class:`SyntheticCloudRemovalDataset`
    The fully-offline CPU dataset: procedurally generates clear 3-band LISS-IV-like
    scenes (fractal terrain + per-band spectral texture) **or** loads clear scenes
    from a directory, then injects synthetic clouds + shadows
    (:mod:`cloudremoval.data.synthetic_clouds`), synthesises a SAR proxy and a DEM,
    and assembles the ``SAMPLE`` dict. Needs **no external data** and runs on the
    minimal stack -- it drives the CPU smoke.
:class:`Sen12msCrDataset`
    Thin adapter for the SEN12MS-CR benchmark (S1 SAR + cloudy/clear S2). Reads a
    manifest/dir of GeoTIFFs via lazy ``rasterio``; documented, not required on the
    smoke path.
:class:`LissivNerDataset`
    Thin adapter for our own Bhoonidhi LISS-IV NER tiles (per-band GeoTIFFs +
    ``BAND_META.txt``). Lazy ``rasterio``; documented, not on the smoke path.

Factories
---------
``build_dataset(cfg, split="train")``
    Dispatch on ``cfg.data.name`` (or a bare ``DataConfig``); returns a torch
    ``Dataset``. Matches both the B1 task signature ``build_dataset(cfg_data)`` and
    the BUILD_PLAN ``build_dataset(cfg, split)`` form.
``build_dataloader(cfg, split)`` / ``collate_sample(batch)`` / ``make_synthetic_sample(cfg, rng)``
    Per BUILD_PLAN §3.4 -- the collate pads variable ``T`` with a ``temporal_mask``.

The ``CloudRemovalDataset`` alias points at the synthetic dataset so the
BUILD_PLAN §3.4 type name resolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cloudremoval.data import preprocess as pp
from cloudremoval.data.synthetic_clouds import CloudSimConfig, CloudSimulator
from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.utils.data import DataLoader

    from cloudremoval.config import Config, DataConfig

__all__ = [
    "SyntheticCloudRemovalDataset",
    "Sen12msCrDataset",
    "LissivNerDataset",
    "CloudRemovalDataset",
    "build_dataset",
    "build_dataloader",
    "collate_sample",
    "make_synthetic_sample",
]

_log = get_logger(__name__)
_EPS = 1e-6
_C = 3  # Green, Red, NIR (BUILD_PLAN §2)


# --------------------------------------------------------------------------- #
# Config helpers (accept either a full Config or a DataConfig)
# --------------------------------------------------------------------------- #
def _as_data_cfg(cfg: Config | DataConfig) -> DataConfig:
    """Return the ``data`` section whether given a ``Config`` or a ``DataConfig``."""
    return getattr(cfg, "data", cfg)


# --------------------------------------------------------------------------- #
# Synthetic scene generation (clear 3-band LISS-IV-like tile + SAR + DEM)
# --------------------------------------------------------------------------- #
def _fractal_terrain(hw: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A smooth low-frequency height/structure field in ``[0, 1]`` (``[H, W]``)."""
    from cloudremoval.data.synthetic_clouds import fractal_noise_2d

    return fractal_noise_2d(hw, rng, octaves=5, base_cells=2, persistence=0.6)


def _synthetic_clear_scene(
    hw: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Procedurally generate a plausible clear 3-band reflectance tile + a DEM.

    Builds a low-frequency terrain base, derives an NDVI-like vegetation field,
    and maps both into physically-ordered Green/Red/NIR reflectances (vegetation:
    high NIR, low Red; bare/built: flatter spectrum) plus fine per-band texture.

    Args:
        hw: Target ``(H, W)``.
        rng: NumPy random generator (determinism).

    Returns:
        ``(clear[3, H, W] in [0, 1], dem[1, H, W] normalized)``.
    """
    from cloudremoval.data.synthetic_clouds import fractal_noise_2d

    h, w = hw
    terrain = _fractal_terrain(hw, rng)  # elevation proxy in [0, 1]
    veg = fractal_noise_2d(hw, rng, octaves=4, base_cells=4, persistence=0.55)
    texture = fractal_noise_2d(hw, rng, octaves=6, base_cells=8, persistence=0.5)

    # Vegetation fraction modulated by (lower) elevation; tea/forest in valleys.
    veg_frac = np.clip(veg * (1.0 - 0.4 * terrain), 0.0, 1.0)

    # Reflectances (rough but physically ordered for NER land cover).
    green = 0.08 + 0.10 * veg_frac + 0.06 * texture
    red = 0.07 + 0.05 * (1.0 - veg_frac) + 0.05 * texture
    nir = 0.18 + 0.45 * veg_frac + 0.08 * texture
    clear = np.stack([green, red, nir], axis=0).astype(np.float32)
    clear = np.clip(clear + 0.02 * (terrain[None] - 0.5), 0.0, 1.0)

    dem = ((terrain - terrain.mean()) / (terrain.std() + _EPS)).astype(np.float32)[None, ...]
    return clear.astype(np.float32), dem


def _synthetic_sar(clear: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthesise a Sentinel-1-like VV/VH SAR proxy correlated with structure.

    SAR backscatter tracks surface roughness/structure (it sees through cloud,
    ``research/06`` §D.5), so we derive it from edges of the clear scene plus
    multiplicative speckle. Output is normalised, shaped ``[2, H, W]``.
    """
    nir = clear[pp.NIR_IDX]
    gy, gx = np.gradient(nir.astype(np.float32))
    edges = np.sqrt(gy**2 + gx**2)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + _EPS)

    base = 0.4 * edges + 0.3 * nir
    # Speckle (exponential-ish multiplicative noise).
    speckle_vv = rng.gamma(shape=4.0, scale=0.25, size=nir.shape).astype(np.float32)
    speckle_vh = rng.gamma(shape=3.0, scale=0.25, size=nir.shape).astype(np.float32)
    vv = base * speckle_vv
    vh = 0.6 * base * speckle_vh
    sar = np.stack([vv, vh], axis=0).astype(np.float32)
    # Per-channel normalise to ~[0, 1].
    for i in range(2):
        ch = sar[i]
        sar[i] = (ch - ch.min()) / (ch.max() - ch.min() + _EPS)
    return sar.astype(np.float32)


def make_synthetic_sample(cfg: Config | DataConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Build one deterministic ``SAMPLE`` dict (§2) from synthetic LISS-IV data.

    Generates a clear 3-band tile + DEM, a SAR proxy, optional temporal clear
    references, injects synthetic clouds/shadows, and fills ``meta`` (including
    ``band_stats`` and presence flags). Absent modalities are **zero-filled**, never
    missing, per the contract.

    Args:
        cfg: A :class:`~cloudremoval.config.Config` or its ``data`` section.
        rng: A seeded NumPy generator (per-index determinism is the caller's job).

    Returns:
        The ``SAMPLE`` dict with ``float32`` channel-first tensors.
    """
    data_cfg = _as_data_cfg(cfg)
    size = int(getattr(data_cfg, "tile_size", 64))
    hw = (size, size)
    norm = str(getattr(data_cfg, "norm", "zscore"))
    cloud_mode = str(getattr(data_cfg, "cloud_mode", "mixed"))
    use_sar = bool(getattr(data_cfg, "use_sar", True))
    use_dem = bool(getattr(data_cfg, "use_dem", True))
    use_temporal = bool(getattr(data_cfg, "use_temporal", False))
    max_temporal = int(getattr(data_cfg, "max_temporal", 3))

    clear, dem = _synthetic_clear_scene(hw, rng)

    # Sun geometry (NER ~10:30 AM descending pass; plausible azimuth/elevation).
    sun_elev = float(rng.uniform(35.0, 60.0))
    sun_az = float(rng.uniform(120.0, 160.0))

    sim = CloudSimulator(CloudSimConfig(cloud_mode=cloud_mode, coverage=0.35))
    cloudy_refl, cloud_mask, shadow_mask = sim(
        clear, sun_azimuth=sun_az, sun_elevation=sun_elev, rng=rng
    )

    # Temporal references: extra clear scenes (nearest-clear-pass priors).
    t = int(rng.integers(1, max_temporal + 1)) if use_temporal and max_temporal > 0 else 0
    temporal_list: list[np.ndarray] = []
    for _ in range(t):
        ref_clear, _ = _synthetic_clear_scene(hw, rng)
        temporal_list.append(ref_clear)

    sar = _synthetic_sar(clear, rng) if use_sar else np.zeros((2, size, size), np.float32)
    dem_arr = dem if use_dem else np.zeros((1, size, size), np.float32)

    # Per-band normalization (store stats in meta so train == inference).
    clear_n, stats = pp.normalize_bands(clear, norm=norm)
    cloudy_n = pp.normalize(cloudy_refl, stats, norm=norm)
    temporal_n = [pp.normalize(r, stats, norm=norm) for r in temporal_list]

    if temporal_n:
        temporal_t = torch.from_numpy(np.stack(temporal_n, axis=0).astype(np.float32))
    else:
        temporal_t = torch.zeros(0, _C, size, size, dtype=torch.float32)

    cloud_frac = float(cloud_mask.mean())
    cloud_type = "thin" if cloud_mode == "thin" else "thick" if cloud_mode == "thick" else "mixed"

    sample: dict[str, Any] = {
        "optical_cloudy": pp.to_tensor(cloudy_n),
        "optical_clear": pp.to_tensor(clear_n),
        "cloud_mask": torch.from_numpy(cloud_mask.astype(np.float32)),
        "shadow_mask": torch.from_numpy(shadow_mask.astype(np.float32)),
        "sar": pp.to_tensor(sar),
        "dem": pp.to_tensor(dem_arr),
        "temporal_refs": temporal_t,
        "meta": {
            "scene_id": f"synthetic_{int(rng.integers(0, 1_000_000)):06d}",
            "date": "2024-01-15",
            "sun_elevation": sun_elev,
            "sun_azimuth": sun_az,
            "crs": "EPSG:32646",
            "transform": (5.8, 0.0, 0.0, 0.0, -5.8, 0.0),
            "band_stats": stats,
            "norm": norm,
            "cloud_type": cloud_type,
            "cloud_fraction": cloud_frac,
            "has_target": True,
            "has_sar": use_sar,
            "has_dem": use_dem,
            "has_temporal": t > 0,
        },
    }
    return sample


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
class SyntheticCloudRemovalDataset(Dataset):
    """Fully-offline synthetic cloud-removal dataset returning the ``SAMPLE`` dict.

    Either procedurally generates clear scenes (default) or, if ``cfg.data.root``
    points at a directory of clear images (``.npy`` / GeoTIFF), loads those as the
    clear targets; in both cases synthetic clouds + shadows + a SAR proxy + DEM are
    added. Determinism: sample ``idx`` is seeded from ``(seed, split, idx)`` so the
    dataset is reproducible across epochs and workers.

    Args:
        cfg: A :class:`~cloudremoval.config.Config` or its ``data`` section.
        split: ``"train"`` / ``"val"`` / ``"test"`` (affects the seed namespace and
            the count).
    """

    def __init__(self, cfg: Config | DataConfig, split: str = "train") -> None:
        self.data_cfg = _as_data_cfg(cfg)
        self.split = split
        self.size = int(getattr(self.data_cfg, "tile_size", 64))
        self._base_seed = _split_seed(split)
        self._n = int(getattr(self.data_cfg, "synthetic_n", 32))
        if split in ("val", "test"):
            self._n = max(2, self._n // 4)

        root = getattr(self.data_cfg, "root", None)
        self._clear_paths: list[Path] = []
        if root is not None:
            self._clear_paths = _list_clear_files(Path(root))
            if self._clear_paths:
                self._n = len(self._clear_paths)
                _log.info("SyntheticCloudRemovalDataset: %d clear files in %s", self._n, root)
            else:
                _log.info("no clear files under %s; generating scenes procedurally", root)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self._n
        rng = np.random.default_rng(self._base_seed + idx)
        if self._clear_paths:
            return self._sample_from_file(self._clear_paths[idx], rng)
        return make_synthetic_sample(self.data_cfg, rng)

    def _sample_from_file(self, path: Path, rng: np.random.Generator) -> dict[str, Any]:
        """Load a clear scene from disk and clothe it with clouds/SAR/DEM/meta."""
        clear = _load_clear_array(path, self.size)
        # Reuse the synthetic machinery with the loaded clear as the target.
        data_cfg = self.data_cfg
        norm = str(getattr(data_cfg, "norm", "zscore"))
        cloud_mode = str(getattr(data_cfg, "cloud_mode", "mixed"))
        sun_elev = float(rng.uniform(35.0, 60.0))
        sun_az = float(rng.uniform(120.0, 160.0))
        sim = CloudSimulator(CloudSimConfig(cloud_mode=cloud_mode, coverage=0.35))
        cloudy_refl, cloud_mask, shadow_mask = sim(
            clear, sun_azimuth=sun_az, sun_elevation=sun_elev, rng=rng
        )
        sar = (
            _synthetic_sar(clear, rng)
            if bool(getattr(data_cfg, "use_sar", True))
            else np.zeros((2, self.size, self.size), np.float32)
        )
        dem = np.zeros((1, self.size, self.size), np.float32)
        clear_n, stats = pp.normalize_bands(clear, norm=norm)
        cloudy_n = pp.normalize(cloudy_refl, stats, norm=norm)
        return {
            "optical_cloudy": pp.to_tensor(cloudy_n),
            "optical_clear": pp.to_tensor(clear_n),
            "cloud_mask": torch.from_numpy(cloud_mask.astype(np.float32)),
            "shadow_mask": torch.from_numpy(shadow_mask.astype(np.float32)),
            "sar": pp.to_tensor(sar),
            "dem": pp.to_tensor(dem),
            "temporal_refs": torch.zeros(0, _C, self.size, self.size, dtype=torch.float32),
            "meta": {
                "scene_id": path.stem,
                "date": "unknown",
                "sun_elevation": sun_elev,
                "sun_azimuth": sun_az,
                "crs": "EPSG:32646",
                "transform": (5.8, 0.0, 0.0, 0.0, -5.8, 0.0),
                "band_stats": stats,
                "norm": norm,
                "cloud_type": cloud_mode,
                "cloud_fraction": float(cloud_mask.mean()),
                "has_target": True,
                "has_sar": bool(getattr(data_cfg, "use_sar", True)),
                "has_dem": False,
                "has_temporal": False,
            },
        }


# --------------------------------------------------------------------------- #
# Real-source adapters (documented; lazy rasterio; not on the smoke path)
# --------------------------------------------------------------------------- #
class Sen12msCrDataset(Dataset):
    """Adapter for SEN12MS-CR (S1 SAR + cloudy/clear S2 triplets).

    Expects ``cfg.data.root`` to contain a manifest or directory of co-located
    GeoTIFF triplets ``{s1, s2_cloudy, s2_clear}``. Sentinel-2 bands ``B3/B4/B8``
    are mapped to LISS-IV ``Green/Red/NIR`` (``research/06`` §B.3) and SAR to the
    ``sar`` channel. GeoTIFF reads use lazy ``rasterio``; this class is **not**
    exercised on the CPU smoke (requires the dataset + ``rasterio``).

    Args:
        cfg: Config (or ``DataConfig``) whose ``root`` locates the dataset.
        split: Dataset split as named in the manifest.
    """

    #: Indices of S2 bands B3 (Green), B4 (Red), B8 (NIR) within a 13-band stack.
    S2_BAND_INDICES = (2, 3, 7)

    def __init__(self, cfg: Config | DataConfig, split: str = "train") -> None:
        self.data_cfg = _as_data_cfg(cfg)
        self.split = split
        self.size = int(getattr(self.data_cfg, "tile_size", 256))
        root = getattr(self.data_cfg, "root", None)
        if root is None:
            raise ValueError("Sen12msCrDataset requires cfg.data.root to be set")
        self.root = Path(root)
        self._records = _read_triplet_manifest(self.root, split)
        _log.info("Sen12msCrDataset: %d records (split=%s)", len(self._records), split)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from cloudremoval.utils.io import read_array

        rec = self._records[idx]
        norm = str(getattr(self.data_cfg, "norm", "zscore"))
        cloudy_full = read_array(rec["s2_cloudy"]).astype(np.float32)
        clear_full = read_array(rec["s2_clear"]).astype(np.float32)
        cloudy = cloudy_full[list(self.S2_BAND_INDICES)]
        clear = clear_full[list(self.S2_BAND_INDICES)]
        sar = (
            read_array(rec["s1"]).astype(np.float32)
            if "s1" in rec and bool(getattr(self.data_cfg, "use_sar", True))
            else np.zeros((2, *clear.shape[1:]), np.float32)
        )

        clear_n, stats = pp.normalize_bands(clear, norm=norm)
        cloudy_n = pp.normalize(cloudy, stats, norm=norm)
        from cloudremoval.data.masking import detect_clouds

        cloud_mask = detect_clouds(cloudy).numpy() if torch.is_tensor(cloudy) else None
        if cloud_mask is None:
            cm = detect_clouds(cloudy)
            cloud_mask = cm if isinstance(cm, np.ndarray) else cm.numpy()
        h, w = clear.shape[1:]
        return {
            "optical_cloudy": pp.to_tensor(cloudy_n),
            "optical_clear": pp.to_tensor(clear_n),
            "cloud_mask": torch.from_numpy(np.asarray(cloud_mask, np.float32)),
            "shadow_mask": torch.zeros(1, h, w, dtype=torch.float32),
            "sar": pp.to_tensor(sar),
            "dem": torch.zeros(1, h, w, dtype=torch.float32),
            "temporal_refs": torch.zeros(0, _C, h, w, dtype=torch.float32),
            "meta": {
                "scene_id": rec.get("id", f"sen12mscr_{idx}"),
                "norm": norm,
                "band_stats": stats,
                "cloud_type": "mixed",
                "has_target": True,
                "has_sar": "s1" in rec,
                "has_dem": False,
                "has_temporal": False,
            },
        }


class LissivNerDataset(Dataset):
    """Adapter for our own Bhoonidhi LISS-IV NER tiles (per-band GeoTIFFs).

    Expects ``cfg.data.root`` to contain scene directories, each with per-band
    GeoTIFFs (``BAND2/3/4.tif``) and a ``BAND_META.txt`` (parsed via
    :mod:`cloudremoval.data.sources.bhoonidhi`). DN is converted to TOA reflectance
    (:mod:`cloudremoval.data.preprocess`) using metadata sun-elevation, then either
    paired with a real clear scene (if the manifest provides one) or synthetically
    clouded. Reads use lazy ``rasterio``; **not** on the smoke path.

    Args:
        cfg: Config (or ``DataConfig``) whose ``root`` locates the LISS-IV scenes.
        split: Split name.
    """

    def __init__(self, cfg: Config | DataConfig, split: str = "train") -> None:
        self.data_cfg = _as_data_cfg(cfg)
        self.split = split
        self.size = int(getattr(self.data_cfg, "tile_size", 256))
        root = getattr(self.data_cfg, "root", None)
        if root is None:
            raise ValueError("LissivNerDataset requires cfg.data.root to be set")
        self.root = Path(root)
        self._scenes = _list_lissiv_scenes(self.root)
        _log.info("LissivNerDataset: %d scenes under %s", len(self._scenes), self.root)

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from cloudremoval.data.sources.bhoonidhi import read_lissiv_scene

        scene_dir = self._scenes[idx]
        norm = str(getattr(self.data_cfg, "norm", "zscore"))
        cloud_mode = str(getattr(self.data_cfg, "cloud_mode", "mixed"))
        # Returns reflectance [3, H, W] and parsed metadata.
        clear, meta = read_lissiv_scene(scene_dir)
        rng = np.random.default_rng(abs(hash(str(scene_dir))) % (2**32))
        sim = CloudSimulator(CloudSimConfig(cloud_mode=cloud_mode, coverage=0.35))
        sun_elev = float(meta.get("sun_elevation", 45.0))
        sun_az = float(meta.get("sun_azimuth", 135.0))
        cloudy_refl, cloud_mask, shadow_mask = sim(
            clear, sun_azimuth=sun_az, sun_elevation=sun_elev, rng=rng
        )
        clear_n, stats = pp.normalize_bands(clear, norm=norm)
        cloudy_n = pp.normalize(cloudy_refl, stats, norm=norm)
        h, w = clear.shape[1:]
        return {
            "optical_cloudy": pp.to_tensor(cloudy_n),
            "optical_clear": pp.to_tensor(clear_n),
            "cloud_mask": torch.from_numpy(cloud_mask.astype(np.float32)),
            "shadow_mask": torch.from_numpy(shadow_mask.astype(np.float32)),
            "sar": torch.zeros(2, h, w, dtype=torch.float32),
            "dem": torch.zeros(1, h, w, dtype=torch.float32),
            "temporal_refs": torch.zeros(0, _C, h, w, dtype=torch.float32),
            "meta": {
                "scene_id": meta.get("scene_id", scene_dir.name),
                "date": meta.get("date", "unknown"),
                "sun_elevation": sun_elev,
                "sun_azimuth": sun_az,
                "crs": meta.get("crs", "unknown"),
                "band_stats": stats,
                "norm": norm,
                "cloud_type": cloud_mode,
                "has_target": True,
                "has_sar": False,
                "has_dem": False,
                "has_temporal": False,
            },
        }


#: BUILD_PLAN §3.4 type name; the synthetic dataset is the canonical implementation.
CloudRemovalDataset = SyntheticCloudRemovalDataset


# --------------------------------------------------------------------------- #
# Factories (BUILD_PLAN §3.4)
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "synthetic": SyntheticCloudRemovalDataset,
    "sen12mscr": Sen12msCrDataset,
    "sen12ms_cr": Sen12msCrDataset,
    "lissiv_ner": LissivNerDataset,
    "lissiv": LissivNerDataset,
}


def build_dataset(cfg: Config | DataConfig, split: str = "train") -> Dataset:
    """Construct the dataset named ``cfg.data.name`` (BUILD_PLAN §3.4).

    Accepts either a full :class:`~cloudremoval.config.Config` or a bare
    :class:`~cloudremoval.config.DataConfig` (so the B1 ``build_dataset(cfg_data)``
    form works too). Dispatch is on the dataset ``name``.

    Args:
        cfg: Config or DataConfig selecting the dataset and its parameters.
        split: ``"train"`` / ``"val"`` / ``"test"``.

    Returns:
        A torch :class:`~torch.utils.data.Dataset` yielding ``SAMPLE`` dicts.

    Raises:
        KeyError: If ``name`` is not a known dataset.
    """
    data_cfg = _as_data_cfg(cfg)
    name = str(getattr(data_cfg, "name", "synthetic")).lower()
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. Known: {sorted(set(_REGISTRY))}")
    return _REGISTRY[name](data_cfg, split=split)


def collate_sample(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate ``SAMPLE`` dicts into a batch, padding variable ``T`` (§2 collate).

    Stacks fixed-shape tensors on dim 0. ``temporal_refs`` are padded to the max
    ``T`` in the batch with zeros and a ``temporal_mask[B, T]`` (1 = real frame) is
    added. ``meta`` becomes a ``list[dict]``.

    Args:
        batch: List of ``SAMPLE`` dicts.

    Returns:
        A batched dict with tensors on dim 0, ``meta`` a list, plus
        ``temporal_mask`` ``[B, T_max]``.
    """
    out: dict[str, Any] = {}
    fixed_keys = [
        "optical_cloudy",
        "optical_clear",
        "cloud_mask",
        "shadow_mask",
        "sar",
        "dem",
    ]
    for key in fixed_keys:
        out[key] = torch.stack([s[key] for s in batch], dim=0)

    # Variable-length temporal refs -> pad to max T with a mask.
    t_list = [s["temporal_refs"] for s in batch]
    t_max = max((t.shape[0] for t in t_list), default=0)
    b = len(batch)
    if t_max == 0:
        c, h, w = out["optical_cloudy"].shape[1:]
        out["temporal_refs"] = torch.zeros(b, 0, c, h, w, dtype=torch.float32)
        out["temporal_mask"] = torch.zeros(b, 0, dtype=torch.float32)
    else:
        c, h, w = t_list[0].shape[1:] if t_list[0].shape[0] > 0 else out["optical_cloudy"].shape[1:]
        padded = torch.zeros(b, t_max, c, h, w, dtype=torch.float32)
        mask = torch.zeros(b, t_max, dtype=torch.float32)
        for i, refs in enumerate(t_list):
            t = refs.shape[0]
            if t > 0:
                padded[i, :t] = refs
                mask[i, :t] = 1.0
        out["temporal_refs"] = padded
        out["temporal_mask"] = mask

    out["meta"] = [s["meta"] for s in batch]
    return out


def build_dataloader(cfg: Config | DataConfig, split: str = "train") -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` over the configured dataset.

    Uses :func:`collate_sample`, shuffles on ``train`` only, and honours
    ``cfg.data.batch_size`` / ``num_workers``. Applies training augmentation
    (:func:`cloudremoval.data.transforms.build_transforms`) lazily by wrapping the
    dataset so transforms run in the worker.

    Args:
        cfg: Config or DataConfig.
        split: ``"train"`` / ``"val"`` / ``"test"``.

    Returns:
        A configured ``DataLoader`` yielding batched ``SAMPLE`` dicts.
    """
    from torch.utils.data import DataLoader

    from cloudremoval.data.transforms import build_transforms

    data_cfg = _as_data_cfg(cfg)
    base = build_dataset(data_cfg, split=split)
    transform = build_transforms(data_cfg, split=split)
    dataset: Dataset = _TransformedDataset(base, transform)

    return DataLoader(
        dataset,
        batch_size=int(getattr(data_cfg, "batch_size", 4)),
        shuffle=(split == "train"),
        num_workers=int(getattr(data_cfg, "num_workers", 0)),
        collate_fn=collate_sample,
        drop_last=False,
    )


class _TransformedDataset(Dataset):
    """Wrap a dataset so a ``transform(sample) -> sample`` runs per ``__getitem__``."""

    def __init__(self, base: Dataset, transform: Any) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.transform(self.base[idx])


# --------------------------------------------------------------------------- #
# Small I/O helpers
# --------------------------------------------------------------------------- #
def _split_seed(split: str) -> int:
    """Deterministic per-split seed namespace."""
    return {"train": 0, "val": 1_000_000, "test": 2_000_000}.get(split, 0)


def _list_clear_files(root: Path) -> list[Path]:
    """List ``.npy``/GeoTIFF clear scenes under ``root`` (sorted)."""
    if not root.exists():
        return []
    exts = {".npy", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _load_clear_array(path: Path, size: int) -> np.ndarray:
    """Load a clear scene to a ``[3, size, size]`` reflectance array in ``[0, 1]``."""
    from cloudremoval.utils.io import read_array

    arr = np.asarray(read_array(path), dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    if arr.shape[0] < _C:
        arr = np.concatenate([arr, arr[-1:].repeat(_C - arr.shape[0], axis=0)], axis=0)
    arr = arr[:_C]
    arr = pp.resample_array(arr, (size, size), order=1)
    mx = float(arr.max())
    if mx > 1.5:  # looks like DN/255 scale; rescale to [0, 1]
        arr = arr / (mx + _EPS)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _read_triplet_manifest(root: Path, split: str) -> list[dict[str, str]]:
    """Read a SEN12MS-CR triplet manifest (JSON list) or glob a directory.

    A ``manifest.json`` (optionally per-split ``manifest_<split>.json``) with a
    list of ``{id, s1, s2_cloudy, s2_clear}`` is preferred; otherwise paths are
    inferred from sibling files. Documented convention -- not run on the smoke path.
    """
    import json

    for cand in (root / f"manifest_{split}.json", root / "manifest.json"):
        if cand.exists():
            with cand.open("r", encoding="utf-8") as fh:
                records = json.load(fh)
            return [
                {k: str((root / v) if not Path(v).is_absolute() else v) for k, v in r.items()}
                for r in records
            ]
    _log.warning("no SEN12MS-CR manifest under %s; returning empty record list", root)
    return []


def _list_lissiv_scenes(root: Path) -> list[Path]:
    """List LISS-IV scene directories (those containing a ``BAND_META.txt``)."""
    if not root.exists():
        return []
    scenes = [p.parent for p in root.rglob("BAND_META.txt")]
    if scenes:
        return sorted(set(scenes))
    # Fallback: treat immediate subdirectories as scenes.
    return sorted(p for p in root.iterdir() if p.is_dir())
