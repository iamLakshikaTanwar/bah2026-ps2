"""Typed configuration schema for the ``cloudremoval`` framework.

This module owns the **frozen pydantic v2 configuration contract** defined in
``docs/BUILD_PLAN.md`` §3.3. Every stage (data, model, training, evaluation,
inference, serving) reads its parameters from the corresponding section of
:class:`Config`. The schema is intentionally flat and permissive so that a
single YAML file can describe both the CPU-smoke regime and a full GPU run,
switching only values (never structure).

Loaders
-------
``Config.from_yaml`` / :func:`load_config`
    Parse a YAML file into a validated :class:`Config`, supporting an optional
    ``base:`` include (relative to the file) that is deep-merged underneath the
    current document, plus an ``overrides`` mapping deep-merged on top.
``Config.to_yaml`` / :func:`save_config`
    Serialise a :class:`Config` back to YAML (paths rendered as strings).

Only the standard library, ``pyyaml`` and ``pydantic`` are imported here, so the
module is importable on the minimal CPU-smoke stack.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "TrainConfig",
    "EvalConfig",
    "InferConfig",
    "ServeConfig",
    "Config",
    "load_config",
    "save_config",
    "deep_merge",
]


# --------------------------------------------------------------------------- #
# Section schemas (BUILD_PLAN §3.3 — frozen)
# --------------------------------------------------------------------------- #
class DataConfig(BaseModel):
    """Dataset / data-engine configuration (consumed by ``data`` package)."""

    model_config = ConfigDict(extra="allow")

    name: str = "synthetic"  # synthetic | lissiv_ner | sen12mscr
    root: Path | None = None
    tile_size: int = 128
    halo: int = 16
    bands: list[str] = Field(default_factory=lambda: ["green", "red", "nir"])
    use_sar: bool = True
    use_dem: bool = True
    use_temporal: bool = False
    max_temporal: int = 3
    norm: str = "zscore"  # zscore | percentile
    batch_size: int = 4
    num_workers: int = 0
    synthetic_n: int = 32  # number of synthetic samples (CPU smoke)
    cloud_mode: str = "mixed"  # thin | thick | mixed


class ModelConfig(BaseModel):
    """Model configuration. Concrete models receive *this* object in ``__init__``.

    Fields outside a model's concern (e.g. diffusion timesteps for the UNet) are
    simply ignored by that model. ``extra`` carries model-specific knobs that do
    not warrant a first-class field.
    """

    model_config = ConfigDict(extra="allow")

    name: str = "unet"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 3
    use_sar: bool = True  # whether this instance ingests SAR
    use_dem: bool = False
    use_temporal: bool = False
    uncertainty: bool = False  # add aleatoric head
    # diffusion-specific (ignored by others)
    timesteps: int = 1000
    sample_steps: int = 5  # DDIM steps at inference (cpu smoke = small)
    mean_reverting: bool = False
    # gan-specific (ignored by others)
    gan_lambda_l1: float = 100.0
    extra: dict[str, Any] = Field(default_factory=dict)


class LossConfig(BaseModel):
    """Weights for the shared loss terms summed by ``models.losses.LossBundle``.

    A weight of ``0.0`` disables that term (and skips its computation).
    """

    model_config = ConfigDict(extra="allow")

    l1: float = 1.0
    carl: float = 0.0  # cloud-adaptive regularized L1 weight
    ssim: float = 0.0
    sam: float = 0.0  # spectral angle loss weight
    perceptual: float = 0.0
    adversarial: float = 0.0
    nll: float = 0.0  # uncertainty negative-log-likelihood
    tv: float = 0.0  # total-variation regulariser (extension of the contract)


class TrainConfig(BaseModel):
    """Training-loop configuration (consumed by ``training.trainer``)."""

    model_config = ConfigDict(extra="allow")

    epochs: int = 1
    max_steps: int | None = 4  # cpu smoke caps steps
    lr: float = 2e-4
    weight_decay: float = 0.0
    device: str = "cpu"  # cpu | cuda
    precision: str = "32"  # 32 | 16 | bf16
    use_lightning: bool = False
    grad_clip: float | None = 1.0
    log_every: int = 1
    ckpt_dir: Path = Path("outputs/ckpt")
    seed: int = 1337
    loss: LossConfig = Field(default_factory=LossConfig)


class EvalConfig(BaseModel):
    """Evaluation configuration (consumed by ``evaluation`` package)."""

    model_config = ConfigDict(extra="allow")

    masked: bool = True
    strata: list[str] = Field(default_factory=lambda: ["whole", "cloud", "shadow", "thin", "thick"])
    metrics: list[str] = Field(default_factory=lambda: ["psnr", "ssim", "sam", "ergas", "ndvi_mae"])
    out_dir: Path = Path("outputs/eval")


class InferConfig(BaseModel):
    """Tiled-inference configuration (consumed by ``inference`` package)."""

    model_config = ConfigDict(extra="allow")

    tile_size: int = 128
    halo: int = 16
    blend: str = "hann"  # hann | gaussian | none
    device: str = "cpu"
    backend: str = "torch"  # torch | onnx | tensorrt
    out_cog: Path = Path("outputs/recon.tif")
    write_uncertainty: bool = True


class ServeConfig(BaseModel):
    """Serving configuration (consumed by ``serving`` package)."""

    model_config = ConfigDict(extra="allow")

    host: str = "0.0.0.0"
    port: int = 8000
    cog_dir: Path = Path("outputs")
    cache: str = "memory"  # memory | redis
    redis_url: str | None = None


class Config(BaseModel):
    """Root configuration aggregating all stage sections.

    Construct directly for defaults (``Config()``), or load from YAML with
    :meth:`from_yaml` / :func:`load_config`.
    """

    model_config = ConfigDict(extra="allow")

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    # ----- loaders ------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
        """Load a :class:`Config` from ``path``.

        Supports an optional top-level ``base:`` key naming another YAML file
        (resolved relative to ``path``) whose contents are deep-merged
        *underneath* the current document, enabling ``cpu_smoke.yaml`` to extend
        ``base.yaml``. ``overrides`` (a nested mapping) is deep-merged on top of
        everything.

        Args:
            path: Path to the YAML configuration file.
            overrides: Optional nested mapping merged last (highest priority).

        Returns:
            A validated :class:`Config` instance.
        """
        merged = _load_yaml_with_base(Path(path))
        if overrides:
            merged = deep_merge(merged, overrides)
        return cls.model_validate(merged)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this config to ``path`` as YAML (paths -> strings)."""
        data = self.model_dump(mode="json")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Nested dicts are merged key-by-key; any non-dict value (or a key absent from
    ``base``) is taken from ``override``. ``base`` is not mutated.

    Args:
        base: The lower-priority mapping.
        override: The higher-priority mapping.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    """Read a YAML file and resolve a single optional ``base:`` include.

    The ``base`` key (if present) is popped and its referenced file loaded first
    (recursively), then the current document is merged on top. This keeps
    ``base.yaml`` as the shared substrate for ``cpu_smoke.yaml`` / ``gpu_full``.
    """
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"config root must be a mapping, got {type(doc)} in {path}")

    base_ref = doc.pop("base", None)
    if base_ref is None:
        return doc

    base_path = (path.parent / str(base_ref)).resolve()
    base_doc = _load_yaml_with_base(base_path)
    return deep_merge(base_doc, doc)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Functional alias for :meth:`Config.from_yaml` (BUILD_PLAN §3.3)."""
    return Config.from_yaml(path, overrides=overrides)


def save_config(cfg: Config, path: str | Path) -> None:
    """Functional alias for :meth:`Config.to_yaml` (BUILD_PLAN §3.3)."""
    cfg.to_yaml(path)
