# BUILD PLAN — Parallel Implementation Contract

**BAH 2026 PS2 · `cloudremoval`.** This is the **binding contract** for parallel implementation agents. Each builder sees ONLY this document, not each other's code. The interfaces below are the integration surface: code to them exactly and independent modules will snap together. Do not change a signature without it being changed here first.

**Golden rules**
1. **Scaffold builder (B0) runs FIRST and alone.** It creates all `__init__.py`, the shared base classes, registry, config schema, utils, and all root files. Every other builder imports from B0's modules and must not redefine them.
2. **Own only your files.** File ownership is disjoint (§4). Never edit a file you don't own. If you need a change in someone else's file, it's already specified here — if not, stop and report.
3. **Everything runs on CPU with synthetic data and tiny configs**, and scales to GPU by config only. No model or stage may be GPU-only. CI runs the CPU-smoke path.
4. **Code to the SAMPLE dict and `BaseCloudRemovalModel`** — trainer/evaluator/inference/serving never import a concrete model; they use the registry.
5. **Standards:** Python 3.10+, full type hints, **pydantic v2** configs, PyTorch (+ optional Lightning), structured logging (`utils.logging`), `pytest`, `ruff`+`black`, docstrings on every public function/class. Determinism via `utils.seed.seed_everything`.

---

## 1. Canonical Repo Layout (full tree — authoritative)

```
bah2026-ps2/
├── ARCHITECTURE.md                         # [exists]
├── README.md                               # B0
├── pyproject.toml                          # B0  (deps, ruff/black/pytest cfg, entry: cloudremoval=cloudremoval.cli:app)
├── requirements.txt                        # B0
├── Makefile                                # B0  (setup lint test smoke serve docker)
├── Dockerfile                              # B0
├── docker-compose.yml                      # B0  (api + redis [+ optional worker])
├── .gitignore                              # B0
├── .dockerignore                           # B0
├── .pre-commit-config.yaml                 # B0
├── .github/workflows/ci.yml                # B0  (ruff+black+pytest+cpu smoke train+benchmark)
├── configs/                                # B0 creates base.yaml,cpu_smoke.yaml,gpu_full.yaml + dirs; owners add model/data leafs
│   ├── base.yaml                           # B0
│   ├── cpu_smoke.yaml                      # B0
│   ├── gpu_full.yaml                       # B0
│   ├── data/synthetic.yaml                 # B1
│   ├── data/lissiv_ner.yaml                # B1
│   ├── data/sen12mscr.yaml                 # B1
│   ├── model/unet.yaml                     # B2
│   ├── model/dsen2cr.yaml                  # B2
│   ├── model/spagan.yaml                   # B2
│   ├── model/restormer.yaml                # B2
│   ├── model/diffusion.yaml                # B2
│   ├── model/uncertainty.yaml              # B2
│   └── serve/serve.yaml                    # B4
├── src/cloudremoval/
│   ├── __init__.py                         # B0
│   ├── config.py                           # B0  (pydantic Config + load_config/save_config)
│   ├── cli.py                              # B0  (Typer app; subcommands delegate to scripts/owners)
│   ├── data/
│   │   ├── __init__.py                     # B0
│   │   ├── preprocess.py                   # B1
│   │   ├── coregister.py                   # B1
│   │   ├── masking.py                      # B1
│   │   ├── synthetic_clouds.py             # B1
│   │   ├── datasets.py                     # B1
│   │   ├── transforms.py                   # B1
│   │   ├── cog_tiling.py                   # B1
│   │   └── sources/
│   │       ├── __init__.py                 # B0
│   │       ├── bhoonidhi.py                # B1
│   │       ├── gee.py                      # B1
│   │       ├── stac.py                     # B1
│   │       ├── sentinel.py                 # B1
│   │       └── dem.py                      # B1
│   ├── models/
│   │   ├── __init__.py                     # B0  (imports concrete models so registry self-populates)
│   │   ├── base.py                         # B0  (BaseCloudRemovalModel, ModelOutput)
│   │   ├── registry.py                     # B0  (register_model, build_model, list_models)
│   │   ├── losses.py                       # B0  (loss fns + LossBundle; signatures frozen here)
│   │   ├── unet.py                         # B2
│   │   ├── dsen2cr_fusion.py               # B2
│   │   ├── gan_spagan.py                   # B2
│   │   ├── transformer_restormer.py        # B2
│   │   ├── diffusion.py                    # B2
│   │   └── uncertainty.py                  # B2
│   ├── training/
│   │   ├── __init__.py                     # B0
│   │   ├── trainer.py                      # B3
│   │   ├── lightning_module.py             # B3
│   │   └── callbacks.py                    # B3
│   ├── evaluation/
│   │   ├── __init__.py                     # B0
│   │   ├── metrics.py                      # B3  (signatures frozen in §3.5)
│   │   ├── evaluator.py                    # B3
│   │   ├── benchmark.py                    # B3
│   │   └── report.py                       # B3
│   ├── inference/
│   │   ├── __init__.py                     # B0
│   │   ├── tiled.py                        # B4
│   │   ├── blending.py                     # B4
│   │   ├── postprocess.py                  # B4
│   │   └── cog_writer.py                   # B4
│   ├── serving/
│   │   ├── __init__.py                     # B0
│   │   ├── app.py                          # B4  (FastAPI)
│   │   ├── tiles.py                        # B4
│   │   └── schemas.py                      # B4
│   └── utils/
│       ├── __init__.py                     # B0
│       ├── geo.py                          # B0
│       ├── io.py                           # B0
│       ├── logging.py                      # B0
│       └── seed.py                         # B0
├── scripts/
│   ├── download.py                         # B1
│   ├── preprocess.py                       # B1
│   ├── simulate.py                         # B1
│   ├── train.py                            # B3
│   ├── eval.py                             # B3
│   ├── benchmark.py                        # B3
│   ├── infer.py                            # B4
│   └── serve.py                            # B4
├── tests/
│   ├── conftest.py                         # B0  (synthetic SAMPLE + tiny Config fixtures)
│   ├── test_smoke_pipeline.py              # B5  (end-to-end CPU: simulate→train1step→eval→infer)
│   ├── test_data.py                        # B5
│   ├── test_models.py                      # B5  (every registered model: forward+loss+predict shapes)
│   ├── test_metrics.py                     # B5
│   ├── test_inference.py                   # B5
│   └── test_serving.py                     # B5
├── docs/
│   ├── COMPARATIVE_ASSESSMENT.md           # [exists]
│   ├── BUILD_PLAN.md                       # [this file]
│   ├── DATA_CARD.md                        # B5
│   ├── MODEL_CARD.md                        # B5
│   └── API.md                              # B5
└── notebooks/                              # B5 (optional, not in CI)
```

---

## 2. The SAMPLE schema (the universal data contract)

A `Dataset.__getitem__` returns a `dict[str, Any]` with these keys. All image tensors are `torch.float32`, channel-first, reflectance-normalized per-band (stats in `meta`). Optional modalities that are absent are **zero-filled** tensors of the correct shape (never missing keys) so models can unconditionally index them; presence flags live in `meta`.

```python
# C = 3 (Green, Red, NIR), H = W = tile size, T = number of temporal refs (may be 0)
SAMPLE = {
    "optical_cloudy": Tensor,   # [C, H, W]  model input (cloud-contaminated)
    "optical_clear":  Tensor,   # [C, H, W]  target (zeros + meta["has_target"]=False at inference)
    "cloud_mask":     Tensor,   # [1, H, W]  1.0 = cloud, 0.0 = clear (float)
    "shadow_mask":    Tensor,   # [1, H, W]  1.0 = cloud-shadow
    "sar":            Tensor,   # [2, H, W]  VV, VH (zeros if absent; meta["has_sar"])
    "dem":            Tensor,   # [1, H, W]  normalized elevation/slope (zeros if absent; meta["has_dem"])
    "temporal_refs":  Tensor,   # [T, C, H, W] nearest-clear references (empty [0,C,H,W] if none)
    "meta":           dict,     # see below
}
# meta keys (all optional consumers tolerate missing): 
#   scene_id:str, date:str(ISO), sun_elevation:float, crs:str, transform:tuple(6),
#   band_stats:dict{"mean":[3],"std":[3]} | {"p2":[3],"p98":[3]}, norm:str("zscore"|"percentile"),
#   cloud_type:str("thin"|"thick"|"mixed"), has_target:bool, has_sar:bool, has_dem:bool, has_temporal:bool
```

A batch (from the default collate) stacks tensors on dim 0 → `optical_cloudy:[B,C,H,W]`, `temporal_refs:[B,T,C,H,W]`, and `meta` becomes a `list[dict]`. **B1 provides a `collate_sample` function** handling variable `T` (pad to max T in batch, with a `temporal_mask`), exported from `data/datasets.py`.

---

## 3. Frozen Interface Contracts (code against these exactly)

### 3.1 `models/base.py` (owned by B0)
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import Tensor, nn

@dataclass
class ModelOutput:
    """Standard return of every model's forward()."""
    reconstruction: Tensor                       # [B, C, H, W] predicted clear optical (normalized)
    uncertainty: Tensor | None = None            # [B, 1, H, W] per-pixel aleatoric std/var, if produced
    aux: dict[str, Tensor] = field(default_factory=dict)  # attention maps, discriminator logits, etc.

class BaseCloudRemovalModel(nn.Module, ABC):
    """All cloud-removal models implement this. Trainer/evaluator/inference use ONLY this API.

    Subclasses MUST:
      - accept a parsed ModelConfig (pydantic) in __init__,
      - register via @register_model("<name>"),
      - set self.name,
      - implement forward(sample) and loss(sample, output).
    predict() has a default (forward + denorm-safe passthrough) and may be overridden
    (e.g. diffusion overrides predict() to run DDIM sampling instead of a single forward).
    """
    name: str = "base"

    def __init__(self, cfg: "ModelConfig") -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, sample: dict[str, Any]) -> ModelOutput:
        """Differentiable forward used during training. Reads keys from the SAMPLE dict."""
        ...

    @abstractmethod
    def loss(self, sample: dict[str, Any], output: ModelOutput) -> "LossDict":
        """Return {'total': Tensor, '<name>': Tensor, ...}; 'total' is backpropagated."""
        ...

    @torch.no_grad()
    def predict(self, sample: dict[str, Any]) -> ModelOutput:
        """Inference entry point (no grad). Default = self.eval()+forward(). Override for samplers."""
        self.eval()
        return self.forward(sample)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

# Type alias used across the codebase
LossDict = dict[str, Tensor]   # MUST contain key "total"
```

GAN models with separate optimizers expose two extra optional hooks (B3's trainer checks with `hasattr`):
```python
    def generator_parameters(self) -> "Iterable[Tensor]": ...      # optional
    def discriminator_parameters(self) -> "Iterable[Tensor]": ...  # optional
    def discriminator_loss(self, sample, output) -> LossDict: ...  # optional (D step)
```

### 3.2 `models/registry.py` (owned by B0)
```python
from typing import Callable, Type
_REGISTRY: dict[str, Type["BaseCloudRemovalModel"]] = {}

def register_model(name: str) -> Callable[[Type], Type]:
    """Class decorator: @register_model('unet')."""
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco

def build_model(cfg: "Config") -> "BaseCloudRemovalModel":
    """Instantiate the model named cfg.model.name with cfg.model."""
    if cfg.model.name not in _REGISTRY:
        raise KeyError(f"unknown model '{cfg.model.name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.model.name](cfg.model)

def list_models() -> list[str]:
    return sorted(_REGISTRY)
```
`models/__init__.py` (B0) imports each concrete module (`from . import unet, dsen2cr_fusion, gan_spagan, transformer_restormer, diffusion, uncertainty`) so importing `cloudremoval.models` self-populates the registry. **If a concrete module is missing during early parallel work, B0 wraps each import in try/except ImportError + a debug log** so a partial repo still imports.

### 3.3 `config.py` — pydantic schema (owned by B0)
```python
from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path

class DataConfig(BaseModel):
    name: str = "synthetic"                 # synthetic | lissiv_ner | sen12mscr
    root: Path | None = None
    tile_size: int = 128
    halo: int = 16
    bands: list[str] = ["green", "red", "nir"]
    use_sar: bool = True
    use_dem: bool = True
    use_temporal: bool = False
    max_temporal: int = 3
    norm: str = "zscore"                    # zscore | percentile
    batch_size: int = 4
    num_workers: int = 0
    synthetic_n: int = 32                   # number of synthetic samples (CPU smoke)
    cloud_mode: str = "mixed"              # thin | thick | mixed

class ModelConfig(BaseModel):
    name: str = "unet"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 3
    use_sar: bool = True                    # whether this instance ingests SAR
    use_dem: bool = False
    use_temporal: bool = False
    uncertainty: bool = False               # add aleatoric head
    # diffusion-specific (ignored by others)
    timesteps: int = 1000
    sample_steps: int = 5                   # DDIM steps at inference (cpu smoke = small)
    mean_reverting: bool = False
    # gan-specific (ignored by others)
    gan_lambda_l1: float = 100.0
    extra: dict = Field(default_factory=dict)

class LossConfig(BaseModel):
    l1: float = 1.0
    carl: float = 0.0                       # cloud-adaptive regularized L1 weight
    ssim: float = 0.0
    sam: float = 0.0                        # spectral angle loss weight
    perceptual: float = 0.0
    adversarial: float = 0.0
    nll: float = 0.0                        # uncertainty negative-log-likelihood

class TrainConfig(BaseModel):
    epochs: int = 1
    max_steps: int | None = 4               # cpu smoke caps steps
    lr: float = 2e-4
    weight_decay: float = 0.0
    device: str = "cpu"                     # cpu | cuda
    precision: str = "32"                   # 32 | 16 | bf16
    use_lightning: bool = False
    grad_clip: float | None = 1.0
    log_every: int = 1
    ckpt_dir: Path = Path("outputs/ckpt")
    seed: int = 1337
    loss: LossConfig = LossConfig()

class EvalConfig(BaseModel):
    masked: bool = True
    strata: list[str] = ["whole", "cloud", "shadow", "thin", "thick"]
    metrics: list[str] = ["psnr", "ssim", "sam", "ergas", "ndvi_mae"]
    out_dir: Path = Path("outputs/eval")

class InferConfig(BaseModel):
    tile_size: int = 128
    halo: int = 16
    blend: str = "hann"                     # hann | gaussian | none
    device: str = "cpu"
    backend: str = "torch"                  # torch | onnx | tensorrt
    out_cog: Path = Path("outputs/recon.tif")
    write_uncertainty: bool = True

class ServeConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cog_dir: Path = Path("outputs")
    cache: str = "memory"                   # memory | redis
    redis_url: str | None = None

class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    infer: InferConfig = InferConfig()
    serve: ServeConfig = ServeConfig()

def load_config(path: str | Path, overrides: dict | None = None) -> Config: ...   # YAML -> Config, deep-merge overrides
def save_config(cfg: Config, path: str | Path) -> None: ...
```

### 3.4 Dataset contract (owned by B1, in `data/datasets.py`)
```python
from torch.utils.data import Dataset

class CloudRemovalDataset(Dataset):
    """Returns the SAMPLE dict (§2). MUST work with name='synthetic' fully offline on CPU."""
    def __init__(self, cfg: DataConfig, split: str = "train") -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...   # the SAMPLE dict

def build_dataset(cfg: DataConfig, split: str) -> CloudRemovalDataset: ...
def build_dataloader(cfg: DataConfig, split: str) -> "DataLoader": ...   # uses collate_sample
def collate_sample(batch: list[dict]) -> dict: ...                       # pads temporal_refs, lists meta
def make_synthetic_sample(cfg: DataConfig, rng) -> dict: ...             # deterministic fake LISS-IV tile
```
The **synthetic generator** must fabricate a plausible 3-band reflectance tile (smooth low-freq base + texture), a SAR pair, a DEM, inject clouds via `synthetic_clouds`, and set masks/meta — so the entire pipeline runs with zero external data.

### 3.5 Metric function signatures (owned by B3, in `evaluation/metrics.py`) — FROZEN
All take `pred, target` as `[B,C,H,W]` or `[C,H,W]` float tensors (normalized or reflectance — caller is consistent) and an optional boolean/float `mask` `[*,1,H,W]` selecting pixels (cloud region). Return a Python `float` (mean over selected pixels) unless noted.
```python
def psnr(pred, target, mask=None, data_range=1.0) -> float: ...
def ssim(pred, target, mask=None) -> float: ...            # per-band mean
def sam(pred, target, mask=None) -> float: ...             # spectral angle, degrees (lower better)
def ergas(pred, target, mask=None, ratio=1.0) -> float: ...
def ndvi_mae(pred, target, mask=None, nir_idx=2, red_idx=1) -> float: ...
def rmse(pred, target, mask=None) -> float: ...
def mae(pred, target, mask=None) -> float: ...
def sid(pred, target, mask=None) -> float: ...
def lpips(pred, target, mask=None, net="alex") -> float: ...   # RGB render; lazy-import, returns nan if unavailable
def per_band_bias(pred, target, mask=None) -> list[float]: ...
def compute_all(pred, target, masks: dict[str, "Tensor|None"], cfg: EvalConfig) -> dict[str, dict[str, float]]:
    """Returns {metric: {stratum: value}} for strata in cfg.strata (whole/cloud/shadow/thin/thick)."""
```
Metrics must **degrade gracefully**: missing optional deps (lpips/sewar) → return `float('nan')`, never crash. Prefer pure-NumPy/torch implementations for psnr/ssim/sam/ergas/ndvi so Tier-0 has zero heavy deps.

### 3.6 Loss function signatures (owned by B0, in `models/losses.py`) — FROZEN
Models import these; weights come from `LossConfig`. Each returns a scalar `Tensor`.
```python
def l1_loss(pred, target, mask=None) -> Tensor: ...
def carl_loss(pred, target, cloud_mask, clear_weight=0.1) -> Tensor: ...  # cloud-adaptive regularized L1 (DSen2-CR)
def ssim_loss(pred, target) -> Tensor: ...                                 # 1 - SSIM
def sam_loss(pred, target, mask=None) -> Tensor: ...                       # mean spectral angle (radians)
def perceptual_loss(pred, target) -> Tensor: ...                           # VGG features on RGB render; lazy
def adversarial_loss(disc_logits, is_real: bool) -> Tensor: ...            # BCE/hinge
def gaussian_nll_loss(pred, target, log_var, mask=None) -> Tensor: ...     # uncertainty head
class LossBundle:
    """Sums weighted losses from LossConfig into a LossDict with 'total'."""
    def __init__(self, cfg: LossConfig) -> None: ...
    def __call__(self, sample: dict, output: "ModelOutput") -> "LossDict": ...
```
`LossBundle` is the default a model's `.loss()` can delegate to; specialized models (GAN, diffusion) may compute their own and still must return a `LossDict` with `"total"`.

### 3.7 Trainer contract (owned by B3, in `training/trainer.py`)
```python
class Trainer:
    def __init__(self, cfg: Config) -> None: ...
    def fit(self, model: BaseCloudRemovalModel, train_loader, val_loader=None) -> dict: ...  # returns history
    def validate(self, model, val_loader) -> dict: ...
    def save_checkpoint(self, model, path) -> None: ...
    def load_checkpoint(self, model, path) -> None: ...
# Lightning path (optional, cfg.train.use_lightning): training/lightning_module.py wraps the same model+LossBundle.
```
The trainer must honor `cfg.train.max_steps` (cap for CPU smoke), detect GAN models via `hasattr(model,"discriminator_parameters")` and alternate G/D steps, and move the SAMPLE dict tensors to `cfg.train.device` via a shared `utils.io.move_to_device(sample, device)` (B0 provides).

### 3.8 Inference contract (owned by B4, in `inference/tiled.py`)
```python
class TiledInferenceEngine:
    def __init__(self, model: BaseCloudRemovalModel, cfg: InferConfig) -> None: ...
    def run_array(self, stack: dict[str, "np.ndarray"]) -> dict[str, "np.ndarray"]:
        """stack has same keys as SAMPLE (numpy, full-scene). Returns {'reconstruction','uncertainty'} full-scene."""
    def run_cog(self, input_paths: dict[str, str], out_path: str) -> str: ...
# blending.py: hann_window(tile, halo), gaussian_window(...), stitch(tiles, positions, weights) -> array
# postprocess.py: histogram_match(recon, reference, mask), per_band_bias_correct(...), to_reflectance(...)
# cog_writer.py: write_cog(array, profile, out_path, overviews=True, uncertainty=None) -> str ; write_stac_item(...)
```

### 3.9 Serving contract (owned by B4)
FastAPI app in `serving/app.py` exposing: `GET /health`, `POST /reconstruct` (body = `serving/schemas.py:ReconstructRequest{cog_url|scene_id, model, params}` → `ReconstructResponse{cog_url, stac_item, stats}`), and XYZ tiles `GET /tiles/{model}/{z}/{x}/{y}.png` via `serving/tiles.py` (rio-tiler over the output COG, Redis/memory cache keyed by content hash). Must boot and answer `/health` with `cache="memory"` and no GPU.

### 3.10 CLI surface (owned by B0 in `cli.py`; subcommands delegate to `scripts/`)
Single entry point `cloudremoval` (Typer). Each subcommand takes `--config PATH` and optional `--set key=value` overrides:
```
cloudremoval download   --config ... [--aoi KML] [--source bhoonidhi|stac|gee]
cloudremoval preprocess --config ...
cloudremoval simulate   --config ...        # synthetic-cloud paired generation
cloudremoval train      --config ...
cloudremoval eval       --config ... [--ckpt PATH]
cloudremoval benchmark  --config ...        # all registered models -> leaderboard
cloudremoval infer      --config ... [--ckpt PATH] [--input ...] [--out ...]
cloudremoval serve      --config ...
```
B0 implements `cli.py` as thin Typer commands that import and call the corresponding `scripts/<name>.main(cfg, **kw)`. Each `scripts/<name>.py` exposes `def main(cfg: Config, **kwargs) -> None`.

---

## 4. Work Breakdown — 6 Disjoint Builder Assignments

Dependency order: **B0 first (blocking)**, then B1/B2 in parallel, then B3, then B4, then B5. Each builder owns exactly the files listed; no overlap.

### B0 — Scaffold & Shared Contracts (RUNS FIRST, ALONE)
**Mission:** create the skeleton so all others compile against stable contracts. Implement the frozen interfaces fully (not stubs): `BaseCloudRemovalModel`, `ModelOutput`, registry, pydantic `Config` + `load_config/save_config`, all loss functions + `LossBundle`, utils, CLI wiring, root/build files, all `__init__.py`, and `tests/conftest.py` fixtures. Provide a working `make smoke` target that runs once `unet` exists (it may import-guard).
**Owns:**
- Root: `README.md`, `pyproject.toml`, `requirements.txt`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.gitignore`, `.dockerignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`
- `configs/base.yaml`, `configs/cpu_smoke.yaml`, `configs/gpu_full.yaml` (+ create `configs/data`, `configs/model`, `configs/serve` dirs with `.gitkeep`)
- `src/cloudremoval/__init__.py`, `config.py`, `cli.py`
- `src/cloudremoval/models/__init__.py`, `base.py`, `registry.py`, `losses.py`
- `src/cloudremoval/{data,data/sources,training,evaluation,inference,serving,utils}/__init__.py`
- `src/cloudremoval/utils/{geo,io,logging,seed}.py`
- `tests/conftest.py` (fixtures: `tiny_config`, `synthetic_sample`, `synthetic_batch`)
**Done when:** `pip install -e .`, `ruff`, `black --check`, `pytest tests/conftest.py`-importing tests, and `python -c "import cloudremoval; from cloudremoval.models.registry import list_models"` all succeed on CPU.

### B1 — Data Layer (depends on B0)
**Mission:** SAMPLE-producing datasets, synthetic-cloud engine, preprocessing, masking, co-registration, multi-source ingestion, COG tiling. Synthetic path must be fully offline/CPU; real sources are optional-dependency guarded.
**Owns:**
- `src/cloudremoval/data/preprocess.py` (DN→TOA→surface refl via Lmax/ESUN/sun-elev; per-band norm; SBAF/histogram-match helpers)
- `src/cloudremoval/data/coregister.py` (AROSICS wrapper + skimage/OpenCV phase-corr fallback)
- `src/cloudremoval/data/masking.py` (OmniCloudMask wrapper for G/R/NIR + DEM geometric shadow projection; threshold fallback)
- `src/cloudremoval/data/synthetic_clouds.py` (SatelliteCloudGenerator + CloudSEN12 copy-paste + Perlin fallback; thin/thick; returns cloud+shadow masks)
- `src/cloudremoval/data/datasets.py` (`CloudRemovalDataset`, `build_dataset`, `build_dataloader`, `collate_sample`, `make_synthetic_sample`)
- `src/cloudremoval/data/transforms.py` (Albumentations geometric aug; spectral aug minimal)
- `src/cloudremoval/data/cog_tiling.py` (windowed tiling with halo; per-band norm sidecar)
- `src/cloudremoval/data/sources/{bhoonidhi,gee,stac,sentinel,dem}.py` (loaders: Bhoonidhi ZIP/GeoTIFF+BAND_META & BIL fallback; STAC via pystac-client; GEE optional; S1/S2 fetch; DEM GLO-30)
- `configs/data/{synthetic,lissiv_ner,sen12mscr}.yaml`
- `scripts/{download,preprocess,simulate}.py` (each `def main(cfg, **kw)`)
**Done when:** `build_dataloader(DataConfig(name='synthetic'), 'train')` yields valid SAMPLE batches on CPU with no network; `cloudremoval simulate --config configs/cpu_smoke.yaml` writes paired tiles.

### B2 — Models (depends on B0)
**Mission:** the six registered models, each subclassing `BaseCloudRemovalModel`, `@register_model`, CPU-runnable at tiny config, GPU-scalable. Use `LossBundle` (or own loss returning `LossDict`). Optional external-weight adapters guarded by try/except.
**Owns:**
- `src/cloudremoval/models/unet.py` → `@register_model("unet")` (encoder-decoder, optical-only, L1+SSIM via LossBundle)
- `src/cloudremoval/models/dsen2cr_fusion.py` → `@register_model("dsen2cr")` (file is `dsen2cr_fusion.py`; ResNet residual-correction, early SAR concat, CARL loss; optical-only if `use_sar=False`)
- `src/cloudremoval/models/gan_spagan.py` → `@register_model("spagan")` (file is `gan_spagan.py`; spatial-attention generator + PatchGAN; exposes generator/discriminator params + `discriminator_loss`)
- `src/cloudremoval/models/transformer_restormer.py` → `@register_model("restormer")` (file is `transformer_restormer.py`; MDTA channel-attention core + optional GLF-CR SAR cross-attention + Align-CR deformable align hook)
- `src/cloudremoval/models/diffusion.py` → `@register_model("diffusion")` (DDPM train / DDIM `predict()` override; SR3 concat conditioning; optional SAR channels; `mean_reverting` option; consistency-distill hook)
- `src/cloudremoval/models/uncertainty.py` → `@register_model("uncertainty")` (wraps a configurable backbone, adds aleatoric `log_var` head, NLL loss; populates `ModelOutput.uncertainty`)
- `configs/model/{unet,dsen2cr,spagan,restormer,diffusion,uncertainty}.yaml`

> **Naming convention:** registry key = short name (the `@register_model(...)` argument / config `model.name` / CLI selector, e.g. `dsen2cr`, `spagan`, `restormer`); the source file may use the longer descriptive name (`dsen2cr_fusion.py`, `gan_spagan.py`, `transformer_restormer.py`).

**Done when:** for every name in `list_models()`, `build_model` + one `forward(synthetic_batch)` + `.loss(...)['total'].backward()` + `.predict(...)` run on CPU and `ModelOutput.reconstruction` has shape `[B,3,H,W]`.

### B3 — Training & Evaluation (depends on B0, B1, B2)
**Mission:** the training loop (+ optional Lightning), callbacks, the full MVES metrics, evaluator with masked/stratified reporting, and the benchmark leaderboard runner.
**Owns:**
- `src/cloudremoval/training/trainer.py` (`Trainer`; honors `max_steps`; GAN-aware G/D alternation; device move)
- `src/cloudremoval/training/lightning_module.py` (optional `LightningModule` wrapper)
- `src/cloudremoval/training/callbacks.py` (checkpoint, early-stop, structured-logging, optional MLflow/W&B)
- `src/cloudremoval/evaluation/metrics.py` (all signatures in §3.5; pure torch/numpy Tier-0; lazy heavy deps)
- `src/cloudremoval/evaluation/evaluator.py` (`Evaluator.evaluate(model, loader) -> dict`; builds cloud/shadow/thin/thick masks from SAMPLE; whole+masked columns)
- `src/cloudremoval/evaluation/benchmark.py` (`run_benchmark(cfg) -> leaderboard`; iterates all registered models, identical seeds/tiling, adds compute/speed columns)
- `src/cloudremoval/evaluation/report.py` (leaderboard + qualitative pack: triptychs, error/ΔNDVI heatmaps, spectral profiles → CSV/Markdown/HTML)
- `scripts/{train,eval,benchmark}.py`
**Done when:** `cloudremoval train --config configs/cpu_smoke.yaml` trains `unet` for `max_steps` and checkpoints; `cloudremoval eval` prints masked+whole PSNR/SSIM/SAM/ERGAS/NDVI-MAE; `cloudremoval benchmark` emits a leaderboard over all registered models on CPU.

### B4 — Inference, Serving, Postprocess (depends on B0, B2; integrates B1 I/O)
**Mission:** tiled blended inference, spectral postprocess, COG/STAC writers, FastAPI serving with tiles + cache, ONNX/TensorRT export path (guarded), FAISS clear-reference retrieval (guarded).
**Owns:**
- `src/cloudremoval/inference/tiled.py` (`TiledInferenceEngine.run_array/run_cog`; sliding-window+halo; batched; `backend` torch/onnx/tensorrt)
- `src/cloudremoval/inference/blending.py` (`hann_window`, `gaussian_window`, `stitch`)
- `src/cloudremoval/inference/postprocess.py` (`histogram_match`, `per_band_bias_correct`, `to_reflectance`)
- `src/cloudremoval/inference/cog_writer.py` (`write_cog` w/ overviews + ZSTD/LERC + uncertainty band; `write_stac_item`)
- `src/cloudremoval/serving/app.py` (FastAPI: `/health`, `/reconstruct`, mount tiles)
- `src/cloudremoval/serving/tiles.py` (rio-tiler XYZ; H3/R-tree lookup; content-addressed + Redis/memory cache; optional FAISS retrieval)
- `src/cloudremoval/serving/schemas.py` (`ReconstructRequest`, `ReconstructResponse`, tile params)
- `configs/serve/serve.yaml`
- `scripts/{infer,serve}.py`
**Done when:** `cloudremoval infer --config configs/cpu_smoke.yaml --ckpt <unet.ckpt>` writes a valid COG (+uncertainty band) from a synthetic scene on CPU; `cloudremoval serve` boots, `/health` returns 200, `/reconstruct` returns a COG URL with `cache="memory"`.

### B5 — Tests, Docs, CI Polish (depends on ALL; runs last)
**Mission:** the test suite proving CPU-only end-to-end works, plus docs cards and CI hardening. May only ADD test/doc files and the CI yaml content B0 stubbed (coordinate: B5 fills test bodies, B0 owns the ci.yml file — B5 proposes its final content to B0 if changes needed; default is B5 owns `tests/*` and `docs/{DATA_CARD,MODEL_CARD,API}.md` only).
**Owns:**
- `tests/test_smoke_pipeline.py` (simulate → train 1 step → eval → infer COG, all CPU, asserts shapes/files)
- `tests/test_data.py` (SAMPLE schema conformance; collate; synthetic determinism)
- `tests/test_models.py` (parametrized over `list_models()`: forward/loss/backward/predict shapes on CPU)
- `tests/test_metrics.py` (known-input metric values; mask correctness; nan-graceful)
- `tests/test_inference.py` (tiling+blend reconstructs an identity passthrough within tolerance; COG validity)
- `tests/test_serving.py` (FastAPI TestClient `/health`, `/reconstruct` happy path)
- `docs/DATA_CARD.md`, `docs/MODEL_CARD.md`, `docs/API.md`
- `notebooks/` (optional demos; excluded from CI)
**Done when:** `pytest -q` is green on CPU end-to-end and `docs/*` describe data provenance, model registry, and API.

---

## 5. Integration invariants (so parallel work merges cleanly)
- **Single import direction:** `utils → config/base/registry/losses → data/models → training/evaluation/inference/serving → scripts → cli`. No back-imports (e.g. models must not import training).
- **No concrete-model imports** outside `models/__init__.py`; everyone else uses `build_model`/`list_models`.
- **SAMPLE keys and metric/loss signatures are frozen** in this doc; changing one requires editing this doc first.
- **CPU-smoke is the contract test:** if `configs/cpu_smoke.yaml` end-to-end (simulate→train→eval→infer→serve `/health`) passes on CPU with no network, integration is correct.
- **Optional heavy deps** (GDAL/rasterio/arosics/omnicloudmask/onnx/tensorrt/faiss/redis/lpips/sewar) are **lazy-imported** inside functions with graceful fallback or clear error; they must NOT be required for the synthetic CPU path or for `import cloudremoval`.
- **Determinism:** every entry point calls `utils.seed.seed_everything(cfg.train.seed)` first.

---

## 6. Coding standards (enforced by ruff/black/CI)
Python 3.10+, `from __future__ import annotations`, full type hints, pydantic v2 for all config, PyTorch (+ optional Lightning behind `use_lightning`), structured logging via `utils.logging.get_logger(__name__)` (no bare `print`), docstrings (Google or NumPy style) on public APIs, `pytest` for tests, `ruff` + `black` clean, line length 100. No network or GPU required to import the package or run the CPU-smoke suite.
