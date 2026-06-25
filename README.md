<div align="center">

# cloudremoval

**Model-agnostic GenAI cloud removal & reconstruction for LISS-IV imagery**

_BAH 2026 · Problem Statement 2 · Resourcesat-2/2A LISS-IV (5.8 m, 3-band G/R/NIR)_

[![CI](https://img.shields.io/badge/CI-lint%20%2B%20cpu--smoke-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![Config: pydantic v2](https://img.shields.io/badge/config-pydantic%20v2-e92063)](src/cloudremoval/config.py)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![Format: black](https://img.shields.io/badge/code%20style-black-000000)](https://black.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#license)

</div>

---

`cloudremoval` is a **unified, model-agnostic framework** for removing clouds from
and reconstructing **LISS-IV** optical satellite imagery using generative AI. It
hosts **six GenAI model families** behind one interface so they can be trained
and benchmarked head-to-head on identical data and metrics — the comparative
assessment the problem statement demands.

The framework is built around one hard constraint: **it runs end-to-end on a
CPU-only machine with synthetic data and zero network access** (for CI, laptops,
and the jury's box), and **scales to GPU + real Bhoonidhi data by changing only
the config**. Every heavy dependency (GDAL/rasterio, ONNX/TensorRT, FAISS, Redis,
Lightning, …) is **optional and lazily imported**, so `import cloudremoval` and
the synthetic pipeline work on a minimal stack of `numpy + torch + pydantic +
pyyaml + fastapi + typer`.

> **Why LISS-IV is hard.** Only 3 VNIR bands (Green, Red, NIR) — no blue, no
> SWIR, no cirrus band — at 5.8 m, over the persistently-cloudy North-East India
> monsoon belt, with **no public cloud-removal benchmark**. We solve the data
> scarcity with synthetic-cloud injection + multi-satellite transfer learning,
> fuse SAR/temporal/DEM evidence, and evaluate spectral fidelity first.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full technical design,
[`docs/COMPARATIVE_ASSESSMENT.md`](docs/COMPARATIVE_ASSESSMENT.md) for the 30+
method scoring, and [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) for the interface
contract.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture at a glance](#architecture-at-a-glance)
- [The six model families](#the-six-model-families)
- [Installation](#installation)
- [Quickstart — CPU smoke](#quickstart--cpu-smoke)
- [CLI usage](#cli-usage)
- [Configuration](#configuration)
- [The `SAMPLE` contract](#the-sample-contract)
- [Documentation](#documentation)
- [Results (smoke)](#results-smoke)
- [Status / limitations](#status--limitations)
- [Extending: add a model](#extending-add-a-model)
- [Repository map](#repository-map)
- [Development](#development)
- [License](#license)

---

## Highlights

- **One interface, many architectures.** Every model subclasses
  `BaseCloudRemovalModel` and self-registers via `@register_model("name")`. The
  trainer, evaluator, benchmark runner, and inference engine talk **only** to the
  interface and a standard `SAMPLE` dict — never to a concrete model. Fair
  comparison is built in.
- **CPU-first, GPU-scalable.** The same code path runs tiny synthetic tiles on
  CPU (seconds, no network) and full-resolution real data on GPU — switching
  `configs/cpu_smoke.yaml` ↔ `configs/gpu_full.yaml`.
- **Minimal core, optional everything-else.** Core = `numpy, torch, pydantic,
  pyyaml, fastapi, uvicorn, typer`. Geospatial / accelerator / serving / training
  extras are installed on demand and imported lazily.
- **Spectral fidelity over raw PSNR.** Masked, stratified MVES metrics
  (PSNR/SSIM/**SAM/ERGAS**/NDVI-MAE) with cloud/shadow/thin/thick strata; CARL,
  spectral-angle, and uncertainty losses resist radiometric hallucination.
- **Analysis-ready outputs.** Tiled blended inference → Cloud-Optimized GeoTIFF
  (+ uncertainty band) + STAC, served over an **O(1)-per-tile** FastAPI stack —
  COG ranged-GET reads + content-addressed cache (Redis/in-memory LRU) + FAISS
  nearest-clear retrieval + ONNX/TensorRT acceleration (`ARCHITECTURE.md` §6).
- **Multi-satellite by design.** LISS-IV is the only target; Sentinel-1 SAR
  (structure under cloud), Sentinel-2 (band-matched spectral proxy/pretrain),
  LISS-III/AWiFS (SWIR teacher + temporal), MODIS/INSAT (cloud climatology) and a
  Copernicus DEM (shadow geometry) cross-verify the reconstruction
  ([`docs/DATA_CARD.md`](docs/DATA_CARD.md)).
- **Reproducible.** Typed `pydantic v2` configs, deterministic seeding, `ruff` +
  `black`, a CPU smoke test that gates every commit in CI, and an executable
  [quickstart notebook](notebooks/01_quickstart.ipynb).

---

## Architecture at a glance

```
download → preprocess → simulate → train → eval → benchmark → infer → serve
   │           │            │         │       │        │          │       │
 sources    radiometry   synthetic  Trainer  MVES   leaderboard  tiled  FastAPI
 (S1/S2/    co-reg +     clouds +   (+ GAN/  masked  (all models  COG +  + tiles
  DEM/LISS) masking      SAR/DEM    Lightning) metrics  head-to-  STAC   + Redis
                         pairing                         head)
```

```
        ┌──────────────────────────── models/base.py ────────────────────────────┐
        │   BaseCloudRemovalModel  ·  forward() · loss() · predict() · name        │
        └───────────────▲───────────────────────────────────────▲─────────────────┘
                        │ @register_model(...)                   │ build_model(cfg)
      ┌───────┬──────┴───┬───────────┬───────────┬────────────┴─┐
    unet   dsen2cr   spagan    restormer   diffusion    uncertainty
                        │                                       │
              ┌─────────┴─────────┬───────────────┬─────────────┴────────┐
           Trainer            Evaluator     Benchmark runner       Tiled inference
```

Single import direction (no back-imports):
`utils → config/base/registry/losses → data/models → training/evaluation/inference/serving → scripts → cli`.

Full design: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## The six model families

Chosen to cover the fidelity/speed/data-efficiency frontier and satisfy
"≥1 per family + a baseline" (full justification in
[`docs/COMPARATIVE_ASSESSMENT.md`](docs/COMPARATIVE_ASSESSMENT.md)):

| Registry name   | Family            | Role |
| --------------- | ----------------- | ---- |
| `unet`          | CNN baseline      | Time-tested encoder-decoder, optical-only (Charbonnier + SSIM + SAM). The control that quantifies how much GenAI adds. |
| `dsen2cr`       | SAR-optical ResNet| Residual correction, early SAR concat, **CARL loss**. The robust thick-cloud workhorse for persistent NER cloud. |
| `spagan`        | GAN               | Spatial-attention generator + PatchGAN. Fine 5.8 m texture; thin-cloud specialist. |
| `restormer`     | Transformer       | MDTA channel-attention core (+ optional GLF-CR SAR cross-attention). Best long-range structure. |
| `diffusion`     | Diffusion         | DDPM train / DDIM sample; SR3 concat conditioning; mean-reverting option. Best generative prior under total occlusion. |
| `uncertainty`   | Uncertainty head  | Wraps a U-Net trunk, adds per-pixel aleatoric variance (Gaussian NLL). The operational confidence map. |

> All six are **implemented and registered** (pure `torch`, CPU-runnable). The
> registry keys above are the canonical names used by the CLI (`-s model.name=…`),
> the benchmark, and the serving API. Confirm the live list with
> `cloudremoval.models.registry.list_models()` →
> `['diffusion', 'dsen2cr', 'restormer', 'spagan', 'uncertainty', 'unet']`.
> Each optionally loads external pretrained weights at runtime, with a from-scratch
> fallback. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full per-model card.

---

## Installation

Requires **Python 3.10+**. CPU-only PyTorch is recommended for the smoke path.

```bash
# 1. CPU PyTorch from the official index (skip if you want CUDA torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. The package (core stack only — enough for the synthetic CPU pipeline)
pip install -e .

# …or with optional extras as needed (all lazily imported):
pip install -e ".[geo]"     # rasterio, rio-cogeo, rioxarray, arosics, omnicloudmask, ...
pip install -e ".[data]"    # pystac-client, planetary-computer, earthengine-api, sentinelhub
pip install -e ".[serve]"   # rio-tiler, redis, pillow (dynamic tiling + cache)
pip install -e ".[accel]"   # faiss-cpu, onnxruntime (ANN retrieval + accelerated inference)
pip install -e ".[train]"   # pytorch-lightning, mlflow, tensorboard, lpips, matplotlib
pip install -e ".[dev]"     # pytest, ruff, black, pre-commit, httpx
pip install -e ".[all]"     # everything (geo,data,serve,accel,train,dev)
```

Or use the Makefile (creates a venv, installs CPU torch + dev extras):

```bash
make setup
```

The package is **importable without installation** too — `tests/conftest.py`
inserts `src/` onto `sys.path`, and the `src/` layout is declared in
`pyproject.toml`.

---

## Quickstart — CPU smoke

The entire pipeline runs on synthetic data, on CPU, with **no network**:

```bash
# Sanity: the package imports on the minimal stack and the registry loads
python -c "import cloudremoval; print(cloudremoval.__version__)"
python -c "from cloudremoval.models.registry import list_models; print(list_models())"

# The CLI is available as `cloudremoval` (or `python -m cloudremoval.cli`)
cloudremoval --help

# End-to-end smoke (simulate → train → eval → infer) on tiny synthetic tiles
make smoke
# equivalently:
cloudremoval simulate --config configs/cpu_smoke.yaml
cloudremoval train    --config configs/cpu_smoke.yaml
cloudremoval eval     --config configs/cpu_smoke.yaml
cloudremoval infer    --config configs/cpu_smoke.yaml

# Run the test suite (CPU, synthetic)
make test         # = pytest -q
```

`configs/cpu_smoke.yaml` uses 64×64 patches, batch size 2, a couple of training
steps, and `device: cpu` — it finishes in seconds and is the **contract test**
exercised by CI.

### Run the demo notebook

[`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) is a self-contained,
CPU-runnable walkthrough: synthetic LISS-IV scene → `CloudSimulator` pairs → build a
registry model → a few train steps → reconstruction + uncertainty map → the masked
MVES metric suite → a 2-model `run_benchmark` leaderboard → a FastAPI serving demo.

```bash
pip install -e ".[train]"   # pulls matplotlib (also: matplotlib jupyter nbconvert)
jupyter notebook notebooks/01_quickstart.ipynb
# or run it headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_quickstart.ipynb
```

---

## CLI usage

Single entry point `cloudremoval` (Typer). Every subcommand takes `--config/-c`
and optional repeatable `--set/-s key=value` overrides:

```text
cloudremoval download   --config CFG [--aoi KML] [--source bhoonidhi|stac|gee|sentinel|dem]
cloudremoval preprocess --config CFG
cloudremoval simulate   --config CFG                 # synthetic cloudy/clear pairs
cloudremoval train      --config CFG [--ckpt PATH]
cloudremoval eval       --config CFG [--ckpt PATH]   # masked + whole MVES metrics
cloudremoval benchmark  --config CFG                 # all registered models → leaderboard
cloudremoval infer      --config CFG [--ckpt PATH] [--input ...] [--out ...]   # → COG
cloudremoval serve      --config CFG                 # FastAPI app
```

Examples:

```bash
# Override the model and cap steps inline
cloudremoval train -c configs/cpu_smoke.yaml -s model.name=dsen2cr -s train.max_steps=1

# Point at a real-data config (requires the [geo] extra + data)
cloudremoval preprocess -c configs/gpu_full.yaml
```

Each subcommand seeds all RNGs from `train.seed` before running.

---

## Configuration

Configs are **typed and validated** by `pydantic v2`
([`src/cloudremoval/config.py`](src/cloudremoval/config.py)). A YAML file may
include a `base:` key naming another YAML, which is deep-merged underneath — so
`cpu_smoke.yaml` and `gpu_full.yaml` both extend `base.yaml` and change only
values, never structure.

| File                      | Purpose |
| ------------------------- | ------- |
| `configs/base.yaml`       | Shared substrate (all sections at defaults). |
| `configs/cpu_smoke.yaml`  | Tiny synthetic CPU run (CI + demo + jury box). |
| `configs/gpu_full.yaml`   | Full real-data GPU run (Bhoonidhi/Copernicus). |
| `configs/{data,model,serve}/` | Per-dataset / per-model / serving leaf configs (added by the respective stages). |

Sections mirror the schema exactly: `data`, `model`, `train` (with nested
`loss`), `eval`, `infer`, `serve`. Load programmatically:

```python
from cloudremoval.config import load_config
cfg = load_config("configs/cpu_smoke.yaml", overrides={"train": {"max_steps": 1}})
print(cfg.model.name, cfg.data.tile_size, cfg.train.device)
```

---

## The `SAMPLE` contract

Every dataset `__getitem__` returns one `SAMPLE` dict; every model reads the
subset of keys it needs and ignores the rest. Absent optional modalities are
**zero-filled** (never missing), with presence flags in `meta`:

```python
SAMPLE = {
    "optical_cloudy": Tensor,   # [C, H, W]      model input (cloud-contaminated)
    "optical_clear":  Tensor,   # [C, H, W]      target (zeros at inference)
    "cloud_mask":     Tensor,   # [1, H, W]      1.0 = cloud
    "shadow_mask":    Tensor,   # [1, H, W]      1.0 = cloud-shadow
    "sar":            Tensor,   # [2, H, W]      VV, VH (zeros if absent)
    "dem":            Tensor,   # [1, H, W]      normalized elevation/slope (zeros if absent)
    "temporal_refs":  Tensor,   # [T, C, H, W]   nearest-clear refs ([0,C,H,W] if none)
    "meta":           dict,     # scene_id, date, sun_elevation, crs, transform,
                                # band_stats, norm, cloud_type, has_{target,sar,dem,temporal}
}
# C = 3 (Green, Red, NIR)
```

Models return a `ModelOutput(reconstruction=[B,C,H,W], uncertainty=[B,1,H,W]|None,
aux={...})`. Full spec: [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) §2.

---

## Documentation

| Document | What's in it |
| -------- | ------------ |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Master technical design — data strategy, unified framework, pipeline, evaluation, O(1) serving. |
| [`docs/COMPARATIVE_ASSESSMENT.md`](docs/COMPARATIVE_ASSESSMENT.md) | 30+ GenAI methods scored; the 6 implemented + why (the required comparative-assessment deliverable). |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | Multi-satellite data strategy, band-mapping, synthetic-cloud + transfer, access (Bhoonidhi/GEE/STAC/CDSE), licenses, data commands. |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | The 6 registry models (family, inputs, key idea, params), intended/out-of-scope use, metrics, limitations & risks. |
| [`docs/API.md`](docs/API.md) | CLI reference (all 8 commands) + REST API (endpoints/schemas/curl/httpx) + the O(1) serving design. |
| [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) | The frozen interface/work contract (SAMPLE, registry, losses, config). |
| [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Executable CPU walkthrough end-to-end. |
| [`research/`](research/) | The 6 grounding research briefs (methods/metrics in `01–03,05`; datasets/satellites in `04`; LISS-IV/ISRO/Bhoonidhi/NER in `06`). |

## Results (smoke)

The CI/CPU smoke (`configs/cpu_smoke.yaml`) trains, evaluates, and benchmarks every
registered model on **synthetic** data in seconds, and the notebook prints a live
`whole` vs `cloud`-region MVES table + a 2-model leaderboard. **These numbers are
illustrative of the plumbing, not of real-world accuracy** — they come from tiny
synthetic tiles and a handful of steps. Real metrics require the Bhoonidhi LISS-IV +
Sentinel-1/2 + DEM workflow described in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Status / limitations

- **What's shipped:** the full framework — 6 registered GenAI models, the 8-command
  CLI, masked/stratified MVES metrics, the benchmark leaderboard, tiled inference, and
  the FastAPI serving app — all CPU-runnable on synthetic data, with a green CI smoke
  and 50 passing tests.
- **What's synthetic vs real:** the smoke path is **synthetic** (procedural LISS-IV-like
  scenes + injected clouds). Real-data training/eval on **Bhoonidhi** LISS-IV (with
  co-registered Sentinel-1/2 + DEM) is wired via config + lazy `[geo]`/`[data]` extras
  but requires downloaded scenes and is **not** exercised by the smoke.
- **Known hard cases:** spectral hallucination under thick cloud, the synthetic→real and
  S2→LISS-IV domain gaps, snow≈cloud in 3 bands, and uncertainty calibration — mitigated
  (SAM/CARL losses, uncertainty head, SBAF, cross-verification) but not eliminated. See
  [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) §6. **Not for safety-critical use without
  validation on real LISS-IV pairs.**

---

## Extending: add a model

```python
from torch import nn
from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput
from cloudremoval.models.registry import register_model
from cloudremoval.models.losses import LossBundle

@register_model("my_model")
class MyModel(BaseCloudRemovalModel):
    def __init__(self, cfg):                 # cfg is the `model` section (ModelConfig)
        super().__init__(cfg)
        self.net = nn.Conv2d(cfg.in_channels, cfg.out_channels, 3, padding=1)
        self.bundle = LossBundle(cfg_loss)   # or compute your own LossDict

    def forward(self, sample) -> ModelOutput:
        return ModelOutput(reconstruction=self.net(sample["optical_cloudy"]))

    def loss(self, sample, output):          # must return a dict containing "total"
        return self.bundle(sample, output)
```

`build_model(cfg)` instantiates `cfg.model.name`; the trainer/evaluator pick it
up automatically. GAN models additionally expose
`generator_parameters` / `discriminator_parameters` / `discriminator_loss`, which
the trainer detects via `hasattr`.

---

## Repository map

```
bah2026-ps2/
├── ARCHITECTURE.md                  # master technical design
├── README.md                        # this file
├── pyproject.toml                   # package, deps (core + [geo|accel|serve|train|dev]), tooling
├── requirements.txt                 # minimal core stack only
├── Makefile                         # setup / smoke / train / eval / serve / test / lint / docker
├── Dockerfile  docker-compose.yml   # CPU API image; api + redis services
├── .github/workflows/ci.yml         # ruff + black + compile + cpu-smoke pytest
├── configs/                         # base.yaml, cpu_smoke.yaml, gpu_full.yaml (+ data/model/serve)
├── src/cloudremoval/
│   ├── __init__.py  config.py  cli.py
│   ├── models/   base.py · registry.py · losses.py · <six model families>
│   ├── data/     datasets · synthetic_clouds · preprocess · masking · sources/…
│   ├── training/ trainer · lightning_module · callbacks
│   ├── evaluation/ metrics · evaluator · benchmark · report
│   ├── inference/ tiled · blending · postprocess · cog_writer
│   ├── serving/  app (FastAPI) · tiles · schemas
│   └── utils/    geo · io · logging · seed
├── scripts/                         # thin CLI delegates (download/preprocess/…/serve)
├── tests/                           # conftest fixtures + CPU end-to-end suite (50 tests)
├── notebooks/                       # 01_quickstart.ipynb (executable CPU walkthrough)
├── research/                        # 01–06 grounding briefs (methods · metrics · datasets · LISS-IV/NER)
└── docs/                            # COMPARATIVE_ASSESSMENT · BUILD_PLAN · DATA_CARD · MODEL_CARD · API
```

---

## Development

```bash
make lint        # ruff check + black --check
make format      # black + ruff --fix
make typecheck   # mypy
make test        # pytest -q (CPU, synthetic)
pre-commit install   # ruff + black on every commit
```

Standards (enforced by CI): Python 3.10+, `from __future__ import annotations`,
full type hints, `pydantic v2`, structured logging via
`cloudremoval.utils.logging.get_logger` (no bare `print`), docstrings on public
APIs, line length 100. **No network or GPU is required** to import the package or
run the CPU-smoke suite.

---

## License

Released under the **MIT License**. Third-party methods with restrictive or
unverified licenses are **reimplemented** rather than vendored; external
pretrained weights are loaded at runtime via adapters when the user supplies
them, with a from-scratch fallback always available (see
[`ARCHITECTURE.md`](ARCHITECTURE.md) §1.4, §7).

---

<div align="center">
<sub>Built for BAH 2026 PS2 — GenAI cloud removal & reconstruction for LISS-IV.
Read <a href="ARCHITECTURE.md">ARCHITECTURE.md</a> ·
<a href="docs/COMPARATIVE_ASSESSMENT.md">COMPARATIVE_ASSESSMENT.md</a> ·
<a href="docs/BUILD_PLAN.md">BUILD_PLAN.md</a>.</sub>
</div>
