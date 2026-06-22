# Model Card — `cloudremoval` (BAH 2026 PS2)

**Product:** Model-agnostic GenAI cloud removal & reconstruction for **LISS-IV** imagery.
**Framework:** one `BaseCloudRemovalModel` interface + a model registry hosting **six families** trained
and benchmarked head-to-head on identical data/metrics — the comparative assessment the problem statement
demands.

> This card follows the model-card convention. It documents the *shipped* registry models (verified
> against `src/cloudremoval/models/`), grounded in [`docs/COMPARATIVE_ASSESSMENT.md`](COMPARATIVE_ASSESSMENT.md)
> (30+ methods scored) and `research/01–05`. Data provenance is in [`DATA_CARD.md`](DATA_CARD.md); the
> operational API is in [`API.md`](API.md).

---

## 1. Overview — the unified, model-agnostic framework

All models implement a single abstract class and self-register, so the trainer, evaluator, benchmark
runner and inference engine are written **only against the interface** and a standard `SAMPLE` dict —
they never import a concrete model (`ARCHITECTURE.md` §3, [`BUILD_PLAN.md`](BUILD_PLAN.md) §3.1).

- **Interface** — `cloudremoval.models.base.BaseCloudRemovalModel(nn.Module, ABC)`:
  - `__init__(self, cfg: ModelConfig)` — receives the `model` section of the config.
  - `forward(sample) -> ModelOutput` — differentiable; reads the keys it needs from `SAMPLE`.
  - `loss(sample, output) -> LossDict` — returns a dict whose `"total"` key is back-propagated.
  - `predict(sample) -> ModelOutput` — default `eval()` + single `forward`; diffusion overrides it
    with DDIM sampling.
  - Optional GAN hooks (`generator_parameters` / `discriminator_parameters` / `discriminator_loss`),
    detected by the trainer via `hasattr`.
- **Output** — `ModelOutput(reconstruction=[B,C,H,W], uncertainty=[B,1,H,W]|None, aux={...})`.
- **Registry** — `@register_model("name")` + `build_model(cfg)`; `list_models()` returns the live keys.
  Importing `cloudremoval.models` triggers registration.

The **registry keys are the canonical model names** used everywhere (CLI `--set model.name=…`,
benchmark, serving `model_name`):

```python
>>> from cloudremoval.models.registry import list_models
>>> list_models()
['diffusion', 'dsen2cr', 'restormer', 'spagan', 'uncertainty', 'unet']
```

---

## 2. The six implemented models

One model per family + a baseline (`ARCHITECTURE.md` §3.3, `COMPARATIVE_ASSESSMENT.md` §9). Each is a
clean, **from-scratch, CPU-runnable** implementation in pure `torch` that *optionally* loads external
pretrained weights at runtime (license-safe). "Inputs" = which `SAMPLE` keys it consumes; all read
`optical_cloudy` and most concat the `cloud_mask`. Param counts are at the CPU-smoke-ish setting
`base_channels=16, depth=2` (illustrative — capacity scales with config).

| Registry key (file) | Family | SAR | Temporal | DEM | Key idea | When to use | Params* |
|---|---|:--:|:--:|:--:|---|---|--:|
| **`unet`** (`unet.py`) | CNN baseline | – | – | – | Encoder–decoder + GroupNorm/SiLU residual blocks; long-skip residual over the cloudy input; loss = Charbonnier + SSIM + SAM (+ CARL when a mask is present) | The **control** — quantifies how much GenAI adds and bounds "how much a plain CNN can invent". Always CPU-fast | ~116 K |
| **`dsen2cr`** (`dsen2cr_fusion.py`) | SAR-optical fusion ResNet | ✓ | – | ✓ (opt.) | DSen2-CR: early channel-concat of SAR(VV/VH)(+DEM) + cloudy optical → residual trunk (`num_res_blocks = max(4, 2·depth)`) → long global skip; loss = **CARL** (mask-weighted L1) + SAM | The **thick-cloud workhorse** for persistent NER cloud; non-adversarial = stable, low hallucination | ~22 K |
| **`spagan`** (`gan_spagan.py`) | GAN | – | – | – | SpA-GAN: spatial-attention (SPANet/CBAM-style) U-Net generator + **PatchGAN** critic; loss = adversarial + L1/Charbonnier + SSIM (+ optional perceptual) | **Thin-cloud / texture specialist** — supplies fine 5.8 m sharpness an L1 core blurs | ~248 K |
| **`restormer`** (`transformer_restormer.py`) | Transformer | ✓ (opt.) | – | – | Restormer: **MDTA** channel-attention (linear in pixels, no window seams) + **GDFN** gated FFN; optional **GLF-CR**-style SAR cross-attention at the bottleneck; loss = Charbonnier + SAM + SSIM | Best **long-range structure** reconstruction; turn on `use_sar` for SAR-guided fusion | ~331 K |
| **`diffusion`** (`diffusion.py`) | Diffusion | ✓ (opt.) | – | – | Conditional DDPM (ε-prediction) with **SR3 concat** conditioning on the cloudy image (+SAR); **DDIM** few-step sampling at inference; `mean_reverting`/consistency hooks documented | Best **generative prior under total occlusion**; `sample_steps` trades speed↔quality | ~237 K |
| **`uncertainty`** (`uncertainty.py`) | Uncertainty head | – | – | – | UnCRtainTS-style wrapper over a U-Net trunk with **mean + log-variance heads**; trained with **Gaussian NLL**; `ModelOutput.uncertainty = exp(log_var)` | The **operational differentiator** — ships a per-pixel confidence map that feeds the §6 cross-verification | ~116 K |

\* From `sum(p.numel() for p in build_model(cfg).parameters())` at `base_channels=16, depth=2`.

**Cross-cutting fusion mechanics** (shared across models, `ARCHITECTURE.md` §3.3): SAR conditioning
guides *structure only*, never radiometry directly (early concat in `dsen2cr`, cross-attention in
`restormer`); DEM-predicted shadows forbid spurious bright fills; the `uncertainty` head is composable on
any backbone (default-on for the operational config). The shared loss library
`cloudremoval.models.losses` provides `l1/charbonnier/carl/ssim/sam/perceptual/adversarial/
gaussian_nll/total_variation` losses and a config-driven `LossBundle`.

> **Naming note.** Earlier design docs (`ARCHITECTURE.md`, the original README table) referred to these by
> longer descriptive names (e.g. `dsen2cr_fusion`, `gan_spagan`, `transformer_restormer`). The **shipped
> registry keys are the short forms above** (`dsen2cr`, `spagan`, `restormer`) — use those with the CLI,
> benchmark, and serving API. The longer names survive only as source filenames.

---

## 3. Intended use

- **In scope:** research-grade cloud removal & reconstruction of LISS-IV (and band-matched S2-surrogate)
  optical imagery; the **comparative assessment** of GenAI architectures on identical LISS-IV-domain data
  and metrics; producing analysis-ready reconstructions + per-pixel uncertainty for downstream NDVI/LULC.
- **Audience:** EO researchers and the BAH 2026 jury; ISRO/NRSC analysts evaluating de-clouding for NER.
- **Mode:** runs CPU-only on synthetic data (CI/laptop/jury demo) and scales to GPU + real Bhoonidhi data
  by config switch only.

### Out-of-scope

- **Not for safety-critical or operational decision-making without validation** on real LISS-IV pairs and
  domain expert review. Reconstructions in thick-cloud regions are *plausible reconstructions*, not
  measurements.
- **Not an atmospheric-correction engine** — outputs are TOA-reflectance-space (DOS/Py6S optional);
  surface reflectance is approximate.
- **Not a generic EO platform** — scope is cloud removal + reconstruction + its evaluation/serving.
- **Not validated for** sub-5 m sensors, non-VNIR bands, or regions outside the NER design envelope.

---

## 4. Training data

- **Synthetic pairs (primary on the smoke path):** `CloudSimulator` pastes physically-plausible thin/thick
  clouds + cast shadows onto clear LISS-IV-like scenes to manufacture `(cloudy, clear)` supervision.
- **Transfer (for real training):** pretrain on **AllClear** / **SEN12MS-CR(-TS)** (SAR+optical), then
  domain-adapt to LISS-IV (3-band, SBAF-aligned, MAE self-supervised), masks from **CloudSEN12+** (CC0).
- **Real (held-out eval):** Bhoonidhi LISS-IV NER tiles co-registered with Sentinel-1/2 + DEM; real cloudy
  scenes reserved strictly for evaluation.

Full provenance, the band-mapping table, DN→TOA calibration, access (Bhoonidhi/GEE/STAC/CDSE) and licenses
are in [`DATA_CARD.md`](DATA_CARD.md). All models consume the universal `SAMPLE` dict (zero-filled absent
modalities), so any data source plugs in without model changes.

---

## 5. Evaluation protocol & metrics

The evaluator (`cloudremoval.evaluation.evaluator.Evaluator`) runs `model.predict` over a loader, builds
**stratum masks** from the `SAMPLE` dict, and aggregates the **mask-aware** metric panel
(`cloudremoval.evaluation.metrics`, MVES Tier-0/1). Every metric accepts a `mask` and is reported per
**stratum**.

### Metrics (real function names / directions)

| Metric | Direction | Why it matters for LISS-IV |
|---|:--:|---|
| `psnr`, `ssim`, `ms_ssim` | ↑ | Pixel/structural fidelity (per-band SSIM, finite PSNR cap for identical images). |
| `sam` (degrees) | ↓ | **Spectral-angle gate** — captures spectral *shape* drift independent of gain; protects NDVI/NIR. |
| `ergas` | ↓ | Global relative spectral error across bands — the spectral-fidelity gate. |
| `sid` | ↓ | Spectral information divergence (secondary; unstable on near-zero 3-band vectors). |
| `ndvi_mae` (red=1, nir=2) | ↓ | The most reviewer-legible "is it analysis-ready" proxy — a model can match RGB yet wreck NDVI. |
| `rmse`, `mae` | ↓ | Reflectance error. |
| `spectral_correlation`, `per_band_correlation`, `per_band_bias` | ↑ / – | Per-band agreement and **systematic bias** that SAM/correlation hide. |
| `lpips` | ↓ | Perceptual distance on an RGB render (lazy; `nan` on the minimal stack — the `[train]` extra adds it). |

Downstream-task validation (documented protocol, `ARCHITECTURE.md` §5.3): **LULC consistency (Cohen's κ)**
on real-vs-reconstructed, **ΔNDVI** error maps, and change-detection false-change rate.

### Why masked + spectral metrics (the non-negotiable rules)

- **Masked, not whole-image.** Most of a scene is already clear, so whole-image metrics are inflated by the
  easy pixels. The evaluator reports **`whole` / `cloud` / `shadow` / `thin` / `thick`** strata
  (`EvalConfig.strata`); **the `cloud` column is the real score**. `build_stratum_masks` splits thin/thick
  by soft-cloud opacity or per-sample `meta["cloud_type"]`.
- **Spectral fidelity over raw PSNR.** Because only 3 bands feed NDVI/LULC, reflectance correctness is the
  deliverable axis — hence SAM/ERGAS/SID/NDVI-MAE are first-class and `per_band_bias` is reported separately.

### Comparative assessment → leaderboard

`cloudremoval.evaluation.benchmark.run_benchmark(models, loader, cfg, ...)` evaluates several models (by
registry name, per-model `Config`, or constructed instance) on the **same** loader + seed, measures
`params` / `latency_ms` / `throughput_sps`, ranks per-metric and overall (default `rank_metric="psnr"`,
`rank_stratum="cloud"`), and writes `leaderboard.json` / `leaderboard.csv`. `run_benchmark_all` does it
for every registered model. This *is* the comparative-assessment artifact (reproducible per-commit on the
synthetic set; see the [quickstart notebook](../notebooks/01_quickstart.ipynb) §6).

---

## 6. Limitations & risks

| Risk | Why it matters | Mitigation in this product |
|---|---|---|
| **Spectral hallucination on thick cloud** | Wrong reflectance → broken NDVI/LULC; the worst failure mode | `mean_reverting`/consistency diffusion hooks; **CARL + SAM** losses; histogram-match to clear pixels; **always report SAM/ERGAS/NDVI-MAE**; the `unet` control bounds invention by contrast; uncertainty flagging |
| **Scarce real LISS-IV pairs** | No supervised real-pair training is possible | Synthetic clouds + SEN12MS-CR/AllClear transfer + MAE self-sup + SAR fusion (`DATA_CARD.md` §4) |
| **Domain gap (synthetic→real, S2→LISS-IV)** | Smoke metrics overstate real accuracy; S2 weights bias NIR | SBAF + histogram matching; `real_alpha` copy-paste path; re-benchmark on the NER set, report relative-to-baseline |
| **Uncertainty calibration** | An over/under-confident confidence map misleads | Heteroscedastic Gaussian-NLL head (`uncertainty`); ECE/coverage in the Tier-2 protocol; fused with the cross-verification checks |
| **3 bands, no SWIR/cirrus → masks fail; snow ≈ cloud** | Mask errors propagate into reconstruction | 3-band cloud logic (G/R/NIR brightness + low NDVI + texture); SAR/temporal cues; DEM/elevation priors + temporal persistence for snow |
| **SAR speckle / mis-registration** | Speckle or misalignment injects artifacts in fusion | SAR = structure not radiometry; despeckle before fusion; mask-weighted losses tolerate residual misregistration; Align-CR deformable-alignment hook |
| **Diffusion latency at scene scale** | Iterative sampling is slow | Few-step DDIM (`sample_steps`); documented consistency-distillation (CM-CR) + ONNX/TensorRT export hooks |
| **GAN instability** | Adversarial training can drift spectrally | Strong L1/Charbonnier core + curriculum; report SAM; prefer non-adversarial cores for the operational path |
| **Not for safety-critical use** | Reconstructions are plausible, not measured | Ship per-pixel uncertainty + quality mask; require real-pair validation + expert review before operational use |

### Key mitigations summarized

- **Uncertainty head** (`uncertainty`) → per-pixel aleatoric variance, the operational confidence signal.
- **Mean-reverting / consistency diffusion** hooks (`mean_reverting`, CM-CR) → spectral fidelity + speed.
- **Always report SAM/ERGAS/NDVI-MAE** masked to the cloud region — spectral correctness is the gate.

---

## 7. Reproducibility & provenance

- **Deterministic seeding** (`cloudremoval.utils.seed.seed_everything` from `train.seed`) before every
  stage; the synthetic dataset seeds per `(split, idx)`.
- **Typed configs** (`pydantic v2`, `configs/model/*.yaml` per model) — one schema scales CPU↔GPU.
- **CI gate** — the CPU smoke (`configs/cpu_smoke.yaml`) trains/evals/benchmarks every model on synthetic
  data each commit; 50 pytest tests cover the contract.
- **License discipline** (`ARCHITECTURE.md` §7) — MIT/Apache/CC0 sources preferred; GPL/unverified methods
  reimplemented from scratch; external weights loaded via runtime adapters with a from-scratch fallback.

> **Smoke caveat.** Numbers produced by the CPU smoke (tiny tiles, 2 steps, synthetic clouds) are
> **illustrative of the pipeline, not of real-world accuracy**. Real evaluation requires the Bhoonidhi
> LISS-IV + Sentinel-1/2 + DEM workflow of [`DATA_CARD.md`](DATA_CARD.md).
