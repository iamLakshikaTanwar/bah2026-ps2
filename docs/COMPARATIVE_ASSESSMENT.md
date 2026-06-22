# Comparative Assessment of GenAI Architectures for LISS-IV Cloud Removal

**BAH 2026 PS2 — required deliverable.** A synthesized decision matrix of **30+ methods** drawn from `research/01–06`, scored on the axes that matter for LISS-IV (5.8 m, 3-band G/R/NIR, NER, no public benchmark). Concludes with **which models we implement in the product and why** (≥1 per family + a baseline) and **which we cite as comparison-only.**

> **Reading the scores.** Ratings are **Low / Med / High / V.High** relative judgments synthesized from the research, **not** measured LISS-IV numbers (none exist; published PSNR/SSIM/SAM are mostly Sentinel-2 13-band/10 m and do **not** transfer directly). License/weights/code reflect what the research files verified `[V]` or flagged unverified `[?]`. "LISS-IV fit" folds in band overlap (G/R/NIR), resolution proximity, data-efficiency, and deployability.

## Scoring axes
- **Inputs** — what conditioning the method requires (mono optical / +SAR / +temporal / +mask / prompt).
- **Data-eff.** — ability to train with scarce paired data (transfer weights, self-sup, unpaired, small-data tricks).
- **Spectral fidelity** — resistance to radiometric/NDVI hallucination (the deliverable-defining axis).
- **Spatial detail** — fine 5.8 m texture/structure preservation.
- **Thick-cloud** — capability under total occlusion (needs SAR/temporal to score high).
- **Speed** — compute / inference throughput at scene scale.
- **Weights** — public pretrained checkpoints available?
- **License** — redistribution posture.
- **LISS-IV fit** — overall suitability.

---

## 1. Diffusion / score-based family

| # | Method | Inputs | Data-eff. | Spectral | Spatial | Thick | Speed | Weights | License | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | DDPM / Score-SDE | mono (uncond) | Low | Med | Med | Low | Low (~1000 step) | many | MIT | Substrate only |
| D2 | SR3 | mono concat | Low | Med | High | Low | Low | unofficial | MIT | Template (concat recipe) |
| D3 | Palette | mono concat | Low | Med | High | Med | Low | unofficial | MIT | Template (inpaint proof) |
| D4 | RePaint | mask + frozen prior | **High** (self-sup prior) | Med | Med | Med | V.Low (resample) | yes | research | Self-sup LISS-IV prior |
| D5 | SeqDMs | temporal+SAR | Med | Med | Med | High | Low | — | — | If time series |
| D6 | DDPM-CR | SAR+optical (feat.) | Med | Med | Med | High | Med (feature) | partial | — | Heavy params |
| D7 | **DiffCR** ⭐ | mono | **High** (SEN12MS-CR weights) | Med-High | High | Med | **High** (5% FLOPs, few-step) | **HF weights** | `[?]` confirm | **V.High** |
| D8 | IDF-CR | mono (ControlNet) | Med | Med (RGB latent) | High | High | Med (+INR) | yes | — | Med-High |
| D9 | DE / CUHK-CR | mono | Med | Med | **V.High** (0.5 m detail) | Med | Low (coarse-fine) | yes | research | High (detail) |
| D10 | **EMRDM** ⭐ | mono / temporal (mean-revert) | High | **High** (starts from cloudy radiometry) | High | Med-High | High (EDM ODE few-step) | yes | — | **V.High** |
| D11 | DC4CR | prompt+optical (LoRA) | **High** (grouped/small-data) | Med (RGB latent) | High | High | Med | — | — | High (small-data) |
| D12 | SADER | temporal | Med | Med | High | High | Med (det. resample) | — | — | Med (very new) |
| D13 | SAR→Opt cond. diffusion | SAR | Med | Low (ill-posed) | Med | **V.High** | Low | yes | — | Thick-cloud aux |
| D14 | Color-supervised SAR→Opt | SAR | Med | Med (color-sup) | Med | High | Low | — | — | Aux (anti color-shift) |
| D15 | **CM-CR** ⭐ | SAR+optical (consistency) | Med | Med-High | High | **V.High** | **V.High** (~8–16 step) | paper | `[?]` | **V.High (fast+fused)** |
| D16 | SAR-DeCR / EDM-CR | SAR+optical (latent) | Med | Med | High | **V.High** | Med | — | — | High (thick) |

**Family takeaway.** Diffusion gives the best generative prior under total occlusion and the best perceptual quality, at the cost of iterative sampling. The frontier we exploit: **DiffCR** (transferable SEN12MS-CR weights, efficiency), **EMRDM** (mean-reverting → spectral fidelity + few-step), **CM-CR** (SAR-fused + consistency-distilled → fast thick-cloud). RePaint enables a self-supervised LISS-IV prior from abundant clear scenes (no pairs).

---

## 2. GAN family

| # | Method | Inputs | Data-eff. | Spectral | Spatial | Thick | Speed | Weights | License | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | Pix2Pix | mono paired | Low | Med | Med (blur) | Low | High | task-dep | research | **Baseline** |
| G2 | pix2pixHD | mono HR paired | Low | Med | High | Low | Med | generic | research | Recipe reuse (multi-scale D) |
| G3 | McGAN | mono multispec paired | Med (NIR-aided, synth) | Med | Med | Low | High | no | MIT `[?]` | High (NIR + synth clouds) |
| G4 | **SpA-GAN** ⭐ | mono paired | **High** (pretrained RICE) | Med | High (attention) | Low-Med | **High** (light) | **yes** | **MIT** | **Top optical pick** |
| G5 | AMGAN-CR | mono (SAR variant) | Med | Med | High | Med | Med | no | — | High (no mask needed) |
| G6 | CR-GAN-PM | mono **unpaired**+physics | **V.High** (no pairs) | Med-High (physical) | Med | Low | Med | no | `[?]` | High (thin, no pairs) |
| G7 | Cloud-GAN | mono **unpaired** | High (no pairs) | Low-Med | Med | Low | Med | no | `[?]` | Unpaired baseline |
| G8 | **DSen2-CR** ⭐ | **SAR+optical** | **High** (SEN12MS-CR weights, CARL) | **High** (CARL mask-wt, residual) | Med-High | **V.High** | Med | **yes** | **GPL-3** (reimpl.) | **Top fusion pick** |
| G9 | SAR-Opt-cGAN | SAR→opt | Med | Low (ill-posed) | Med | **V.High** | High | partial | scattered | Core (opaque) |
| G10 | Simulation-Fusion GAN | SAR+optical (2-step) | Med | Med-High (simulate-then-fuse) | High | High | Med | no | `[?]` | Strong design template |
| G11 | SAR2Opt baselines | SAR | Med | Low | Med | High | High | yes | various | Comparison |
| G12 | AttentionGAN (SAR) | SAR+optical attn | Med | Med | High | High | Med | no | `[?]` | Secondary fusion |
| G13 | STGAN | multitemporal | Med | Med | High | Med-High | Med | partial | `[?]` | Strong (revisit) |
| G14 | CTGAN | multitemporal (3) | Med | Med | High | Med | Low (heavy) | yes | `[?]` | Beaten by PMAA |

**Family takeaway.** GANs are lighter and faster to train than diffusion and several ship weights. **DSen2-CR** (technically a non-adversarial ResNet, grouped here as the canonical SAR-fusion engine) is our thick-cloud workhorse: CARL mask-weighted loss directly fits imperfect pairs and residual learning preserves clear-pixel radiometry. **SpA-GAN** (MIT, pretrained, attention, light) is our optical/thin-cloud + texture-refinement pick. Unpaired CR-GAN-PM is the thin-cloud fallback when no clear reference exists. Risk: instability + spectral drift → curriculum (L1 core first, adversarial head last) and SAM evaluation.

---

## 3. CNN / restoration / inpainting family (non-adversarial)

| # | Method | Inputs | Data-eff. | Spectral | Spatial | Thick | Speed | Weights | License | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | RSC-Net (residual CNN) | mono | Med | Med | Med (blur) | Low | High | varies | varies | Control |
| C2 | **PMAA** ⭐ | multitemporal | High (0.5% params) | Med | High (detail) | Med-High | **V.High** (0.5% params of CTGAN) | **yes** | repo `[?]` | High (efficient temporal) |
| C3 | PartialConv | mono+mask | Med | Med | Med | Low (optical-only) | High | yes | NVIDIA research | Block reuse (irregular holes) |
| C4 | DeepFill v2 (gated) | mono+mask | Med | Med | Med | Low | High | yes | `[?]` | Block reuse (soft mask) |
| C5 | EdgeConnect | mono+mask | Med | Med | High (edges) | Low | Med | yes | research | Boundary preservation |
| C6 | MPRNet | mono | Med | Med | High | Low (no holes) | Med | **yes** | `[?]` | Thin-cloud/haze |
| C7 | RDN / ResNet | mono | Med | Med | Med | Low | High | yes | various | Ablation baseline |

**Family takeaway.** Non-adversarial cores are stable and quantify hallucination by contrast. **PMAA** is the efficient multitemporal option (matches CTGAN at 0.5% params). PartialConv/gated-conv layers are reusable building blocks for irregular cloud holes. MPRNet/RDN are clean restoration baselines. A plain **UNet** from this family is our mandatory control.

---

## 4. Transformer / attention / fusion family

| # | Method | Inputs | Data-eff. | Spectral | Spatial | Thick | Speed | Weights | License | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | **Restormer** ⭐ | mono | Med | Med-High | **High** (MDTA channel attn) | Low (no mask) | **High** (linear in pixels) | yes | permissive `[?]` | **High (core)** |
| T2 | SwinIR | mono | Med | Med | High (window seams) | Low | High | yes | Apache-2.0 | Med-High |
| T3 | Uformer | mono | Med | Med | High | Low | High | yes | MIT | Med-High |
| T4 | Swin Transformer | mono | Med | — | — | — | High | yes | MIT | Backbone primitive |
| T5 | MAT | mono+mask | Med | Med | High | Med (large holes) | Med | yes | permissive `[?]` | Med (mask-aware idea) |
| T6 | ICT | mono+mask | Low | Med | High | Med | Low (heavy) | yes | `[?]` | Med (token completion idea) |
| T7 | **GLF-CR** ⭐ | **SAR+optical** | Med | High | **V.High** (global+local fusion) | **V.High** | Med | repo | `[?]` | **V.High (fusion core)** |
| T8 | **Align-CR** ⭐ | SAR(lo)+opt(hi) | Med | High | High | **V.High** | Med | — | `[?]` | **V.High (misalign-tolerant)** |
| T9 | MSDA-CR | optical | Med | Med-High (distortion-aware) | High | Med | Med | — | `[?]` | Med-High (thin cloud) |

**Family takeaway.** **Restormer**'s channel-attention is cheap at full resolution (no window seams) and the cleanest restoration core to add input channels to. **GLF-CR** is the purpose-built SAR-optical fusion template (global structure consistency + local SAR injection + dynamic speckle filtering); **Align-CR** adds deformable warping that tolerates the LISS-IV↔S1 misregistration we will actually face. We adopt Restormer as the implementable transformer core and fold in GLF-CR/Align-CR fusion mechanics.

---

## 5. Multi-temporal & uncertainty family

| # | Method | Inputs | Data-eff. | Spectral | Spatial | Thick | Speed | Weights | License | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | **UnCRtainTS** ⭐ | S1+S2 temporal | Med | High | High | High | High (light TAE) | yes | permissive `[?]` | **V.High (uncertainty)** |
| M2 | PMAA (temporal) | multitemporal | High | Med | High | Med-High | V.High | yes | `[?]` | High |
| M3 | CTGAN | multitemporal | Med | Med | High | Med | Low | yes | `[?]` | Medium |
| M4 | U-TILISE | temporal seq2seq | Med | Med-High | High | High | Med | yes | `[?]` | High (temporal) |
| M5 | Temporal U-Net / Bishift / tensor-completion | temporal | Med | Med | Med | Med-High | High | partial | various | Classical sanity checks |

**Family takeaway.** When a temporal stack exists, temporal attention beats single-image. **UnCRtainTS** is decisive: per-pixel **aleatoric uncertainty** (no GT at inference) turns the system into a credible operational product and feeds the cross-verification confidence map. Its uncertainty head is the pattern we wrap around any backbone.

---

## 6. Geospatial foundation models (transfer-learning backbones)

| # | FM | Pretrain | Bands | License | LISS-IV fit |
|---|---|---|---|---|---|
| F1 | **Prithvi-EO-2.0** (IBM-NASA) | HLS, temporal, MAE | B,G,R,NarrowNIR,SWIR1,SWIR2 | **Apache-2.0** | **V.High** (G/R/NIR map cleanly; drop blue/SWIR, re-init stem) |
| F2 | **DOFA** | multimodal incl. SAR, hypernetwork | wavelength-conditioned (arbitrary) | open | **V.High** (native arbitrary bands + SAR, no stem surgery) |
| F3 | SatMAE++ | fMoW/Sentinel MS, multi-scale | MS | open | High (init) |
| F4 | Scale-MAE | RGB high-res + GSD pos-enc | RGB | open | High (cross-resolution fusion) |
| F5 | Clay v1.5 | multi-sensor, loc/time aware | flexible | **Apache-2.0** | High (permissive, flexible bands) |
| F6 | SpectralGPT | 1M spectral, 3D tokens | spectral | open | Med (heavy; if multispectral) |
| F7 | SatMAE | fMoW MS/temporal | MS | open | High (init) |

**Family takeaway.** FMs are our hedge against scarce pairs. **Prithvi-EO-2.0** (Apache-2.0, temporal, bands already include G/R/Narrow-NIR) and **DOFA** (wavelength hypernetwork natively ingests LISS-IV's odd 3-band + SAR) are the top transfer picks; we use them as encoder initializers + MAE self-supervised domain adaptation on unlabeled NER tiles.

---

## 7. Classical baselines (comparison-only, some reusable as components)

| # | Method | Use | LISS-IV role |
|---|---|---|---|
| K1 | Dark Channel Prior (dehaze) | physics, thin cloud | Baseline + thin-cloud component |
| K2 | Multitemporal median/best-pixel compositing | de-facto operational | Baseline + clear-target source |
| K3 | Histogram matching / SBAF | radiometric harmonization | **Component** (spectral-shift corrector on any output) |
| K4 | Poisson / seamless blending | gradient-domain donor blend | Baseline + seam fixer |

These quantify how much GenAI actually adds, and K2/K3/K4 are useful pipeline components regardless of the learned core.

---

## 8. Cloud/shadow masking (enabling, not reconstruction)

| Method | Sensors | LISS-IV fit |
|---|---|---|
| **OmniCloudMask** | sensor-agnostic, G/R/NIR @~5 m | **Primary** — only mask that runs on LISS-IV's exact 3 bands |
| s2cloudless / KappaMask / Fmask / Cloud-Net | S2 / Landsat (need blue/SWIR/thermal) | Run on the **S2 proxy**, transfer masks; Cloud-Net (R/G/B/NIR) retrainable on LISS-IV |
| DEM geometric shadow projection | any + DEM | **Primary** for cloud-shadow (sun-azimuth/elevation + terrain) |

---

## 9. Synthesis — Decision

### 9.1 What we IMPLEMENT in the product (the unified registry)
We implement **one model per family plus a baseline (six total)**, behind the single `BaseCloudRemovalModel` interface, so the comparative assessment is head-to-head on identical data/metrics. Each is a *clean, from-scratch, CPU-runnable* implementation that **optionally** loads external pretrained weights at runtime (license-safe):

| Registry name | Family | Synthesized from | Why this one |
|---|---|---|---|
| `unet` | CNN baseline | UNet (C7/C1), DSen2 lineage | **Mandatory control**; quantifies GenAI value-add and hallucination; always CPU-fast |
| `dsen2cr` | SAR-optical fusion (CNN/GAN) | **DSen2-CR (G8)** + Simulation-Fusion (G10) idea | Thick-cloud workhorse; **CARL mask-weighted loss** fits imperfect pairs; residual preserves spectra; SEN12MS-CR transfer; non-adversarial = stable |
| `spagan` | GAN | **SpA-GAN (G4)** + Pix2Pix (G1) | MIT, pretrained, spatial-attention texture/sharpness; thin-cloud specialist; light/CPU-trainable |
| `restormer` | Transformer / fusion | **Restormer (T1)** + **GLF-CR (T7)** + **Align-CR (T8)** fusion mechanics | Channel-attention core (no seams) + SAR cross-attention + deformable alignment for real misregistration; best long-range structure |
| `diffusion` | Diffusion | **DiffCR (D7)** + **EMRDM (D10)** mean-reverting + **CM-CR (D15)** consistency + SR3 (D2) concat | Best occlusion prior; DDIM few-step; mean-reverting for spectral fidelity; optional SAR conditioning + distillation for speed |
| `uncertainty` | Multitemporal / uncertainty wrapper | **UnCRtainTS (M1)** | Per-pixel aleatoric variance head wrappable on any backbone; the operational differentiator; feeds cross-verification confidence |

> **Naming convention.** The "Registry name" column is the short key (`dsen2cr`, `spagan`, `restormer`) used by the CLI / config `model.name` / serving API; the source file may carry the longer descriptive name (`dsen2cr_fusion.py`, `gan_spagan.py`, `transformer_restormer.py`).

**Cross-cutting, implemented as shared infrastructure (not separate registry models):**
- **Transfer backbone:** Prithvi-EO-2.0 / DOFA (F1/F2) as optional encoder initializer + MAE self-sup (`research/03 §5`).
- **Masking:** OmniCloudMask + DEM shadow projection (`research/02 §5`, `research/05 B.3`).
- **Synthetic clouds:** SatelliteCloudGenerator + CloudSEN12 copy-paste (`research/05 B.5`).
- **Temporal head & RePaint self-sup prior:** available as options on the diffusion/uncertainty paths.
- **Classical components:** histogram matching (K3) + Poisson blend (K4) in postprocess; compositing (K2) as a temporal prior and target source.

### 9.2 Why this set satisfies the deliverable
It covers **all required families** — diffusion, GAN, transformer-restoration, SAR-optical fusion, multitemporal/uncertainty, **plus a UNet/DSen2-CR baseline** — and each pick is the research-justified frontier point for its axis: DSen2-CR (thick-cloud + spectral safety), SpA-GAN (texture + data-efficiency + permissive license), Restormer/GLF-CR (structure + fusion), diffusion (occlusion prior + spectral-fidelity via mean-reverting), UnCRtainTS (uncertainty), UNet (control). Together they map the fidelity × speed × data-efficiency × thick-cloud frontier the jury wants compared.

### 9.3 What we CITE as comparison-only (not implemented, referenced in the report)
To keep scope tractable while demonstrating breadth, these are documented and benchmarked-against-via-published-numbers but not re-implemented:
- **Diffusion:** IDF-CR, DE/CUHK-CR, DC4CR, SeqDMs, DDPM-CR, SADER, SAR→optical/color-supervised/distilled variants (D5,D6,D8,D9,D11,D12,D13,D14,D16). DiffCR/EMRDM/CM-CR weights may be loaded as adapters if licenses confirm.
- **GAN/CNN:** pix2pixHD, McGAN, AMGAN-CR, CR-GAN-PM, Cloud-GAN, SAR-Opt-cGAN, Simulation-Fusion GAN, SAR2Opt, AttentionGAN, STGAN, CTGAN, PMAA, PartialConv, DeepFill v2, EdgeConnect, MPRNet, RDN (G2,G3,G5,G6,G7,G9,G10,G11,G12,G13,G14,C2–C7). PMAA / CR-GAN-PM are "fast-follow" candidates if a temporal/unpaired regime is prioritized.
- **Transformer/temporal:** SwinIR, Uformer, MAT, ICT, MSDA-CR, U-TILISE, Bishift/tensor-completion (T2,T3,T5,T6,T9,M3,M4,M5).
- **Foundation models:** Prithvi/DOFA/Clay/SatMAE++/Scale-MAE/SpectralGPT used as *backbones/initializers*, not as standalone CR entries.
- **Classical:** DCP, compositing, histogram matching, Poisson blending (K1–K4) as baselines/components.

This gives the jury a **30+-method comparative landscape** with a focused, defensible **6-model implemented product** that exercises every GenAI family on identical LISS-IV-domain data and metrics.
