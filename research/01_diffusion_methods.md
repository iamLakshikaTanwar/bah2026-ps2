# Diffusion-Model-Based Cloud Removal & Satellite Image Reconstruction — Research Report

**Project:** BAH 2026 PS2 — Generative-AI Cloud Removal for LISS-IV (Resourcesat-2/2A, 3-band G/R/NIR, 5.8 m, North-East India).
**Scope of this report:** Diffusion / score-based generative methods only (GANs, transformers, pure-CNN baselines are covered in sibling reports).
**Date:** 2026-06-22. **Method verification:** Items marked **[verified]** were confirmed via web search/fetch in June 2026; items marked **[memory]** rely on the author's training knowledge (cutoff Jan 2026) and should be re-checked; **[unverified]** flags any specific figure I could not confirm online. Do not treat numeric metrics as exact unless cross-checked against the cited paper.

---

## 1. Why diffusion for cloud removal?

Cloud removal (CR) is a conditional image-restoration / inpainting problem: given a cloudy optical scene (plus optional auxiliary SAR, multitemporal, or DEM data), reconstruct the cloud-free surface reflectance. Diffusion models learn a stochastic denoising trajectory and have three properties that matter for our problem:

1. **Mode coverage & stable training** vs. GANs — no adversarial collapse, more reliable convergence on the small paired datasets typical of CR. The 2025 comparative review explicitly finds GANs prone to suboptimal image quality, while diffusion gives better stability **[verified]**.
2. **Strong priors for plausible texture** under thick cloud / total occlusion, where the surface signal is gone and the model must *hallucinate* a consistent surface — exactly the NER (North-Eastern Region) thick-cloud regime.
3. **Flexible conditioning** — concatenation, cross-attention, ControlNet, mean-reverting endpoints, and SAR fusion all plug into the same denoiser.

The central tension for us: diffusion's iterative sampling is expensive, and high-res 3-band LISS-IV with limited paired data stresses both data efficiency and spectral fidelity. The methods below are ordered roughly foundations → image-to-image → CR-specific → fusion → acceleration.

---

## 2. Foundational diffusion methods (the building blocks)

### 2.1 DDPM / Score-SDE (foundations)
- **Year/venue:** DDPM — Ho et al., NeurIPS 2020; Score-SDE — Song et al., ICLR 2021. **[memory]**
- **Core idea:** Forward process gradually adds Gaussian noise over T steps; a U-Net learns to predict the noise (ε-prediction) / score so the reverse process can denoise pure noise back into a sample.
- **Conditioning:** Unconditional originally; conditioning added downstream via concatenation, guidance, or cross-attention.
- **Backbone:** Time-conditioned U-Net (sinusoidal time embeddings, residual blocks, self-attention at low resolutions).
- **Loss:** Simplified ε-MSE (`L_simple`), optionally variational `L_vlb`.
- **Relevance to LISS-IV:** The substrate every CR method below inherits. We will implement the denoiser as a 3-band U-Net; input channels trivially set to 3 (or 3+SAR). Pixel-space DDPM at 256–512 px is too slow for production but fine as a research baseline.

### 2.2 SR3 — Image Super-Resolution via Iterative Refinement
- **Year/venue:** Saharia et al., Google Brain — arXiv 2021, IEEE TPAMI 2022. **[verified]**
- **Core idea:** Adapts DDPM to image-to-image translation. The low-resolution (here: degraded/conditional) image is **concatenated** to the noisy target at every step; a U-Net iteratively refines from Gaussian noise conditioned on the LR input.
- **Conditioning:** Mono-image (LR → HR). Directly generalizes to (cloudy → cloud-free) by swapping the conditioning image.
- **Loss:** ε-MSE (the paper found L1 on noise competitive). **Datasets:** CelebA-HQ, ImageNet, natural images. **Metrics:** ~50% human "fool rate" on 8× CelebA face SR, beating GAN baselines (≤34%) **[verified]**.
- **Code:** Unofficial but widely used `Janspiry/Image-Super-Resolution-via-Iterative-Refinement` (MIT) **[memory]**.
- **Applicability:** **High.** SR3's concatenation-conditioning recipe is the template most CR papers (incl. DiffCR) adopt. It also doubles as a super-resolution path if we ever want to sharpen LISS-IV or fuse with coarser Sentinel-2.
- **Compute:** Pixel-space, 1000-step training; slow inference (mitigate with DDIM). Single 256² model trains on one 16–24 GB GPU.

### 2.3 Palette — Image-to-Image Diffusion
- **Year/venue:** Saharia et al., SIGGRAPH 2022. **[verified]**
- **Core idea:** A single multi-task image-to-image diffusion framework for colorization, **inpainting**, uncropping, and JPEG restoration; conditions by direct concatenation of the reference image with the noisy target. Establishes that one generic conditional diffusion recipe beats task-specific GANs.
- **Conditioning:** Mono-image, concatenation. Loss: L1 ε-prediction found best. No task-specific architecture changes.
- **Applicability:** **High (conceptual).** Cloud removal is essentially guided inpainting + restoration — Palette is the canonical proof that this works. We inherit its "concatenate the cloudy image, predict noise with L1" recipe. No RS pretrained weights, so transfer is conceptual, not weights-level.
- **Code:** Unofficial `Janspiry/Palette-Image-to-Image-Diffusion-Models` (MIT) **[memory]**.

### 2.4 RePaint — DDPM Inpainting via Resampling
- **Year/venue:** Lugmayr et al. (ETH Zürich), CVPR 2022. **[verified]**
- **Core idea:** Uses a **pretrained, frozen unconditional DDPM** as a prior. Only the reverse process is altered: known (unmasked) pixels are sampled from the input image's forward diffusion, generated pixels are denoised normally, and a **resampling ("jump back") schedule** harmonizes the boundary. Handles extreme/free-form masks; beats AR and GAN inpainters on 5/6 mask types **[verified]**.
- **Conditioning:** Mask + known pixels (no retraining). Backbone: any DDPM. Code: `andreas128/RePaint` (research license; also in HF `diffusers`) **[verified]**.
- **Applicability:** **Medium.** Cloud masks make CR a masked-inpainting task, and RePaint's training-free conditioning is attractive when paired data is scarce — we could pretrain an *unconditional* LISS-IV diffusion prior on abundant cloud-free NER scenes (no pairs needed!) and inpaint cloudy regions. **Weakness:** assumes the masked region is truly unknown; it ignores residual signal under thin cloud/haze and the resampling loop multiplies inference cost (often 5–10× the step budget).

---

## 3. Diffusion methods purpose-built for cloud removal

### 3.1 SeqDMs — Sequential-based Diffusion Models
- **Year/venue:** Zhao & Jia, *Remote Sensing* (MDPI) 2023 (rs15112861). **[verified]**
- **Core idea:** Multi-modal diffusion models (MmDMs) + a **Sequential Training & Inference Strategy (SeqTIS)** that integrates temporal information across arbitrary-length sequences of the main (optical) and auxiliary (SAR) modalities **without retraining** for different sequence lengths.
- **Conditioning:** **Multitemporal + SAR** (multi-modal). Backbone: conditional U-Net. **Datasets:** SEN12MS-CR-TS. **Metrics:** reported competitive PSNR/SSIM on multitemporal CR **[unverified exact values]**.
- **Applicability:** **Medium-High** *if* we have time series. NER revisit (Resourcesat ~5 day for LISS-III; LISS-IV ~5–24 day depending on mode) means multitemporal stacks are feasible but cloud-contaminated; SeqTIS's variable-length handling suits irregular clear-date availability. **Weakness:** needs co-registered multitemporal LISS-IV — non-trivial to assemble for NER.

### 3.2 DDPM-CR — Denoising Diffusion Probabilistic Feature network
- **Year/venue:** Jin/Zhu et al., *Remote Sensing* (MDPI) 2023 (rs-15-2217), "DDPM Feature-Based Network for Cloud Removal in Sentinel-2." **[verified]**
- **Core idea:** Uses a DDPM as a **feature extractor**: clouded optical + auxiliary SAR are fed through the DDPM to extract multi-scale "DDPM features," which a separate CR head fuses to remove both thin and thick cloud. Diffusion priors as features rather than end-to-end sampling.
- **Conditioning:** **SAR + optical fusion.** Backbone: DDPM encoder + CR network; loss combines reconstruction + cloud-aware terms. **Datasets:** Sentinel-2 + Sentinel-1.
- **Applicability:** **Medium.** The "diffusion-as-feature-extractor" trick sidesteps slow sampling — attractive for deployability. But it needs SAR co-registration and a higher parameter count; the 2025 literature notes DDPM-CR improves restoration "at the expense of substantially increased parameters/compute" **[verified]**.

### 3.3 DiffCR — Fast Conditional Diffusion for CR  ⭐
- **Year/venue:** Zou, Li et al., **IEEE TGRS 2024** (arXiv 2308.04417). **[verified]**
- **Core idea:** A *fast* conditional diffusion CR framework. Introduces (a) a **decoupled NAFNet-based encoder** that extracts a robust color/appearance representation from the cloudy conditional image, and (b) a **Time-and-Condition Fusion Block (TCFBlock)** that cheaply aligns condition↔target. Achieves SOTA while using only ~5.1% params and ~5.4% FLOPs of the prior best **[verified]**.
- **Conditioning:** Mono-image (cloudy → cloud-free), with a NAFNet double-encoder + split-channel U-Net. Loss: ε / restoration MSE. **Datasets:** Sen2_MTC_Old, Sen2_MTC_New, **SEN12MS-CR** **[verified]**. **Metrics:** SOTA PSNR/SSIM on those benchmarks (exact per-dataset values in the TGRS paper; e.g. high-20s to low-30s dB PSNR range typical for these sets) **[unverified exact values]**.
- **Code & weights:** `github.com/XavierJiezou/DiffCR` — **pretrained checkpoints on HuggingFace for all three datasets** (incl. a SEN12MS-CR model built on UnCRtainTS) **[verified]**. License not stated in repo front matter — **must confirm before reuse** **[verified: license unspecified]**.
- **Applicability to LISS-IV:** **Very high.** Designed precisely for the efficiency/quality trade-off we need; mono-image (no SAR/time-series dependency) matches our minimal-data scenario; lightweight enough to run tiled over large scenes. The **SEN12MS-CR pretrained weights are reusable for transfer learning** to LISS-IV's 3 bands (Sen-2 RGB+NIR overlap is large). Top deployability candidate.

### 3.4 IDF-CR — Iterative Diffusion, Divide-and-Conquer
- **Year/venue:** Wang, Song, Wei et al., **IEEE TGRS 2024** (arXiv 2403.11870). **[verified]**
- **Core idea:** Two-stage divide-and-conquer: a **pixel-space CR module (Pixel-CR)** does coarse cloud reduction, then a **latent-space diffusion stage** (Stable Diffusion refined with **ControlNet**) refines detail. Adds an **unsupervised Iterative Noise Refinement (INR)** module to optimize the predicted-noise distribution for detail recovery.
- **Conditioning:** Mono-image; ControlNet provides spatial conditioning on the coarse CR result. Backbone: SD + ControlNet. **Datasets:** optical RS CR benchmarks (incl. RICE). **Metrics:** SOTA vs. CR + inpainting baselines **[verified qualitatively; exact values unverified]**.
- **Applicability:** **Medium-High.** The latent-space stage is far cheaper than pixel-space at high res, and the ControlNet pattern is a clean way to inject DEM/SAR later. **Weakness:** depends on SD's RGB latent space (3-band) — NIR handling and spectral fidelity need care; INR adds iterations.

### 3.5 DE (Diffusion Enhancement) — Ultra-Resolution CR (CUHK-CR)
- **Year/venue:** Sui et al., **IEEE TGRS 2024** (arXiv 2401.15105). **[verified]**
- **Core idea:** Progressive **texture-detail recovery** diffusion for *ultra-resolution* RS; adds a **Weight-Allocation (WA) network** for dynamic feature-fusion weighting and a **coarse-to-fine training** schedule to cut training cost. Ships the **CUHK-CR benchmark at 0.5 m** with thin/thick splits **[verified]**.
- **Conditioning:** Mono-image. **Datasets:** CUHK-CR (0.5 m), RICE. **Metrics:** beats prior DL CR on perceptual + fidelity on CUHK-CR/RICE **[verified qualitatively]**.
- **Code/weights:** `github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal` (dataset + method) **[verified]**.
- **Applicability:** **High.** This is the **closest match to our high-res, fine-texture LISS-IV regime** — 0.5 m is even sharper than 5.8 m, so its detail-preservation machinery directly addresses our "preserve fine spatial detail" requirement. Reusable dataset + method for benchmarking. **Weakness:** RGB-centric; ultra-res training is heavy.

### 3.6 EMRDM — Improved Mean-Reverting Diffusion (Elucidated)  ⭐
- **Year/venue:** Liu et al., **CVPR 2025** (arXiv 2503.23717). **[verified]**
- **Core idea:** A **mean-reverting diffusion model (MRDM)**: instead of diffusing to pure noise, the forward process reverts the *cloud-free* image toward the *cloudy* image, so sampling is a **direct cloudy→clear** trajectory (preserves residual surface signal). Reformulated as an **EDM-style "elucidated design space"** — modular, preconditioned denoiser, ODE-based deterministic + stochastic samplers, and a multi-image denoiser for **multitemporal** CR **[verified]**.
- **Conditioning:** Mono- and multi-temporal (no explicit SAR in base). Backbone: preconditioned EDM denoiser. **Datasets:** SEN12MS-CR / SEN12MS-CR-TS **[verified]**. **Metrics:** reports superior PSNR/SSIM/SAM/MAE vs. prior diffusion CR incl. DiffCR **[verified qualitatively; exact values unverified]**.
- **Code:** `github.com/Ly403/EMRDM` **[verified]**.
- **Applicability:** **Very high.** Mean-reverting endpoints = **better spectral fidelity** (the model starts from the cloudy radiometry, not noise) and **few sampling steps** thanks to EDM ODE samplers — both critical for LISS-IV's spectral-consistency and deployability requirements. Strong candidate; the multitemporal denoiser is an option if we build time series.

### 3.7 DC4CR — Diffusion Control for CR (prompt-driven)
- **Year/venue:** "When Cloud Removal Meets Diffusion Model in Remote Sensing," arXiv 2504.14785, 2025. **[verified]**
- **Core idea:** A **multimodal, prompt-driven** diffusion control framework enabling **selective** removal of thin vs. thick cloud **without pre-generated masks**. Uses **LoRA** for cheap fine-tuning, **subject-driven generation** for generalization, and **grouped learning** to perform well on **small datasets**; designed as a plug-and-play module **[verified]**.
- **Conditioning:** Prompt + optical (multimodal). Backbone: controllable diffusion (SD-family). **Metrics:** SOTA on RS CR benchmarks **[verified qualitatively]**.
- **Applicability:** **High for our data-scarce setting.** Its explicit emphasis on **small-dataset training (grouped learning) + LoRA efficiency** directly targets our "limited paired data" constraint, and prompt control could let operators choose thin-vs-thick behavior over NER. **Weakness:** SD-RGB latent; prompt-control reproducibility on 3-band scientific data is unproven.

### 3.8 SADER — Structure-Aware Diffusion w/ Deterministic Resampling
- **Year/venue:** arXiv 2602.00536 (2026), multi-temporal RS CR. **[verified — very recent]**
- **Core idea:** Structure-aware diffusion with a **deterministic resampling** scheme for **multi-temporal** CR (RePaint-style resampling made deterministic to cut stochastic cost while preserving structure).
- **Conditioning:** Multitemporal optical. **Applicability:** **Medium** (watch this one — it is the newest multitemporal SOTA as of 2026; details still thin). **[memory/early]**

---

## 4. SAR-conditioned & SAR-to-optical diffusion (fusion)

SAR penetrates cloud, so SAR→optical translation and SAR-fused CR are the backbone of *thick-cloud* reconstruction. Auxiliary Sentinel-1 is explicitly available in our PS2 brief.

### 4.1 Conditional Diffusion for SAR-to-Optical Translation
- **Year/venue:** Bai et al., IEEE GRSL 2024 (RG 376025587). **[verified]**
- **Core idea:** SAR image injected as a **conditional constraint in the sampling process** to translate SAR→optical. Backbone: conditional DDPM. **Datasets:** SEN12 (282,384 S1/S2 pairs) **[verified]**. Code: `github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation` **[verified]**.
- **Applicability:** **High for thick-cloud zones** in NER where optical is fully lost — SAR is the only surface evidence. **Weakness:** SAR→optical is severely ill-posed (radiometric ambiguity); spectral fidelity (esp. NIR) is hard. Best as an *auxiliary* conditioner, not sole input.

### 4.2 Color-Supervised SAR-to-Optical Diffusion
- **Year/venue:** arXiv 2407.16921, 2024. **[verified]** — adds **color supervision** to sharpen boundaries and **reduce color shift** in SAR→optical diffusion. Directly addresses the spectral-shift failure mode that would corrupt LISS-IV reflectance. **[verified qualitatively]**

### 4.3 Adversarial-Consistency-Distilled SAR-to-Optical
- **Year/venue:** arXiv 2407.06095, 2024 — **accelerates** SAR→optical diffusion via **adversarial consistency distillation** (few-step student). Bridges to §6 acceleration. **[verified]**

### 4.4 CM-CR — SAR-Conditioned Consistency Model  ⭐ (fast)
- **Year/venue:** *Remote Sensing* (MDPI) 2025 (rs-17-22-3721). **[verified]**
- **Core idea:** **Distills a SAR-conditioned diffusion teacher into a consistency-model student** for **few-step** (≈8–16 step) CR, reporting SOTA PSNR/SSIM/SAM/MAE with far fewer sampling steps **[verified]**. Conditioning: **SAR + optical**.
- **Applicability:** **Very high for deployability.** Combines the two things we need most — SAR fusion for thick cloud *and* near-real-time inference. Likely the best "fast + fused" template for a production LISS-IV service. **[fetch blocked by paywall — metrics from search summary, re-verify]**.

### 4.5 EDM-CR / SAR-DeCR — Latent SAR-fused thick-cloud diffusion (2025)
- **EDM-CR:** efficient diffusion fusing S1 SAR + S2 optical (ScienceDirect, *RSE*-adjacent, 2025) **[verified title]**. **SAR-DeCR:** **latent** diffusion for SAR-fused **thick**-cloud removal (*Remote Sensing* 2025, rs17132241) **[verified]**. Both confirm the 2025 trend: **latent-space + SAR fusion + thick-cloud focus** — the regime that matches NER best.

---

## 5. Comparative table of diffusion CR methods

| Method | Yr | Venue | Conditioning | Backbone | Space | Steps (infer) | Datasets | Code/Weights | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|---|
| DDPM / Score-SDE | '20/'21 | NeurIPS/ICLR | uncond. | U-Net | pixel | ~1000 | — | many (MIT) | substrate |
| SR3 | '21 | TPAMI'22 | mono (concat) | U-Net | pixel | ~100–1000 | CelebA/INet | unofficial MIT | template |
| Palette | '22 | SIGGRAPH | mono (concat) | U-Net | pixel | ~1000 | multi-task | unofficial MIT | template |
| RePaint | '22 | CVPR | mask (frozen prior) | DDPM | pixel | ↑↑ (resample) | ImageNet/CelebA | andreas128 (research) | self-sup. prior |
| SeqDMs | '23 | Remote Sens. | multitemporal+SAR | U-Net | pixel | ~1000 | SEN12MS-CR-TS | — | if time series |
| DDPM-CR | '23 | Remote Sens. | SAR+optical (feat.) | DDPM-enc+head | pixel | n/a (feat.) | S2+S1 | partial | medium |
| **DiffCR** ⭐ | '24 | **TGRS** | mono | NAFNet+TCFBlock | pixel | few (fast) | Sen2_MTC, SEN12MS-CR | **HF weights** | **very high** |
| IDF-CR | '24 | TGRS | mono (ControlNet) | SD+ControlNet | latent | mod.+INR | RICE/optical | yes | med-high |
| DE (CUHK-CR) | '24 | TGRS | mono | DDPM+WA net | pixel | coarse-to-fine | CUHK-CR 0.5 m, RICE | littlebeen | high |
| **EMRDM** ⭐ | '25 | **CVPR** | mono/multitemp (mean-revert) | EDM denoiser | pixel | few (ODE) | SEN12MS-CR(-TS) | Ly403 | **very high** |
| DC4CR | '25 | arXiv | prompt+optical (LoRA) | SD-control | latent | mod. | RS CR sets | — | high (small-data) |
| SADER | '26 | arXiv | multitemporal | struct.-aware diff. | — | det. resample | multitemp RS | — | medium (new) |
| SAR→Opt cond. diff | '24 | GRSL | SAR | DDPM | pixel | ~1000 | SEN12 | Coordi777 | thick-cloud aux |
| **CM-CR** ⭐ | '25 | Remote Sens. | SAR+optical (consistency) | distilled student | latent? | **~8–16** | SAR/opt CR | (paper) | **very high (fast+fused)** |
| SAR-DeCR / EDM-CR | '25 | RS / RSE | SAR+optical | latent diffusion | latent | mod. | S1+S2 | — | high (thick) |

*Metrics column omitted from table because exact PSNR/SSIM/SAM vary per dataset/split and several could not be verified online; see per-method notes and §8. As anchors on SEN12MS-CR, non-diffusion SAR-fused GLF-CR reports PSNR ≈ 28.6 dB / SSIM ≈ 0.885 **[verified]**, and diffusion methods (DiffCR, EMRDM) report parity-to-better with far fewer params (DiffCR) or better fidelity (EMRDM).*

---

## 6. Fast / near-O(1) inference techniques

Diffusion's Achilles heel is the T-step loop. For a deployable LISS-IV service we stack the following:

- **Latent vs. pixel space.** Running diffusion in a VAE latent (SD/LDM, IDF-CR, SAR-DeCR, DC4CR) cuts spatial dims ~8× per side → ~64× fewer FLOPs/step and enables higher resolution. **Caveat:** the SD VAE is RGB-trained; for 3-band G/R/NIR with reflectance fidelity we must **retrain/fine-tune the VAE** or risk spectral leakage — a real cost and risk.
- **Step reduction.**
  - **DDIM** (deterministic, non-Markovian) → 20–50 steps with minimal quality loss; drop-in for any ε-model. First thing to enable.
  - **Consistency models / consistency distillation** → 1–4 steps (CM-CR already does this for SAR-fused CR; adversarial-consistency distillation for SAR→optical). Best speed/quality frontier.
  - **EDM ODE samplers** (Heun, 2nd-order) → ~10–35 NFE; EMRDM's elucidated design uses these for few-step deterministic sampling.
  - **Cold-diffusion / mean-reverting** endpoints (EMRDM) shorten the trajectory because sampling starts from the cloudy image, not noise.
  - **Progressive/coarse-to-fine** (DE) reduces effective steps at high res.
- **Patch-wise tiled inference.** LISS-IV scenes are huge (tens of thousands of px). Tile into 256/512 patches with **overlap + feathered (Gaussian/Hann) blending** to avoid seams; **fix the noise seed per tile** for consistency; mask-aware tiling skips clear tiles entirely. Essential for whole-scene throughput.
- **Classifier-free guidance (CFG) cost.** CFG doubles NFE (cond + uncond pass). For CR we often **don't need CFG** (the cloudy image is strong conditioning); drop it or use low guidance to halve cost.
- **Latent / feature caching.** Cache the VAE-encoded conditioning latent and SAR/DEM feature maps once per tile (reused across all steps). Cache time-invariant condition encoders (DiffCR's decoupled encoder runs once, not per step).
- **Engine-level.** **fp16/bf16** inference (≈2× + memory), **ONNX export → TensorRT** for the U-Net (fused kernels, INT8 calibration optional), **torch.compile**, and batched-tile execution on the GPU. A NAFNet-light backbone (DiffCR) + 4-step consistency + fp16 TensorRT + tiled inference is a realistic path to seconds-per-scene.

---

## 7. Top 2–3 diffusion picks for the LISS-IV pipeline

**Pick #1 — DiffCR (efficiency + transferable weights).**
Mono-image (no hard SAR/time-series dependency), TGRS-validated, ~5% params/FLOPs of prior SOTA, and — crucially — **public SEN12MS-CR pretrained checkpoints** we can fine-tune to LISS-IV's 3 bands (large RGB+NIR overlap with Sentinel-2). Best *data-efficiency × deployability* per unit of engineering effort. **Action:** confirm repo license, fine-tune the SEN12MS-CR model on whatever paired LISS-IV/Bhoonidhi we assemble, run tiled + DDIM/few-step.

**Pick #2 — EMRDM (spectral fidelity + few-step).**
Mean-reverting + EDM ODE sampling gives (a) better **spectral consistency** (sampling begins from the cloudy radiometry, not noise — directly serves our reflectance-fidelity requirement) and (b) genuinely **few-step** inference. CVPR-2025 quality, public code, optional multitemporal denoiser if we later build NER time series. Best *spectral fidelity* candidate.

**Pick #3 — CM-CR (SAR-fused, fast) for the thick-cloud / NER regime.**
NER is dominated by thick, persistent cloud where optical signal is gone; SAR fusion is then essential, and CM-CR delivers **SAR-conditioned CR in ~8–16 steps** via consistency distillation — fusion *and* speed together. Use it (or distill our own DiffCR/EMRDM teacher into a consistency student) as the production thick-cloud path; pair with **color-supervised SAR→optical** ideas to suppress spectral shift.

**Strategy:** Start from **DiffCR** (fast win, transfer weights), benchmark against **EMRDM** for fidelity, and add **SAR conditioning + consistency distillation** (CM-CR pattern) for thick-cloud NER and real-time inference. RePaint/Palette inform a **self-supervised unconditional LISS-IV prior** to offset scarce paired data; ControlNet (IDF-CR/DC4CR) is the clean hook for DEM/SAR conditioning later.

---

## 8. Open problems & pitfalls for diffusion CR on high-res 3-band data

1. **Spectral fidelity / hallucination.** Generative priors can produce visually plausible but **radiometrically wrong** reflectance — fatal for NDVI and downstream analysis-ready products. Mitigate with mean-reverting models (EMRDM), spectral/SAM losses, color supervision, and physical consistency checks. Always report **SAM** alongside PSNR/SSIM.
2. **RGB-latent / pretrained mismatch.** SD VAE and most pretrained diffusion weights are 3-channel **RGB** sRGB; LISS-IV is **G/R/NIR reflectance**. Naïvely loading weights mis-maps the NIR band. Requires VAE/first-conv surgery or full retrain — a real data/compute cost and a top integration risk.
3. **Scarce paired LISS-IV data over NER.** Few cloud-free references in persistently cloudy regions → overfitting. Lean on transfer (DiffCR/EMRDM SEN12MS-CR weights), self-supervised priors (RePaint-style), LoRA + grouped learning (DC4CR), and synthetic cloud augmentation.
4. **Thick cloud = ill-posed reconstruction.** Under total occlusion only SAR/temporal priors carry surface info; SAR→optical is radiometrically ambiguous (esp. NIR). Treat thick-cloud outputs as **uncertain**; consider uncertainty quantification (cf. UnCRtainTS) and per-pixel confidence masks.
5. **Inference cost vs. scene size.** Whole LISS-IV scenes need tiling; naïve diffusion is too slow. Step reduction (DDIM/consistency/EDM) + fp16/TensorRT + latent caching are mandatory, not optional.
6. **Tiling artifacts & global consistency.** Independent tiles cause seams and inconsistent global illumination/haze; needs overlap-feathering, shared seeds, and possibly a low-res global guidance pass.
7. **Cloud/shadow mask dependence & co-registration.** Many methods need clean masks and pixel-accurate SAR/temporal co-registration; LISS-IV↔Sentinel-1 mis-registration and terrain (NER is mountainous → SAR layover/shadow) degrade fusion. Mask-free approaches (DC4CR) and DEM-aware terrain correction help.
8. **Evaluation gaps.** No LISS-IV CR benchmark exists; SEN12MS-CR/RICE/CUHK-CR are proxies with different sensors/resolutions. We will likely need to **build a small NER LISS-IV eval set** and report PSNR/SSIM/SAM + a downstream task (e.g., NDVI/land-cover agreement).

---

## Sources (verified online, June 2026)
- DiffCR — arXiv 2308.04417; IEEE TGRS 2024; repo `github.com/XavierJiezou/DiffCR` (HF weights).
- IDF-CR — arXiv 2403.11870; IEEE TGRS 2024.
- EMRDM — arXiv 2503.23717; CVPR 2025; repo `github.com/Ly403/EMRDM`.
- DE / CUHK-CR — arXiv 2401.15105; IEEE TGRS 2024; repo `github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal`.
- DC4CR / "When Cloud Removal Meets Diffusion Model in RS" — arXiv 2504.14785 (2025).
- SADER — arXiv 2602.00536 (2026).
- SeqDMs — *Remote Sensing* 2023, doi 10.3390/rs15112861.
- DDPM-CR — *Remote Sensing* 2023, MDPI rs-15-2217.
- SR3 — arXiv 2104.07636; IEEE TPAMI 2022; project `iterative-refinement.github.io`.
- Palette — Saharia et al., SIGGRAPH 2022.
- RePaint — arXiv 2201.09865; CVPR 2022; repo `github.com/andreas128/RePaint`; HF `diffusers`.
- Consistency Models — arXiv 2303.01469; EDM — Karras et al. 2022.
- SAR→optical conditional diffusion — RG 376025587 (GRSL 2024); repo `github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation`; color-supervised arXiv 2407.16921; adversarial-consistency distillation arXiv 2407.06095.
- CM-CR (SAR-conditioned consistency) — *Remote Sensing* 2025, MDPI rs-17-22-3721.
- SAR-DeCR — *Remote Sensing* 2025, doi 10.3390/rs17132241; EDM-CR — ScienceDirect S0034425725004535 (2025).
- SEN12MS-CR / SEN12MS-CR-TS datasets — `patricktum.github.io/cloud_removal`; arXiv 2201.09613.
- GLF-CR (SAR-fused baseline anchor) — arXiv 2206.02850; ISPRS J. 2022.
- Comparative review GANs vs diffusion CR — ScienceDirect S2667393225000298 (2025).
