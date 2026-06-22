# 03 — Transformer / Attention, SAR–Optical Fusion, and Multi-Temporal Cloud Removal

**Project:** BAH 2026 PS2 — Generative-AI cloud removal & reconstruction for LISS-IV (5.8 m, 3-band G/R/NIR, Resourcesat-2/2A) over the persistently cloudy North-Eastern Region (NER).
**This file covers three pillars:** (a) transformer/attention restoration backbones, (b) SAR–optical multimodal fusion, (c) multi-temporal reconstruction. Plus geospatial foundation models (FMs), fast-inference tricks, top picks, and pitfalls.

**Provenance tags:** `[V]` = verified this session via web search; `[M]` = from internal knowledge (treat as memory, double-check before citing); `[U]` = uncertain / unverified (do not hard-cite URLs). Metrics are dataset-specific (mostly Sentinel-2, 13-band, 10 m) and are **not** directly transferable to LISS-IV — use as relative signal only.

---

## 1. Why these three families matter for LISS-IV/NER

NER is among the cloudiest regions on Earth; long stretches have **no clean mono-temporal optical view**. That makes pure single-image inpainting fragile under thick cloud. Three levers help:

1. **Long-range attention** reconstructs large occluded regions using distant cloud-free context and learned scene priors (transformers >> CNN receptive fields).
2. **SAR (Sentinel-1 C-band) penetrates cloud**, giving an all-weather *structural* anchor where optical is blind — the single biggest win for thick-cloud NER.
3. **Multi-temporal** stacks exploit the fact that a pixel cloudy on date *t* is often clear on *t±k*; the network borrows real radiometry instead of hallucinating.

A strong LISS-IV pipeline likely combines all three: a transformer restoration core, SAR feature-level fusion for structure, and temporal references for spectra.

---

## 2. Transformer / attention restoration backbones

### 2.1 Restormer `[V]`
- **Year/venue:** CVPR 2022 (Oral). Zamir et al.
- **Core idea:** Efficient encoder–decoder transformer for *high-resolution* restoration **without** splitting into windows; applies attention across **channels** instead of spatial tokens to keep cost linear in pixels.
- **Architecture:** **MDTA** (Multi-Dconv-head Transposed Attention) — self-attention over the channel dimension, depth-wise convs inject locality; **GDFN** (Gated-Dconv FFN) for controlled feature transform. Multi-scale (4-level) U-shape.
- **Inputs:** mono-image (no native SAR/temporal — must be adapted).
- **Losses:** L1 (task-dependent).
- **Datasets/metrics:** SOTA on deraining/deblurring/denoising (e.g. Rain100L ~ 38–39 dB PSNR) `[M]`. Not a cloud paper.
- **Code/license:** github.com/swz30/Restormer, MIT-style/permissive `[U-license]`.
- **Strengths:** Channel attention → cheap at full res (no window seams); excellent texture fidelity; easy to add input channels.
- **Weaknesses:** No explicit large-hole/mask handling; channel attention is weaker at *very* long spatial reasoning than true spatial attention.
- **Compute:** Moderate; scales gracefully to 256–512 tiles.
- **LISS-IV fit:** **High** — strong, well-engineered restoration core; change first conv to 3-band (+SAR channels), train as conditional reconstructor.

### 2.2 SwinIR `[V]`
- **Year/venue:** ICCV 2021 Workshop (AIM). Liang et al.
- **Core idea:** Swin-Transformer backbone for SR/denoise/JPEG; shallow conv + **RSTB** (Residual Swin Transformer Blocks) + reconstruction head.
- **Architecture:** Window self-attention (e.g. 8×8) + **shifted windows** for cross-window flow.
- **Inputs:** mono-image.
- **Losses:** L1 / Charbonnier; +perceptual/GAN for "real" variants.
- **Datasets/metrics:** SR/denoise benchmarks (Set5/14, BSD); strong PSNR/SSIM `[M]`.
- **Code/license:** github.com/JingyunLiang/SwinIR, Apache-2.0 `[M]`.
- **Strengths:** Proven, compact, lots of community variants/pretrained weights.
- **Weaknesses:** Window-boundary truncation (corner tokens lose context); fixed window size; quadratic within windows.
- **Compute:** Light–moderate.
- **LISS-IV fit:** **Medium-high** — great efficient backbone, but window seams matter for the fine 5.8 m texture we must preserve.

### 2.3 Uformer `[V]`
- **Year/venue:** CVPR 2022. Wang et al.
- **Core idea:** U-Net with **locally-enhanced window attention (LeWin)** + **locally-enhanced FFN (LeFF)** (depth-wise conv refers to neighbor pixels).
- **Architecture:** 8×8 windows; hierarchical U encoder/decoder; optional modulators.
- **Inputs:** mono-image.
- **Losses:** Charbonnier.
- **Datasets/metrics:** Denoise/derain/deblur SOTA-class `[M]`.
- **Code/license:** github.com/ZhendongWang6/Uformer, MIT `[M]`.
- **Strengths:** Multi-scale + locality; efficient; the skip-connection U-shape suits inpainting-like reconstruction.
- **Weaknesses:** Same window-attention limits as SwinIR.
- **LISS-IV fit:** **Medium-high** — good base for a SAR-conditioned encoder–decoder.

### 2.4 Swin Transformer (general backbone) `[V]`
- **Year/venue:** ICCV 2021 (Best Paper). Foundation for SwinIR/Uformer/many RS FMs.
- **Relevance:** Hierarchical shifted-window ViT; the de-facto efficient attention primitive. Many geospatial FMs (Satlas, some Prithvi variants) use Swin/ViT. Useful when adopting a pretrained Swin encoder.

### 2.5 MAT — Mask-Aware Transformer (inpainting) `[V]`
- **Year/venue:** CVPR 2022. Li et al.
- **Core idea:** Large-hole inpainting; attention aggregates non-local info **only from valid tokens** via a **dynamic mask** — directly analogous to "valid = cloud-free pixels."
- **Architecture:** Conv+transformer hybrid; "fusion learning" replaces LayerNorm for stable optimization; style manipulation for diverse fills.
- **Inputs:** image + binary mask (mono).
- **Params:** ~62 M (lighter than ICT 150 M / CoModGAN 109 M) `[V]`.
- **Datasets/metrics:** Places2, CelebA-HQ SOTA inpainting (FID/P-IDS) `[V]`.
- **Code/license:** github.com/fenglinglwb/MAT, permissive `[U-license]`.
- **Strengths:** **Cloud masks map exactly to inpainting masks**; handles very large holes; valid-token attention avoids contaminating fills with cloud pixels.
- **Weaknesses:** Trained on faces/natural scenes; geo-domain gap; no SAR/temporal; can hallucinate plausible-but-wrong structure (bad for science).
- **LISS-IV fit:** **Medium** — borrow the *mask-aware valid-token attention* idea rather than the weights.

### 2.6 ICT — Image Completion Transformer `[M]`
- **Year/venue:** ICCV 2021. Two-stage: transformer predicts low-res **appearance priors / tokens** (diverse, global), CNN upsamples/guides texture.
- **Relevance:** Token-based inpainting paradigm; ~150 M params (heavy). Concept (coarse global token completion → fine CNN refinement) is reusable for our coarse-to-fine GenAI stage. `[U-license]`.

### 2.7 MAE for remote sensing — SatMAE / Scale-MAE / SatMAE++ `[V]`
Masked-Autoencoder pretext (mask 60–75% patches, reconstruct) is the dominant **self-supervised** recipe for RS and is directly relevant: cloud removal *is* masked reconstruction.
- **SatMAE** (NeurIPS 2022): temporal + multispectral MAE; **band-grouping** by resolution + spectral-aware encoding; temporal embeddings. `[V]`
- **Scale-MAE** (ICCV 2023): **GSD positional encoding** (encodes true ground scale, not pixel count) + Laplacian-pyramid decoder for multi-scale recon. Good when fusing 5.8 m LISS-IV with 10 m Sentinel-2. `[V]`
- **SatMAE++** (CVPR 2024): multi-scale pretraining + conv upsampling, works for optical **and** multispectral; +2.5 % mAP BigEarthNet. github.com/techmn/satmae_pp. `[V]`
> Use these as **encoder initializers** and as the **pretraining strategy** (MAE on unlabeled LISS-IV/Sentinel tiles) to beat the scarce-pairs problem.

---

## 3. SAR–optical fusion methods (cloud-removal-specific)

### 3.1 DSen2-CR — fusion baseline `[V]`
- **Year/venue:** ISPRS J. 2020 (ISPRS Best Paper / U.V. Helava award). Meraner et al.
- **Core idea:** Deep **ResNet** predicts a residual correction to cloudy Sentinel-2; **Sentinel-1 SAR concatenated as prior** to guide reconstruction under optically opaque cloud.
- **Architecture:** EDSR/DSen2-derived ResNet (~16 res-blocks), early **channel-concatenation** fusion (13 S2 + 2 S1 bands).
- **Losses:** Cloud-masked L1 (CARL loss emphasizing cloudy pixels).
- **Datasets/metrics:** SEN12MS-CR; strong MAE/PSNR baseline (the reference everyone compares to). `[V]`
- **Code/license:** github.com/ameraner/dsen2-cr (Keras), permissive `[U-license]`.
- **Strengths:** Simple, robust, the canonical SAR-fusion baseline; residual learning preserves clear regions.
- **Weaknesses:** Early concat is a *weak* fusion (no cross-modal attention); CNN receptive field limits large holes; SAR speckle leaks through.
- **LISS-IV fit:** **High as a baseline** — first thing to reproduce (adapt 13→3 bands). Sets the bar.

### 3.2 GLF-CR — Global-Local Fusion transformer `[V]` ⭐
- **Year/venue:** ISPRS J. 2022. Xu et al. (TUM).
- **Core idea:** SAR-enhanced cloud removal with **two fusion paths**: **global fusion** keeps recovered structure consistent across all optical windows; **local fusion** injects SAR texture into cloudy areas, with **dynamic filtering** to suppress SAR speckle.
- **Architecture:** Transformer (Swin-style windows) + SAR-guided local feature transfer; explicitly tackles optical↔SAR domain gap.
- **Inputs:** SAR (S1) + single-date optical (S2).
- **Metrics:** ~**+1.7 dB PSNR** over prior SOTA on **SEN12MS-CR**. `[V]`
- **Code/license:** Official repo released (search "GLF-CR github"); license **unverified** `[U]`.
- **Strengths:** Purpose-built SAR-optical fusion; speckle-aware; attention handles structure consistency — *exactly our problem shape*.
- **Weaknesses:** Mono-temporal; assumes decent SAR-optical co-registration; window cost.
- **LISS-IV fit:** **Very high** — top conceptual template for our fusion stage (structure-from-SAR + spectra-from-optical with despeckling).

### 3.3 Align-CR — alignment-robust fusion `[V]`
- **Year/venue:** IEEE TGRS 2024 (M3R-CR benchmark, "Multi-modal & multi-resolution"). Xu et al.
- **Core idea:** **Low-res SAR-guided high-res optical** cloud removal that **progressively warps & fuses** multi-resolution features using **deformable convolution** to fix misalignment — instead of assuming pixel-perfect registration.
- **Datasets/metrics:** **M3R-CR** (high-res, multi-resolution) — best on most benchmarks. `[V]`
- **Strengths:** Directly targets the **co-registration problem** that will bite us (S1 10 m ↔ LISS-IV 5.8 m, different geometry/orbit).
- **Weaknesses:** More complex; deformable convs add cost.
- **LISS-IV fit:** **Very high** — its multi-resolution, misalignment-tolerant fusion is *more realistic for LISS-IV* than aligned-pair methods. Strong candidate to adapt.

### 3.4 MSDA-CR — Multiscale Distortion-Aware CR `[V]`
- **Year/venue:** ~2023–24 (RS journal). Multiscale grid of **Cloud-Distortion-Aware Representation Learning (CDARL)** modules; learnable **cloud distortion control functions** model how clouds corrupt imaging; attention fuses across scales.
- **Inputs:** optical (visible/multispectral); thin-cloud/haze oriented.
- **Strengths:** Explicitly models cloud-induced distortion (good for thin cloud/haze common in NER).
- **Weaknesses:** Less about thick-cloud/SAR. `[V, license U]`

### 3.5 SAR–optical similarity-attention & GAN translation `[V/M]`
- **Multimodal similarity attention** (ISPRS J. 2025) computes cross-modal similarity to weight SAR vs optical per-pixel. `[V]`
- **SAR-to-optical GAN translation** (e.g. Bermudez, Fuentes-Reyes) synthesizes fake optical from SAR to fill holes — fast but **spectrally unreliable**, prone to hallucination; use only as auxiliary prior. `[M]`

### 3.6 Diffusion fusion (rising SOTA) `[V]`
- **DiffCR** (IEEE TGRS 2024): fast conditional diffusion; **decoupled encoder** for color/condition; only ~5 % params/FLOPs of prior best, SOTA on benchmarks. github.com/XavierJiezou/DiffCR. `[V]`
- **SAR-DeCR / Multimodal Diffusion Bridge (2024-25):** latent diffusion with **attention-based SAR fusion** for thick cloud. `[V]`
- **Relevance:** Diffusion gives best perceptual quality but slower inference; cross-reference the diffusion research file. Listed here for the **SAR-fusion-via-cross-attention** mechanism.

---

## 4. Multi-temporal cloud removal

### 4.1 SEN12MS-CR-TS dataset `[V]`
- **Year/venue:** IEEE TGRS 2023. Ebel et al. (TUM).
- **What:** Multi-modal **multi-temporal** CR benchmark — **30 samples/ROI** time series of S1+S2. The standard for temporal CR. github.com/PatrickTUM/SEN12MS-CR-TS. `[V]`

### 4.2 UnCRtainTS — uncertainty-aware temporal CR `[V]` ⭐
- **Year/venue:** CVPRW 2023 (EarthVision). Ebel et al.
- **Core idea:** **Attention-based** network maps a **sequence of cloudy obs → one cloud-free image** AND outputs **per-pixel aleatoric uncertainty** (no GT needed at inference).
- **Architecture:** Lightweight temporal-attention encoder (L-TAE-style) + conv; probabilistic output head. Multi-modal (can take S1).
- **Losses:** Negative log-likelihood (uncertainty) + reconstruction.
- **Datasets/metrics:** SEN12MS-CR-TS SOTA-class; reports PSNR/SSIM/SAM + calibration. `[V]`
- **Code/license:** github.com/PatrickTUM/UnCRtainTS, permissive `[U-license]`.
- **Strengths:** **Uncertainty maps are gold for an operational ISRO product** (flag unreliable reconstructions); efficient; multi-modal.
- **Weaknesses:** Needs a temporal stack (NER revisit gaps); outputs can be smooth/conservative.
- **LISS-IV fit:** **Very high** if we can assemble LISS-IV (or LISS-IV + Sentinel) time series; the uncertainty head is a differentiator for the jury.

### 4.3 PMAA — Progressive Multi-scale Attention Autoencoder `[V]`
- **Year/venue:** ECAI 2023 (Oral). Zou et al.
- **Core idea:** Multi-temporal CR with **Multi-scale Attention Module (MAM)** (long-range, global+local) + **Local Interaction Module (LIM)** (fine detail).
- **Metrics:** Beats CTGAN on two benchmarks with only **0.5 % params & 14.6 % FLOPs** of CTGAN. `[V]`
- **Code/license:** github.com/XavierJiezou/PMAA, permissive `[U-license]`.
- **Strengths:** **Extremely efficient** multi-temporal model; good fine-detail retention (matters for 5.8 m).
- **Weaknesses:** Optical-only temporal (no SAR by default).
- **LISS-IV fit:** **High** — efficiency + detail focus aligns with our constraints.

### 4.4 CTGAN — Cloud Transformer GAN `[V]`
- **Year/venue:** 2022 (IEEE). Takes **3 temporal cloudy images → 1 cloud-free**; transformer token generator (group-conv multi-level features) + progressive-upsampling decoder learning **time-invariant** features; adversarial loss. `[V]`
- **Strengths:** Sharp temporal results.
- **Weaknesses:** GAN instability, hallucination in heavy cloud, **heavy** (PMAA dominates it on efficiency). `[U-license]`
- **LISS-IV fit:** **Medium** — useful baseline; prefer PMAA/UnCRtainTS.

### 4.5 Temporal U-Net / Bishift / tensor-completion `[V/M]`
- **Temporal U-Net + cloud-cover evolution simulation** (Sci. Rep. 2025) `[V]`; **Bishift Networks** (2023) and **frequency-spectrum tensor completion** (RS 2023) for multitemporal thick cloud `[V]`. Classical-leaning baselines; good sanity checks.

### 4.6 Temporal mechanics we must implement
- **Reference-frame selection:** pick the temporally **nearest low-cloud-fraction** scene (per-pixel cloud mask), or let **temporal attention** weight frames automatically (L-TAE / UnCRtainTS approach — more robust than hand-picking). `[V]`
- **Alignment:** image-level **optical-flow warping** (e.g. pretrained FlowNet warps frames to target date) + feature-level **deformable-conv** alignment for residual misregistration. `[V]`
- **Composite + GenAI refine:** build a robust cloud-free **median/least-cloud composite**, then a transformer/diffusion model **refines** seams, fills remaining gaps, and harmonizes radiometry (see §6).

---

## 5. Geospatial foundation models for transfer learning

Fine-tuning a pretrained geospatial FM is our best hedge against **scarce LISS-IV cloud/clear pairs**.

| FM | Year | Arch / pretext | Pretrain bands & data | License | Notes for 3-band LISS-IV |
|---|---|---|---|---|---|
| **Prithvi-EO-2.0** (IBM-NASA) | 2024 | ViT + **3D** patch/pos embeds, MAE; multi-temporal | 6 bands **B,G,R,NarrowNIR,SWIR1,SWIR2**; HLS V2 30 m, 4.2 M samples | open (Apache-2.0 on HF) `[V]` | Strongest temporal FM; bands **include G,R,NIR → maps to LISS-IV**; drop SWIR/Blue, re-init patch-embed for 3 bands. **Top transfer pick.** `[V]` |
| **SatMAE++** | 2024 (CVPR) | ViT-MAE, multi-scale + conv upsampling | fMoW-RGB/Sentinel multispectral | open (repo) `[V]` | Good multi-scale init; supports MS; adapt input stem to 3 bands. |
| **Scale-MAE** | 2023 (ICCV) | ViT-MAE + **GSD pos-enc**, Laplacian decoder | RGB high-res (fMoW etc.) `[V]` | open (repo) `[M]` | Best when mixing 5.8 m + 10 m scales; GSD encoding helps cross-resolution fusion. |
| **Clay v1.5** | 2024-25 | ViT, location+time aware, MAE-style | multi-sensor (S2/S1/Landsat/NAIP...) | **Apache-2.0** (weights+data) `[V]` | Permissive, commercial-OK; flexible band/sensor inputs; good general embeddings. `[V]` |
| **DOFA** (Dynamic-One-For-All) | 2024 | ViT + **hypernetwork** generating band-adaptive weights from wavelengths | multimodal (optical+SAR, varying bands) | open repo (zhu-xlab/DOFA) `[V]` | **Wavelength-conditioned** → natively handles arbitrary band sets incl. **SAR**; elegant for LISS-IV's odd 3-band + S1 mix. `[V]` |
| **SpectralGPT** | 2024 (TPAMI) | ViT-MAE, **3D spatial-spectral tokens**, >600 M params | 1 M spectral RS images | open (repo danfenghong/...) `[V]` | Spectral focus; heavier; useful if we lean multispectral (add Sentinel-2). `[V]` |
| **SatMAE** | 2022 (NeurIPS) | ViT-MAE, band-group + temporal | fMoW multispectral/temporal | open `[V]` | Predecessor of ++; still a solid multispectral/temporal init. |

**How to fine-tune on 3-band LISS-IV (G/R/NIR):**
1. Keep the pretrained transformer trunk; **replace the patch-embedding conv** to accept 3 channels (or, for DOFA, pass LISS-IV wavelengths to the hypernetwork — no surgery needed).
2. **Channel mapping:** LISS-IV G,R,NIR ≈ Prithvi's Green, Red, Narrow-NIR — initialize new stem weights from those pretrained channels (don't random-init).
3. Stage 1: continue **MAE self-supervised** pretraining on *unlabeled* LISS-IV + co-located Sentinel tiles (cheap, abundant) → domain-adapt the FM.
4. Stage 2: attach a lightweight decoder, fine-tune **supervised** on the scarce cloudy/clear (+SAR/temporal) pairs with cloud-masked losses.
5. Freeze early blocks first; unfreeze progressively (avoids overfitting tiny paired sets).

---

## 6. Focused: SAR–optical fusion for cloud removal — *why and how*

**Why.** Optical sensors see **spectra** (vegetation, water, soil via G/R/NIR) but are **blocked by cloud**. Sentinel-1 C-band SAR sees **geometry/roughness/moisture** and is **cloud-immune, day/night**. Principle: **structure-from-SAR + spectra-from-optical** — under cloud, SAR tells us *where* the edges/fields/rivers are; the network then paints physically plausible *spectra* consistent with surrounding cloud-free optical and the temporal history. This is exactly what DSen2-CR (concat), GLF-CR (global-local attention), and Align-CR (deformable warp) operationalize, in increasing fusion sophistication.

**How — preprocessing Sentinel-1 GRD for LISS-IV fusion `[V]`:**
1. **Acquire** S1 **GRD**, dual-pol **VV + VH** (and the VV/VH ratio as a 3rd channel). Cloud filtering is irrelevant for SAR. `[V]`
2. **SNAP pipeline:** orbit file → **radiometric calibration** (σ0) → **terrain correction** (Range-Doppler using **DEM** — Cartosat/SRTM; mandatory in hilly NER) → **speckle filtering** (Refined Lee / multi-temporal denoise). `[V]`
3. **Despeckle** before fusion (or let the network do dynamic-filter speckle suppression, GLF-CR-style) — raw speckle leaks into reconstructions.
4. **Co-register S1 → LISS-IV grid** (5.8 m). Expect sub-pixel/multi-pixel misalignment from different orbits/geometry/resolution → use **deformable / optical-flow** feature alignment (Align-CR) rather than assuming perfect overlay. Co-registration is the **#1 practical risk**. `[V]`
5. **Fusion level (choose):**
   - *Early* (channel concat, DSen2-CR): simplest, weak.
   - *Feature-level* (separate encoders, fuse mid-network): better.
   - *Cross-attention* (SAR keys/values attend optical queries, GLF-CR / SAR-DeCR): **recommended** — lets the model selectively pull SAR structure only inside cloudy regions while ignoring speckle elsewhere.

**Caveat:** SAR and optical are **physically different** — never force SAR to dictate radiometry directly; use it as a *structural guidance/conditioning* signal. Translation-GAN "fake optical from SAR" is spectrally unreliable.

---

## 7. Focused: Multi-temporal reconstruction — composites + GenAI refinement

**Step 1 — Per-pixel cloud/shadow masking** of every date in the stack (s2cloudless-style for Sentinel; train a small classifier for LISS-IV).
**Step 2 — Reference selection:** for each pixel, candidate set = all dates where it's clear. Prefer **temporally nearest clear** (minimize land-cover change) OR learn weights via **temporal attention** (UnCRtainTS / L-TAE) — attention is more robust to irregular NER revisits than fixed rules. `[V]`
**Step 3 — Alignment:** optical-flow warp all frames to the target acquisition geometry; deformable-conv for residual misalignment. `[V]`
**Step 4 — Composite:** robust **median / least-cloud-percentile** mosaic → a coarse, mostly-gap-free image (may still have all-dates-cloudy holes + seams + radiometric jumps).
**Step 5 — GenAI refinement:** a transformer (Restormer/Uformer core) or diffusion model takes {composite, target cloudy optical, SAR, per-frame masks} and (a) fills residual holes using SAR + learned priors, (b) **de-seams** and harmonizes radiometry to the target date, (c) restores 5.8 m fine detail. Train with cloud-masked L1 + SSIM + spectral (SAM) + perceptual losses; add the **uncertainty head** for QA.

This **composite-then-refine** design degrades gracefully: with a temporal stack it leans on real radiometry; with only SAR (no clear date) it leans on structure-from-SAR; pure-optical mono fallback still works via the transformer core.

---

## 8. Fast / low-cost inference techniques `[V]`

Operational ISRO scale (whole NER scenes) demands throughput.

- **Windowed attention** (Swin/Uformer): O(N) in pixels at fixed window — already efficient, but **boundary seams**; mitigate with shifted windows / overlap.
- **Channel-/transposed attention** (Restormer MDTA): attention over channels → linear in spatial size; ideal for full-res tiles.
- **Linear-attention drop-ins:** **PnP-Nystra** (Nyström, *training-free*) replaces MHSA in SwinIR/Uformer/Dehazeformer → **1.8–3.6× GPU / up to 7× CPU** speedup, minimal quality loss `[V]`; **ELFATT** ~4–7× over softmax attention `[V]`; **LinearSR** for stable linear-attention SR `[V]`.
- **FlashAttention:** exact attention, tiled in SRAM, no full matrix materialized → big memory/throughput win on long token sequences (use in any ViT/transformer trunk). `[V]`
- **Patch/tiling inference:** process large scenes in **overlapping tiles** + feather/blend (Gaussian weight) to hide seams; standard for RS at scale.
- **Knowledge distillation / compression / diffusion-step reduction:** distill a heavy diffusion/transformer teacher into a 1–few-step student; DiffCR-style lightweight conditioning already cuts FLOPs ~20×. `[V]`
- **Efficiency by design:** PMAA proves multi-temporal CR at **0.5 % params** of CTGAN — pick efficient architectures first, optimize attention second. `[V]`

---

## 9. Comparative table (relative; metrics are dataset-specific, **not** LISS-IV)

| Method | Yr | Family | Inputs | Attention/fusion | Key metric signal | Code license | LISS-IV fit |
|---|---|---|---|---|---|---|---|
| Restormer | 22 | Transformer core | mono | MDTA (channel) + GDFN | SOTA restoration `[M]` | permissive `[U]` | High (core) |
| SwinIR | 21 | Transformer core | mono | shifted-window | SOTA SR/denoise `[M]` | Apache-2.0 `[M]` | Med-high |
| Uformer | 22 | Transformer core | mono | LeWin window + LeFF | SOTA restoration `[M]` | MIT `[M]` | Med-high |
| MAT | 22 | Inpainting ViT | mono+mask | mask-aware valid-token | SOTA inpaint `[V]` | permissive `[U]` | Med (idea) |
| ICT | 21 | Inpainting ViT | mono+mask | token completion | strong inpaint `[M]` | `[U]` | Med (idea) |
| DSen2-CR | 20 | SAR-fusion CNN | S1+S2 mono | early concat (ResNet) | strong baseline `[V]` | permissive `[U]` | High (baseline) |
| **GLF-CR** | 22 | SAR-fusion transf. | S1+S2 mono | global+local fusion | **+1.7 dB PSNR** `[V]` | repo `[U]` | **Very high** |
| **Align-CR** | 24 | SAR-fusion (multi-res) | S1(lo)+opt(hi) | deformable warp fusion | best on M3R-CR `[V]` | `[U]` | **Very high** |
| MSDA-CR | 23 | Distortion-aware | optical | multiscale CDARL attn | strong (thin cloud) `[V]` | `[U]` | Med-high |
| CTGAN | 22 | Temporal GAN | 3× optical | token GAN | beaten by PMAA `[V]` | `[U]` | Medium |
| **PMAA** | 23 | Temporal attn AE | multi-temp opt | MAM+LIM | >CTGAN, 0.5% params `[V]` | repo `[U]` | High |
| **UnCRtainTS** | 23 | Temporal + uncert. | S1+S2 series | temporal attention | SOTA-class + uncert. `[V]` | permissive `[U]` | **Very high** |
| DiffCR | 24 | Diffusion | optical(+cond) | decoupled cond enc | SOTA, 5% FLOPs `[V]` | repo `[U]` | High (xref diffusion) |
| Prithvi-EO-2.0 | 24 | Geo-FM (ViT-MAE) | 6-band, temporal | 3D MAE | transfer FM `[V]` | Apache-2.0 `[V]` | **Very high (init)** |
| Scale-MAE / SatMAE++ | 23/24 | Geo-FM | MS / multi-scale | MAE | transfer FM `[V]` | open `[V]` | High (init) |
| Clay / DOFA | 24 | Geo-FM | multimodal | ViT / hypernet | transfer FM `[V]` | Apache-2.0 / open `[V]` | High (init; DOFA→SAR+3band) |

---

## 10. Top 2–3 transformer/fusion picks for our pipeline

1. **GLF-CR-style SAR–optical fusion transformer as the core CR engine.** It is purpose-built for our exact problem (SAR structure + optical spectra, speckle-aware global-local attention) and reports concrete gains on SEN12MS-CR. *Justification:* thick-cloud NER needs the SAR anchor; GLF-CR's local-fusion + dynamic speckle filtering + global structure-consistency is the cleanest published template. **Pair with Align-CR's deformable alignment** because LISS-IV↔S1 co-registration will be imperfect.

2. **UnCRtainTS for the multi-temporal head + uncertainty.** *Justification:* whenever a temporal stack exists, temporal attention beats single-image; and **per-pixel uncertainty maps** turn a research demo into a credible operational ISRO product (the jury will value "we flag where the reconstruction is unreliable"). Multi-modal (can ingest S1) and efficient.

3. **Prithvi-EO-2.0 (or DOFA) as the pretrained backbone / weight-init.** *Justification:* directly attacks **scarce LISS-IV pairs** via transfer learning; Prithvi's bands already include Green/Red/Narrow-NIR (clean mapping to LISS-IV) and it is Apache-2.0 + multi-temporal. **DOFA** is the wildcard — its wavelength-conditioned hypernetwork natively ingests arbitrary bands **and SAR**, eliminating patch-embed surgery.

**Concrete recommended stack:** Prithvi/DOFA-initialized encoder → Restormer/Uformer-style decoder → **cross-attention SAR fusion (GLF-CR) + deformable alignment (Align-CR)** → optional **temporal-attention + uncertainty (UnCRtainTS)** → composite-then-refine training (§7) → linear-attention/tiling for fast inference (§8).

---

## 11. Pitfalls

- **Co-registration error (LISS-IV 5.8 m ↔ S1 10 m, different geometry/orbit)** is the dominant practical failure; assume misalignment and use deformable/optical-flow alignment, not naive overlay.
- **SAR speckle** leaks into reconstructions — despeckle and/or use dynamic-filter/speckle-aware fusion.
- **Domain gap:** pretrained FMs/inpainters trained on RGB/natural scenes or 30 m HLS may not transfer cleanly to 5.8 m 3-band NER — *always* do MAE self-supervised domain-adaptation on LISS-IV first.
- **Metric transfer fallacy:** PSNR/SSIM/SAM in the table are on Sentinel-2 13-band/10 m datasets — they do **not** predict LISS-IV numbers; re-benchmark on our data.
- **Hallucination:** GANs/diffusion can invent plausible-but-false structures under thick cloud — guard with SAR conditioning, masked losses, and **uncertainty reporting**; never present hallucinated pixels as measured.
- **Spectral consistency:** optimize SAM/spectral-angle and per-band losses, not just RGB PSNR — NIR fidelity is critical for downstream vegetation/water science.
- **Window seams** (Swin/Uformer) at scene scale → overlap-tile + blend.
- **Temporal land-cover change:** distant reference frames may show genuine change (harvest, flood) — prefer nearest-clear + learned temporal weighting; don't average across seasons blindly.
- **Scarce pairs / overfitting:** tiny paired sets overfit fast — freeze-then-unfreeze FM blocks, heavy augmentation, self-supervised pretraining.
- **License diligence:** several CR repos have **unverified** licenses (`[U]`) — confirm before any redistribution/commercial ISRO use.

---

*Cross-references:* diffusion/GAN cloud-removal specifics, datasets (SEN12MS-CR, SEN12MS-CR-TS, M3R-CR, Bhoonidhi LISS-IV), and loss/metric definitions are covered in the companion research files.
