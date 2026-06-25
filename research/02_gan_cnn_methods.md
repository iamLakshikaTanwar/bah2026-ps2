# GAN-Based and CNN-Based Cloud Removal & Reconstruction (non-Transformer / non-Diffusion)

**Project:** BAH 2026 PS2 — Generative-AI Cloud Removal for LISS-IV (5.8 m, 3-band G/R/NIR, Resourcesat-2/2A) over the North-Eastern Region (NER).
**Scope of this file:** GAN and CNN families only. Transformer/diffusion families (GLF-CR, DiffCR, UnCRtainTS, CTGAN-Transformer) are covered by another agent and only cross-referenced here.
**Provenance tags:** [V] = verified this session via web search/fetch; [M] = from internal knowledge (treat as memory, re-verify before citing in the final report); links marked *(unverified)* were not opened this session.

---

## 1. Why GAN/CNN methods matter for LISS-IV

LISS-IV gives us 3 bands (Green, Red, NIR), 5.8 m GSD, frequent thick monsoon cloud over NER, and — critically — **very few perfectly co-registered cloudy/cloud-free pairs**. That constraint shapes everything below. GAN/CNN methods are attractive because they are (a) lighter and faster to train than diffusion models, (b) have abundant open-source reference code, and (c) several ship **pretrained weights** we can fine-tune. The main risks are GAN training instability and spectral/radiometric hallucination, which matter a lot when downstream users need physically meaningful NIR/NDVI values.

We organise methods into four buckets: (A) **mono-temporal optical-only GAN/CNN**, (B) **SAR–optical fusion GAN/CNN** (best for persistent cloud, since SAR sees through clouds), (C) **multitemporal GAN/CNN**, and (D) **generic image-inpainting/restoration backbones** repurposable for cloud holes.

---

## 2. Method-by-method review

### A. Mono-temporal, optical-only GAN/CNN

**1. Pix2Pix (Isola et al., CVPR 2017)** [V core/M details]
Core idea: paired image-to-image translation with a conditional GAN. Generator = U-Net encoder-decoder; discriminator = PatchGAN (70×70 receptive field). Loss = conditional adversarial + L1. Inputs: mono, paired. It is the *baseline* nearly every cloud-removal GAN compares against. Datasets: generic (maps, façades), but trivially retargetable to cloudy→clear. Strengths: simple, stable-ish, huge community. Weaknesses: needs pixel-aligned pairs, blurs fine texture, no spectral constraint. Repo: `phillipi/pix2pix`, `junyanz/pytorch-CycleGAN-and-pix2pix` (BSD-ish/non-commercial research license). **LISS-IV fit:** excellent didactic baseline; our pair scarcity is the limiting factor.

**2. pix2pixHD (Wang et al., CVPR 2018)** [M]
Core idea: high-resolution (2048×1024) paired translation. Coarse-to-fine generator (global + local enhancer), multi-scale discriminators, feature-matching + VGG perceptual loss. Inputs: mono, paired. Strengths: sharp high-res output, the multi-scale discriminator idea is reused widely (e.g., SAR-to-optical). Weaknesses: heavy, still paired. Repo: `NVIDIA/pix2pixHD` (BSD, non-commercial research). **LISS-IV fit:** the multi-scale-discriminator + feature-matching recipe is worth importing for 5.8 m detail preservation, even if we don't use the whole model.

**3. McGAN — Multispectral conditional GAN (Enomoto et al., CVPRW 2017)** [V]
Core idea: first dedicated cloud-removal cGAN; extends Pix2Pix to multispectral and uses NIR to recover visible bands; trains on *synthetic* (Perlin-noise) clouds over RGB+NIR. Generator U-Net, PatchGAN discriminator, adversarial+L1. Inputs: mono, multispectral, paired (synthetic). Datasets: synthetic-cloud satellite imagery. Strengths: shows NIR helps thin/filmy cloud; simple. Weaknesses: synthetic clouds ≠ real thick cloud; no SAR. Repos: `enomotokenji/mcgan-cvprw2017-pytorch` and `-chainer` (license: check, likely MIT/none) *(unverified license)*. **LISS-IV fit:** directly relevant — we also have NIR and can synthesise clouds when real pairs are missing; good warm-start strategy.

**4. SpA-GAN — Spatial Attention GAN (Pan, arXiv 2020)** [V]
Core idea: insert a **spatial attention network (SPANet-style)** into a cGAN generator so the model focuses on cloud-occluded pixels. Generator = attention-augmented (residual + spatial-attention blocks); discriminator = PatchGAN. Losses: adversarial + L1 + attention loss. Inputs: mono optical, paired. Datasets: RICE1/RICE2 + synthetic Perlin. **Reported [V]: RICE1 PSNR 30.23 / SSIM 0.954; RICE2 PSNR 28.37 / SSIM 0.906** — beats cGAN and CycleGAN. Repo: **`Penn000/SpA-GAN_for_cloud_removal`, MIT license, PyTorch, pretrained weights for RICE1 & RICE2 included** [V]. Compute: light (single GPU). **LISS-IV fit:** STRONG — MIT-licensed, pretrained, attention helps localise NER cloud, easy to fine-tune on 3-band data. A top pick (see §6).

**5. AMGAN-CR — Attention-Mechanism GAN for Cloud Removal (Xu et al., RSE/IEEE TGRS 2022)** [V]
Core idea: attentive-recurrent network generates cloud attention maps, then an attentive-residual + reconstruction network removes cloud — **no separate cloud detector needed**. Generator uses ConvLSTM + residual blocks guided by attention; GAN-trained on cloudy/clear pairs. Inputs: mono optical, paired. Datasets: Landsat (simulated + real cloud). Strengths: self-localising attention, strong on Landsat. Weaknesses: paired, recurrent attention is heavier; optical-only so thick cloud is hard. There is also a SAR+optical **AttentionGAN** variant (Zhang et al., *Pattern Recognition Letters* 2023, repo `Shuaizhang7/AttentionGAN-for-Cloud-removal`) [V]. **LISS-IV fit:** good if we lack reliable masks; the SAR variant is more promising for thick NER cloud.

**6. CR-GAN-PM — GAN + Physical Model of cloud distortion (Li et al., ISPRS J. P&RS 2020)** [V]
Core idea: **semi-supervised, UNPAIRED**. Decompose a cloudy image into cloud-free background + cloud-distortion layers via GANs + image-decomposition priors, then recompose through a physical cloud-distortion (absorption-aware) model. Inputs: mono optical, **unpaired** (different regions OK). Losses: adversarial + decomposition/reconstruction + physical-consistency. **Reported [V]: per-band SSIM 0.72–0.83 (visible+NIR)** — comparable to supervised end-to-end, better than classical. Repo: `Neooolee/CR-GAN-PM` *(license unverified)*. **LISS-IV fit:** VERY relevant for thin cloud/haze because it needs no pairs — directly addresses our pair-scarcity problem; physical model gives some spectral safety. Thin-cloud focus is the caveat.

**7. RSC-Net (Residual Symmetrical Concatenation / related CNN cloud-removal nets)** [M]
Core idea: deep residual CNN with symmetric skip concatenations for single-image thin-cloud/haze removal (encoder-decoder, L1/L2 + perceptual). Inputs: mono optical. Strengths: stable (no adversarial instability), fast. Weaknesses: tends to blur, struggles with thick opaque cloud. *(Specific paper not re-verified this session — treat name as [M]; several "RSC-Net" variants exist.)* **LISS-IV fit:** useful as a non-adversarial control to benchmark against GANs and quantify hallucination.

### B. SAR–optical fusion (best for persistent NER cloud)

**8. DSen2-CR (Meraner et al., ISPRS J. P&RS 2020)** [V] — *ISPRS Best Paper / U.V. Helava Award*
Core idea: **deep ResNet (no GAN)** that fuses Sentinel-1 SAR + cloudy Sentinel-2 to predict cloud-free S2. ~16+ residual blocks, long global skip connection; **CARL loss** (Cloud-Adaptive Regularised L1) that weights cloudy vs. clear pixels by a cloud mask. Inputs: SAR + mono optical (13-band S2). Datasets: **SEN12MS-CR (~150 k S1/S2 patches, global, all seasons)** [V]. Strengths: robust to thick cloud (SAR carries structure), stable training, **pretrained checkpoints released** (full CARL, L1-only, no-SAR variants) [V]. Weaknesses: Keras/TF1 codebase (older), GPL-3.0 (copyleft — matters if we redistribute) [V]. Repo: **`ameraner/dsen2-cr`, GPL-3.0, Keras/TF, pretrained weights** [V]. **LISS-IV fit:** TOP PICK. SAR-fusion is the single most reliable way through persistent NER cloud, CARL loss is mask-weighted (exactly our need), and pretrained weights give a warm start. Adaptation work: S2→LISS-IV band mapping (use S2 Green/Red/NIR-equivalent bands), and pair S1 with LISS-IV via co-registration.

**9. SAR-Opt-cGAN (Bermúdez, Happ, Feitosa, Oliveira; also Schmitt-school work, 2018–2019)** [V]
Core idea: conditional GAN that **synthesises optical from SAR (+ optional contaminated optical)** to fill cloudy areas. Pix2Pix-style U-Net generator + PatchGAN, trained on the **SEN1-2** SAR/optical dataset (Schmitt et al.). Inputs: SAR (mono or multitemporal) → optical. Strengths: foundational SAR→optical translation; works where optical is fully cloud-covered. Weaknesses: SAR↔optical is ill-posed → texture/color artifacts, speckle leakage. Repo: scattered re-implementations *(no single canonical repo verified)*. **LISS-IV fit:** conceptually core — when cloud is opaque, only SAR informs the fill; expect spectral caution.

**10. Simulation-Fusion GAN (Gao et al., *Remote Sensing* MDPI 2020)** [V]
Core idea: **two-step**. Step 1: a CNN translates SAR → *simulated* optical (object-to-object). Step 2: a GAN fuses {simulated optical, SAR, cloudy optical} to reconstruct only the occluded area; contrast/luminance of the simulated image is randomly jittered for robustness. Inputs: SAR + mono optical. Datasets: Sentinel-1/2, GF-2/3, airborne SAR/optical. Strengths: decoupling simulation from fusion reduces hallucination; **outperforms other SAR-aux methods** [V]. Weaknesses: two networks to train. Repo: *(not verified)*. **LISS-IV fit:** STRONG design template — the "simulate then fuse only the hole" idea preserves clear-area radiometry, which protects LISS-IV spectral fidelity.

**11. SAR2Opt / general SAR-to-optical GAN baselines (CycleGAN/pix2pix variants, 2019–2023)** [M]
Core idea: family of paired (pix2pix) and unpaired (CycleGAN) SAR→optical translators; the **SAR2Opt** benchmark dataset is commonly used to compare them. Inputs: SAR→optical. Strengths: many pretrained baselines, good for ablation. Weaknesses: same ill-posedness as #9. **LISS-IV fit:** use as comparison points, not as the primary engine.

**12. AttentionGAN (SAR+optical) (Zhang et al., Pattern Recognit. Lett. 2023)** [V]
Core idea: attention-mechanism GAN fusing SAR + cloudy optical; attention picks cloud regions and SAR-informative features. Repo: **`Shuaizhang7/AttentionGAN-for-Cloud-removal`** [V] *(license unverified)*. **LISS-IV fit:** modern SAR-fusion GAN with code; a good secondary candidate alongside DSen2-CR.

### C. Multitemporal GAN/CNN

**13. STGAN — Spatiotemporal GAN (Sarukkai, Jain, Uzkent, Ermon; WACV 2020)** [V]
Core idea: cloud removal as conditional synthesis from a **stack of multitemporal cloudy images**; spatiotemporal generator (Pix2Pix-derived, encoder over time). Inputs: multitemporal optical. Datasets: a **97,640-pair** global Sentinel-2 spatiotemporal set [V]. Strengths: exploits temporal redundancy (clouds move), realistic results, single+multi-image variants. Weaknesses: needs co-registered time series with at least partially clear looks. Repo: **`ermongroup/STGAN`, PyTorch** (built on a Pix2Pix clone) [V] *(license: check)*. **LISS-IV fit:** strong if revisit cadence over a NER tile yields ≥2–3 partly-clear scenes; LISS-IV revisit is ~5 days (with pointing), so feasible per AOI.

**14. CTGAN — Cloud "Transformer" GAN (Huang & Wu, ICIP 2022)** [V]
Core idea: 3-time-step cloudy → cloud-free sequence; spatiotemporal feature joint-extraction + interaction + spatiotemporal discriminator. NOTE: name says "Transformer" but it is GAN-centric; the heavy-transformer line is the other agent's. Inputs: multitemporal optical. Datasets: **Sen2_MTC** (4× larger than STGAN's) [V]. Repo: **`come880412/CTGAN`** [V]. **LISS-IV fit:** relevant if we go multitemporal; note it is parameter-heavy (PMAA, below, beats it with far fewer params).

**15. PMAA — Progressive Multi-scale Attention Autoencoder (Zou et al., ECAI 2023, Oral)** [V]
Core idea: **CNN autoencoder (no adversarial)** for multitemporal cloud removal; Multi-scale Attention Module (global) + Local Interaction Module (local). Inputs: multitemporal optical. Strengths: **matches/beats CTGAN with ~0.5% of params and ~14.6% of the compute** [V] — extremely efficient. Repo: **`XavierJiezou/PMAA`, PyTorch, pretrained models** [V]. **LISS-IV fit:** STRONG efficient multitemporal option; great for limited compute and a non-GAN control. A top-pick contender (§6).

### D. Generic inpainting / restoration backbones (repurposable for cloud holes)

**16. Partial Convolutions (Liu et al., NVIDIA, ECCV 2018)** [V]
Core idea: mask-aware convolution renormalised on valid pixels, with an auto-updating mask propagated through layers — designed for **irregular holes** (exactly cloud shapes). Loss: L1 (hole+valid) + perceptual + style + total-variation. Inputs: mono image + binary mask. Repo: `NVIDIA/partialconv` (NVIDIA non-commercial research license) [V]. **LISS-IV fit:** the partial-conv layer is a natural fit because our cloud masks are irregular; can be dropped into a generator. Limitation: pure inpainting hallucinates where no auxiliary (SAR/temporal) info exists — combine with SAR.

**17. Gated Convolutions / DeepFill v2 (Yu et al., ICCV 2019)** [V]
Core idea: **learnable** gating generalises partial conv (soft mask per channel/location); free-form inpainting with contextual attention + SN-PatchGAN. Inputs: mono image + free-form mask. Repo: `JiahuiYu/generative_inpainting` (TF). **LISS-IV fit:** gated convs handle soft/thin-cloud transparency better than hard masks; useful generator building block.

**18. EdgeConnect (Nazeri et al., ICCV-W 2019)** [V/M]
Core idea: two-stage — predict edges in the hole, then inpaint guided by edges (adversarial + feature-matching). Inputs: mono + mask. Repo: `knazeri/edge-connect`. **LISS-IV fit:** edge-first helps preserve field boundaries / rivers in NER; the "structure-then-texture" idea can reduce blur, but optical-only edge guess under thick cloud is risky.

**19. MPRNet — Multi-Stage Progressive Image Restoration (Zamir et al., CVPR 2021)** [V]
Core idea: 3-stage encoder-decoder restoration with Supervised Attention Module (SAM) + Cross-Stage Feature Fusion; SOTA deblur/derain/denoise. Inputs: mono degraded image. Repo: **`swz30/MPRNet`, pretrained weights** [V]. **LISS-IV fit:** thin-cloud/haze can be cast as a restoration (derain-like) task; strong, stable, pretrained backbone to adapt. Not designed for opaque holes.

**20. RDN / ResNet / RDN-inpainting baselines** [M]
Core idea: Residual Dense Networks and plain ResNet encoder-decoders used as non-adversarial reconstruction baselines (L1/L2 ± perceptual). Strengths: stable, easy. Weaknesses: blur, no semantics. **LISS-IV fit:** mandatory ablation baselines to show GAN value-add and quantify hallucination vs. fidelity.

---

## 3. Comparative table — GAN/CNN cloud-removal methods

| # | Method | Year / Venue | Type | Inputs | Backbone (G / D) | Key losses | Datasets | Reported metrics | Code (license) | Pretrained? | LISS-IV fit |
|---|--------|--------------|------|--------|------------------|-----------|----------|------------------|----------------|-------------|-------------|
| 1 | Pix2Pix | 2017 CVPR | Paired GAN | mono | U-Net / PatchGAN | adv+L1 | generic | — | junyanz repo (research) | task-dep. | Baseline |
| 2 | pix2pixHD | 2018 CVPR | Paired GAN | mono HR | coarse-fine / multi-D | adv+FM+VGG | generic | — | NVIDIA (research) | yes (generic) | Recipe reuse |
| 3 | McGAN | 2017 CVPRW | Paired cGAN | mono multispec | U-Net / PatchGAN | adv+L1 | synthetic | — | enomotokenji (MIT?) | no | High (NIR) |
| 4 | SpA-GAN | 2020 arXiv | Paired attn-GAN | mono | attn-resnet / PatchGAN | adv+L1+attn | RICE1/2 | **30.23/0.954; 28.37/0.906** | Penn000 (**MIT**) | **yes** | **Top pick** |
| 5 | AMGAN-CR | 2022 RSE | Paired attn-GAN | mono (SAR variant) | ConvLSTM+resid / GAN | adv+attn+L1 | Landsat | beats 5 SOTA | Shuaizhang7 (SAR var.) | no | High |
| 6 | CR-GAN-PM | 2020 ISPRS | **Unpaired** GAN+physics | mono | decomp GANs | adv+recon+phys | unpaired RS | SSIM 0.72–0.83 | Neooolee (?) | no | **High (no pairs)** |
| 7 | RSC-Net [M] | ~2019 | CNN | mono | resid enc-dec | L1+perc | RS thin-cloud | — | varies | no | Control |
| 8 | DSen2-CR | 2020 ISPRS | **CNN ResNet (SAR-fusion)** | SAR+mono | ResNet / — | **CARL** (mask-wt) | SEN12MS-CR | award; high PSNR/SSIM | ameraner (**GPL-3**) | **yes** | **Top pick** |
| 9 | SAR-Opt-cGAN | 2018–19 | SAR→opt GAN | SAR(+opt) | U-Net / PatchGAN | adv+L1 | SEN1-2 | — | scattered | partial | Core (opaque) |
| 10 | Simulation-Fusion GAN | 2020 RS | SAR-fusion 2-step | SAR+mono | CNN→GAN | adv+L1 | S1/2,GF,air | beats SAR-aux SOTA | (?) | no | **Strong design** |
| 11 | SAR2Opt baselines | 2019–23 | SAR→opt GAN | SAR | pix2pix/CycleGAN | adv(+cyc) | SAR2Opt | — | various | yes | Comparison |
| 12 | AttentionGAN (SAR) | 2023 PRL | SAR-fusion attn-GAN | SAR+mono | attn / GAN | adv+attn+L1 | S1/2 | — | Shuaizhang7 (?) | no | Secondary |
| 13 | STGAN | 2020 WACV | Multitemporal GAN | multitemp | spatiotemp / PatchGAN | adv+L1 | 97.6k S2 | high PSNR/SSIM | ermongroup (?) | partial | Strong (revisit) |
| 14 | CTGAN | 2022 ICIP | Multitemporal GAN | multitemp(3) | spatiotemp / spatiotemp-D | adv+L1 | Sen2_MTC | — | come880412 (?) | yes | Heavy |
| 15 | PMAA | 2023 ECAI | **CNN autoenc (multitemp)** | multitemp | MAM+LIM autoenc | L1+perc | Sen2_MTC/STGAN | **≈CTGAN, 0.5% params** | XavierJiezou (?) | **yes** | **Strong/efficient** |
| 16 | PartialConv | 2018 ECCV | Inpaint CNN | mono+mask | UNet partial-conv | L1+perc+style+TV | generic | — | NVIDIA (research) | yes | Block reuse |
| 17 | DeepFill v2 (gated) | 2019 ICCV | Inpaint GAN | mono+mask | gated / SN-PatchGAN | adv+L1 | generic | — | JiahuiYu (?) | yes | Block reuse |
| 18 | EdgeConnect | 2019 ICCVW | Inpaint GAN | mono+mask | edge+inpaint / PatchGAN | adv+FM+perc | generic | — | knazeri (?) | yes | Edges/boundaries |
| 19 | MPRNet | 2021 CVPR | CNN restoration | mono | 3-stage SAM+CSFF | Charbonnier+edge | derain/deblur | SOTA restoration | swz30 (?) | **yes** | Thin-cloud |
| 20 | RDN/ResNet | 2018+ | CNN | mono | resid-dense | L1/L2 | generic | — | various | yes | Ablation |

*(Metric cells left "—" where no single canonical cloud-removal number is reliably attributable; do not invent. Licenses with "(?)" were not opened this session — verify before redistribution.)*

---

## 4. Paired (Pix2Pix-style) vs Unpaired (CycleGAN-style) for LISS-IV

LISS-IV's hardest constraint is the **scarcity of perfectly co-registered cloudy/cloud-free pairs** over the same NER ground at near-identical phenology. This drives the paired-vs-unpaired choice:

- **Paired (Pix2Pix, pix2pixHD, McGAN, SpA-GAN, AMGAN-CR, DSen2-CR, STGAN):** need pixel-aligned (cloudy, clear) targets. Pros: direct L1/SSIM supervision → best fidelity and spectral control. Cons: hard to assemble at scale for LISS-IV; mis-registration or phenology drift between the cloudy and "clear" date injects label noise (the model learns to also change land cover, not just remove cloud). Mitigations: (a) **synthesise clouds** on clear LISS-IV scenes (McGAN/SpA-GAN Perlin approach) to manufacture perfect pairs; (b) **cloud-mask-weighted loss (CARL)** so only occluded pixels are penalised against the imperfect reference; (c) tight temporal windows (<10–15 days, as RICE-II does) + DEM-aided co-registration.

- **Unpaired (CycleGAN, Cloud-GAN, CR-GAN-PM):** learn cloudy↔clear domain mapping from *unaligned* pools. Pros: dramatically lower data-curation burden — exactly what we lack pairs for; CR-GAN-PM adds a physical model to keep results plausible. Cons: cycle-consistency does not guarantee the *correct* surface is reconstructed (mode collapse, content hallucination, spectral drift); weaker per-pixel fidelity. Mitigations: add SAR or temporal conditioning to constrain content; add identity + spectral-consistency losses; restrict to thin-cloud/haze where the problem is closer to dehazing.

**Recommendation for LISS-IV:** a **hybrid** — build a *paired* core via (i) cloud synthesis on clear scenes and (ii) SAR-fusion with mask-weighted loss (DSen2-CR/SpA-GAN style), and keep an *unpaired* CR-GAN-PM/CycleGAN branch as a thin-cloud fallback and for tiles where no clear reference exists at all.

**Cloud-GAN (Singh & Komodakis, IGARSS 2018)** [V] is the canonical unpaired exemplar: CycleGAN with two generator-discriminator pairs learning cloudy↔clear **without pairs and without SAR**. Useful as our unpaired baseline; weakness is exactly the content-fidelity risk above.

---

## 5. Cloud & cloud-shadow DETECTION / masking (the mask feeds the generator)

Reconstruction quality depends on the **mask** that tells the generator which pixels to trust/replace. Options:

| Method | Year | Type | Sensors | Classes | Notes / fit |
|--------|------|------|---------|---------|-------------|
| **Simple spectral thresholds** | — | Rule-based | any (needs NIR/SWIR) | cloud (rough) | LISS-IV lacks SWIR/thermal/cirrus → thresholds are weak; usable only as a crude prior over G/R/NIR + brightness/NDVI. [M] |
| **Fmask / CFMask** [V] | 2012/2015; v4.0 2019 | Rule+stats (object-based) | Landsat, Sentinel-2 | cloud, shadow, snow | ~96% cloud acc on Landsat; **needs thermal/cirrus bands LISS-IV doesn't have** → not directly runnable on LISS-IV, but run it on co-registered Sentinel-2 to transfer masks. Repo `GERSL/Fmask`. |
| **s2cloudless** [V] | 2018 | Gradient-boosted ML (pixel) | Sentinel-2 (10 bands) | cloud prob | Fast, popular; cloud-only (no shadow). ~63% dice in KappaMask study. S2-specific → use on auxiliary S2. |
| **Cloud-Net / CloudFCN** [V] | 2019 IGARSS | FCN (deep) | Landsat-8 RGB+NIR | cloud | **Uses exactly R/G/B/NIR → 4-band input is close to LISS-IV's 3-band G/R/NIR**; +8.7% Jaccard over prior. Repo `SorourMo/Cloud-Net...`. Strong candidate to retrain on LISS-IV. CloudFCN (Francis et al. 2019) is a sibling FCN. |
| **KappaMask** [V] | 2021 | U-Net | Sentinel-2 L1C/L2A | clear, shadow, semi-transparent, cloud, invalid | Best multi-class on S2 (80% dice L2A, +17% over s2cloudless); good shadow handling. S2-specific. Repo from KappaZeta. |
| **OmniCloudMask (OCM)** [V] | 2024–25 | Sensor-agnostic CNN | S2/Landsat/PlanetScope/Maxar (10–50 m, →5 m) | clear, cloud, shadow | **Generalises across sensors via dynamic Z-score norm + mixed-res training** — 96.9/98.8/97.4% balanced acc — *most promising for LISS-IV* because it is designed to run on sensors it wasn't trained on. PyPI `omnicloudmask`, repo `DPIRD-DMA/OmniCloudMask`. |
| **CloudS2Mask** [V] | 2023 | CNN | Sentinel-2 L1C | cloud, shadow | Same authors as OCM; S2-only sibling. |

**Cloud-shadow** is the weak spot: Fmask/KappaMask/OCM/CloudSEN12-trained nets handle shadow; s2cloudless and naive thresholds do not. For LISS-IV (no thermal, no cirrus), the realistic pipeline is: **(1) OmniCloudMask directly on LISS-IV G/R/NIR for cloud+shadow** (sensor-agnostic), **and/or (2) transfer masks from co-registered Sentinel-2 (s2cloudless/KappaMask/Fmask)**, optionally **(3) retrain Cloud-Net on a small LISS-IV labelled set** since its 4-band R/G/B/NIR input is closest to ours. Use **CloudSEN12 / 38-Cloud** as label sources. [V for datasets]

---

## 6. Top 2–3 GAN/CNN picks for our pipeline

1. **DSen2-CR (SAR-optical ResNet + CARL loss)** — *primary engine for thick/persistent NER cloud.* [V]
   Justification: SAR fusion is the only reliable signal under opaque monsoon cloud; the **cloud-mask-weighted CARL loss** is precisely the right objective for imperfect pairs; **pretrained weights + the global SEN12MS-CR dataset** give a strong warm start; non-adversarial → stable, no GAN collapse, less hallucination. Caveats: port Keras/TF1 (or reimplement in PyTorch), map 13-band S2 → LISS-IV 3-band, co-register S1 with LISS-IV, and respect GPL-3.0 if redistributing.

2. **SpA-GAN (spatial-attention GAN)** — *strong, data-efficient, trainable optical baseline + thin-cloud specialist.* [V]
   Justification: **MIT-licensed, PyTorch, pretrained on RICE**, best-in-class PSNR/SSIM among classic optical GANs (30.23/0.954), attention localises cloud, light compute → ideal to fine-tune on synthetic-cloud LISS-IV pairs and to provide the GAN texture/sharpness that DSen2-CR's L1 objective lacks. Use it where SAR is unavailable or for thin cloud/haze.

3. **PMAA (multitemporal CNN autoencoder)** *or* **CR-GAN-PM (unpaired GAN+physics)** — *third pick, chosen by data regime.* [V]
   - If we can stack ≥2–3 partly-clear LISS-IV revisits per AOI → **PMAA**: matches CTGAN at ~0.5% params, pretrained, exploits temporal redundancy, non-adversarial (stable). Best compute/accuracy trade-off for multitemporal.
   - If pairs are essentially absent and cloud is thin → **CR-GAN-PM**: unpaired + physical cloud model keeps spectral plausibility without co-registered targets.

Supporting masking: **OmniCloudMask** (sensor-agnostic cloud+shadow on LISS-IV directly), with **s2cloudless/KappaMask** mask-transfer from auxiliary Sentinel-2.

**Suggested architecture:** mask (OCM) → SAR-fusion reconstruction (DSen2-CR-style backbone, CARL/mask-weighted loss) → optional GAN refinement head (SpA-GAN attention + adversarial+perceptual) for fine 5.8 m texture → spectral-consistency post-check on clear pixels. Train DSen2-CR/PMAA branches first (stable), add the adversarial head last.

---

## 7. Pitfalls and mitigations

- **GAN training instability (mode collapse, oscillation).** Mitigate: spectral-norm or WGAN-GP discriminator; TTUR; start from a pretrained generator (SpA-GAN, pix2pixHD); train an L1/CARL CNN first, then add the adversarial head (curriculum). Prefer non-adversarial cores (DSen2-CR, PMAA, MPRNet) for the bulk of fidelity and use GAN only for texture refinement.
- **Spectral / radiometric color shift (NDVI, NIR drift).** This is the biggest scientific risk — hallucinated reflectance breaks downstream analytics. Mitigate: **cloud-mask-weighted losses (CARL)** so clear pixels are preserved exactly; add an explicit **spectral-consistency / SAM loss** and per-band histogram matching to a clear reference; physical cloud models (CR-GAN-PM, Simulation-Fusion) constrain plausibility; evaluate with **SAM** alongside PSNR/SSIM, not PSNR alone.
- **Hallucination of content under opaque cloud** (inventing fields/rivers/structures). Mitigate: condition on **SAR and/or temporal** data so the fill is informed, not invented; never inpaint opaque regions from optical-only; flag/mask low-confidence pixels (UnCRtainTS-style uncertainty — transformer agent) and propagate a quality layer.
- **Domain gap (S2/Landsat training → LISS-IV).** 5.8 m vs 10/30 m, 3 vs 13 bands, NER land cover. Mitigate: fine-tune on LISS-IV; synthesise clouds on clear LISS-IV scenes for paired data; band-mapping/normalisation; sensor-agnostic tools (OmniCloudMask) for masks.
- **Mis-registration of pairs / SAR–optical.** Mitigate: DEM-aided ortho + sub-pixel co-registration; mask-weighted losses tolerate residual misalignment in clear areas.
- **Compute.** SpA-GAN/PMAA are light (single-GPU); DSen2-CR is moderate; multitemporal/transformer models cost more — favour PMAA over CTGAN (0.5% params) when going temporal.

---

## 8. Classical baselines (for comparison only)

- **Dark Channel Prior / haze removal (He et al., 2009)** [M] — physics-based single-image dehazing; decent for **thin** cloud/haze, fails on thick cloud. Strong, parameter-free baseline.
- **Multitemporal compositing** (median/best-pixel mosaicking over a time stack) [M] — robust, simple, the de-facto operational method; needs enough clear looks; no learning. Natural baseline + a source of cloud-free targets for training.
- **Histogram matching** [M] — radiometric harmonisation between reference and reconstructed/donor scene; cheap spectral-shift corrector to bolt onto any method's output.
- **Poisson / seamless blending** [M] — gradient-domain blending of a clear donor patch into the cloud hole; eliminates seams; classic inpainting baseline, no semantic generation.

These quantify how much GAN/CNN methods actually add, and several (compositing, histogram matching, Poisson blending) are useful **components** in the final pipeline regardless of the learned core.

---

### Key sources (verify links before final citation)
- SpA-GAN: arXiv 2009.13015; repo github.com/Penn000/SpA-GAN_for_cloud_removal (MIT) [V]
- DSen2-CR: ISPRS JP&RS 166:333–346 (2020); repo github.com/ameraner/dsen2-cr (GPL-3) [V]
- STGAN: arXiv 1912.06838 / WACV 2020; repo github.com/ermongroup/STGAN [V]
- CR-GAN-PM: ISPRS JP&RS 166:373–389 (2020); repo github.com/Neooolee/CR-GAN-PM [V]
- McGAN: arXiv 1710.04835 / CVPRW 2017; repo github.com/enomotokenji/mcgan-cvprw2017-pytorch [V]
- AMGAN-CR / AttentionGAN(SAR): RSE 2022 / PRL 2023; repo github.com/Shuaizhang7/AttentionGAN-for-Cloud-removal [V]
- Simulation-Fusion GAN: Remote Sensing 12(1):191 (2020) [V]
- PMAA: arXiv 2303.16565 / ECAI 2023; repo github.com/XavierJiezou/PMAA [V]
- CTGAN: ICIP 2022; repo github.com/come880412/CTGAN [V]
- PartialConv: arXiv 1804.07723 / ECCV 2018; repo github.com/NVIDIA/partialconv [V]
- DeepFill v2 (gated conv): arXiv 1806.03589 / ICCV 2019 [V]
- MPRNet: arXiv 2102.02808 / CVPR 2021; repo github.com/swz30/MPRNet [V]
- Cloud-Net: arXiv 1901.10077 / IGARSS 2019; repo github.com/SorourMo/Cloud-Net... [V]
- Fmask 4.0: RSE 2019; repo github.com/GERSL/Fmask [V]
- KappaMask: Remote Sensing 13(20):4100 (2021) [V]
- OmniCloudMask: RSE 2025; repo github.com/DPIRD-DMA/OmniCloudMask; PyPI omnicloudmask [V]
- Datasets: SEN12MS-CR & -TS (patricktum.github.io/cloud_removal), RICE (arXiv 1901.00600), CloudSEN12 (Nature Sci. Data 2022), 38-Cloud [V]
