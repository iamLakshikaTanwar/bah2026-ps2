# 06 — LISS-IV / Resourcesat / ISRO Data Ecosystem, Bhoonidhi Access & North-East India Cloud Climatology

> **Project:** BAH 2026 PS2 — Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery
> **Scope of this brief:** Deep domain grounding on the LISS-IV sensor, the Resourcesat family, how to actually obtain data from ISRO/Bhoonidhi, the cloud reality over North-East India (NER), and the concrete constraints our cloud-removal pipeline must respect.
> **Provenance convention:** Each major fact is tagged **[verified]** (confirmed this session against ESA eoPortal, ESA Earth Online RESOURCESAT-2 Data Users' Handbook, NRSC Bhoonidhi, USGS EROS, peer-reviewed literature) or **[memory]** (internal knowledge, not re-verified this session) or **[unverified]** (plausible but I could not confirm; treat as a hypothesis to check). Sources are listed at the end.

---

## A. The LISS-IV Sensor — Exact Definition

LISS-IV (Linear Imaging Self-Scanning Sensor-4) is the high-resolution optical payload on **Resourcesat-2 (launched 20 Apr 2011)** and **Resourcesat-2A (launched 07 Dec 2016)**. It is a pushbroom sensor operating in the **visible/near-infrared (VNIR)** only. This is the single most important fact for our problem: **LISS-IV has no blue band and no SWIR band** — only Green, Red and NIR. Every design choice downstream flows from this. **[verified]**

### A.1 Spectral bands (exact)

| Band | Name | Wavelength (µm) | Notes |
|------|------|-----------------|-------|
| **B2** | Green | **0.52 – 0.59** | Maps to Sentinel-2 B3 |
| **B3** | Red | **0.62 – 0.68** | Maps to Sentinel-2 B4 |
| **B4** | NIR | **0.77 – 0.86** | Maps to Sentinel-2 B8/B8A |

There is no "B1"; ISRO numbering inherits the historic IRS scheme where B1 (blue) existed on older LISS sensors but was dropped for LISS-IV. **[verified — ESA eoPortal & Data Users' Handbook]**

### A.2 Geometry and radiometry

- **Spatial resolution:** ≤ **5.8 m** at nadir (all three bands). **[verified]**
- **Swath & modes:** LISS-IV has two acquisition modes — **[verified]**
  - **Mono (Pan) mode:** ~**70 km** swath, a single band selectable on command (historically B3/Red is the default for the wide mono strip).
  - **Multispectral (MX / "Mx") mode:** ~**23.9 km** swath (often cited as ~23–24 km), all three bands B2/B3/B4 together. **For cloud removal we need the MX product** because we need all three bands co-registered; the 70 km mono strip is single-band and not directly usable for multispectral reconstruction.
- **Quantization:** **10-bit** on Resourcesat-2/2A (DN 0–1023). Note the lineage: **Resourcesat-1 (2003) LISS-IV was 7-bit**; the radiometric depth was upgraded to **10-bit** on Resourcesat-2 onward. If anyone hands us "old IRS-P6 LISS-IV", expect 7-bit. **[verified]**
- **Steering:** the camera is steerable **±26°** across-track, which is how the nominal **24-day repeat** orbit is compressed to a **~5-day effective revisit** for a given target. Steering means off-nadir geometry → larger GSD than 5.8 m and parallax over terrain (relevant in NER hills). **[verified]**
- **Revisit with two satellites:** with Resourcesat-2 **and** 2A both operating, the LISS-IV revisit improved to roughly **25–26 days repeat** (and ~5 days with steering per satellite); LISS-III ~12–13 days; AWiFS ~2–3 days. **[verified]**

### A.3 Orbit

Sun-synchronous, **~817 km** altitude, inclination **98.78°**, **descending-node equator crossing ~10:30 AM local**, period ~101.35 min, **24-day repeat cycle**. The 10:30 AM crossing matters: it is *close* to Sentinel-2's ~10:30 descending node and Landsat's ~10:00–10:15, which helps cross-sensor transfer (similar solar geometry). **[verified]**

### A.4 Product levels, formats, projection

- **Product levels (NRSC):** radiometrically-corrected **system-level (Standard, "L1")** products; **geo-referenced / geo-coded** products; **terrain-corrected / ortho-rectified** products. For our work, **ortho / terrain-corrected** is strongly preferred in NER because of relief displacement in the Himalayan foothills. **[verified]**
- **Delivery format (Bhoonidhi, current):** a **ZIP** per scene containing **one GeoTIFF per band** — typically `BANDx.tif` style filenames (e.g. `BAND2.tif`, `BAND3.tif`, `BAND4.tif`) **plus a `BAND_META.txt` metadata file** (and/or XML). Historic IRS deliveries used **BIL/BSQ with a separate header**; modern Bhoonidhi LISS-IV is GeoTIFF. Plan our loader to accept *both* (GeoTIFF preferred, BIL/`.img` fallback). **[verified — Bhoonidhi delivery via Spatial Thoughts processing writeup]**
- **Projection / datum:** **UTM / WGS84** for standard geocoded products (older IRS standard products historically used **Polyconic / Everest or Polyconic/WGS84**; LISS-IV ortho is UTM/WGS84). Always read the projection from the metadata rather than assuming. **[verified for UTM/WGS84; Polyconic note = memory]**
- **Metadata fields of interest** (`BAND_META.txt`): `OTSProductID`, `DateOfPass` (DD-MMM-YYYY), `SunElevationAtCenter`, scene corner coordinates, satellite/sensor, path/row. **[verified]**

### A.5 DN → radiance → reflectance (calibration concept)

NRSC does **not** ship surface reflectance; products are DN. To compute **Top-of-Atmosphere (TOA) reflectance** we use per-band **saturation (max) radiance** coefficients `Lmax` (Lmin ≈ 0 for these sensors):

```
L_λ      = (Lmax_λ / DN_max) × DN            # DN_max = 1023 for 10-bit
ρ_TOA,λ  = (π · L_λ · d²) / (ESUN_λ · cos θ_z)   # θ_z from SunElevation; d = Earth–Sun distance (AU)
```

Representative LISS-IV `Lmax` (saturation radiance, mW·cm⁻²·sr⁻¹·µm⁻¹, Resourcesat-2 class): **B2 ≈ 53.0, B3 ≈ 47.0, B4 ≈ 31.5**. ESUN per band and `d²` from day-of-year complete the conversion. **Implication:** for deep-learning cloud removal we should at minimum **normalize to TOA reflectance** (or robust per-scene percentile scaling) so the model is not fed raw, scene-dependent 10-bit DN — this is essential for transfer to/from Sentinel-2. **[verified for the Lmax values & DN range; exact ESUN constants = memory — read from the product/handbook per scene]**

---

## B. Resourcesat Family & Companion Sensors

LISS-IV never flies alone — the same platform carries **LISS-III** and **AWiFS**, and other ISRO assets (Cartosat) and external missions (Sentinel-2, Landsat) provide complementary data. This matters because the single best mitigation for LISS-IV's data scarcity is **multi-sensor reference**.

### B.1 Resourcesat sensor table **[verified]**

| Spec | **LISS-IV** | **LISS-III** | **AWiFS** |
|------|-------------|--------------|-----------|
| Resolution (nadir) | **5.8 m** | 23.5 m | 56 m |
| Swath | 70 km (mono) / ~23.9 km (MX) | 141 km | 740 km |
| Bands | B2,B3,B4 (G,R,NIR) — **3** | B2,B3,B4,B5 (G,R,NIR,**SWIR**) — 4 | B2,B3,B4,B5 (same as LISS-III) — 4 |
| Green | 0.52–0.59 | 0.52–0.59 | 0.52–0.59 |
| Red | 0.62–0.68 | 0.62–0.68 | 0.62–0.68 |
| NIR | 0.77–0.86 | 0.77–0.86 | 0.77–0.86 |
| SWIR | — (none) | 1.55–1.70 | 1.55–1.70 |
| Quantization | 10-bit | 10-bit | 10-bit |
| Repeat / revisit | 24-day repeat, ~5-day steered | 24 days | ~5-day repeat |

**Key insight:** LISS-III/AWiFS share LISS-IV's **exact VNIR band passbands** (G/R/NIR identical to 2 decimal places). That makes them **perfect spectral siblings** — an AWiFS or LISS-III scene a few days off can serve as a *radiometric/temporal reference* for the same area, and critically **they carry the SWIR band LISS-IV lacks**. SWIR (1.55–1.70 µm) is the single most useful band for cloud/snow discrimination and cirrus, so **LISS-III/AWiFS SWIR can be used as auxiliary/teacher signal** even though LISS-IV itself cannot supply it.

### B.2 Cartosat for pan-sharpening

**Cartosat-1 (2.5 m PAN)** and **Cartosat-2 series (~0.6–1 m PAN)** provide panchromatic imagery that can sharpen or co-register LISS-IV. PAN is single-band (broad VNIR) so it is a structural/edge prior, not spectral. Useful for **co-registration anchoring and super-resolution priors**, less so for spectral reconstruction. **[memory]**

### B.3 Band-matching Sentinel-2 (and Landsat) to LISS-IV

This is the linchpin for transfer learning and for borrowing cloud-free fill. **[verified band centers]**

| LISS-IV | Sentinel-2 (MSI) | Landsat 8/9 (OLI) | Notes |
|---------|------------------|-------------------|-------|
| **B2 Green** 0.52–0.59 µm | **B3 Green** ~0.560 µm (560 nm, 36 nm BW), 10 m | **B3 Green** 0.53–0.59 µm, 30 m | Near-identical passband |
| **B3 Red** 0.62–0.68 µm | **B4 Red** ~0.665 µm (665 nm, 31 nm BW), 10 m | **B4 Red** 0.64–0.67 µm, 30 m | Near-identical |
| **B4 NIR** 0.77–0.86 µm | **B8 NIR** ~0.833 µm (833 nm, **106 nm** wide), 10 m **— or — B8A** ~0.865 µm (21 nm), 20 m | **B5 NIR** 0.85–0.88 µm, 30 m | LISS-IV NIR (0.77–0.86) is **wider** than S2-B8A and offset from narrow bands; **S2-B8 (wide) is the closer match by bandwidth**, S2-B8A by center. Apply a spectral band-adjustment factor (SBAF). |

**Practical mapping:** to build a 3-band Sentinel-2 surrogate of LISS-IV use **S2 {B3, B4, B8}** (all native 10 m, closest to LISS-IV's 5.8 m), and resample S2 10 m → LISS-IV 5.8 m grid. Because passbands are *close but not equal* (especially NIR), expect a small linear **SBAF/histogram-matching** step before any domain-transfer or synthetic-cloud training. **[verified band specs; SBAF recommendation = standard practice]**

---

## C. Bhoonidhi Portal & ISRO Data Access

### C.1 What Bhoonidhi is

**Bhoonidhi** (`https://bhoonidhi.nrsc.gov.in`) is **NRSC/ISRO's EO data dissemination portal** — the successor to the old "Bhuvan/NDC ordering" flow and the canonical place to obtain LISS-IV. It is operated by the **National Remote Sensing Centre (NRSC)**, Hyderabad, under the **National Data Centre (NDC)**. **[verified]**

### C.2 Open-data policy (the good news)

Under the **Indian Space Policy 2023** (implemented at Bhoonidhi), **EO data of 5 m and coarser GSD from Resourcesat-1/2/2A is OPEN DATA for all users**. LISS-IV at **5.8 m is therefore in the open-data tier** — i.e. **free to download** after registration, not priced. Higher-resolution (<5 m, e.g. Cartosat-2) remains priced/restricted. **This is decisive for the hackathon: LISS-IV is obtainable free.** **[verified — ISRO Bhoonidhi 2024 newsletter / Space Policy 2023]**

> ⚠️ **Caveat to verify in-portal:** open-data still requires a **registered Indian-policy-compliant account** and EULA acceptance; some archive depths or NRT feeds may have additional gating. Confirm the exact tier for the specific scenes you order. **[unverified — check live].**

### C.3 How to search & order (workflow)

1. **Register** at `bhoonidhi.nrsc.gov.in/bhoonidhi/registration.html`; accept the **EULA**. **[verified link]**
2. **Data Discovery & Download** from the dashboard. **[verified]**
3. **Define AOI:** draw a box on the map, **upload KML/GeoJSON**, or enter coordinates. **[verified]**
4. **Filter:** Satellite (**Resourcesat-2 / Resourcesat-2A**), Sensor (**LISS-IV MX**), **date range**, and **cloud-cover %**. **[verified — cloud% filter exists]**
5. **Browse/preview** quick-looks → **add to cart / place order** → download **ZIP (GeoTIFF + metadata)**. **[verified]**

### C.4 Alternatives & samples

- **USGS EROS** hosts **IRS Resourcesat-1/2 LISS-3** (historic, US distribution) — **LISS-IV is generally NOT on USGS**; do not rely on USGS for LISS-IV. **[verified — USGS EROS page is LISS-3 only]**
- **Bhuvan** (`bhuvan.nrsc.gov.in`) for visualization and some thematic products. **[memory]**
- **GitHub/community samples:** small LISS-IV classification datasets exist (e.g. community repos), useful for code testing but **not** a substitute for matched cloudy/cloud-free pairs. **[verified such repos exist; contents = unverified]**
- **Tutorials:** NRSC Bhoonidhi help/tutorials + community walkthroughs (incl. a "How to download LISS-IV from Bhoonidhi" video) exist for onboarding. **[verified existence]**

### C.5 The hard part for cloud removal

Bhoonidhi lets us filter by cloud %, but **getting a matched (cloudy, cloud-free) pair of the *same* LISS-IV footprint close in time is hard** because of the 24-day repeat and NER's persistent cloud. **Expect to (a) mine the archive for the rare clear scene per tile, and (b) rely on synthetic clouds + cross-sensor data** (Section E). **[reasoned from verified revisit + climatology]**

---

## D. North-East Region (NER) Context — Why Optical Data Is Scarce

### D.1 The eight states

**Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura, Sikkim** (the "Eight Sisters"; Sikkim added to NER administratively). **[memory]**

### D.2 Cloud climatology (the core problem)

NER sits under the **South-West monsoon (June–September)** and gets some of the **highest rainfall on Earth** (Mawsynram/Cherrapunji, Meghalaya). Consequences for optical EO: **[verified climatology direction; numbers below mixed]**

- During the **Jul–Aug monsoon peak**, mean cloud cover over monsoon-affected Indian districts has been measured at **~90–92%**, leaving **~23–25% of even surface-water areas unmappable** by optical sensors in those months. NER is at the wetter end of this distribution. **[verified — PLOS One 2024 study on monsoon India]**
- Practically, **usable cloud-free LISS-IV scenes over NER are concentrated in the dry/post-monsoon/winter window (roughly Nov–Mar)**; the Jun–Sep window is near-unusable optically. **[reasoned from verified climatology + 24-day repeat]**
- This is precisely why the problem statement exists: **clear-sky LISS-IV revisits over NER are rare, so reconstruction/cloud-removal is not a nicety but a necessity.** **[reasoned]**

### D.3 Terrain implications

- **Himalayan/sub-Himalayan foothills** (Arunachal, Sikkim, Nagaland, Mizoram) → strong **relief displacement, terrain shadow, and parallax**, amplified by LISS-IV's ±26° steering. **Use ortho/terrain-corrected products and a DEM** (e.g. **Cartosat DEM / CartoDEM**, or SRTM/Copernicus DEM 30 m) for co-registration and **topographic/cast-shadow** discrimination (cloud shadow vs terrain shadow is a real confusion in this terrain). **[memory/standard practice]**
- **Brahmaputra and Barak valleys** → flat, agricultural, **flood-dynamic**, frequent haze/fog; braided river channels shift between dates (a co-registration and "change vs cloud" hazard). **[memory]**

### D.4 Land cover

Predominantly **dense evergreen/semi-evergreen forest** (high NIR, low red — bright in NDVI), **shifting cultivation (jhum) and terraced/valley agriculture**, **tea estates** (Assam), **water bodies/wetlands**, and **snow/ice at high elevation** (Sikkim/Arunachal). High vegetation fraction means **NDVI-based context is informative** but also that **bright cloud-free vegetation can be spectrally confused with some thin-cloud/haze conditions** in a 3-band sensor. **[memory]**

### D.5 Why SAR & temporal methods dominate here

Because optical is blocked for ~4 months/year, **Sentinel-1 C-band SAR (cloud-penetrating, 10 m, ~6–12 day revisit)** is the workhorse "always-available" signal, and **multi-temporal optical compositing** is the other pillar. For our pipeline this argues strongly for **SAR-guided cloud removal** (SAR as the conditioning input that "sees through" cloud) and/or **temporal infilling** from the nearest clear LISS-IV/Sentinel-2 acquisition. **[verified — monsoon-India SAR-vs-optical literature]**

---

## E. Practical Implications for OUR Pipeline

### E.1 "3 bands, no blue, no SWIR" — what breaks and what to do

Most off-the-shelf cloud masks assume blue and/or SWIR/cirrus bands. **They will not run on LISS-IV unmodified.** **[verified — sensor lacks those bands]**

- **Blue-band brightness/whiteness tests (Fmask-style, S2 B1/B2):** **unavailable.** Adapt to **Green-based brightness** + NIR + Green/Red ratios; clouds are bright and spectrally flat across G/R/NIR, so use **high reflectance in all three bands + low NDVI + texture** as the cloud cue.
- **Cirrus band (1.38 µm) and SWIR-based snow/cloud (NDSI):** **unavailable** on LISS-IV. Thin cirrus detection is intrinsically hard with 3 VNIR bands — lean on **SAR consistency** and **temporal anomaly** instead, and consider **LISS-III/AWiFS SWIR as auxiliary teacher** where same-area near-date scenes exist.
- **Snow vs cloud:** without SWIR/NDSI, **snow (Sikkim/Arunachal) will be a notorious false-positive for cloud.** Mitigate with DEM/elevation priors and temporal persistence (snow is static over days; clouds move). **[reasoned]**

### E.2 Spectral indices we *can* compute

- **NDVI = (NIR−Red)/(NIR+Red)** ✅ — fully available (B4,B3). Primary vegetation/context index.
- **GNDVI = (NIR−Green)/(NIR+Green)** ✅ — available (B4,B2).
- **SR / simple ratios, GR ratio** ✅.
- **NDWI (McFeeters, Green−NIR)** ✅ available; **MNDWI (needs SWIR)** ❌ not available — water masking will be weaker than with Sentinel-2.
- **NDSI (snow, needs SWIR)** ❌ — must be replaced by elevation+temporal logic. **[verified from band availability]**

### E.3 Co-registration challenges

- LISS-IV at **5.8 m vs Sentinel-1/2 at 10 m** → resample/align carefully; sub-pixel misregistration over NER terrain will create artifacts in any SAR-optical fusion. Use **DEM-aware ortho + phase-correlation / AROP-style tie-pointing**, and validate on stable features (road/river junctions). Steered (off-nadir) LISS-IV strips need extra care. **[memory/standard practice]**
- **Cloud-shadow geometry:** with the ~10:30 AM overpass and known sun elevation (`SunElevationAtCenter` in metadata), **predict shadow offset direction** to separate cloud shadow from terrain shadow. **[reasoned]**

### E.4 Recommended auxiliary data for NER

| Need | Recommended source |
|------|--------------------|
| Cloud-penetrating signal / conditioning | **Sentinel-1 GRD (C-band, 10 m)** — free, Copernicus |
| Cross-sensor cloud-free fill & training pairs | **Sentinel-2 L2A {B3,B4,B8}** — free, near-same overpass time |
| Missing SWIR proxy / temporal reference | **LISS-III / AWiFS** (same VNIR + SWIR), **same Resourcesat platform** |
| Terrain / shadow / ortho | **CartoDEM (30 m)** or **Copernicus GLO-30 / SRTM 30 m** |
| Structural prior / sharpening | **Cartosat PAN** (optional) |

### E.5 Data-scarcity mitigation (the heart of the approach)

1. **Synthetic clouds:** since real matched (cloudy, clear) LISS-IV pairs are scarce, **paste physically-plausible clouds + shadows onto clear LISS-IV scenes** to create supervised pairs (perlin/real cloud masks, varying opacity for haze→thick). **[standard practice]**
2. **Transfer from Sentinel-2 / SEN12MS-CR(-TS):** pre-train on the **SEN12MS-CR** (122,218 SAR+cloudy+clear triplets) / **SEN12MS-CR-TS** (multi-temporal) benchmark, then **fine-tune to LISS-IV's 3-band space**. Because S2 {B3,B4,B8} ≈ LISS-IV {B2,B3,B4}, the feature transfer is favorable after SBAF. **[verified dataset exists & composition]**
3. **Domain adaptation LISS-IV ↔ Sentinel-2:** histogram matching / SBAF + adversarial or CycleGAN-style alignment to bridge the small spectral/radiometric gap, so a model trained mostly on abundant S2 data generalizes to scarce LISS-IV. **[reasoned/standard]**
4. **SAR-optical fusion (Sentinel-1 conditioning):** strongest mitigation in NER monsoon — the network learns to reconstruct optical reflectance from co-registered SAR where cloud blocks the optical. **[verified — SAR cloud-removal literature, e.g. DSen2-CR]**
5. **Temporal infilling:** use nearest-date clear LISS-IV/S2 of the same tile as a reconstruction prior (caution: NER land cover/water changes between widely-spaced dates). **[reasoned]**

---

## Data Acquisition Playbook (Team Checklist)

- [ ] **Register on Bhoonidhi** (`bhoonidhi.nrsc.gov.in`), accept EULA, confirm **LISS-IV (5.8 m) is open/free** under Space Policy 2023.
- [ ] Pick **2–4 NER AOIs** spanning land-cover types: a Brahmaputra-valley agri tile (Assam), a forested hill tile (Meghalaya/Mizoram), a high-relief/snow tile (Sikkim/Arunachal).
- [ ] In Discovery & Download, set **Satellite = Resourcesat-2 & 2A**, **Sensor = LISS-IV MX**, upload **AOI KML/GeoJSON**.
- [ ] **Harvest CLEAR scenes:** filter **cloud% low**, prioritise **Nov–Mar** dates → these are the **targets / "ground truth"**.
- [ ] **Harvest CLOUDY scenes:** same AOIs, **monsoon Jun–Sep**, mid/high cloud% → these are **real-world test inputs**.
- [ ] Download as **ZIP → GeoTIFF (BAND2/3/4) + BAND_META.txt**; record `OTSProductID`, `DateOfPass`, `SunElevationAtCenter`, projection.
- [ ] **Pull matched Sentinel-1 GRD + Sentinel-2 L2A** ({B3,B4,B8}) for the **same AOIs & nearest dates** (Copernicus) — for fusion + training pairs + SBAF calibration.
- [ ] **Pull a DEM** (CartoDEM/GLO-30) per AOI for ortho check & shadow logic.
- [ ] **(Optional) LISS-III/AWiFS** same-AOI scenes to obtain a **SWIR reference** band.
- [ ] **Standardize:** reproject all to a common **UTM/WGS84** grid, resample to **5.8 m** (or a chosen working GSD), **DN→TOA reflectance** (Lmax/ESUN/sun-elevation), tile into patches.
- [ ] Build the **synthetic-cloud generator** on clear LISS-IV; reserve real cloudy scenes strictly for **held-out evaluation**.
- [ ] **Pre-train on SEN12MS-CR(-TS)**, fine-tune on LISS-IV (3-band, SBAF-aligned).

---

## Key Risks / Constraints to Design Around

| # | Risk / Constraint | Why it matters | Mitigation |
|---|-------------------|----------------|------------|
| 1 | **Only 3 VNIR bands (no blue, no SWIR, no cirrus)** | Standard cloud masks (Fmask/S2cloudless/sen2cor) don't run; thin cirrus & snow hard | Custom 3-band cloud logic; SAR + temporal cues; LISS-III/AWiFS SWIR as teacher |
| 2 | **Severe data scarcity of matched cloudy/clear LISS-IV** (24-day repeat × monsoon) | Can't train supervised on real pairs alone | Synthetic clouds + SEN12MS-CR transfer + SAR fusion |
| 3 | **Snow ≈ cloud confusion in 3 bands** (Sikkim/Arunachal) | False positives in cloud mask | DEM/elevation prior + temporal persistence |
| 4 | **Cloud-shadow vs terrain-shadow** in NER hills | Mislabeled "missing" regions | Sun-geometry-predicted shadow offset + DEM |
| 5 | **Co-registration: 5.8 m LISS-IV vs 10 m S1/S2 over relief, off-nadir steering** | Fusion artifacts | Ortho products + DEM-aware sub-pixel alignment, validate on stable features |
| 6 | **Spectral mismatch LISS-IV NIR (wide, 0.77–0.86) vs S2 B8/B8A** | Naive transfer biases NIR/NDVI | SBAF + histogram matching before transfer |
| 7 | **Raw 10-bit DN, no surface reflectance shipped; scene-dependent radiometry** | Model sees inconsistent scales | DN→TOA reflectance normalization per scene from metadata |
| 8 | **Projection/format heterogeneity** (UTM/WGS84 GeoTIFF, possibly legacy Polyconic/BIL) | Loader breakage | Read projection from metadata; support GeoTIFF + BIL/`.img` |
| 9 | **Bhoonidhi access gating / EULA / Indian-policy account** | Could stall data collection | Register early; confirm open-data tier for the exact scenes; keep S1/S2 as free fallback |
| 10 | **NER land-cover/water change between sparse dates** | Temporal infill injects false content | Prefer nearest-date references; weight SAR over stale optical |

---

## Sources

**Verified this session:**
- ESA eoPortal — *ResourceSat-2*: https://www.eoportal.org/satellite-missions/resourcesat-2
- ESA Earth Online — *RESOURCESAT-2 Data Users' Handbook (Dec 2011)*: https://earth.esa.int/eogateway/documents/20142/37627/ResourceSat-2-Data-User-Handbook.pdf
- ESA Earth Online — *IRS-R2 (ResourceSat-2)*: https://earth.esa.int/eogateway/missions/resourcesat-2
- NRSC Bhoonidhi portal & registration: https://bhoonidhi.nrsc.gov.in/ and https://bhoonidhi.nrsc.gov.in/bhoonidhi/registration.html
- ISRO — *Bhoonidhi Newsletter 2024 (Space Policy 2023 / open data ≤5 m)*: https://www.isro.gov.in/media_isro/pdf/Bhoonidhi_NewsLetter_2024_Edition1.pdf
- USGS EROS — *ISRO Resourcesat-1/2 LISS-3 archive* (confirms LISS-3, not LISS-IV, on USGS): https://www.usgs.gov/centers/eros/science/usgs-eros-archive-isro-resourcesat-1-and-resourcesat-2-liss-3
- Wikipedia — *Resourcesat-2A* (launch 2016, revisit): https://en.wikipedia.org/wiki/Resourcesat-2A
- Spatial Thoughts — *LISS4 processing (Bhoonidhi GeoTIFF format, BAND_META.txt, Lmax, DN→TOA)*: https://spatialthoughts.com/2023/12/25/liss4-processing-xarray/
- Sentinel-2 band centers (B3/B4/B8/B8A): http://homepage.ntu.edu.tw/~choying/Sentinel-2_QS.pdf ; ClearSKY: https://clearsky.vision/knowledge/sentinel2-spectral-bands
- Landsat 8 OLI bands — USGS: https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
- *Radar vs optical: cloud cover mapping seasonal surface water in monsoon India* (PLOS One 2024, ~90–92% Jul–Aug cloud): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0314033
- SEN12MS-CR / SEN12MS-CR-TS dataset: https://patricktum.github.io/cloud_removal/sen12mscr/ and https://patricktum.github.io/cloud_removal/sen12mscrts/
- *Synergistic Use of Radar and Optical for Monsoon Cropland Mapping in India* (MDPI RS): https://www.mdpi.com/2072-4292/12/3/522
- *Cloud removal in Sentinel-2 using residual net + SAR-optical fusion (DSen2-CR)*: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7386944/

**Marked memory/unverified in-text:** Resourcesat-1 LISS-IV 7-bit lineage; legacy Polyconic/BIL formats; exact ESUN constants; CartoDEM specifics; NER state list & land-cover detail; precise in-portal open-data gating (confirm live on Bhoonidhi).
