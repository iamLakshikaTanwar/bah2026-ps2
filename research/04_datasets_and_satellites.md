# 04 — Datasets & Satellites: A Comprehensive Catalogue for Cloud Removal

**Project:** BAH 2026 PS2 — Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery
**Target sensor:** Resourcesat-2/2A **LISS-IV** (5.8 m, 3 bands: Green/Red/NIR)
**Focus region:** North-Eastern Region (NER), India — persistently cloudy
**Strategy:** Use MANY satellites/datasets worldwide to cross-verify and fill gaps that LISS-IV alone cannot.

> **Verification legend:** `[V]` = spec verified via web this session · `[M]` = from internal knowledge (memory), treat as high-confidence but re-confirm before production · `[?]` = uncertain / URL unverified — confirm manually.
> **Cloud-removal role legend:** **TGT** target to reconstruct · **AUX** auxiliary fusion (SAR/DEM) · **TMP** temporal reference · **XV** cross-verification / cloud climatology.

This catalogue covers **~30 sensors/platforms** and **~18 benchmark datasets**, plus concrete data-access recipes (portals, GEE collection IDs, STAC, COG). It is meant to be the single reference for what data we can pull, how, and why it helps.

---

## Part A — Sensors & Platforms

### A.1 ISRO / NRSC sensors

| Sensor (platform) | Bands & wavelengths (µm) | Res. | Revisit | Swath | Radiometry | Role |
|---|---|---|---|---|---|---|
| **LISS-IV** (Resourcesat-2/2A) `[V]` | G 0.52–0.59, R 0.62–0.68, NIR 0.77–0.86 (3-band Mx) | **5.8 m** | 5 d (24 d full) | 70 km mono / 23.5 km Mx | 10-bit (7-bit transmitted) | **TGT** |
| **LISS-III** (Resourcesat-2/2A) `[V]` | G, R, NIR + SWIR 1.55–1.70 (4-band) | 23.5 m | 24 d | 141 km | 10-bit | **TMP / XV** (same orbit, recent clear scenes) |
| **AWiFS** (Resourcesat-2/2A) `[V]` | G, R, NIR, SWIR (4-band) | 56 m | 5 d | 740 km | 10/12-bit | **XV / TMP** (frequent wide-area context) |
| **Cartosat-2 series** `[M]` | PAN 0.50–0.85; some MX | PAN ~0.65 m / MX ~1.6 m | ~4–5 d (agile) | ~10 km | 10–11-bit | AUX (high-res structure priors) |
| **Cartosat-3** `[V]` | PAN + 4-band MX + hyperspectral/SWIR payloads | **PAN 0.25 m**, MX 1 m | agile | ~16 km | 11-bit | AUX (sharpest optical priors) |
| **EOS-04 / RISAT-1A** `[V]` | **C-band SAR** 5.35 GHz, HH/VV/HV/VH | 1–50 m (mode-dependent) | ~12–25 d | mode-dependent | — | **AUX** (cloud-penetrating structure) |
| **RISAT-1/-2B** `[M]` | C-band / X-band SAR | 1–50 m | variable | variable | — | **AUX** (all-weather backscatter) |
| **EOS-06 / Oceansat-3 (OCM-3)** `[M]` | 13 VNIR ocean-colour bands | 360 m / 1 km | 2 d | ~1500 km | 12-bit | XV (atmosphere/water context, coarse) |
| **INSAT-3D / 3DR / 3DS** `[M]` | Imager VIS/SWIR/MIR/TIR (6-ch); Sounder | VIS ~1 km, TIR ~4 km | **30 min (geostationary)** | full-disk | 10-bit | **XV** (cloud climatology, hourly cloud mask, shadow geometry) |

**Notes:** LISS-IV is the *only* target. LISS-III/AWiFS share the Resourcesat platform — the most natural same-family temporal references. INSAT-3D/3DR (geostationary, 30-min cadence) is uniquely valuable for **cloud-cover climatology and acquisition timing** over NER (pick low-cloud windows; characterize persistent cloudiness).

### A.2 ESA Copernicus sensors

| Sensor | Bands & wavelengths | Res. | Revisit | Swath | Radiometry | Role |
|---|---|---|---|---|---|---|
| **Sentinel-2 A/B/C MSI** `[V]` | 13 bands: B2 490, B3 560, B4 665 (10 m); B5 705, B6 740, B7 783, B8A 865, B11 1610, B12 2190 (20 m); B1 443, B9 945, B10 1375 (60 m) | **10/20/60 m** | **5 d** (2–3 sats) | 290 km | 12-bit | **XV / PROXY-TGT** — band-match to LISS-IV |
| **Sentinel-1 A/B/C SAR** `[V]` | **C-band 5.405 GHz**, dual-pol VV+VH (IW); GRD & SLC | 5×20 m (IW GRD ~10 m grid) | 6–12 d | 250 km (IW) | — | **AUX** (cloud-penetrating, primary SAR modality) |
| **Sentinel-3 OLCI/SLSTR** `[M]` | OLCI 21 bands (400–1020 nm); SLSTR VIS–TIR | 300 m (OLCI) / 500 m–1 km (SLSTR) | ~1–2 d | 1270 km | — | XV (cloud/atmosphere, coarse) |

**Why Sentinel-2 is the linchpin:** S2 bands **B3 (Green 560), B4 (Red 665), B8/B8A (NIR 842/865)** map almost 1:1 onto LISS-IV's G/R/NIR. S2 is free, global, 5-day, and underpins *most* benchmark cloud-removal datasets — making it the best domain bridge for LISS-IV transfer.

### A.3 NASA / USGS sensors

| Sensor | Bands | Res. | Revisit | Swath | Role |
|---|---|---|---|---|---|
| **Landsat-8/9 OLI/(TIRS)** `[V]` | 11 bands: Coastal 443, Blue 482, Green 561, Red 655, NIR 865, SWIR1 1609, SWIR2 2201, Pan 590, Cirrus 1373, TIR 10.9/12 µm | 30 m (Pan 15 m, TIR 100 m) | 16 d (8 d combined L8+L9) | 185 km | **TMP / XV** (cirrus + QA bands aid cloud masking) |
| **MODIS (Terra/Aqua)** `[M]` | 36 bands 0.4–14.4 µm | 250/500/1000 m | **1–2 d** | 2330 km | **XV** — cloud products (MOD35 mask, MOD06 cloud props), daily context |
| **VIIRS (Suomi-NPP/JPSS)** `[M]` | 22 bands (DNB + M + I) | 375/750 m | ~1 d | 3060 km | XV (cloud mask continuity post-MODIS) |
| **ASTER (Terra)** `[M]` | VNIR 3-band 15 m, SWIR 6-band 30 m, TIR 5-band 90 m | 15–90 m | ~16 d (on-demand) | 60 km | AUX/DEM (GDEM source, terrain) |
| **HLS (Harmonized Landsat-Sentinel)** `[V]` | L30 (Landsat-8/9) + S30 (Sentinel-2) harmonized to common bands | **30 m** | **2–3 d** combined | tile-based | **TMP / XV** — analysis-ready harmonized time series |

**MODIS cloud products** (MOD35_L2 mask, MOD06_L2 cloud props, MOD09 reflectance) give *daily* cloud climatology to complement INSAT and label scene cloudiness. HLS is the cleanest dense, calibrated optical **temporal stack** for consistency checks.

### A.4 Commercial / other sensors

| Sensor | Bands | Res. | Revisit | Role | Access/cost |
|---|---|---|---|---|---|
| **PlanetScope / Dove (SuperDove)** `[M]` | 8-band (incl. G/R/NIR + RedEdge) | **3–5 m** | ~daily | AUX/TMP — near-LISS-IV res, daily revisit (gap-fill) | Commercial; free research via Planet/NICFI for tropics `[?]` |
| **RapidEye** `[M]` | 5-band (incl. RedEdge) | 5 m | ~5.5 d | TMP (archive 2009–2020) | Commercial archive |
| **SPOT-6/7** `[M]` | PAN + 4 MS | PAN 1.5 m / MS 6 m | ~1–3 d | AUX (high-res optical) | Commercial (Airbus) |
| **WorldView-2/3 / Maxar** `[M]` | PAN + 8–16 MS (incl. SWIR on WV-3) | PAN 0.31 m / MS 1.2 m | ~1 d | AUX (very-high-res priors, eval) | Commercial |
| **Capella / ICEYE** `[M]` | **X-band SAR** | ≤0.5–1 m spotlight | tasking | AUX (very-high-res SAR structure) | Commercial |

**Use commercial data sparingly**: PlanetScope is most relevant (3–5 m, near-daily) for filling small temporal gaps over NER; the rest serve as high-res evaluation references or qualitative priors.

### A.5 DEMs (terrain / shadow modelling — **AUX**)

| DEM | Source | Res. | Coverage | GEE ID `[V]` | Role |
|---|---|---|---|---|---|
| **Copernicus DEM GLO-30** | TanDEM-X (DSM) | **30 m** | global | `COPERNICUS/DEM/GLO30` | **Preferred** — best modern global DSM, terrain + shadow geometry |
| **SRTM** | C-band radar (DTM) | 30 m | 60°N–56°S | `USGS/SRTMGL1_003` | Baseline elevation, slope/aspect |
| **NASADEM** | SRTM reprocessed | 30 m | 60°N–56°S | `NASA/NASADEM_HGT/001` | Improved-accuracy SRTM (void-fill, ICESat) |
| **ALOS PALSAR / AW3D30** | L-band SAR | 30 m | global | `JAXA/ALOS/AW3D30/V*` `[?]` | Alt. DSM, good in vegetation |
| **CartoDEM** (ISRO) | Cartosat stereo | 10–30 m | India | Bhoonidhi | National DEM for India (terrain consistency with LISS-IV) |

**Why DEM matters for cloud removal:** clouds cast **shadows**; terrain casts **topographic shadows**. A DEM + sun-geometry lets us (a) predict/flag shadow regions, (b) condition the generator on terrain so reconstructed reflectance respects slope/aspect, and (c) sanity-check that "filled" pixels are physically plausible.

### A.6 Cloud masks / cloud climatology (**XV**)

| Product | Source | GEE/portal `[V]` | Use |
|---|---|---|---|
| **MODIS MOD35_L2** | Terra/Aqua | LP DAAC / GEE MOD09/MOD35 | Daily cloud mask, climatology |
| **Sentinel-2 Cloud Probability** | s2cloudless (LightGBM) | `COPERNICUS/S2_CLOUD_PROBABILITY` | Per-pixel cloud prob 0–100 |
| **Cloud Score+** (Google/DeepMind) | S2-derived | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | SOTA per-pixel quality/cloud score |
| **Sentinel-2 SCL** | Sen2Cor L2A | S2_SR scene classification band | Cloud/shadow/snow class |
| **ERA5 / ERA5-Land cloud cover** | ECMWF reanalysis | `ECMWF/ERA5_LAND/*`, CDS | Total-cloud-cover climatology, planning |
| **INSAT-3D/3DR cloud mask** | ISRO geostationary | MOSDAC | 30-min cloud climatology over NER |

---

## Part B — Benchmark Cloud-Removal & Cloud-Detection Datasets

> All sizes/specs verified this session unless marked `[M]`. "Suitability for LISS-IV transfer" is rated **High / Med / Low** based on band overlap (G/R/NIR), resolution proximity to 5.8 m, and whether SAR/temporal structure is included.

### B.1 Paired / multimodal cloud-**removal** datasets

| Dataset | Contents / sensors | Paired? | Size | Res. | Bands | License | URL | LISS-IV fit |
|---|---|---|---|---|---|---|---|---|
| **SEN12MS-CR** `[V]` | S1 SAR + cloudy & clear S2 (triplets) | SAR+optical paired | **122,218** patches 256² | 10 m | S2 13 + S1 2 | CC-BY `[?]` | patricktum.github.io/cloud_removal/sen12mscr/ | **High** (S2 G/R/NIR + SAR; our core fusion analog) |
| **SEN12MS-CR-TS** `[V]` | Multi-temporal S1+S2 time series, cloudy↔clear | SAR+optical, temporal | 53 ROIs, dense TS | 10 m | S2 13 + S1 2 | CC-BY `[?]` | patricktum.github.io/cloud_removal/sen12mscrts/ | **High** (temporal+SAR; best methodological match) |
| **AllClear** `[V]` | S2 + Landsat-8/9 + S1 SAR + cloud/landcover aux; per-ROI 2022 time series | multimodal, temporal | **23,742 ROIs / ~4M images** | 10–30 m | multi | open (research) | allclear.cs.cornell.edu | **High** (largest, multimodal, modern — primary pretraining) |
| **SEN2-MTC (Old)** `[V]` | Multi-temporal S2 only (3 cloudy → 1 clear) | temporal optical | 945 tiles, **3130 pairs** | 10 m | 4 (RGB+NIR) | research `[?]` | via CTGAN repo | **Med-High** (4-band incl. NIR; pure optical temporal) |
| **SEN2-MTC (New)** `[V]` | Cleaner S2 multi-temporal | temporal optical | ~50 tiles, ~3500 pairs 256² | 10 m | 4 (RGB+NIR) | research `[?]` | Huang et al. repo `[?]` | **Med-High** (improved labels) |
| **RICE-I** `[V]` | Google-Earth RGB, thin/film cloud↔clear + mask | paired (synthetic clouds) | **500** pairs 512² | varies | 3 (RGB) | research | github.com/BUPTLdy/RICE_DATASET | **Low-Med** (RGB only, no NIR) |
| **RICE-II** `[V]` | Landsat-8 cloudy↔clear (≤15 d apart) + mask | paired temporal | 450 / **736** groups 512² | 30 m | 3 (RGB) | research | github.com/BUPTLdy/RICE_DATASET | **Low-Med** (RGB; real Landsat pairs) |
| **T-Cloud** `[V]` | Landsat-8 cloudy↔clear | paired | **2939** pairs 256² | 30 m | 3 (RGB) `[?]` | research | (Nature s41598-025-87296-x) `[?]` | **Low-Med** |
| **WHU Cloud Dataset** `[V]` | Landsat-8 cloudy + historical clear + cloud/shadow masks, 6 regions | paired temporal | 6 regions | 30 m | 3 (RGB, B4/3/2) | research | gpcv.whu.edu.cn/data/WHU_Cloud_Dataset.html | **Low-Med** (RGB only) |
| **CUHK-CR1 / CR2** `[V]` | **Jilin-1** ultra-res, thin (CR1) & thick (CR2) | paired | CR1 668 / CR2 559 imgs | **0.5 m** | **4 (RGB+NIR)** | research | github.com/littlebeen/Cloud-removal-model-collection | **Med** (has NIR + high-res; res mismatch but useful for detail) |

### B.2 Cloud-**detection / mask** datasets (for masking, simulation, eval)

| Dataset | Sensors | Size | Res. | Classes/bands | License | URL | Use |
|---|---|---|---|---|---|---|---|
| **CloudSEN12 / CloudSEN12+** `[V]` | S2 L1C+L2A + S1 SAR + aux | **49,400** patches (509² & 2000²) | 10 m | 4 classes (clear/thick/thin/shadow), 13+2 bands | **CC0** (public domain) | huggingface.co/datasets/tacofoundation/cloudsen12 ; cloudsen12.github.io | **Best mask source** (free, S2, includes SAR) |
| **KappaSet (KappaZeta)** `[V]` | S2 L1C | **9251** sub-tiles (512²) from 1038 products | 10 m | cloud/shadow masks | CC `[?]` | zenodo.org/records/7100327 | Global S2 mask labels |
| **38-Cloud** `[V]` | Landsat-8 | 8400 train / 9201 test patches 384² (38 scenes) | 30 m | binary cloud, 4 bands (R/G/B/NIR) | research | github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset | Cloud seg (has NIR) |
| **95-Cloud** `[V]` | Landsat-8 (75 train + 20 test scenes) | **34,701** train patches 384² | 30 m | binary cloud, 4 bands | research | github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset | Larger 38-Cloud extension |
| **Sentinel-2 Cloud Mask Catalogue** `[M]` | S2 L1C | 513 subscenes 1022² (20 hand-labelled) | 20/60 m | cloud/shadow masks | CC-BY | zenodo (Francis et al.) `[?]` | ESA reference masks |
| **SEN12MS** `[V]` (parent) | S1 SAR + S2 + MODIS landcover | 180,662 triplets 256² | 10 m | S2 13 + S1 2 + LC | CC-BY | github.com/schmitt-muc/SEN12MS `[?]` | Landcover/pretrain backbone |
| **PixBox** `[M]` | Multi-sensor expert pixel labels (EUMETSAT) | pixel-level | n/a | cloud types | EUMETSAT `[?]` | EUMETSAT `[?]` | Validation reference `[?]` |

### B.3 LISS-IV-specific datasets

- **No widely-published, ready-made LISS-IV cloud-removal benchmark exists** `[M]`. The practical path is to **build our own** paired/temporal set from **Bhoonidhi** (LISS-IV target + LISS-III/AWiFS temporal references) and **co-register Sentinel-2 / Sentinel-1** over the same NER tiles. NRSC's **Bhuvan/Bhoonidhi** archives provide LISS-IV scenes; pairs must be constructed by temporal matching (clear vs. cloudy passes) and/or **synthetic cloud injection** (see Part E). Re-confirm at proposal time whether NRSC has released any AI-ready cloud benchmark `[?]`.

---

## Part C — Data Access: Concrete How-To

### C.1 ISRO Bhoonidhi (primary LISS-IV source) `[V]`
- **Portal:** `https://bhoonidhi.nrsc.gov.in` — ISRO/NRSC Open Data Access hub.
- **Registration:** free account required; sign in to browse/order. Per India Space Policy 2023, **5 m Resourcesat-2/2A LISS-IV is now open data**.
- **Workflow:** Browse & Order → define AOI (draw NER polygon) → filter sensor=LISS-IV, date, cloud%. Online-available scenes download instantly; others are queued ("delayed download") with email/SMS notification.
- **Format:** GeoTIFF (Georeferenced TIFF, `.tif`), band-separated; standard products are radiometrically corrected & geo-referenced.
- Also: **Bhuvan** (`bhuvan.nrsc.gov.in`) for visualization; **MOSDAC** (`mosdac.gov.in`) for INSAT-3D/3DR cloud products. CartoDEM via Bhoonidhi.

### C.2 USGS EarthExplorer `[V]`
- `https://earthexplorer.usgs.gov` — free login. Sources: **Landsat-8/9 C2 L1/L2**, MODIS, ASTER, SRTM, and the **ISRO Resourcesat LISS-III** archive (US distribution). Bulk via `EE` + USGS `M2M` API / `landsatxplore`.

### C.3 Copernicus Data Space Ecosystem (CDSE) `[V]`
- `https://dataspace.copernicus.eu` — **replaces the old SciHub** (deprecated end-Oct 2023). Free account.
- APIs: **STAC** & **OpenSearch** (catalog), **OData** (full product download), **Sentinel Hub** API, **openEO** (server-side processing), and **S3** access. Use OData for whole S2/S1 granules; Sentinel Hub/openEO for on-the-fly subsets.

### C.4 Google Earth Engine (GEE) — exact collection IDs `[V]`
| Data | GEE Collection ID |
|---|---|
| Sentinel-2 **SR** (L2A, harmonized) | `COPERNICUS/S2_SR_HARMONIZED` |
| Sentinel-2 **TOA** (L1C, harmonized) | `COPERNICUS/S2_HARMONIZED` |
| Sentinel-2 cloud probability | `COPERNICUS/S2_CLOUD_PROBABILITY` |
| **Cloud Score+** (best S2 cloud QA) | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |
| Sentinel-1 SAR GRD | `COPERNICUS/S1_GRD` |
| Landsat-8 L2 SR | `LANDSAT/LC08/C02/T1_L2` |
| Landsat-9 L2 SR | `LANDSAT/LC09/C02/T1_L2` |
| MODIS surface reflectance (Terra) | `MODIS/061/MOD09GA` |
| MODIS vegetation/quality | `MODIS/061/MOD13Q1` |
| Copernicus DEM GLO-30 | `COPERNICUS/DEM/GLO30` |
| SRTM 30 m | `USGS/SRTMGL1_003` |
| NASADEM | `NASA/NASADEM_HGT/001` |
| ERA5-Land | `ECMWF/ERA5_LAND/HOURLY` |

GEE gives **server-side, on-the-fly** access (no bulk download) — ideal for building cloudy/clear composites, band-matching, and exporting LISS-IV-aligned tiles.

### C.5 Microsoft Planetary Computer (STAC) `[V]`
- `https://planetarycomputer.microsoft.com` — petabyte STAC catalog on Azure (**COG**-converted). Datasets incl. `sentinel-2-l2a`, `landsat-c2-l2`, `hls2-l30/-s30`, `sentinel-1-grd`, `cop-dem-glo-30`. Query via `pystac-client` + sign URLs with `planetary-computer` package; load lazily with `odc-stac`/`stackstac`.

### C.6 AWS Open Data / others `[M]`
- **Sentinel-2 COGs:** `s3://sentinel-cogs/` (Element 84 Earth Search STAC: `https://earth-search.aws.element84.com/v1`); **Sentinel-1**, **Landsat** (`s3://usgs-landsat/`, requester-pays). **sentinelhub-py** and **openEO** clients for subset/processing. These enable **range-request** reads of single bands.

---

## Part D — Master Comparison Table (all sensors)

| Platform / Sensor | Type | Res. | Bands | Revisit | Swath | Access | Role |
|---|---|---|---|---|---|---|---|
| Resourcesat-2/2A **LISS-IV** | Optical | **5.8 m** | 3 (G/R/NIR) | 5 d | 70/23.5 km | Bhoonidhi | **TGT** |
| Resourcesat LISS-III | Optical | 23.5 m | 4 | 24 d | 141 km | Bhoonidhi/EE | TMP/XV |
| Resourcesat AWiFS | Optical | 56 m | 4 | 5 d | 740 km | Bhoonidhi | XV/TMP |
| Cartosat-2 | Optical | 0.65 m | PAN(+MX) | ~5 d | 10 km | Bhoonidhi | AUX |
| Cartosat-3 | Optical | 0.25 m | PAN+4MX | agile | 16 km | Bhoonidhi | AUX |
| EOS-04 / RISAT-1A | **SAR (C)** | 1–50 m | — | ~12–25 d | var | Bhoonidhi | **AUX** |
| EOS-06 / Oceansat-3 | Optical | 360 m | 13 | 2 d | 1500 km | Bhoonidhi | XV |
| INSAT-3D/3DR | Optical(GEO) | 1–4 km | 6 | **30 min** | full-disk | MOSDAC | **XV** |
| **Sentinel-2** | Optical | **10/20/60 m** | **13** | **5 d** | 290 km | CDSE/GEE/MPC/AWS | **XV/PROXY** |
| **Sentinel-1** | **SAR (C)** | 5×20 m | 2 (VV/VH) | 6–12 d | 250 km | CDSE/GEE/MPC | **AUX** |
| Sentinel-3 | Optical | 300 m–1 km | 21 | 1–2 d | 1270 km | CDSE | XV |
| Landsat-8/9 OLI | Optical | 30 m (15 pan) | 11 | 16/8 d | 185 km | EE/GEE/MPC | TMP/XV |
| MODIS Terra/Aqua | Optical | 250–1000 m | 36 | 1–2 d | 2330 km | LP DAAC/GEE | **XV** (cloud) |
| VIIRS | Optical | 375/750 m | 22 | 1 d | 3060 km | LAADS/GEE | XV |
| ASTER | Optical | 15–90 m | 14 | ~16 d | 60 km | EE/GEE | AUX/DEM |
| **HLS** (L30+S30) | Optical | 30 m | harmonized | 2–3 d | tile | LP DAAC/MPC | TMP/XV |
| PlanetScope | Optical | 3–5 m | 8 | ~daily | swath | Planet | AUX/TMP |
| RapidEye | Optical | 5 m | 5 | 5.5 d | 77 km | Planet archive | TMP |
| SPOT-6/7 | Optical | 1.5/6 m | 5 | 1–3 d | 60 km | Airbus | AUX |
| WorldView-2/3 | Optical | 0.31 m | 8–16 | ~1 d | 13 km | Maxar | AUX |
| Capella / ICEYE | **SAR (X)** | ≤0.5–1 m | — | tasking | spot | Commercial | AUX |
| Copernicus DEM GLO-30 | DEM (DSM) | 30 m | — | static | global | GEE/MPC | AUX |
| SRTM / NASADEM | DEM | 30 m | — | static | near-global | GEE/EE | AUX |
| ALOS PALSAR/AW3D30 | DEM/SAR | 30 m | — | static | global | JAXA/GEE | AUX |
| CartoDEM | DEM | 10–30 m | — | static | India | Bhoonidhi | AUX |

---

## Part E — Cross-Verification & Gap-Filling Strategy

**Core idea:** LISS-IV alone cannot see through clouds, has no SWIR/cirrus band for cloud detection, and revisits only every ~5–24 days — so over cloudy NER, single-date reconstruction is under-constrained. We close the gaps with a **multi-sensor evidence stack**, each layer answering what LISS-IV cannot:

1. **Sentinel-2 → spectral/spatial proxy & verifier (XV).** Band-match S2 **B3→Green, B4→Red, B8/B8A→NIR** to LISS-IV. Use S2 (a) as a *clean-sky reference* to validate reconstructed reflectance values, (b) to **pretrain/transfer** generators (since most benchmarks are S2), and (c) for **super-resolution-style domain transfer** (S2 10 m → LISS-IV 5.8 m). S2's SWIR (B11/B12) + cirrus (B10) drive robust cloud masking (s2cloudless / Cloud Score+ / SCL) that LISS-IV's 3 bands cannot do alone.

2. **Sentinel-1 (and EOS-04/RISAT) SAR → cloud-penetrating structure (AUX).** C-band backscatter is **unaffected by clouds**, providing the *true* land structure under the cloud. Feed co-registered VV+VH as conditioning so the generator hallucinates texture consistent with real ground geometry — the single most important signal for **thick-cloud** reconstruction (the SEN12MS-CR / AllClear paradigm).

3. **LISS-III / AWiFS / Landsat-8/9 / HLS → temporal reference (TMP).** The nearest **clear-sky pass** (same or adjacent dates) gives the actual recent appearance of the surface. Use these as: temporal priors for the generator, change-detection guards (don't invent change that contradicts a clear neighbour), and ground-truth surrogates when constructing pairs. LISS-III/AWiFS are ideal because they share LISS-IV's platform/orbit/spectral design.

4. **MODIS / INSAT-3D / ERA5 → cloud climatology & acquisition planning (XV).** INSAT-3D's 30-min cadence and MODIS daily masks characterize *when* NER is least cloudy, flag which LISS-IV scenes are recoverable, and provide cloud-fraction labels. ERA5 total-cloud-cover supports site/date selection and difficulty stratification.

5. **DEM (Copernicus GLO-30 / CartoDEM) → shadow & terrain consistency (AUX).** With sun-azimuth/elevation, compute hillshade to (a) distinguish **cloud shadow vs. terrain shadow**, (b) condition reconstruction on slope/aspect so filled reflectance is physically plausible, and (c) co-register all sensors to a common terrain frame.

**Verification loop:** reconstruct LISS-IV → check against (i) SAR structure (edges/texture agree?), (ii) nearest clear S2/LISS-III (radiometry within tolerance?), (iii) DEM-predicted shadows (no spurious bright fills in shadowed terrain?), (iv) MODIS/INSAT cloud mask (was this pixel actually cloudy?). Disagreement flags low-confidence pixels.

---

## Part F — Recommended Dataset Mix for Training / Eval

**Priority order (with rationale):**

1. **AllClear** *(pretrain backbone)* — largest, multimodal (S2+Landsat+S1+aux), temporal, 2022, open. Best for learning general multimodal cloud-removal priors. **`[V]`**
2. **SEN12MS-CR-TS** *(multimodal-temporal method development)* — the canonical SAR+optical temporal benchmark; directly mirrors our LISS-IV+S1+temporal design. **`[V]`**
3. **SEN12MS-CR** *(single-pass SAR-optical baseline)* — train/validate the core "SAR-conditioned single-date" path. **`[V]`**
4. **CloudSEN12+** *(cloud/shadow masking + simulation)* — CC0, S2+SAR, expert masks; supplies masks for loss weighting, evaluation, and realistic cloud-mask sampling. **`[V]`**
5. **Our LISS-IV NER set** *(domain adaptation + final eval)* — Bhoonidhi LISS-IV target tiles co-registered with S2/S1/LISS-III over NER; the **only** data in the true target domain. Use for fine-tuning and as the held-out evaluation set. **`[M]`**
6. **SEN2-MTC New / CUHK-CR** *(optical-temporal & high-detail aux)* — 4-band (incl. NIR) optical temporal and ultra-res detail priors. **`[V]`**
7. **38-/95-Cloud, KappaSet, RICE-II, T-Cloud, WHU** *(auxiliary cloud-detection / RGB pairs)* — supplementary masks and Landsat RGB pairs for robustness; lower priority (RGB-only / res mismatch). **`[V]`**

**Cloud simulation (when real pairs are scarce — essential for LISS-IV):** because matched cloudy/clear LISS-IV pairs are rare, generate **synthetic clouds** on clear LISS-IV (and S2) scenes:
- Sample realistic cloud/shadow masks from **CloudSEN12+ / KappaSet / S2-Cloud-Mask-Catalogue**, composite with Perlin-noise + physically-plausible opacity/cirrus models (the SEN12MS-CR / RICE-I approach), and pair with the original clear image as ground truth.
- Validate that synthetic cloud statistics match **INSAT/MODIS** NER cloud climatology so the simulated distribution is regionally realistic.

**Evaluation metrics:** PSNR / SSIM / SAM (spectral angle, critical for 3-band radiometry) / per-band MAE on held-out NER LISS-IV, with **separate thin- vs. thick-cloud** breakdowns and SAR-only vs. SAR+temporal ablations.

---

## Part G — O(1) / Fast-Access Note

To pull terabytes without bulk-downloading:
- **Cloud-Optimized GeoTIFF (COG) + HTTP range requests:** read only the bands/windows you need (single LISS-IV-aligned tile) directly from object storage (`s3://sentinel-cogs/`, Planetary Computer, AWS Landsat) via `rasterio`/`rioxarray` with GDAL `/vsicurl/` and `VSI` block caching.
- **STAC search → lazy load:** `pystac-client` against CDSE/MPC/Earth-Search STAC endpoints, then `odc-stac`/`stackstac` to build an xarray cube on demand (no full-scene download).
- **GEE / Planetary Computer server-side:** do band-matching, cloud-masking, compositing, and reprojection-to-LISS-IV-grid **in the cloud**, export only the final aligned chips. This turns "find + fetch + preprocess" into near-**O(1)** per training tile.
- **Cache layer:** materialize the NER training tiles (LISS-IV + S1 + S2 + DEM stacks) once as local COGs / WebDataset shards for fast epoch iteration.

---

### Sources (verified this session)
- ISRO Bhoonidhi: https://bhoonidhi.nrsc.gov.in/ · User Manual: https://bhoonidhi.nrsc.gov.in/bhoonidhi_resources/help/docs/Bhoonidhi_ISROEOHub_UserManual_V2.0.pdf
- LISS/IRS specs (eoPortal): https://directory.eoportal.org/satellite-missions/irs
- EOS-04/RISAT-1A: https://en.wikipedia.org/wiki/EOS-04 · Cartosat-3: https://en.wikipedia.org/wiki/Cartosat-3
- SEN12MS-CR / -TS: https://patricktum.github.io/cloud_removal/
- AllClear: https://allclear.cs.cornell.edu/ (arXiv 2410.23891)
- CloudSEN12: https://cloudsen12.github.io/ · https://huggingface.co/datasets/tacofoundation/cloudsen12 · Nature: https://www.nature.com/articles/s41597-022-01878-2
- RICE: https://github.com/BUPTLdy/RICE_DATASET · WHU: https://gpcv.whu.edu.cn/data/WHU_Cloud_Dataset.html
- CUHK-CR: https://github.com/littlebeen/Cloud-removal-model-collection (arXiv 2401.15105)
- 38-/95-Cloud: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset · https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset
- KappaSet: https://zenodo.org/records/7100327
- GEE catalog: S2_SR_HARMONIZED, S1_GRD, LC08/LC09 C02 T1_L2, COPERNICUS/DEM/GLO30, USGS/SRTMGL1_003, NASA/NASADEM_HGT/001, COPERNICUS/S2_CLOUD_PROBABILITY, GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED (developers.google.com/earth-engine/datasets)
- CDSE: https://dataspace.copernicus.eu/ · Planetary Computer: https://planetarycomputer.microsoft.com/ · EarthExplorer: https://earthexplorer.usgs.gov/
