# Data Card — `cloudremoval` (BAH 2026 PS2)

**Product:** Model-agnostic GenAI cloud removal & reconstruction for **LISS-IV** imagery.
**Target sensor:** Resourcesat-2/2A **LISS-IV** (5.8 m, 3-band Green/Red/NIR, 10-bit), North-East India (NER).
**Scope of this card:** what data the system consumes, where it comes from, how paired
(cloudy, clear) supervision is manufactured, how sources cross-verify each other, and the
exact configs / CLI commands that exercise the data layer.

> Verified facts are grounded in [`research/04_datasets_and_satellites.md`](../research/04_datasets_and_satellites.md)
> and [`research/06_lissiv_isro_bhoonidhi_ner.md`](../research/06_lissiv_isro_bhoonidhi_ner.md);
> the design rationale is in [`ARCHITECTURE.md`](../ARCHITECTURE.md) §2. The companion
> [`MODEL_CARD.md`](MODEL_CARD.md) documents the models trained on this data.

---

## 1. Motivation — why a multi-satellite data strategy at all

LISS-IV is uniquely hard to de-cloud, and **no public LISS-IV cloud-removal benchmark exists**
(`research/04` §B.3, `research/06` §C.5). Three facts drive the entire data strategy:

1. **Only 3 VNIR bands — no blue, no SWIR, no cirrus.** Green 0.52–0.59, Red 0.62–0.68,
   NIR 0.77–0.86 µm (`research/06` §A.1). Standard cloud masks (Fmask, s2cloudless, Sen2Cor)
   assume blue/SWIR/cirrus bands and **cannot run unmodified** on LISS-IV.
2. **Severe scarcity of matched (cloudy, clear) LISS-IV pairs.** 24-day repeat (~5-day steered)
   × NER's ~90–92 % Jul–Aug monsoon cloud (`research/06` §A.2, §D.2) means a clear and a cloudy
   acquisition of the *same* footprint close in time is rare. Supervised training on real pairs
   alone is impossible.
3. **Raw 10-bit DN, no surface reflectance shipped** (`research/06` §A.5). Each scene's radiometry
   is scene-dependent; a model must be fed a normalized signal to transfer to/from Sentinel-2.

We therefore **(a) manufacture paired training data** with a synthetic-cloud generator,
**(b) borrow signal from many other satellites** (SAR sees through cloud; S2 is a band-matched
spectral proxy; LISS-III/AWiFS supply the missing SWIR; MODIS/INSAT give cloud climatology; a DEM
gives shadow geometry), and **(c) cross-verify** the reconstruction against that evidence stack.

---

## 2. Source roster (target / auxiliary / temporal / cross-verify)

LISS-IV is the **only** reconstruction target (TGT). Everything else is *evidence* that closes what
LISS-IV cannot see. Roles: **TGT** target · **AUX** auxiliary fusion · **TMP** temporal reference ·
**XV** cross-verification / cloud climatology (`research/04` Part A, Part E).

| Source | Role | Bands / signal | What it contributes | Access |
|---|---|---|---|---|
| **LISS-IV** (Resourcesat-2/2A) | **TGT** | G/R/NIR @ 5.8 m, 10-bit | The scene to reconstruct | Bhoonidhi (open ≤5 m) |
| **Sentinel-1 GRD** (C-band) | **AUX** | VV+VH @ ~10 m | Cloud-penetrating *structure* under thick cloud — the #1 thick-cloud signal | CDSE / GEE / MPC |
| **Sentinel-2 L2A** | **XV / PROXY** | {B3, B4, B8} @ 10 m | Band-matched spectral proxy, pretraining domain, clear-sky verifier | CDSE / GEE / MPC / AWS |
| **LISS-III / AWiFS** | **TMP / XV** | G/R/NIR **+ SWIR** | Same-platform spectral siblings + the **SWIR teacher** band LISS-IV lacks | Bhoonidhi / EarthExplorer (LISS-III) |
| **Landsat-8/9 / HLS** | **TMP / XV** | 11-band incl. cirrus/QA @ 30 m | Calibrated temporal stack; cirrus/QA aids masking on the proxy side | EarthExplorer / GEE / MPC |
| **MODIS / INSAT-3D/3DR / ERA5** | **XV** | cloud products | Cloud climatology (when is NER clear), per-scene cloud fraction, acquisition planning | LP DAAC / MOSDAC / CDS |
| **Copernicus DEM GLO-30 / CartoDEM** | **AUX** | 30 m DSM | Cloud-shadow vs terrain-shadow geometry; slope/aspect conditioning; ortho frame | GEE / MPC / Bhoonidhi |
| **PlanetScope** (optional) | **AUX / TMP** | 8-band @ 3–5 m | Near-daily gap-fill where available | Commercial / NICFI |

Benchmark **transfer datasets** (used to pretrain before domain-adapting to LISS-IV; `research/04`
§B.1, Part F): **SEN12MS-CR** (122,218 SAR+cloudy+clear S2 triplets), **SEN12MS-CR-TS** (multi-temporal),
**AllClear** (~4 M images, largest multimodal/temporal), **CloudSEN12 / CloudSEN12+** (CC0 cloud/shadow
masks), **RICE-I/II** (RGB cloudy↔clear).

---

## 3. The band-mapping table (the linchpin for transfer)

LISS-IV's three bands map almost 1:1 onto Sentinel-2 and the Resourcesat siblings — this is what
makes transfer from S2-based benchmarks viable (`research/06` §B.3, `research/04` §A.2). The code's
S2→LISS-IV map is `cloudremoval.data.sources.gee.S2_TO_LISSIV_BANDS = {"B3": "green", "B4": "red",
"B8": "nir"}`, and `Sen12msCrDataset.S2_BAND_INDICES = (2, 3, 7)` selects exactly B3/B4/B8 from a
13-band S2 stack.

| LISS-IV | λ (µm) | Sentinel-2 (MSI) | Landsat-8/9 (OLI) | LISS-III / AWiFS | Note |
|---|---|---|---|---|---|
| **B2 Green** | 0.52–0.59 | **B3** (560 nm, 10 m) | B3 (0.53–0.59) | Green (0.52–0.59) | near-identical |
| **B3 Red** | 0.62–0.68 | **B4** (665 nm, 10 m) | B4 (0.64–0.67) | Red (0.62–0.68) | near-identical |
| **B4 NIR** | 0.77–0.86 | **B8** (833 nm, 106 nm wide, 10 m) — or B8A (865 nm, 21 nm) | B5 (0.85–0.88) | NIR (0.77–0.86) | LISS-IV NIR is **wide** → apply **SBAF** + histogram match (B8 closer by bandwidth) |
| *(none)* | — | B11/B12 SWIR; B10 cirrus | SWIR / Cirrus | **SWIR 1.55–1.70** | LISS-IV lacks SWIR → use LISS-III/AWiFS or S2 SWIR only on the *proxy* side as masking / teacher signal |

### Why 3-band / no-blue / no-SWIR matters (and what the code does about it)

- **No blue** → blue-band whiteness/brightness cloud tests are unavailable. The masking logic keys on
  high reflectance across **all three** G/R/NIR bands + low NDVI + texture (`research/06` §E.1).
- **No SWIR / cirrus** → NDSI (snow), MNDWI (water), and cirrus detection are impossible from LISS-IV
  alone; **snow ≈ cloud** in 3 bands (a real false-positive hazard in Sikkim/Arunachal). Mitigations:
  SAR + temporal anomaly, LISS-III/AWiFS SWIR as a teacher band, DEM/elevation priors + temporal
  persistence (snow is static; cloud moves) (`research/06` §E.1, Risk table #3).
- **Spectral fidelity is the deliverable axis, not raw PSNR.** Because only 3 bands feed NDVI/LULC,
  the system optimizes and reports **SAM / ERGAS / SID / NDVI-MAE** (see [`MODEL_CARD.md`](MODEL_CARD.md) §5).
  The wide LISS-IV NIR vs narrow S2-B8A means a naive S2→LISS-IV transfer biases NIR/NDVI, so an
  **SBAF + histogram-match** step is required before transfer (`research/06` Risk table #6).

### DN → reflectance calibration (so the model never sees raw scene-dependent DN)

`cloudremoval.data.preprocess.dn_to_toa_reflectance` implements
`ρ_TOA = (π · L · d²) / (ESUN · cos θz)` with `L = (Lmax/1023) · DN`, using the documented LISS-IV
constants (`research/06` §A.5):

- `LISSIV_LMAX = (53.0, 47.0, 31.5)` mW·cm⁻²·sr⁻¹·µm⁻¹ for B2/B3/B4 (`DN_max = 1023` for 10-bit).
- `LISSIV_ESUN = (185.3, 158.4, 109.1)` (defaults; production reads `Lmax`/sun-elevation **per scene**
  from `BAND_META.txt`).

The alternative robust path for real scenes is a **per-scene percentile (2–98 %) stretch**
(`configs/data/lissiv_ner.yaml: norm: percentile`).

---

## 4. How we obtain paired (cloudy, clear) data — the three-pronged data engine

Because no LISS-IV cloud benchmark and almost no real pairs exist, supervision is manufactured
(`ARCHITECTURE.md` §2.3, `research/06` §E.5):

### 4.1 Synthetic cloud injection (the implemented paired-pair generator)

`cloudremoval.data.synthetic_clouds.CloudSimulator` (and the functional `add_synthetic_clouds`) take a
**clear** 3-band scene as ground truth and paste physically-plausible clouds + cast shadows to produce a
perfect `(cloudy, clear, cloud_mask, shadow_mask)` tuple. It is **pure NumPy/torch** (a self-contained
fractal value-noise / fBm field replaces the `noise` package), so it runs on the CPU-smoke stack with no
dependencies. Key `CloudSimConfig` controls:

| Field | Default | Meaning |
|---|---|---|
| `cloud_mode` | `"mixed"` | `"thin"` (haze, alpha-blended, spectrally recoverable) / `"thick"` (opaque core) / `"mixed"` |
| `coverage` (+ `coverage_jitter`) | 0.35 (± 0.15) | target cloud-cover fraction |
| `thickness` / `thin_max_alpha` | 1.0 / 0.6 | peak opacity (thick) / alpha cap (thin keeps surface partly visible) |
| `cloud_brightness` | 0.92 | reflectance the cloud blends toward |
| `shadow_strength` / `shadow_offset_frac` | 0.45 / 0.06 | darkening at the shadow core / cast-shadow displacement (scaled by `1/tan(sun_elevation)`) |
| `add_shadows` | `True` | cast a geometry-offset shadow *away* from the sun |

Thin vs thick are generated **separately** to match the stratified evaluation (thin = recoverable haze,
thick = inpainting/SAR-driven). A **copy-paste** path accepts a real cloud-alpha matte via `real_alpha=`
— the lower synthetic→real domain-gap option, intended to be fed alpha-mattes sampled from
**CloudSEN12+ / KappaSet** and validated against **MODIS/INSAT NER climatology** so opacity/coverage
distributions are regionally realistic (`research/04` Part F, `research/05` §B.5).

> **Honest scope note.** The shipped generator uses fractal (fBm) clouds + a geometry-offset shadow and a
> `real_alpha` hook; copy-paste from real CloudSEN12+ mattes and the climatology validation are documented
> integration points, not bundled assets, because the smoke path must stay offline and dependency-free.

### 4.2 Transfer learning (benchmark → LISS-IV)

Pretrain on **AllClear** (largest multimodal/temporal) and **SEN12MS-CR(-TS)** (single-pass / temporal
SAR-optical), then **domain-adapt to LISS-IV** (3-band, SBAF-aligned, MAE self-supervised pretext on
unlabeled NER tiles), with mask labels from **CloudSEN12+** (CC0). The `Sen12msCrDataset` adapter
(`cloudremoval.data.datasets`) maps S2 {B3,B4,B8} → LISS-IV {Green,Red,NIR} and feeds SAR into the `sar`
channel of the universal `SAMPLE` dict, so a SEN12MS-CR-pretrained model transfers without code changes.

### 4.3 Cross-sensor fusion & domain adaptation (LISS-IV ↔ S2)

Histogram matching / SBAF (`cloudremoval.inference.postprocess.histogram_match`,
`per_band_bias_correct`) plus optional CycleGAN-style alignment bridge the residual spectral gap so
S2-trained weights generalize to scarce LISS-IV. SAR/temporal fusion is the strongest NER mitigation
(`research/06` §D.5, §E.5).

### What the data layer actually emits — the `SAMPLE` dict

Every dataset `__getitem__` returns the universal `SAMPLE` dict (`ARCHITECTURE.md` §3.2,
[`BUILD_PLAN.md`](BUILD_PLAN.md) §2), so the trainer/evaluator/inference never special-case a source.
Absent optional modalities are **zero-filled** (never missing), with presence flags in `meta`:

```python
SAMPLE = {
    "optical_cloudy": Tensor,   # [C, H, W]      model input (cloud-contaminated)
    "optical_clear":  Tensor,   # [C, H, W]      target (zeros at inference)
    "cloud_mask":     Tensor,   # [1, H, W]      1.0 = cloud
    "shadow_mask":    Tensor,   # [1, H, W]      1.0 = cloud-shadow
    "sar":            Tensor,   # [2, H, W]      VV, VH (zeros if absent)
    "dem":            Tensor,   # [1, H, W]      normalized elevation/slope (zeros if absent)
    "temporal_refs":  Tensor,   # [T, C, H, W]   nearest-clear refs ([0,C,H,W] if none)
    "meta":           dict,     # scene_id, date, sun_elevation, sun_azimuth, crs, transform,
                                # band_stats, norm, cloud_type, cloud_fraction, has_{target,sar,dem,temporal}
}
# C = 3 (Green=0, Red=1, NIR=2)
```

The **synthetic** dataset (`SyntheticCloudRemovalDataset`) procedurally generates a clear 3-band
LISS-IV-like scene (fractal terrain → NDVI-ordered G/R/NIR reflectances), derives a SAR proxy from NIR
edges + speckle and a DEM from the terrain field, then injects clouds/shadows — needing **no external
data**. Two real-source adapters are documented and lazy-`rasterio`-guarded (not on the smoke path):
`LissivNerDataset` (Bhoonidhi per-band GeoTIFFs + `BAND_META.txt`) and `Sen12msCrDataset` (GeoTIFF
triplets via a `manifest.json`).

---

## 5. Cross-verification & gap-filling logic

Reconstruction of a 3-band cloudy scene is under-constrained, so the system stacks independent evidence
and **cross-checks the output** (`ARCHITECTURE.md` §2.4, `research/04` Part E):

1. **SAR structure** — do reconstructed edges/texture agree with co-registered Sentinel-1?
2. **Spectral reference** — is reconstructed reflectance within tolerance of the nearest clear
   Sentinel-2 / LISS-III?
3. **DEM shadow geometry** — are there spurious bright fills inside a DEM-predicted shadow?
4. **Cloud climatology** — was this pixel actually cloudy per MODIS/INSAT?

Disagreement on any check lowers that pixel's confidence, propagated alongside the model's own aleatoric
uncertainty head (the `uncertainty` model — see [`MODEL_CARD.md`](MODEL_CARD.md)). This verification loop
is the "many satellites for cross-verification" requirement made operational.

---

## 6. Data access (registration, collection IDs, endpoints)

### 6.1 Bhoonidhi (primary LISS-IV source) — `research/06` §C

- **Portal:** <https://bhoonidhi.nrsc.gov.in> (NRSC/ISRO open-data hub). Code constant:
  `cloudremoval.data.sources.bhoonidhi.BHOONIDHI_PORTAL`.
- **Policy:** under the **Indian Space Policy 2023**, Resourcesat-1/2/2A EO data ≤5 m GSD is **open data**
  — LISS-IV at 5.8 m is **free after registration + EULA**.
- **Workflow** (`order_workflow()` documents it; no live API): register → Data Discovery & Download →
  define AOI (draw / upload KML/GeoJSON / coordinates) → filter Satellite = Resourcesat-2/2A,
  Sensor = **LISS-IV MX**, date range, cloud% → preview quick-looks → order → download **ZIP** of
  per-band GeoTIFFs (`BAND2/3/4.tif`) + `BAND_META.txt`. `read_lissiv_scene()` is the local reader.
- **Note:** USGS EarthExplorer hosts Resourcesat **LISS-III** (not LISS-IV) — do not rely on USGS for the target.

### 6.2 Google Earth Engine — verified collection IDs (`research/04` §C.4)

Exact IDs live in `cloudremoval.data.sources.gee.COLLECTIONS`:

| Data | GEE Collection ID |
|---|---|
| Sentinel-2 SR (L2A, harmonized) | `COPERNICUS/S2_SR_HARMONIZED` |
| Sentinel-2 TOA (L1C, harmonized) | `COPERNICUS/S2_HARMONIZED` |
| Sentinel-2 cloud probability | `COPERNICUS/S2_CLOUD_PROBABILITY` |
| Cloud Score+ (best S2 cloud QA) | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |
| Sentinel-1 SAR GRD | `COPERNICUS/S1_GRD` |
| Landsat-8 / -9 L2 SR | `LANDSAT/LC08/C02/T1_L2` · `LANDSAT/LC09/C02/T1_L2` |
| Copernicus DEM GLO-30 | `COPERNICUS/DEM/GLO30` |
| SRTM 30 m · NASADEM | `USGS/SRTMGL1_003` · `NASA/NASADEM_HGT/001` |
| ERA5-Land | `ECMWF/ERA5_LAND/HOURLY` |

`build_s2_lissiv_composite(aoi, start, end, ...)` builds a cloud-masked S2 {B3,B4,B8} median composite as
a LISS-IV surrogate (apply SBAF downstream); `export_image(..., scale=5.8)` exports at the LISS-IV grid.

### 6.3 STAC / Planetary Computer / EarthExplorer / CDSE (`research/04` §C.2–C.6)

- **Microsoft Planetary Computer** — STAC catalog of COGs (`sentinel-2-l2a`, `landsat-c2-l2`,
  `hls2-l30/-s30`, `sentinel-1-grd`, `cop-dem-glo-30`); query via `pystac-client`, sign URLs with
  `planetary-computer`, lazy-load via `odc-stac`/`stackstac`.
- **Copernicus Data Space Ecosystem (CDSE)** — <https://dataspace.copernicus.eu> (replaces SciHub);
  STAC/OpenSearch catalog, OData download, Sentinel Hub & openEO server-side processing, S3 access.
- **USGS EarthExplorer** — <https://earthexplorer.usgs.gov> for Landsat-8/9 C2 L1/L2, MODIS, ASTER, SRTM,
  Resourcesat LISS-III; bulk via `landsatxplore`/M2M.
- **AWS Open Data** — `s3://sentinel-cogs/`, Element 84 Earth Search STAC, `s3://usgs-landsat/`
  (range-request reads of single bands → O(1) per tile, `research/04` Part G).

### 6.4 Licenses (`ARCHITECTURE.md` §7, `research/04` §B)

Prefer MIT/Apache/CC0 sources. **CloudSEN12+** is **CC0** (public domain) — the preferred mask source.
SEN12MS-CR(-TS) is CC-BY (confirm). Sentinel-1/2 (Copernicus) and Landsat (USGS) are open. LISS-IV is
ISRO open data (≤5 m, Space Policy 2023) after registration. GPL-licensed reference repos (e.g. the
DSen2-CR code) are **reimplemented, not vendored**; external pretrained weights are loaded at runtime via
adapters with a from-scratch fallback.

---

## 7. Configs and commands that exercise the data layer

### 7.1 Data configs (`configs/data/*.yaml`)

| Config | Dataset | Notes |
|---|---|---|
| `configs/data/synthetic.yaml` | `synthetic` | Fully offline; 64² tiles, `synthetic_n: 8`, SAR+DEM on. **Drives the CPU smoke.** |
| `configs/data/lissiv_ner.yaml` | `lissiv_ner` | Bhoonidhi LISS-IV tiles under `root`; 256² tiles, `norm: percentile`. Needs `rasterio` (`[geo]` extra) + downloaded scenes. |
| `configs/data/sen12mscr.yaml` | `sen12mscr` | SEN12MS-CR triplets under `root` + `manifest.json`; S2 {B3,B4,B8} → LISS-IV {G,R,NIR}; SAR on. Pretraining source. |

These are partial `data:` blocks; in practice they are deep-merged under `configs/base.yaml` (or used as
`--set` override sources). The smoke entrypoint `configs/cpu_smoke.yaml` already pins
`data.name: synthetic`, `tile_size: 64`, `synthetic_n: 8`.

### 7.2 CLI — simulate / download / preprocess

```bash
# Generate synthetic cloudy/clear paired tiles (offline, CPU) — exercises CloudSimulator
cloudremoval simulate --config configs/cpu_smoke.yaml
# (scripts.simulate.main also accepts out_dir / n; override n inline:)
cloudremoval simulate --config configs/cpu_smoke.yaml --set data.synthetic_n=16

# Fetch source imagery for an AOI (delegates to scripts.download; real sources need the [data]/[geo] extras)
cloudremoval download --config configs/data/lissiv_ner.yaml --aoi aoi/ner.geojson --source bhoonidhi
cloudremoval download --config configs/gpu_full.yaml --source stac        # or: gee | sentinel | dem

# Radiometric (DN→TOA) + geometric preprocessing + masking (real data; lazy rasterio)
cloudremoval preprocess --config configs/data/lissiv_ner.yaml
```

> On the minimal CPU stack, `download`/`preprocess` against real-source configs report a clean
> "not yet available / install the extra" message rather than crashing; `simulate` runs fully offline.
> See [`API.md`](API.md) §1 for the complete CLI reference and [the quickstart notebook](../notebooks/01_quickstart.ipynb)
> for a runnable `CloudSimulator` demo.

---

## 8. Known limitations of the data

- **Smoke data is synthetic.** The CPU-smoke pipeline uses procedurally-generated LISS-IV-like scenes +
  synthetic clouds; metrics from it are **illustrative of plumbing, not of real-world accuracy**. Real
  validation requires Bhoonidhi LISS-IV + co-registered Sentinel-1/2 + DEM over NER.
- **Synthetic→real domain gap.** fBm clouds approximate, but do not equal, real cloud statistics; the
  `real_alpha` copy-paste path + climatology validation are the documented mitigations.
- **Metric-transfer fallacy.** Published PSNR/SSIM/SAM are mostly Sentinel-2 13-band/10 m and **do not
  transfer directly** to LISS-IV 3-band/5.8 m; report relative-to-baseline on the NER set, not absolute.
- **Co-registration risk.** 5.8 m LISS-IV vs 10 m S1/S2 over NER relief + ±26° steering induces sub-pixel
  misregistration; mask-weighted losses + (documented) AROSICS/Align-CR alignment mitigate but do not
  eliminate it.
- **Snow / thin-cirrus** remain intrinsically hard with 3 VNIR bands; flagged via uncertainty + SAR/temporal
  cross-checks rather than fully solved.
