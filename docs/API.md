# API Reference — `cloudremoval` (BAH 2026 PS2)

Operational reference for the **CLI**, the **REST serving API**, and the **O(1)-per-tile serving design**.
Everything here is verified against `src/cloudremoval/` (CLI: `cli.py`; serving: `serving/{app,schemas,tiles}.py`).

Conventions used by the copy-paste examples:

```bash
pip install -e .                       # core stack (numpy, torch, pydantic, fastapi, typer, httpx)
python -c "import cloudremoval; print(cloudremoval.__version__)"   # 0.1.0
cloudremoval --help                    # or: python -m cloudremoval.cli --help
```

The CPU-smoke config `configs/cpu_smoke.yaml` (64² tiles, 2 steps, `device: cpu`, `model.name: unet`) is
used throughout — it runs offline in seconds.

---

## 1. CLI reference (Typer)

Single entry point `cloudremoval` with eight subcommands
(`download · preprocess · simulate · train · eval · benchmark · infer · serve`). Global option
`--version/-V`. Every subcommand takes:

- `--config / -c PATH` *(required)* — YAML config.
- `--set / -s dotted.key=value` *(repeatable)* — typed override deep-merged on top
  (e.g. `-s model.name=dsen2cr -s train.max_steps=1`); values are coerced to bool/int/float/str/None.

Each subcommand loads the config, **seeds all RNGs from `train.seed`**, then delegates to
`scripts/<name>.py:main(cfg, **kwargs)`. A subcommand whose script module is absent prints a clean
*"not yet implemented"* message (exit code 2) instead of breaking the CLI.

### Command summary

| Command | Extra flags | Delegates to | Purpose |
|---|---|---|---|
| `download` | `--aoi PATH`, `--source {bhoonidhi\|stac\|gee\|sentinel\|dem}` | `scripts.download` | Fetch source imagery for an AOI (real sources need `[data]`/`[geo]`). |
| `preprocess` | – | `scripts.preprocess` | Radiometric (DN→TOA) + geometric preprocessing + masking. |
| `simulate` | – | `scripts.simulate` | Generate synthetic cloudy/clear paired tiles (offline). |
| `train` | `--ckpt PATH` | `scripts.train` | Train the configured model. |
| `eval` | `--ckpt PATH` | `scripts.eval` | Evaluate with masked/stratified MVES metrics. |
| `benchmark` | – | `scripts.benchmark` | Benchmark **all registered models** → leaderboard. |
| `infer` | `--ckpt PATH`, `--input PATH`, `--out PATH` | `scripts.infer` | Tiled inference → reconstructed COG. |
| `serve` | – | `scripts.serve` | Launch the FastAPI serving app. |

### Copy-paste examples (work on the CPU smoke)

```bash
# 0) Sanity: registry loads on the minimal stack
python -c "from cloudremoval.models.registry import list_models; print(list_models())"
# -> ['diffusion', 'dsen2cr', 'restormer', 'spagan', 'uncertainty', 'unet']

# 1) Synthetic paired tiles (offline)
cloudremoval simulate --config configs/cpu_smoke.yaml
cloudremoval simulate --config configs/cpu_smoke.yaml --set data.synthetic_n=16   # more samples

# 2) Train — default model is unet; override per model and cap steps inline
cloudremoval train --config configs/cpu_smoke.yaml
cloudremoval train -c configs/cpu_smoke.yaml -s model.name=dsen2cr -s train.max_steps=1
cloudremoval train -c configs/cpu_smoke.yaml -s model.name=diffusion -s model.sample_steps=2

# 3) Evaluate (masked + whole MVES; PSNR/SSIM/SAM/ERGAS/NDVI-MAE per stratum)
cloudremoval eval --config configs/cpu_smoke.yaml
cloudremoval eval -c configs/cpu_smoke.yaml --ckpt outputs/smoke/ckpt/best.pt

# 4) Benchmark every registered model → leaderboard.{json,csv}
cloudremoval benchmark --config configs/cpu_smoke.yaml

# 5) Tiled inference → reconstructed COG (writes to infer.out_cog or --out)
cloudremoval infer -c configs/cpu_smoke.yaml --input scene.npy --out outputs/recon.tif

# 6) Serve the FastAPI app (host/port from serve config)
cloudremoval serve --config configs/cpu_smoke.yaml      # -> http://127.0.0.1:8000

# Real-data configs (require the [geo]/[data] extras + downloaded data)
cloudremoval download   -c configs/data/lissiv_ner.yaml --source bhoonidhi --aoi aoi/ner.geojson
cloudremoval preprocess -c configs/data/lissiv_ner.yaml
```

> Scripts are also runnable standalone (they bootstrap `src/` onto `sys.path`):
> `python scripts/train.py --config configs/cpu_smoke.yaml`.

The configurable knobs live in `cloudremoval.config` (`pydantic v2`): sections `data`, `model`, `train`
(+ nested `loss`), `eval`, `infer`, `serve`. A YAML may carry a `base:` key that is deep-merged underneath.
See [`DATA_CARD.md`](DATA_CARD.md) §7 and [`MODEL_CARD.md`](MODEL_CARD.md) §2 for data/model specifics.

---

## 2. REST API reference

The app is `cloudremoval.serving.app:app` (built by `create_app(cfg)`). **Import safety is a hard
requirement:** the app imports and `/health` answers `200` on the *minimal* stack — every heavy dependency
(rasterio / rio-tiler / faiss / redis) is imported lazily inside the function that needs it. Models are
loaded **on demand** and cached in-process (`ModelCache`).

Run it:

```bash
cloudremoval serve --config configs/cpu_smoke.yaml
# or directly:  uvicorn cloudremoval.serving.app:app --host 127.0.0.1 --port 8000
```

### Endpoints

| Method & path | Request | Response model | Notes |
|---|---|---|---|
| `GET /health` | – | `HealthResponse` | Liveness — always `200 {"status":"ok","version":...}`. |
| `GET /version` | – | `{"version": str}` | Package version. |
| `GET /info` | – | `InfoResponse` | Capabilities, registered `models`, `device`, `cache_backend`, and which heavy deps are present. |
| `GET /models` | – | `ModelsResponse` | `{"models":[...], "default": models[0]}` (the registry list). |
| `POST /predict` | `PredictRequest` | `PredictResponse` | Lazily load the model, run tiled inference (+ optional composite), return reconstruction + quick metrics. |
| `GET /tiles/{z}/{x}/{y}.png` | query: `scene`, `model_name`, `rescale` | `image/png` | Dynamic XYZ tile via rio-tiler (ranged COG GET); cached by content hash; `503` if rio-tiler absent, 1×1 placeholder if no `scene`. |
| `GET /tiles/info` | – | JSON | Which tile backends (rio-tiler / redis / faiss) are available + the O(1) design string. |

### Schemas (from `serving/schemas.py`, `pydantic v2`)

```jsonc
// PredictRequest
{
  "model_name": "unet",            // REQUIRED to be a registered key; schema default "identity" is NOT registered -> 404
  "array":      { "npy_base64": "<base64 of numpy.save([C,H,W] float)>", "note": null },  // inline input
  "input_ref":  null,               // OR a server-side path / COG URL / scene id (read via utils.io.read_array)
  "cloud_mask": null,               // optional inline [1,H,W] mask (ArraySpec), enables compositing
  "options": {                      // PredictOptions (all optional)
    "tile_size": 256, "overlap": 32, "batch_size": 4,
    "blend": "hann",                // hann | gaussian | linear | none
    "composite_under_mask": true,   // paste recon only under the cloud mask
    "harmonize": false,
    "return_array": true            // echo recon inline (base64 npy) in the response
  },
  "checkpoint": null                // optional state_dict path to load (non-strict)
}

// PredictResponse
{
  "status": "ok",
  "model_name": "unet",
  "output_ref": null,
  "array_base64": "<base64 npy of [C,H,W] reconstruction>",   // present when options.return_array
  "shape": [3, 64, 64],
  "metrics": { "recon_mean": 0.31, "recon_std": 0.12,         // + residual_mae/rmse when a mask is composited
               "...": 0.0 },
  "uncertainty": null,
  "cached": false,
  "message": null
}
```

> **Gotcha:** the schema default `model_name` is `"identity"`, which is **not** a registered model — a bare
> `POST /predict` with no `model_name` returns `404 unknown model 'identity'`. Always pass a real registry
> key (`unet`, `dsen2cr`, `restormer`, `spagan`, `diffusion`, `uncertainty`). `InfoResponse`/`ModelsResponse`
> expose the live list. `ReconstructRequest`/`ReconstructResponse` are BUILD_PLAN §3.9 aliases over the same
> fields (`cog_url`/`scene_id`/`array`/`model`/`params`).

### curl examples

```bash
# Meta endpoints
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/info
curl -s http://127.0.0.1:8000/models
curl -s "http://127.0.0.1:8000/tiles/info"

# Placeholder tile (no scene COG -> valid 1x1 PNG, never 500)
curl -s "http://127.0.0.1:8000/tiles/0/0/0.png?model_name=unet" -o tile.png
```

### Python (`httpx`) example — `POST /predict` with an inline array

```python
import base64, io, numpy as np, httpx

# Build a tiny [C,H,W] cloudy scene and base64-encode it as .npy bytes.
cloudy = np.random.rand(3, 64, 64).astype(np.float32)
buf = io.BytesIO(); np.save(buf, cloudy)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

req = {
    "model_name": "unet",                       # a real registry key (NOT the "identity" default)
    "array": {"npy_base64": b64},
    "options": {"tile_size": 64, "overlap": 8, "blend": "hann", "return_array": True},
}
r = httpx.post("http://127.0.0.1:8000/predict", json=req, timeout=60.0)
r.raise_for_status()
out = r.json()
print(out["shape"], out["metrics"])
recon = np.load(io.BytesIO(base64.b64decode(out["array_base64"])))   # [3,64,64] reconstruction
```

### Python — in-process `TestClient` (no server, used by the test suite & notebook)

```python
from fastapi.testclient import TestClient
from cloudremoval.serving.app import app

client = TestClient(app)
assert client.get("/health").json()["status"] == "ok"
print(client.get("/models").json())                 # live registry list
# POST /predict with the same JSON body as above:
resp = client.post("/predict", json=req)            # req from the httpx example
print(resp.json()["shape"], resp.json()["metrics"])
```

---

## 3. O(1) / fast-serving design

True whole-scene O(1) generation is impossible; the stack engineers **O(1)-per-tile reads, O(1)-ish
lookup, amortized-O(1) serving, and sublinear retrieval** (`ARCHITECTURE.md` §6, `research/05` §C). The
design degrades gracefully on the minimal stack and is fully exercised only with the optional extras.

1. **COG range-reads (O(1) read).** `GET /tiles/{z}/{x}/{y}.png` renders a single XYZ tile straight from a
   Cloud-Optimized GeoTIFF via `rio-tiler`, which issues an HTTP **ranged GET** for just that tile's bytes —
   independent of scene size. Missing `rio-tiler` → `503` with a `pip install 'cloudremoval[serve]'` hint,
   not a crash.
2. **Content-addressed cache (amortized O(1) serve).** `content_key(...)` = SHA-256 of
   `tile_identity + model + params`; identical inputs collide to the same key and are never recomputed
   (`X-Cache: HIT/MISS` header). `TileCache` uses **Redis** when reachable (hot tiles shared across workers /
   fronted by a CDN) and a process-local **LRU `OrderedDict`** otherwise — same interface, so the minimal
   stack transparently uses memory.
3. **FAISS nearest clear-reference retrieval (sublinear ≈ O(1)).** `retrieve_clear_reference(query, index, k)`
   ANN-searches an index of historical clear-patch embeddings to condition the fill; returns `None` (model
   runs unconditioned) when `faiss` is absent or no index is supplied.
4. **Accelerated inference.** ONNX Runtime / TensorRT (`InferConfig.backend ∈ {torch, onnx, tensorrt}`),
   `torch.compile`, batched tiling, and diffusion step-distillation keep the device saturated. Tiled
   inference itself (`cloudremoval.inference.tiled.tiled_inference`) is pure torch/numpy with Hann/Gaussian
   seam blending and runs on the smoke path.

### Scaling out (Docker / compose / extras)

```bash
# Optional extras (installed on demand, imported lazily)
pip install -e ".[serve]"    # rio-tiler, redis, pillow  -> dynamic tiling + shared cache
pip install -e ".[accel]"    # faiss-cpu, onnxruntime     -> ANN retrieval + accelerated inference
pip install -e ".[geo]"      # rasterio, rio-cogeo, ...    -> COG read/write, masking
pip install -e ".[all]"      # everything

# Containerized: api + redis (see docker-compose.yml; ARCHITECTURE.md §6 covers Ray/Dask/Celery scale-out)
docker compose up
```

The whole stack runs identically **on-prem / air-gapped** (MinIO for S3-compatible COG range reads + local
TiTiler + bundled ONNX engines) — important for ISRO. Health/predict/tiles all degrade to functioning
behaviour with none of the heavy deps installed, which is what makes the CPU-smoke and CI paths possible.
