"""B4 self-smoke: minimal stack (heavy deps genuinely absent), inline model."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# This environment IS the minimal CPU-smoke stack: assert every heavy/optional
# dep is absent so the import-safety claim is proven against real absence (the
# most faithful test — no synthetic blocker needed).
_MUST_BE_ABSENT = ["rasterio", "rio_tiler", "rio_cogeo", "faiss", "redis",
                   "scipy", "cv2", "skimage", "onnxruntime"]
_present = [m for m in _MUST_BE_ABSENT if importlib.util.find_spec(m) is not None]
assert not _present, f"expected a minimal stack; these are installed: {_present}"

_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# --- 1. Import all modules + app on the minimal stack ---------------------------
from cloudremoval.inference import blending, cog_writer, postprocess, tiled  # noqa: E402,F401
from cloudremoval.serving import schemas, tiles  # noqa: E402,F401
from cloudremoval.serving.app import app, create_app  # noqa: E402,F401
from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput  # noqa: E402
from cloudremoval.models.registry import register_model  # noqa: E402

print("[1] imports OK on minimal stack (rasterio/rio_tiler/faiss/redis/scipy absent); "
      "app imported with NO heavy dep loaded")


# --- Inline dummy/identity model satisfying predict()->ModelOutput --------------
class _Cfg:
    in_channels = 3
    out_channels = 3


@register_model("identity")
class IdentityModel(BaseCloudRemovalModel):
    """Returns optical_cloudy unchanged (identity passthrough)."""

    def __init__(self, cfg=None):
        super().__init__(cfg or _Cfg())

    def forward(self, sample):
        return ModelOutput(reconstruction=sample["optical_cloudy"])

    def loss(self, sample, output):
        return {"total": torch.tensor(0.0)}


@register_model("blur")
class BlurModel(BaseCloudRemovalModel):
    """Tiny depthwise 3x3 average conv so outputs differ from the input."""

    def __init__(self, cfg=None):
        super().__init__(cfg or _Cfg())
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 9.0)

    def forward(self, sample):
        return ModelOutput(reconstruction=self.conv(sample["optical_cloudy"]))

    def loss(self, sample, output):
        return {"total": torch.tensor(0.0)}


model_id = IdentityModel()
model_blur = BlurModel()

# --- 2. tiled_inference on [3,300,300], tile=128 overlap=32 ---------------------
rng = np.random.default_rng(0)
scene = rng.random((3, 300, 300)).astype(np.float32)

out_id = tiled.tiled_inference(model_id, scene, tile=128, overlap=32, batch_size=4)
assert out_id.shape == (3, 300, 300), out_id.shape
assert np.isfinite(out_id).all()
max_err = float(np.abs(out_id - scene).max())  # Hann partition-of-unity -> ~0
print(f"[2] tiled identity: shape={out_id.shape} finite=True max|out-in|={max_err:.2e}")
assert max_err < 1e-4, f"identity reconstruction error too high: {max_err}"

out_blur = tiled.tiled_inference(model_blur, scene, tile=128, overlap=32, batch_size=4)
assert out_blur.shape == (3, 300, 300) and np.isfinite(out_blur).all()
with torch.no_grad():
    # Reference = SAME conv on the engine's own reflect-padded scene, cropped
    # back. Residual is only the conv's per-tile edge handling leaking through the
    # Hann window's clamped (1e-3) endpoints: tiny, bounded, NON-localized (no seam).
    stride = 128 - 32
    ph = (-(300 - 128)) % stride
    pw = (-(300 - 128)) % stride
    sp = np.pad(scene, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    full_np = model_blur.forward(
        {"optical_cloudy": torch.from_numpy(sp)[None]}
    ).reconstruction[0].numpy()[:, :300, :300]
interior = (slice(None), slice(40, 260), slice(40, 260))
seam_err = float(np.abs(out_blur[interior] - full_np[interior]).max())
col_grad = float(np.abs(np.diff(out_blur[0, 150, 90:170])).max())   # across a seam
input_grad = float(np.abs(np.diff(scene[0, 150, 90:170])).max())
print(f"[2b] tiled blur: interior max|tiled-conv|={seam_err:.2e} (~Hann-floor, no seam) "
      f"seam-band max|d/dx|={col_grad:.2e} << input grad {input_grad:.2e}")
assert seam_err < 5e-3, f"interior error would indicate a seam: {seam_err}"
assert col_grad < input_grad, "blended gradient must not exceed the input's"

small = rng.random((3, 64, 80)).astype(np.float32)   # smaller than tile -> reflect pad
out_small = tiled.tiled_inference(model_id, small, tile=128, overlap=32)
assert out_small.shape == (3, 64, 80), out_small.shape
print(f"[2c] reflect-pad edge: {small.shape} -> {out_small.shape} OK")

# --- 3. composite_under_mask + histogram_match fallback -------------------------
orig = rng.random((3, 64, 64)).astype(np.float32)
recon = rng.random((3, 64, 64)).astype(np.float32)
mask = np.zeros((1, 64, 64), np.float32)
mask[:, 20:40, 20:40] = 1.0
comp = postprocess.composite_under_mask(orig, recon, mask, feather=0)
assert comp.shape == (3, 64, 64)
clear_unchanged = bool(np.allclose(comp[:, 0:15, 0:15], orig[:, 0:15, 0:15]))
under_mask_is_recon = bool(np.allclose(comp[:, 25:35, 25:35], recon[:, 25:35, 25:35]))
print(f"[3] composite_under_mask shape={comp.shape} clear-unchanged={clear_unchanged} "
      f"under-mask==recon={under_mask_is_recon}")
assert clear_unchanged and under_mask_is_recon

# feather_mask must preserve shape (regression: box-blur off-by-one).
fm = postprocess.feather_mask(mask, radius=4)
assert fm.shape == (1, 64, 64), fm.shape

ref = rng.random((3, 64, 64)).astype(np.float32) * 0.5 + 0.25
hm = postprocess.histogram_match(recon, ref)
assert hm.shape == (3, 64, 64) and np.isfinite(hm).all()
pb = postprocess.poisson_blend(orig, recon, mask, feather=2)  # scipy absent -> feather
assert pb.shape == (3, 64, 64) and np.isfinite(pb).all()
print(f"[3b] feather_mask shape ok; histogram_match {hm.shape}; "
      f"poisson feather-fallback {pb.shape} OK")

# --- 4. write_cog fallback (no rasterio) ----------------------------------------
out_dir = Path("outputs/.b4_smoke")
cog_path = cog_writer.write_cog(out_id, out_dir / "recon.tif")
assert cog_path.endswith(".npy") and Path(cog_path).exists()
assert (out_dir / "recon.json").exists()
assert cog_writer.validate_cog(out_dir / "recon.tif") is False  # not a real COG
print(f"[4] write_cog fallback -> {cog_path} (+ recon.json); validate_cog=False")

# --- 5. FastAPI TestClient -------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)
r_health = client.get("/health")
assert r_health.status_code == 200 and r_health.json()["status"] == "ok", r_health.text
r_models = client.get("/models")
assert r_models.status_code == 200
r_info = client.get("/info")
assert r_info.status_code == 200, r_info.text
r_ver = client.get("/version")
assert r_ver.status_code == 200
print(f"[5] /health={r_health.status_code} {r_health.json()}")
print(f"[5b] /models -> {r_models.json()['models']}")
print(f"[5c] /info heavy_deps -> {r_info.json()['heavy_deps']}")
assert r_info.json()["heavy_deps"]["rasterio"] is False

# POST /predict with a tiny inline array + identity model.
import base64  # noqa: E402
import io  # noqa: E402


def _b64(a):
    b = io.BytesIO()
    np.save(b, a.astype(np.float32))
    return base64.b64encode(b.getvalue()).decode("ascii")


tiny = rng.random((3, 32, 32)).astype(np.float32)
mtiny = np.zeros((1, 32, 32), np.float32)
mtiny[:, 8:16, 8:16] = 1.0
payload = {
    "model_name": "identity",
    "array": {"npy_base64": _b64(tiny)},
    "cloud_mask": {"npy_base64": _b64(mtiny)},
    "options": {"tile_size": 16, "overlap": 4, "blend": "hann",
                "composite_under_mask": True, "return_array": True},
}
r_pred = client.post("/predict", json=payload)
assert r_pred.status_code == 200, r_pred.text
body = r_pred.json()
assert body["status"] == "ok" and body["shape"] == [3, 32, 32], body
recon_back = np.load(io.BytesIO(base64.b64decode(body["array_base64"])))
assert recon_back.shape == (3, 32, 32)
print(f"[5d] POST /predict -> status={body['status']} shape={body['shape']} "
      f"metrics={ {k: round(v, 4) for k, v in body['metrics'].items()} }")

# Unknown model -> 404 (registry guard).
r_404 = client.post("/predict", json={"model_name": "does_not_exist",
                                       "array": {"npy_base64": _b64(tiny)}})
assert r_404.status_code == 404, r_404.text

# Tiles router mounted + content-addressed cache O(1) (placeholder PNG path).
r_tile = client.get("/tiles/0/0/0.png")
assert r_tile.status_code == 200 and r_tile.headers["content-type"] == "image/png"
r_tile2 = client.get("/tiles/0/0/0.png")
assert r_tile2.headers.get("x-cache") == "HIT", "cache should hit on 2nd request"
r_tinfo = client.get("/tiles/info")
print(f"[5e] /tiles/0/0/0.png={r_tile.status_code} cache-2nd={r_tile2.headers.get('x-cache')} "
      f"/tiles/info={r_tinfo.json()}")

print("\nALL B4 SMOKE CHECKS PASSED")
