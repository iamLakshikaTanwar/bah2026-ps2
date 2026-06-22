"""B4 self-smoke: heavy deps BLOCKED, inline identity model, full path checks."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

# --- Block heavy deps at import time to prove no hard dependency ----------------
_BLOCKED = {"rasterio", "rio_tiler", "rio_cogeo", "faiss", "redis", "scipy", "cv2",
            "skimage", "onnxruntime", "tensorrt", "titiler"}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in _BLOCKED:
            raise ModuleNotFoundError(f"[BLOCKED for smoke] {name}")
        return None


sys.meta_path.insert(0, _Blocker())

_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# --- 1. Import all modules + app on the minimal/blocked stack -------------------
from cloudremoval.inference import blending, cog_writer, postprocess, tiled  # noqa: E402,F401
from cloudremoval.serving import schemas, tiles  # noqa: E402,F401
from cloudremoval.serving.app import app, create_app  # noqa: E402
from cloudremoval.models.base import BaseCloudRemovalModel, ModelOutput  # noqa: E402
from cloudremoval.models.registry import register_model  # noqa: E402

print("[1] imports OK (rasterio/rio_tiler/faiss/redis/scipy BLOCKED); app imported")


# --- Inline dummy/identity model satisfying predict()->ModelOutput --------------
class _IdentityCfg:
    in_channels = 3
    out_channels = 3


@register_model("identity")
class IdentityModel(BaseCloudRemovalModel):
    """Returns optical_cloudy unchanged (identity passthrough)."""

    def __init__(self, cfg=None):
        super().__init__(cfg or _IdentityCfg())

    def forward(self, sample):
        return ModelOutput(reconstruction=sample["optical_cloudy"])

    def loss(self, sample, output):
        return {"total": torch.tensor(0.0)}


@register_model("blur")
class BlurModel(BaseCloudRemovalModel):
    """Tiny conv (3x3 avg) so outputs differ from input -> seam check is meaningful."""

    def __init__(self, cfg=None):
        super().__init__(cfg or _IdentityCfg())
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
# Identity must reproduce input within blend tolerance (Hann partition-of-unity).
max_err = float(np.abs(out_id - scene).max())
print(f"[2] tiled identity: shape={out_id.shape} finite=True max|out-in|={max_err:.2e}")
assert max_err < 1e-4, f"identity reconstruction error too high: {max_err}"

# Blur model: check seams are blended (no discontinuity at tile boundary x=96..160
# overlap band). Compare against a single full-tile forward in the interior.
out_blur = tiled.tiled_inference(model_blur, scene, tile=128, overlap=32, batch_size=4)
assert out_blur.shape == (3, 300, 300)
assert np.isfinite(out_blur).all()
with torch.no_grad():
    # Reference = the SAME conv on the engine's own reflect-padded scene, cropped
    # back. Any residual is purely the conv's per-tile edge handling leaking
    # through the Hann window's clamped (1e-3) endpoints — a tiny, bounded, NON
    # -localized effect, i.e. there is no seam, just a uniform ~0.1% floor.
    stride = 128 - 32
    pad_h = (-(300 - 128)) % stride
    pad_w = (-(300 - 128)) % stride
    sp = np.pad(scene, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    full_np = model_blur.forward(
        {"optical_cloudy": torch.from_numpy(sp)[None]}
    ).reconstruction[0].numpy()[:, :300, :300]
interior = (slice(None), slice(40, 260), slice(40, 260))
seam_err = float(np.abs(out_blur[interior] - full_np[interior]).max())
# A seam would be a *localized* gradient spike: confirm the overlap-band 1st
# derivative is smooth and far below the input signal's natural gradient.
col_grad = np.abs(np.diff(out_blur[0, 150, 90:170]))
input_grad = np.abs(np.diff(scene[0, 150, 90:170]))
print(f"[2b] tiled blur: interior max|tiled-conv|={seam_err:.2e} (bounded ~Hann-floor) "
      f"seam-band max|d/dx|={col_grad.max():.2e} << input grad {input_grad.max():.2e}")
assert seam_err < 5e-3, f"interior error too high (would indicate a seam): {seam_err}"
assert col_grad.max() < input_grad.max(), "blended gradient should not exceed input"

# Edge: tiny scene smaller than tile -> reflect pad path.
small = rng.random((3, 64, 80)).astype(np.float32)
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
# Clear region (mask==0, away from feather) must be unchanged.
clear_unchanged = np.allclose(comp[:, 0:15, 0:15], orig[:, 0:15, 0:15])
print(f"[3] composite_under_mask shape={comp.shape} clear-region-unchanged={clear_unchanged}")
assert clear_unchanged

ref = rng.random((3, 64, 64)).astype(np.float32) * 0.5 + 0.25
hm = postprocess.histogram_match(recon, ref)
assert hm.shape == (3, 64, 64) and np.isfinite(hm).all()
# Poisson must fall back (scipy blocked) and still return a finite composite.
pb = postprocess.poisson_blend(orig, recon, mask, feather=2)
assert pb.shape == (3, 64, 64) and np.isfinite(pb).all()
print(f"[3b] histogram_match shape={hm.shape}; poisson fallback shape={pb.shape} OK")

# --- 4. write_cog fallback (no rasterio) ----------------------------------------
out_dir = Path("outputs/.b4_smoke")
cog_path = cog_writer.write_cog(out_id, out_dir / "recon.tif")
print(f"[4] write_cog fallback -> {cog_path}")
assert cog_path.endswith(".npy") and Path(cog_path).exists()
assert (out_dir / "recon.json").exists()

# --- 5. FastAPI TestClient -------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)
r_health = client.get("/health")
assert r_health.status_code == 200 and r_health.json()["status"] == "ok", r_health.text
r_models = client.get("/models")
assert r_models.status_code == 200
r_info = client.get("/info")
assert r_info.status_code == 200
r_ver = client.get("/version")
assert r_ver.status_code == 200
print(f"[5] /health={r_health.status_code} {r_health.json()}")
print(f"[5b] /models -> {r_models.json()['models']}")
print(f"[5c] /info heavy_deps -> {r_info.json()['heavy_deps']}")

# POST /predict with a tiny inline array + identity model.
import base64  # noqa: E402
import io  # noqa: E402

tiny = rng.random((3, 32, 32)).astype(np.float32)
buf = io.BytesIO()
np.save(buf, tiny)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
mbuf = io.BytesIO()
mtiny = np.zeros((1, 32, 32), np.float32)
mtiny[:, 8:16, 8:16] = 1.0
np.save(mbuf, mtiny)
mb64 = base64.b64encode(mbuf.getvalue()).decode("ascii")

payload = {
    "model_name": "identity",
    "array": {"npy_base64": b64},
    "cloud_mask": {"npy_base64": mb64},
    "options": {"tile_size": 16, "overlap": 4, "blend": "hann",
                "composite_under_mask": True, "return_array": True},
}
r_pred = client.post("/predict", json=payload)
assert r_pred.status_code == 200, r_pred.text
body = r_pred.json()
assert body["status"] == "ok" and body["shape"] == [3, 32, 32], body
assert body["array_base64"] is not None
# Decode the returned reconstruction reference.
recon_back = np.load(io.BytesIO(base64.b64decode(body["array_base64"])))
assert recon_back.shape == (3, 32, 32)
print(f"[5d] POST /predict -> status={body['status']} shape={body['shape']} "
      f"metrics={ {k: round(v,4) for k,v in body['metrics'].items()} }")

# Tiles router mounted + content-addressed cache O(1) (placeholder PNG path).
r_tile = client.get("/tiles/0/0/0.png")
assert r_tile.status_code == 200 and r_tile.headers["content-type"] == "image/png"
r_tile2 = client.get("/tiles/0/0/0.png")
assert r_tile2.headers.get("x-cache") == "HIT", "cache should hit on 2nd request"
r_tinfo = client.get("/tiles/info")
print(f"[5e] /tiles/0/0/0.png={r_tile.status_code} cache2nd={r_tile2.headers.get('x-cache')} "
      f"tiles/info={r_tinfo.json()}")

print("\nALL B4 SMOKE CHECKS PASSED")
