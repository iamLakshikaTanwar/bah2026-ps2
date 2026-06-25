"""Serving API tests via FastAPI ``TestClient`` (BUILD_PLAN §3.9).

Validates the contract endpoints on the minimal stack:

* ``GET /health`` -> 200 ``{"status": "ok"}``;
* ``GET /models`` lists the six registered models;
* ``GET /info`` -> 200 and reports optional-dep availability;
* ``POST /predict`` round-trips a tiny inline base64-``.npy`` array through a real
  model and returns a finite, correctly-shaped reconstruction;
* importing the app pulls **no** heavy dependency (rasterio / rio-tiler / faiss).

The TestClient uses ``httpx`` (in the minimal stack); no server is launched.
"""

from __future__ import annotations

import base64
import io
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from cloudremoval.serving.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _encode(arr: np.ndarray) -> dict:
    """Encode an array as the ``ArraySpec`` body (base64 ``.npy``)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.float32))
    return {"npy_base64": base64.b64encode(buf.getvalue()).decode("ascii")}


def test_health_ok(client: TestClient) -> None:
    """``/health`` is always 200 ``{"status": "ok"}`` on the minimal stack."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_lists_six(client: TestClient) -> None:
    """``/models`` returns the six registered models."""
    from cloudremoval.models.registry import list_models

    resp = client.get("/models")
    assert resp.status_code == 200
    names = resp.json()["models"]
    assert set(names) == set(list_models())
    assert len(names) == 6


def test_info_ok(client: TestClient) -> None:
    """``/info`` is 200 and reports optional heavy-dep availability flags."""
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "cloudremoval"
    assert "rasterio" in body["heavy_deps"]
    # On the minimal stack rasterio is absent.
    assert body["heavy_deps"]["rasterio"] is False


def test_predict_roundtrip_with_real_model(client: TestClient) -> None:
    """``/predict`` reconstructs a tiny inline array through a registered model."""
    rng = np.random.default_rng(0)
    arr = rng.random((3, 64, 64)).astype(np.float32)
    payload = {
        "model_name": "unet",
        "array": _encode(arr),
        "options": {
            "tile_size": 64,
            "overlap": 8,
            "batch_size": 2,
            "blend": "hann",
            "composite_under_mask": False,
            "return_array": True,
        },
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["shape"] == [3, 64, 64]

    # The inline reconstruction decodes to a finite array of the right shape.
    recon = np.load(io.BytesIO(base64.b64decode(body["array_base64"])))
    assert recon.shape == (3, 64, 64)
    assert np.isfinite(recon).all()


def test_predict_unknown_model_is_404(client: TestClient) -> None:
    """An unregistered model name yields a 404."""
    rng = np.random.default_rng(1)
    arr = rng.random((3, 16, 16)).astype(np.float32)
    resp = client.post(
        "/predict",
        json={"model_name": "does-not-exist", "array": _encode(arr)},
    )
    assert resp.status_code == 404


def test_predict_requires_input(client: TestClient) -> None:
    """Omitting both ``array`` and ``input_ref`` is a 400."""
    resp = client.post("/predict", json={"model_name": "unet"})
    assert resp.status_code == 400


def test_app_import_pulls_no_heavy_deps() -> None:
    """Importing the serving app must not import rasterio / rio-tiler / faiss."""
    import importlib

    # Re-import the app module fresh and assert no heavy module got pulled in.
    importlib.import_module("cloudremoval.serving.app")
    for heavy in ("rasterio", "rio_tiler", "faiss"):
        assert heavy not in sys.modules, f"{heavy} was imported by the serving app"
