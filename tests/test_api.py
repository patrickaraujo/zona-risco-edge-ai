"""Testes dos endpoints com um detector falso (rápidos e sem modelo)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.alerts import AlertDispatcher
from app.detector import Detection, DetectionResult


class FakeDetector:
    model_path = type("P", (), {"name": "fake.onnx"})()
    input_w = input_h = 640

    def detect(self, image):
        h, w = image.shape[:2]
        return DetectionResult(
            detections=[
                Detection(0, "person", 0.91, (0.85 * w, 0.3 * h, 0.95 * w, 0.95 * h)),  # vermelha
                Detection(0, "person", 0.80, (0.30 * w, 0.3 * h, 0.40 * w, 0.90 * h)),  # amarela
                Detection(0, "person", 0.75, (0.02 * w, 0.2 * h, 0.12 * w, 0.60 * h)),  # seguro
            ],
            preprocess_ms=1.0, inference_ms=10.0, postprocess_ms=1.0,
        )


def _jpeg(w=800, h=600) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 127, np.uint8))
    return buf.tobytes()


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        main.state.detector = FakeDetector()
        main.state.alerts = AlertDispatcher(cooldown_s=0)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["zones"] >= 1


def test_detect_json(client):
    r = client.post("/detect", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["overall_risk"] == "perigo"
    assert body["people_count"] == 3
    assert sorted(p["risk"] for p in body["people"]) == ["atencao", "perigo", "seguro"]
    assert {a["risk"] for a in body["alerts"]} == {"atencao", "perigo"}
    assert body["metadata"]["image_size"] == [800, 600]


def test_detect_image_png(client):
    r = client.post("/detect/image", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-overall-risk"] == "perigo"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (600, 800, 3)


def test_invalid_file_returns_415(client):
    r = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_model_missing_returns_503(client):
    main.state.detector = None
    r = client.post("/detect", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 503


def test_put_zones_changes_result(client):
    only_yellow = {"zones": [{"name": "tudo", "level": "amarela",
                              "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}
    original = client.get("/zones").json()
    try:
        assert client.put("/zones", json=only_yellow).status_code == 200
        body = client.post("/detect", files={"file": ("f.jpg", _jpeg(), "image/jpeg")}).json()
        assert body["overall_risk"] == "atencao"
    finally:
        client.put("/zones", json=original)


def test_events_endpoint(client):
    client.post("/detect", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    events = client.get("/events").json()
    assert events and events[0]["zone"]
