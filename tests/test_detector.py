"""Teste de integração com o modelo real (pulado se o modelo não existir)."""

import os
from pathlib import Path

import cv2
import pytest

from app.detector import PersonDetector

MODEL = Path(os.getenv("MODEL_PATH", "models/yolo11n.onnx"))
pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason=f"modelo {MODEL} ausente")


def test_detects_people_in_bus_sample():
    det = PersonDetector(MODEL, num_threads=2)
    image = cv2.imread("samples/bus.jpg")
    result = det.detect(image)
    # A imagem de referência da Ultralytics tem 4 pessoas (uma parcialmente cortada)
    assert len(result.detections) >= 3
    h, w = image.shape[:2]
    for d in result.detections:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert d.class_name == "person"
