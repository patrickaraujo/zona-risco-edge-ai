"""Configuração da aplicação via variáveis de ambiente.

Tudo o que muda entre o notebook de desenvolvimento e a Raspberry Pi 5
(modelo, resolução de entrada, número de threads) é parametrizável aqui,
sem necessidade de alterar código.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Modelo ONNX (exportado a partir do YOLO11n pré-treinado no COCO)
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", "models/yolo11n.onnx"))
    )
    # Limiar de confiança e de IoU para o NMS
    conf_threshold: float = field(default_factory=lambda: _env_float("CONF_THRESHOLD", 0.35))
    iou_threshold: float = field(default_factory=lambda: _env_float("IOU_THRESHOLD", 0.45))
    # Threads de CPU para o ONNX Runtime (a Pi 5 tem 4 núcleos Cortex-A76)
    num_threads: int = field(default_factory=lambda: _env_int("NUM_THREADS", 4))
    # Arquivo de zonas (polígonos normalizados 0–1)
    zones_path: Path = field(
        default_factory=lambda: Path(os.getenv("ZONES_PATH", "config/zones.json"))
    )
    # Tamanho máximo de upload (bytes) — proteção contra imagens gigantes
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    )
    # Quantidade de eventos de alerta mantidos em memória
    max_events: int = field(default_factory=lambda: _env_int("MAX_EVENTS", 200))
    # Arquivo JSONL para registro persistente de eventos (vazio = desativado)
    events_log_path: str = field(default_factory=lambda: os.getenv("EVENTS_LOG_PATH", ""))


settings = Settings()
