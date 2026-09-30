"""Zonas de risco (polígonos) e classificação de criticidade.

Regra central: uma pessoa está "dentro" de uma zona quando o seu PONTO DE
APOIO — o centro da base da caixa delimitadora, ou seja, aproximadamente os
pés — cai dentro do polígono. Com a câmera elevada (3–5 m), o polígono é
desenhado sobre o piso, então o ponto dos pés é a melhor aproximação da
posição real da pessoa no chão. Usar o centro da caixa geraria falsos
positivos quando o tronco "invade" visualmente a zona sem que a pessoa pise nela.

Os polígonos são armazenados em coordenadas NORMALIZADAS (0 a 1), de modo que
a mesma configuração vale para qualquer resolução de câmera.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from app.detector import Detection

logger = logging.getLogger(__name__)


class RiskLevel(IntEnum):
    """Níveis ordenados: comparar com max() devolve o mais crítico."""

    SEGURO = 0
    ATENCAO = 1   # zona amarela
    PERIGO = 2    # zona vermelha

    @property
    def label(self) -> str:
        return self.name.lower()


ZONE_LEVELS = {"amarela": RiskLevel.ATENCAO, "vermelha": RiskLevel.PERIGO}


class ZoneConfigError(ValueError):
    """Configuração de zonas inválida."""


@dataclass
class Zone:
    name: str
    level: RiskLevel
    polygon: list[tuple[float, float]]  # pontos normalizados (x, y) em [0, 1]

    def to_pixels(self, width: int, height: int) -> np.ndarray:
        pts = np.array([(x * width, y * height) for x, y in self.polygon], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def contains(self, point: tuple[float, float], width: int, height: int) -> bool:
        return cv2.pointPolygonTest(self.to_pixels(width, height), point, False) >= 0

    def to_dict(self) -> dict:
        color = "vermelha" if self.level == RiskLevel.PERIGO else "amarela"
        return {"name": self.name, "level": color, "polygon": [list(p) for p in self.polygon]}


@dataclass
class PersonAssessment:
    detection: Detection
    anchor: tuple[float, float]
    risk: RiskLevel
    zones: list[str]


def anchor_point(det: Detection) -> tuple[float, float]:
    """Centro da base da caixa (ponto aproximado dos pés)."""
    x1, _, x2, y2 = det.bbox
    return ((x1 + x2) / 2, y2)


def parse_zones(raw: object) -> list[Zone]:
    """Valida e converte a configuração (lista de dicts) em objetos Zone."""
    if not isinstance(raw, list) or not raw:
        raise ZoneConfigError("A configuração deve ser uma lista não vazia de zonas.")

    zones: list[Zone] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ZoneConfigError(f"Zona #{i}: esperado um objeto.")
        name = str(item.get("name", f"zona_{i}"))
        level_name = str(item.get("level", "")).lower()
        if level_name not in ZONE_LEVELS:
            raise ZoneConfigError(f"Zona '{name}': level deve ser 'amarela' ou 'vermelha'.")
        polygon = item.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ZoneConfigError(f"Zona '{name}': o polígono precisa de ao menos 3 pontos.")
        points: list[tuple[float, float]] = []
        for p in polygon:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ZoneConfigError(f"Zona '{name}': ponto inválido {p!r}.") from exc
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ZoneConfigError(
                    f"Zona '{name}': coordenadas devem estar normalizadas entre 0 e 1."
                )
            points.append((x, y))
        zones.append(Zone(name=name, level=ZONE_LEVELS[level_name], polygon=points))
    return zones


def assess(
    detections: list[Detection], zones: list[Zone], width: int, height: int
) -> list[PersonAssessment]:
    """Classifica cada pessoa pelo nível da zona mais crítica em que está."""
    results: list[PersonAssessment] = []
    for det in detections:
        anchor = anchor_point(det)
        inside = [z for z in zones if z.contains(anchor, width, height)]
        risk = max((z.level for z in inside), default=RiskLevel.SEGURO)
        results.append(PersonAssessment(det, anchor, risk, [z.name for z in inside]))
    return results


class ZoneManager:
    """Mantém as zonas em memória (thread-safe) e persiste em JSON."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._zones = self._load()

    def _load(self) -> list[Zone]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return parse_zones(data.get("zones") if isinstance(data, dict) else data)
        except FileNotFoundError:
            raise ZoneConfigError(f"Arquivo de zonas não encontrado: {self.path}") from None
        except json.JSONDecodeError as exc:
            raise ZoneConfigError(f"JSON inválido em {self.path}: {exc}") from exc

    @property
    def zones(self) -> list[Zone]:
        with self._lock:
            return list(self._zones)

    def update(self, raw: object, persist: bool = False) -> list[Zone]:
        zones = parse_zones(raw)
        with self._lock:
            self._zones = zones
        if persist:
            try:
                self.path.write_text(
                    json.dumps({"zones": [z.to_dict() for z in zones]}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:  # ex.: volume somente leitura
                logger.warning("Zonas atualizadas em memória, mas não persistidas: %s", exc)
        return zones
