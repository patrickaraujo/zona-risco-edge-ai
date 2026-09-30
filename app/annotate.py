"""Desenho das zonas, caixas e rótulos sobre a imagem (saída PNG)."""

from __future__ import annotations

import cv2
import numpy as np

from app.zones import PersonAssessment, RiskLevel, Zone

# Cores em BGR
COLORS = {
    RiskLevel.SEGURO: (80, 175, 76),     # verde
    RiskLevel.ATENCAO: (0, 200, 255),    # amarelo/âmbar
    RiskLevel.PERIGO: (40, 40, 230),     # vermelho
}
BANNER_TEXT = {
    RiskLevel.SEGURO: "SEGURO",
    RiskLevel.ATENCAO: "ATENCAO - pessoa em zona amarela",
    RiskLevel.PERIGO: "PERIGO - pessoa em zona vermelha",
}


def _scale(image: np.ndarray) -> float:
    """Espessuras e fontes proporcionais à resolução."""
    return max(image.shape[0], image.shape[1]) / 1000


def draw(
    image_bgr: np.ndarray,
    zones: list[Zone],
    assessments: list[PersonAssessment],
    overall: RiskLevel,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    s = _scale(out)
    thick = max(1, round(2 * s))
    font_scale = 0.6 * s

    # 1. Zonas semitransparentes (vermelhas por cima das amarelas)
    overlay = out.copy()
    for zone in sorted(zones, key=lambda z: z.level):
        pts = zone.to_pixels(w, h).astype(np.int32)
        cv2.fillPoly(overlay, [pts], COLORS[zone.level])
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    for zone in zones:
        pts = zone.to_pixels(w, h).astype(np.int32)
        cv2.polylines(out, [pts], True, COLORS[zone.level], thick)
        x, y = pts[0][0]
        cv2.putText(out, zone.name, (int(x) + 4, int(y) + round(20 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLORS[zone.level], thick)

    # 2. Pessoas: caixa, rótulo e ponto de apoio (pés)
    for a in assessments:
        color = COLORS[a.risk]
        x1, y1, x2, y2 = (int(v) for v in a.detection.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"pessoa {a.detection.confidence:.2f} | {a.risk.label}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(y1, th + base + 4)
        lx = max(0, min(x1, w - tw - 6))  # mantém o rótulo dentro da imagem
        cv2.rectangle(out, (lx, ty - th - base - 4), (lx + tw + 6, ty), color, -1)
        cv2.putText(out, label, (lx + 3, ty - base - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)
        ax, ay = (int(v) for v in a.anchor)
        cv2.circle(out, (ax, ay), max(3, round(6 * s)), color, -1)
        cv2.circle(out, (ax, ay), max(3, round(6 * s)), (255, 255, 255), max(1, thick // 2))

    # 3. Faixa de status geral no topo
    banner_h = round(36 * s) + 8
    cv2.rectangle(out, (0, 0), (w, banner_h), COLORS[overall], -1)
    cv2.putText(out, BANNER_TEXT[overall], (10, banner_h - round(10 * s) - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), thick)
    return out
