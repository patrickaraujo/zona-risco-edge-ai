"""Registro de eventos e acionamento de respostas de segurança.

Nesta versão os "atuadores" são: log estruturado, histórico em memória
(endpoint /events) e arquivo JSONL opcional. A interface `AlertHandler`
permite plugar, sem alterar o pipeline, as respostas reais da Raspberry Pi 5:
    - GPIO → sinaleiro/sirene/relé de parada (ex.: biblioteca gpiozero)
    - MQTT → integração com o supervisório da planta (ex.: paho-mqtt)

Um "cooldown" por zona evita disparar dezenas de alertas por segundo quando
a câmera processa um fluxo contínuo de frames com a mesma pessoa na zona.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Protocol

from app.zones import PersonAssessment, RiskLevel, Zone

logger = logging.getLogger(__name__)

# Ação esperada no chão de fábrica para cada nível
ACTIONS = {
    RiskLevel.ATENCAO: "alerta_visual_sonoro",
    RiskLevel.PERIGO: "parada_de_emergencia",
}


class AlertHandler(Protocol):
    def __call__(self, event: dict) -> None: ...


class AlertDispatcher:
    def __init__(
        self,
        max_events: int = 200,
        cooldown_s: float = 3.0,
        jsonl_path: str = "",
    ) -> None:
        self._events: deque[dict] = deque(maxlen=max_events)
        self._last_fired: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()
        self._handlers: list[AlertHandler] = []
        self.cooldown_s = cooldown_s
        self.jsonl_path = jsonl_path

    def register(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    def process(
        self, assessments: list[PersonAssessment], zones: list[Zone], source: str = "api"
    ) -> list[dict]:
        """Gera um evento por zona ocupada (com o nível DA ZONA), respeitando o cooldown."""
        zone_levels = {z.name: z.level for z in zones}
        affected: dict[str, tuple[RiskLevel, int]] = {}
        for a in assessments:
            for zone in a.zones:
                level, count = affected.get(zone, (zone_levels[zone], 0))
                affected[zone] = (level, count + 1)

        fired: list[dict] = []
        now = time.monotonic()
        with self._lock:
            for zone, (level, count) in affected.items():
                key = (zone, int(level))
                if now - self._last_fired.get(key, -1e9) < self.cooldown_s:
                    continue
                self._last_fired[key] = now
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "zone": zone,
                    "risk": level.label,
                    "action": ACTIONS[level],
                    "people_in_zone": count,
                    "source": source,
                }
                self._events.append(event)
                fired.append(event)

        for event in fired:
            self._emit(event)
        return fired

    def _emit(self, event: dict) -> None:
        log = logger.error if event["risk"] == RiskLevel.PERIGO.label else logger.warning
        log("ALERTA %s na zona '%s' → %s", event["risk"].upper(), event["zone"], event["action"])
        if self.jsonl_path:
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("Falha ao gravar evento em %s: %s", self.jsonl_path, exc)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:  # um atuador com defeito não pode derrubar a API
                logger.exception("Falha no handler de alerta %r", handler)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._events)[-limit:][::-1]
