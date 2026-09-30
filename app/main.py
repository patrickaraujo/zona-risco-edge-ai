"""API HTTP do sistema de monitoramento de zonas de risco.

Endpoints principais:
    POST /detect        → JSON com pessoas, nível de risco, alertas e métricas
    POST /detect/image  → PNG anotado com zonas, caixas e rótulos
Auxiliares:
    GET  /health  ·  GET/PUT /zones  ·  GET /events
Documentação interativa: /docs (Swagger UI)
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response

from app.alerts import AlertDispatcher
from app.annotate import draw
from app.config import settings
from app.detector import ModelLoadError, PersonDetector
from app.schemas import AlertEvent, DetectResponse, HealthResponse, ZonesPayload
from app.zones import PersonAssessment, RiskLevel, ZoneConfigError, ZoneManager, assess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("zona-risco")


class AppState:
    detector: PersonDetector | None = None
    zones: ZoneManager | None = None
    alerts: AlertDispatcher | None = None
    # Serializa a inferência: na Pi 5 (4 núcleos) rodar duas inferências
    # simultâneas só faz as duas ficarem mais lentas.
    infer_lock = threading.Lock()


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.zones = ZoneManager(settings.zones_path)
    state.alerts = AlertDispatcher(
        max_events=settings.max_events, jsonl_path=settings.events_log_path
    )
    try:
        state.detector = PersonDetector(
            settings.model_path,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            num_threads=settings.num_threads,
        )
    except ModelLoadError as exc:
        # A API sobe em modo degradado para que /health explique o problema
        logger.error("%s", exc)
    yield


app = FastAPI(
    title="Monitoramento de Zonas de Risco — Edge AI",
    description=(
        "Detecta pessoas (YOLO11n/ONNX) e classifica o risco conforme a posição "
        "em zonas amarelas/vermelhas ao redor de máquinas pesadas. "
        "Projetado para Raspberry Pi 5 (ARM64, somente CPU)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------- #
# Núcleo compartilhado pelos dois endpoints de inferência
# ---------------------------------------------------------------------- #
@dataclass
class Analysis:
    image: np.ndarray
    zones: list
    assessments: list[PersonAssessment]
    overall: RiskLevel
    alerts: list[dict]
    metadata: dict


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"Imagem maior que {settings.max_upload_bytes} bytes.")
    if not data:
        raise HTTPException(400, "Arquivo vazio.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "Formato não suportado. Envie JPEG, PNG ou BMP.")
    return image


def _analyze(image: np.ndarray, source: str) -> Analysis:
    if state.detector is None:
        raise HTTPException(503, "Modelo não carregado. Verifique GET /health.")

    t0 = time.perf_counter()
    with state.infer_lock:
        result = state.detector.detect(image)
    h, w = image.shape[:2]
    zones = state.zones.zones  # snapshot: a mesma configuração do início ao fim
    assessments = assess(result.detections, zones, w, h)
    overall = max((a.risk for a in assessments), default=RiskLevel.SEGURO)
    alerts = state.alerts.process(assessments, zones, source=source)
    total_ms = (time.perf_counter() - t0) * 1000

    metadata = {
        "model": state.detector.model_path.name,
        "input_size": (state.detector.input_w, state.detector.input_h),
        "image_size": (w, h),
        "preprocess_ms": round(result.preprocess_ms, 2),
        "inference_ms": round(result.inference_ms, 2),
        "postprocess_ms": round(result.postprocess_ms, 2),
        "total_ms": round(total_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return Analysis(image, zones, assessments, overall, alerts, metadata)


# ---------------------------------------------------------------------- #
# Endpoints
# ---------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["sistema"])
def health() -> HealthResponse:
    loaded = state.detector is not None
    return HealthResponse(
        status="ok" if loaded else "degradado",
        model_loaded=loaded,
        model=state.detector.model_path.name if loaded else None,
        zones=len(state.zones.zones) if state.zones else 0,
        architecture=platform.machine(),
    )


@app.post("/detect", response_model=DetectResponse, tags=["inferência"])
async def detect(
    file: UploadFile = File(..., description="Imagem (JPEG/PNG) de um frame da câmera"),
    source: str = Query("api", description="Identificador da câmera/origem"),
) -> DetectResponse:
    """Retorna as pessoas detectadas, o risco de cada uma e o risco geral da cena."""
    analysis = _analyze(await _read_image(file), source)
    people = [
        {
            "class_name": a.detection.class_name,
            "confidence": round(a.detection.confidence, 4),
            "bbox": dict(zip(("x1", "y1", "x2", "y2"), (round(v, 1) for v in a.detection.bbox))),
            "anchor_point": tuple(round(v, 1) for v in a.anchor),
            "risk": a.risk.label,
            "zones": a.zones,
        }
        for a in analysis.assessments
    ]
    return DetectResponse(
        overall_risk=analysis.overall.label,
        people_count=len(people),
        people=people,
        alerts=[AlertEvent(**e) for e in analysis.alerts],
        metadata=analysis.metadata,
    )


@app.post(
    "/detect/image",
    tags=["inferência"],
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Imagem anotada"}},
)
async def detect_image(
    file: UploadFile = File(..., description="Imagem (JPEG/PNG) de um frame da câmera"),
    source: str = Query("api"),
) -> Response:
    """Retorna a imagem com zonas, caixas, rótulos de risco e faixa de status."""
    analysis = _analyze(await _read_image(file), source)
    annotated = draw(analysis.image, analysis.zones, analysis.assessments, analysis.overall)
    ok, png = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(500, "Falha ao codificar a imagem anotada.")
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={
            "X-Overall-Risk": analysis.overall.label,
            "X-People-Count": str(len(analysis.assessments)),
            "X-Inference-Ms": str(analysis.metadata["inference_ms"]),
        },
    )


@app.get("/zones", tags=["zonas"])
def get_zones() -> dict:
    return {"zones": [z.to_dict() for z in state.zones.zones]}


@app.put("/zones", tags=["zonas"])
def put_zones(payload: ZonesPayload) -> dict:
    """Redefine as zonas em tempo de execução (ex.: reposicionamento da câmera)."""
    try:
        zones = state.zones.update(
            [z.model_dump() for z in payload.zones], persist=payload.persist
        )
    except ZoneConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"zones": [z.to_dict() for z in zones]}


@app.get("/events", response_model=list[AlertEvent], tags=["alertas"])
def events(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    """Histórico recente de alertas (mais recentes primeiro)."""
    return state.alerts.recent(limit)
