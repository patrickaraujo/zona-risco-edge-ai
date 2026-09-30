"""Modelos de resposta da API (documentados automaticamente no Swagger)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLabel = Literal["seguro", "atencao", "perigo"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PersonDetection(BaseModel):
    class_name: str = Field(examples=["person"])
    confidence: float = Field(ge=0, le=1, examples=[0.87])
    bbox: BBox
    anchor_point: tuple[float, float] = Field(
        description="Centro da base da caixa (pés), usado no teste de zona."
    )
    risk: RiskLabel
    zones: list[str] = Field(description="Zonas onde o ponto de apoio está contido.")


class AlertEvent(BaseModel):
    timestamp: str
    zone: str
    risk: RiskLabel
    action: str
    people_in_zone: int
    source: str


class Metadata(BaseModel):
    model: str
    input_size: tuple[int, int]
    image_size: tuple[int, int]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    timestamp: str


class DetectResponse(BaseModel):
    overall_risk: RiskLabel
    people_count: int
    people: list[PersonDetection]
    alerts: list[AlertEvent] = Field(
        description="Alertas disparados nesta requisição (respeitando o cooldown)."
    )
    metadata: Metadata


class ZoneModel(BaseModel):
    name: str = Field(examples=["prensa_hidraulica"])
    level: Literal["amarela", "vermelha"]
    polygon: list[tuple[float, float]] = Field(
        min_length=3, description="Pontos (x, y) normalizados entre 0 e 1."
    )


class ZonesPayload(BaseModel):
    zones: list[ZoneModel] = Field(min_length=1)
    persist: bool = Field(False, description="Gravar no arquivo de configuração.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degradado"]
    model_loaded: bool
    model: str | None
    zones: int
    architecture: str
