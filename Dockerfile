# syntax=docker/dockerfile:1.7
# =============================================================================
# Monitoramento de Zonas de Risco — Edge AI (Raspberry Pi 5 / linux/arm64)
#
# Build multi-arquitetura:
#   docker buildx build --platform linux/amd64,linux/arm64 -t zona-risco:1.0 .
# =============================================================================
ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
# Estágio 1 — exportação do modelo
# Roda na arquitetura NATIVA de quem faz o build ($BUILDPLATFORM), e não sob
# emulação QEMU. Como o .onnx é independente de arquitetura, exportamos uma
# única vez e copiamos o arquivo para as imagens amd64 e arm64.
# O PyTorch (~200 MB, versão CPU) fica só neste estágio e NÃO vai para a imagem final.
# -----------------------------------------------------------------------------
FROM --platform=$BUILDPLATFORM python:${PYTHON_VERSION}-slim AS exporter

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir ultralytics==8.4.166 onnx==1.23.1 onnxslim==0.1.97 onnxruntime==1.24.4

WORKDIR /export
COPY scripts/export_model.py .
# 640 = padrão | 320 = opção rápida para a Pi | INT8 = pesos quantizados
RUN python export_model.py --imgsz 640 320 --int8 --out /export/models

# -----------------------------------------------------------------------------
# Estágio 2 — imagem de execução (construída para cada plataforma alvo)
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/yolo11n.onnx \
    ZONES_PATH=/app/config/zones.json \
    NUM_THREADS=4

WORKDIR /app

# Dependências primeiro: camada reaproveitada em cache quando só o código muda
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=exporter /export/models ./models
COPY app ./app
COPY config ./config
COPY scripts/benchmark.py ./scripts/benchmark.py
COPY samples ./samples

# Usuário sem privilégios (boa prática em dispositivos de campo)
RUN useradd --create-home --uid 1000 edge \
 && mkdir -p /app/data \
 && chown -R edge:edge /app
USER edge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# 1 worker: a inferência já usa os 4 núcleos via threads do ONNX Runtime
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
