#!/usr/bin/env bash
# Build multi-arquitetura (linux/amd64 + linux/arm64) com Docker Buildx.
#
# Uso:
#   ./scripts/build_multiarch.sh                         # build + validação local
#   PUSH=1 IMAGE=usuario/zona-risco ./scripts/build_multiarch.sh   # publica no registry
set -euo pipefail

IMAGE="${IMAGE:-zona-risco-edge-ai}"
TAG="${TAG:-1.0.0}"
PLATFORMS="linux/amd64,linux/arm64"
BUILDER="edge-builder"

# 1. Emuladores QEMU para executar/compilar camadas de outra arquitetura
docker run --privileged --rm tonistiigi/binfmt --install arm64,amd64 >/dev/null

# 2. Builder baseado em container (o driver padrão não gera multi-plataforma)
docker buildx inspect "$BUILDER" >/dev/null 2>&1 || docker buildx create --name "$BUILDER" --driver docker-container
docker buildx use "$BUILDER"
docker buildx inspect --bootstrap | grep -i platforms

if [[ "${PUSH:-0}" == "1" ]]; then
  # 3a. Publica UM manifest list com as duas arquiteturas
  docker buildx build --platform "$PLATFORMS" -t "$IMAGE:$TAG" --push .
  docker buildx imagetools inspect "$IMAGE:$TAG"
else
  # 3b. Sem registry: gera as duas e exporta como arquivo OCI (prova de build)
  mkdir -p dist
  docker buildx build --platform "$PLATFORMS" -t "$IMAGE:$TAG" \
    --output "type=oci,dest=dist/${IMAGE##*/}-${TAG}-multiarch.tar" .
  echo "Arquivo OCI multi-arquitetura: dist/${IMAGE##*/}-${TAG}-multiarch.tar"

  # 4. Carrega a variante arm64 e executa sob QEMU para provar que ela funciona
  docker buildx build --platform linux/arm64 -t "$IMAGE:$TAG-arm64" --load .
  echo "Arquitetura da imagem arm64: $(docker image inspect "$IMAGE:$TAG-arm64" --format '{{.Architecture}}')"
  docker run --rm --platform linux/arm64 "$IMAGE:$TAG-arm64" \
    python -c "import platform, onnxruntime as ort; print('uname -m =', platform.machine(), '| ORT', ort.__version__, ort.get_available_providers())"
fi
