"""Pipeline de detecção de pessoas com YOLO (ONNX Runtime, somente CPU).

Etapas:
    1. Pré-processamento  → letterbox, BGR→RGB, normalização, NCHW
    2. Inferência         → ONNX Runtime (CPUExecutionProvider)
    3. Pós-processamento  → filtro de confiança, NMS, reversão do letterbox

Por que ONNX Runtime e não PyTorch/Ultralytics em produção?
    - A imagem Docker fica ~10x menor (sem torch), o que importa no ARM64.
    - O mesmo arquivo .onnx roda em amd64 e arm64 sem reexportar.
    - Permite trocar o "execution provider" (ex.: OpenVINO, XNNPACK)
      sem mudar o código do pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Classes do COCO (80). O índice 0 ("person") é o foco deste projeto.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
PERSON_CLASS_ID = 0


class ModelLoadError(RuntimeError):
    """Falha ao carregar o modelo ONNX."""


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Caixa em pixels da imagem ORIGINAL: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]


@dataclass
class LetterboxInfo:
    """Guarda o necessário para desfazer o letterbox no pós-processamento."""

    scale: float
    pad_x: float
    pad_y: float


@dataclass
class DetectionResult:
    detections: list[Detection]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class PersonDetector:
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
        target_classes: tuple[int, ...] = (PERSON_CLASS_ID,),
    ) -> None:
        if not Path(model_path).is_file():
            raise ModelLoadError(
                f"Modelo não encontrado em '{model_path}'. "
                "Rode 'python scripts/export_model.py' ou faça o build da imagem Docker."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # erros do ORT não têm hierarquia estável
            raise ModelLoadError(f"Não foi possível carregar '{model_path}': {exc}") from exc

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # Formato esperado: [1, 3, H, W]. Se o eixo for dinâmico, usa 640.
        h, w = model_input.shape[2], model_input.shape[3]
        self.input_h = h if isinstance(h, int) else 640
        self.input_w = w if isinstance(w, int) else 640

        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        logger.info(
            "Modelo carregado: %s (entrada %dx%d, %d threads)",
            self.model_path.name, self.input_w, self.input_h, num_threads,
        )

    # ------------------------------------------------------------------ #
    # 1. Pré-processamento
    # ------------------------------------------------------------------ #
    def preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        """Redimensiona mantendo a proporção (letterbox) e monta o tensor NCHW."""
        orig_h, orig_w = image_bgr.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w, new_h = round(orig_w * scale), round(orig_h * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_w - new_w) / 2
        pad_y = (self.input_h - new_h) / 2
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        # Cinza 114 é o valor de preenchimento usado no treino do YOLO
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = tensor.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor), LetterboxInfo(scale, left, top)

    # ------------------------------------------------------------------ #
    # 2. Inferência
    # ------------------------------------------------------------------ #
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]

    # ------------------------------------------------------------------ #
    # 3. Pós-processamento
    # ------------------------------------------------------------------ #
    def postprocess(
        self, raw: np.ndarray, info: LetterboxInfo, orig_shape: tuple[int, int]
    ) -> list[Detection]:
        """Converte a saída [1, 4+nc, N] do YOLO em detecções na imagem original."""
        preds = raw[0].T  # → [N, 4 + num_classes]
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.conf_threshold
        if self.target_classes:
            mask &= np.isin(class_ids, self.target_classes)
        if not mask.any():
            return []

        boxes_cxcywh, class_ids, confidences = boxes_cxcywh[mask], class_ids[mask], confidences[mask]

        # (cx, cy, w, h) → (x, y, w, h) para o NMS do OpenCV
        xywh = boxes_cxcywh.copy()
        xywh[:, 0] -= xywh[:, 2] / 2
        xywh[:, 1] -= xywh[:, 3] / 2

        keep = cv2.dnn.NMSBoxesBatched(
            xywh.tolist(), confidences.tolist(), class_ids.tolist(),
            self.conf_threshold, self.iou_threshold,
        )
        if len(keep) == 0:
            return []

        orig_h, orig_w = orig_shape
        detections: list[Detection] = []
        for i in np.array(keep).flatten():
            x, y, w, h = xywh[i]
            # Desfaz o letterbox: remove o padding e volta para a escala original
            x1 = np.clip((x - info.pad_x) / info.scale, 0, orig_w)
            y1 = np.clip((y - info.pad_y) / info.scale, 0, orig_h)
            x2 = np.clip((x + w - info.pad_x) / info.scale, 0, orig_w)
            y2 = np.clip((y + h - info.pad_y) / info.scale, 0, orig_h)
            cid = int(class_ids[i])
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid),
                    confidence=float(confidences[i]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------ #
    # Pipeline completo
    # ------------------------------------------------------------------ #
    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        tensor, info = self.preprocess(image_bgr)
        t1 = time.perf_counter()
        raw = self.infer(tensor)
        t2 = time.perf_counter()
        detections = self.postprocess(raw, info, image_bgr.shape[:2])
        t3 = time.perf_counter()
        return DetectionResult(
            detections=detections,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
            postprocess_ms=(t3 - t2) * 1000,
        )
