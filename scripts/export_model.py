"""Exporta o YOLO11n pré-treinado (COCO) para ONNX e, opcionalmente, INT8.

Executado automaticamente no 1º estágio do Dockerfile, mas também pode ser
rodado localmente:

    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    pip install ultralytics onnx onnxslim onnxruntime
    python scripts/export_model.py --imgsz 640 320 --int8

Saídas em models/:
    yolo11n.onnx         → FP32, entrada 640 (padrão da API)
    yolo11n_320.onnx     → FP32, entrada 320 (~4x menos operações)
    yolo11n_int8.onnx    → pesos INT8 (quantização dinâmica), entrada 640

O arquivo .onnx é independente de arquitetura: exportado uma vez em amd64,
roda sem alteração na Raspberry Pi 5 (arm64).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def quantize_int8(src: Path, dst: Path) -> Path:
    """Quantização dinâmica: pesos em INT8, ativações quantizadas em tempo de execução.

    Não exige dataset de calibração. Para ganho máximo na Pi 5, a alternativa é a
    quantização estática (QDQ) com ~200 imagens do próprio ambiente da câmera.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QUInt8)
    pre.unlink(missing_ok=True)
    return dst


def export(weights: str, sizes: list[int], out_dir: Path, opset: int) -> list[Path]:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)  # baixa yolo11n.pt dos releases da Ultralytics se necessário
    stem = Path(weights).stem
    outputs: list[Path] = []
    for i, size in enumerate(sizes):
        exported = Path(
            model.export(format="onnx", imgsz=size, opset=opset, simplify=True, dynamic=False)
        )
        name = f"{stem}.onnx" if i == 0 else f"{stem}_{size}.onnx"
        target = out_dir / name
        shutil.move(str(exported), target)
        outputs.append(target)
        print(f"[ok] {target} (entrada {size}x{size})")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640],
                        help="Tamanhos de entrada; o primeiro vira o modelo padrão.")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--int8", action="store_true", help="Gera também a versão INT8.")
    parser.add_argument("--quantize-only", type=Path,
                        help="Apenas quantiza um .onnx existente (não exige PyTorch).")
    args = parser.parse_args()

    if args.quantize_only:
        dst = args.quantize_only.with_name(args.quantize_only.stem + "_int8.onnx")
        print(f"[ok] {quantize_int8(args.quantize_only, dst)}")
        return 0

    outputs = export(args.weights, args.imgsz, args.out, args.opset)
    if args.int8:
        base = outputs[0]
        dst = base.with_name(base.stem + "_int8.onnx")
        print(f"[ok] {quantize_int8(base, dst)} (INT8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
