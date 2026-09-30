"""Benchmark de latência do pipeline em CPU (sem GPU).

Mede pré-processamento, inferência e pós-processamento separadamente para
um ou mais modelos ONNX e gera uma tabela Markdown pronta para o README.

Exemplos:
    python scripts/benchmark.py --models models/yolo11n.onnx models/yolo11n_int8.onnx
    python scripts/benchmark.py --threads 4 --runs 50          # emula os 4 núcleos da Pi 5
    docker run --rm --cpus=4 --memory=4g <imagem> python scripts/benchmark.py

Na Raspberry Pi 5, rode o mesmo comando dentro do container para obter os
números reais e compará-los com os do notebook.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detector import PersonDetector  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def run(model: Path, images: list[np.ndarray], threads: int, runs: int, warmup: int) -> dict:
    det = PersonDetector(model, num_threads=threads)
    for _ in range(warmup):
        det.detect(images[0])

    pre, inf, post, people = [], [], [], []
    for i in range(runs):
        r = det.detect(images[i % len(images)])
        pre.append(r.preprocess_ms)
        inf.append(r.inference_ms)
        post.append(r.postprocess_ms)
        people.append(len(r.detections))
    total = [a + b + c for a, b, c in zip(pre, inf, post)]
    return {
        "model": model.name,
        "size_mb": round(model.stat().st_size / 1e6, 1),
        "input": f"{det.input_w}x{det.input_h}",
        "threads": threads,
        "pre_ms": round(statistics.mean(pre), 1),
        "inf_p50_ms": round(percentile(inf, 50), 1),
        "inf_p95_ms": round(percentile(inf, 95), 1),
        "post_ms": round(statistics.mean(post), 1),
        "total_p50_ms": round(percentile(total, 50), 1),
        "fps": round(1000 / percentile(total, 50), 1),
        "avg_people": round(statistics.mean(people), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, nargs="+",
                    default=sorted(Path("models").glob("*.onnx")))
    ap.add_argument("--images", type=Path, nargs="+",
                    default=sorted(Path("samples").glob("*.jpg")))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="Salva o resultado em JSON.")
    args = ap.parse_args()

    images = [img for p in args.images if (img := cv2.imread(str(p))) is not None]
    if not images or not args.models:
        print("Nenhuma imagem ou modelo encontrado.", file=sys.stderr)
        return 1

    env = {
        "arch": platform.machine(),
        "processor": platform.processor() or "n/d",
        "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    rows = [run(m, images, args.threads, args.runs, args.warmup) for m in args.models]

    cols = ["model", "size_mb", "input", "threads", "pre_ms", "inf_p50_ms",
            "inf_p95_ms", "post_ms", "total_p50_ms", "fps", "avg_people"]
    print(f"\nAmbiente: {env['arch']} | {env['processor']} | Python {env['python']}\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    if args.out:
        args.out.write_text(json.dumps({"env": env, "results": rows}, indent=2))
        print(f"\nResultado salvo em {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
