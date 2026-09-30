"""Cliente de "tempo real": lê webcam ou vídeo e envia frames para a API.

Simula a câmera instalada sobre a zona da máquina. Na Pi 5, o mesmo papel
seria feito pela Camera Module 3 (via picamera2) no próprio dispositivo.

    python scripts/stream_client.py --source 0                 # webcam
    python scripts/stream_client.py --source video.mp4 --show  # vídeo + janela anotada
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Índice da webcam ou caminho do vídeo")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--fps", type=float, default=5.0, help="Taxa máxima de envio")
    ap.add_argument("--show", action="store_true", help="Exibe o PNG anotado numa janela")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Não foi possível abrir a fonte {args.source!r}", file=sys.stderr)
        return 1

    endpoint = f"{args.api}/detect/image" if args.show else f"{args.api}/detect"
    interval = 1.0 / args.fps
    session = requests.Session()
    try:
        while True:
            start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                break
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            try:
                resp = session.post(
                    endpoint, files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                    params={"source": "stream_client"}, timeout=5,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"[erro] {exc}")
                time.sleep(1)
                continue

            rtt = (time.monotonic() - start) * 1000
            if args.show:
                img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                risk = resp.headers.get("X-Overall-Risk", "?")
                cv2.imshow("Zona de risco (q para sair)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                data = resp.json()
                risk = data["overall_risk"]
                for alert in data["alerts"]:
                    print(f"  >>> ALERTA {alert['risk'].upper()} em {alert['zone']}: {alert['action']}")
            print(f"risco={risk:<8} ida_e_volta={rtt:6.1f} ms")

            time.sleep(max(0.0, interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
