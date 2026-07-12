#!/usr/bin/env python3
"""Новый алгоритм — строится по шагам. РУЧНОЙ триггер: снимок и детект
только по запросу GET /scan (кнопка), не по таймеру.

ШАГ 1 (сейчас): ставим ОДНУ точку ровно в центре экрана. От неё дальше
будет расти алгоритм.

Старт (в фоне):
    python3 detect.py                 # слушает http://127.0.0.1:8133

Триггер скана (то же делает кнопка):
    curl -s http://127.0.0.1:8133/scan
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

HOST = os.environ.get("SG_HOST", "127.0.0.1")
PORT = int(os.environ.get("SG_PORT", "8133"))
PATCH = int(os.environ.get("SG_PATCH", "6"))                 # размер патча для цвета точки, px

_CAP_PATH = os.path.join(OUT_DIR, "screenshot.png")
_COND = threading.Condition()
_LATEST = {"ts": 0.0, "count": 0, "ms": 0.0, "points": [], "colors": [],
           "kinds": [], "numbers": [], "lines": [], "v": 0}


def grab_screen():
    """Снимок основного дисплея -> BGR-массив (или None)."""
    try:
        subprocess.run(["screencapture", "-x", "-t", "png", _CAP_PATH],
                       check=True, capture_output=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"[err] screencapture: {e}", flush=True)
        return None
    img = cv2.imread(_CAP_PATH)
    if img is None:
        print("[err] снимок не прочитался (нет прав Screen Recording?)", flush=True)
    return img


def color_at(img, x, y):
    """Срединный (медианный) цвет патча PATCH×PATCH вокруг (x, y), BGR."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, x0 + PATCH), min(H, y0 + PATCH)
    roi = img[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(roi, axis=0)


def color_hex(c):
    """BGR-массив -> "rrggbb"."""
    b, g, r = c
    return f"{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def detect(img):
    """BGR-кадр -> (points, colors_hex, kinds, numbers, lines).

    ШАГ 1: одна точка в центре экрана (0.5, 0.5)."""
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    color = color_at(img, cx, cy)
    points = [[0.5, 0.5]]
    colors_hex = [color_hex(color)]
    kinds = ["seed"]
    numbers = [0]
    lines = []
    return points, colors_hex, kinds, numbers, lines


def do_scan():
    """Один проход: снимок -> детект -> публикация в _LATEST."""
    t0 = time.time()
    img = grab_screen()
    if img is None:
        return None
    points, colors_hex, kinds, numbers, lines = detect(img)
    with _COND:
        _LATEST.update(ts=round(time.time(), 3), count=len(points),
                       v=_LATEST["v"] + 1, ms=round((time.time() - t0) * 1000, 1),
                       points=points, colors=colors_hex, kinds=kinds,
                       numbers=numbers, lines=lines)
        _COND.notify_all()
    print(f"[scan] {len(points)} точек за {_LATEST['ms']:.0f}мс", flush=True)
    return _LATEST


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            with _COND:
                return self._send(200, {"ok": True, "count": _LATEST["count"]})
        if parsed.path == "/scan":
            result = do_scan()
            if result is None:
                return self._send(500, {"error": "screencapture failed"})
            return self._send(200, {"ok": True, "count": result["count"], "v": result["v"],
                                     "numbers": result["numbers"], "colors": result["colors"]})
        if parsed.path in ("/points", "/"):
            qs = urllib.parse.parse_qs(parsed.query)
            since = int(qs.get("since", ["0"])[0]) if qs.get("since", ["0"])[0].isdigit() else 0
            with _COND:
                if _LATEST["v"] <= since:
                    _COND.wait(timeout=25)
                return self._send(200, dict(_LATEST))
        return self._send(404, {"error": "use /scan, /points или /health"})

    def log_message(self, *args):
        pass


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] слушаю http://{HOST}:{PORT} — детект ручной, по GET /scan "
          f"(Ctrl+C — стоп)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
