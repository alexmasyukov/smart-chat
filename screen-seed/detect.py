#!/usr/bin/env python3
"""Новый алгоритм — строится по шагам. РУЧНОЙ триггер: снимок и детект
только по запросу GET /scan (кнопка), не по таймеру.

ШАГ 4 (сейчас): от центра маршируем ВПРАВО до конца экрана столбцами по 4
точки; между соседними столбцами — 3 кубика. У кубика с преобладающим цветом
углы дописываются в объект blocks (под ключом-цветом), кубик заливается
розовым. Цвет точки — по одному пикселю (без медианы).

Старт (в фоне):
    python3 detect.py                 # слушает http://127.0.0.1:8133

Триггер скана (то же делает кнопка):
    curl -s http://127.0.0.1:8133/scan
"""
import collections
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
STEP = int(os.environ.get("SG_STEP", "40"))                  # шаг между соседними точками, px

_CAP_PATH = os.path.join(OUT_DIR, "screenshot.png")
_COND = threading.Condition()
_LATEST = {"ts": 0.0, "count": 0, "ms": 0.0, "points": [], "colors": [],
           "kinds": [], "numbers": [], "lines": [], "cubes": [], "blocks": {}, "v": 0}


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


def color_px(img, x, y):
    """Цвет ОДНОГО пикселя (x, y), BGR. Пока без медианы — просто пиксель."""
    H, W = img.shape[:2]
    xi = min(max(0, int(round(x))), W - 1)
    yi = min(max(0, int(round(y))), H - 1)
    return img[yi, xi]


def color_hex(c):
    """BGR-массив -> "rrggbb"."""
    b, g, r = c
    return f"{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def predominant(colors4):
    """Преобладающий цвет среди углов кубика: строгий уникальный максимум,
    встречается >= 2 раз. None, если явного преобладания нет (2-2 или все разные)."""
    cnt = collections.Counter(colors4)
    top, n = cnt.most_common(1)[0]
    if n < 2 or list(cnt.values()).count(n) > 1:
        return None
    return top


def detect(img, step=STEP):
    """BGR-кадр -> (points, colors_hex, kinds, numbers, lines, cubes, blocks).

    ШАГ 4: от центра МАРШИРУЕМ ВПРАВО до конца экрана. Столбцы на x = cx, cx+step,
    cx+2·step, ... до правого края; в каждом столбце 4 точки снизу вверх (cy,
    cy-step, cy-2·step, cy-3·step). Между соседними столбцами — 3 КУБИКА по высоте.

    Каждую итерацию (пара соседних столбцов) даёт 3 кубика. У каждого кубика ищем
    ПРЕОБЛАДАЮЩИЙ цвет 4 углов; если он есть — ДОПИСЫВАЕМ его углы в объект blocks
    под ключом-цветом (одинаковый цвет копится под одним ключом) и отдаём кубик на
    заливку розовым. blocks собирается ЗАНОВО на каждый Scan.
    Цвет точки — по одному пикселю (без медианы). «Сверху» = меньше y."""
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    xs = []                                   # x столбцов: от центра вправо до края
    x = cx
    while x < W:
        xs.append(x)
        x += step
    rows = [cy - j * step for j in range(4)]  # 4 точки столбца: j=0 низ, вверх

    coords = [(xi, yj) for xi in xs for yj in rows]   # столбец за столбцом, снизу вверх
    points = [[round(px / W, 4), round(py / H, 4)] for (px, py) in coords]
    colors_hex = [color_hex(color_px(img, px, py)) for (px, py) in coords]
    kinds = ["base"] * len(coords)
    if kinds:
        kinds[0] = "seed"                     # самая первая точка — центр
    numbers = list(range(len(coords)))
    lines = []

    # Пара соседних столбцов i, i+1 -> 3 кубика. Углы кубика k (снизу вверх):
    # (i,k) лв-низ, (i,k+1) лв-верх, (i+1,k) пр-низ, (i+1,k+1) пр-верх.
    blocks = {}
    cubes = []
    for i in range(len(xs) - 1):
        xl = round(xs[i] / W, 4)
        xr = round(xs[i + 1] / W, 4)
        for k in range(3):
            idx = [i * 4 + k, i * 4 + k + 1, (i + 1) * 4 + k, (i + 1) * 4 + k + 1]
            key = predominant([colors_hex[j] for j in idx])
            if key is None:
                continue
            blocks.setdefault(key, []).extend(points[j] for j in idx)
            yt = round((cy - (k + 1) * step) / H, 4)   # верх кубика (меньше y)
            yb = round((cy - k * step) / H, 4)         # низ кубика
            cubes.append([xl, yt, xr, yb])
    return points, colors_hex, kinds, numbers, lines, cubes, blocks


def do_scan(step=STEP):
    """Один проход: снимок -> детект -> публикация в _LATEST."""
    t0 = time.time()
    img = grab_screen()
    if img is None:
        return None
    points, colors_hex, kinds, numbers, lines, cubes, blocks = detect(img, step=step)
    with _COND:
        _LATEST.update(ts=round(time.time(), 3), count=len(points),
                       v=_LATEST["v"] + 1, ms=round((time.time() - t0) * 1000, 1),
                       points=points, colors=colors_hex, kinds=kinds,
                       numbers=numbers, lines=lines, cubes=cubes, blocks=blocks)
        _COND.notify_all()
    print(f"[scan] {len(points)} точек, {len(cubes)} кубиков, {len(blocks)} блоков "
          f"за {_LATEST['ms']:.0f}мс", flush=True)
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
            qs = urllib.parse.parse_qs(parsed.query)
            step_raw = qs.get("step", [""])[0]
            step = int(step_raw) if step_raw.isdigit() else STEP
            result = do_scan(step=step)
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
