#!/usr/bin/env python3
"""Новый алгоритм — строится по шагам. РУЧНОЙ триггер: снимок и детект
только по запросу GET /scan (кнопка), не по таймеру.

ШАГ 6 (сейчас): старт с отступа (20,20) от левого верхнего угла, маршируем
ВПРАВО столбцами по ndown точек ВНИЗ (ползунок «Точек вниз»). Ориентир — каждый
кубик отдельно: у кубика с преобладающим цветом углы дописываются в blocks под
ключом-цветом, а сам кубик заливается ПРОТИВОПОЛОЖНЫМ цветом этого ключа (свой
цвет на каждый ключ). Цвет точки — по одному пикселю (без медианы).

Старт (в фоне):
    python3 detect.py                 # слушает http://127.0.0.1:8133

Триггер скана (то же делает кнопка):
    curl -s http://127.0.0.1:8133/scan
"""
import collections
import colorsys
import json
import os
import random
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
NDOWN = int(os.environ.get("SG_NDOWN", "4"))                 # точек вниз в столбце (ползунок)
START_X = int(os.environ.get("SG_START_X", "20"))            # старт: отступ слева, px
START_Y = int(os.environ.get("SG_START_Y", "20"))            # старт: отступ сверху, px

_CAP_PATH = os.path.join(OUT_DIR, "screenshot.png")
_COND = threading.Condition()
_LATEST = {"ts": 0.0, "count": 0, "ms": 0.0, "points": [], "colors": [],
           "kinds": [], "numbers": [], "lines": [], "cubes": [], "cube_fills": [],
           "blocks": {}, "v": 0}


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


def key_color(key):
    """Случайный ЯРКИЙ цвет для ключа-цвета блока, "rrggbb". Сид — сам ключ,
    поэтому цвет стабилен для одного ключа (и между сканами), но у разных
    ключей — разные насыщенные оттенки. Прозрачность заливки задаёт оверлей."""
    h = random.Random(key).random()
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return f"{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


PREDOM = int(os.environ.get("SG_PREDOM", "2"))               # порог преобладания: цвет >= стольких точек


def predominant(colors4, min_count=PREDOM):
    """Преобладающий цвет среди углов кубика: строгий уникальный максимум,
    встречается >= min_count раз (ползунок «Точек на преобладание», 2..8).
    None, если явного преобладания нет (недобрал порог, или ничья за максимум)."""
    cnt = collections.Counter(colors4)
    top, n = cnt.most_common(1)[0]
    if n < min_count or list(cnt.values()).count(n) > 1:
        return None
    return top


def detect(img, step=STEP, predom=PREDOM, ndown=NDOWN):
    """BGR-кадр -> (points, colors_hex, kinds, numbers, lines, cubes, blocks, cube_fills).

    ШАГ 6: старт с отступа (START_X, START_Y) от левого верхнего угла, МАРШИРУЕМ
    ВПРАВО до края. Столбцы на x = START_X, +step, +2·step, ...; в каждом столбце
    ndown точек ВНИЗ (ползунок «Точек вниз»). Между соседними столбцами — (ndown-1)
    КУБИКОВ по высоте.

    Ориентир — КАЖДЫЙ кубик отдельно (не ряд/блок). У кубика ищем ПРЕОБЛАДАЮЩИЙ
    цвет 4 углов (порог predom); если он есть — дописываем 4 угла в blocks под
    ключом-цветом и заливаем кубик ПРОТИВОПОЛОЖНЫМ цветом этого ключа (свой цвет
    заливки на каждый ключ, чтобы визуально различать). blocks и заливки — заново
    на каждый Scan. Цвет точки — по одному пикселю (без медианы)."""
    H, W = img.shape[:2]
    sx, sy = float(START_X), float(START_Y)   # старт: отступ сверху и слева

    xs = []                                   # x столбцов: от старта вправо
    x = sx
    while x < W:
        xs.append(x)
        x += step
    rows = [sy + j * step for j in range(ndown)]   # ndown точек столбца: j=0 верх, ВНИЗ

    coords = [(xi, yj) for xi in xs for yj in rows]   # столбец за столбцом, сверху вниз
    points = [[round(px / W, 4), round(py / H, 4)] for (px, py) in coords]
    colors_hex = [color_hex(color_px(img, px, py)) for (px, py) in coords]
    kinds = ["base"] * len(coords)
    if kinds:
        kinds[0] = "seed"                     # самая первая точка — старт
    numbers = list(range(len(coords)))
    lines = []

    # Рисуем РЯДАМИ: для каждого ряда k (сверху вниз) идём по кубикам слева
    # направо до конца экрана, потом переходим на следующий ряд. Рядов = ndown-1.
    # Углы кубика (столбцы i, i+1; ряд k): (i,k) лв-верх, (i,k+1) лв-низ,
    # (i+1,k) пр-верх, (i+1,k+1) пр-низ.
    blocks = {}
    cubes = []
    cube_fills = []
    ncols = len(xs)
    for k in range(ndown - 1):                         # ряд за рядом, сверху вниз
        yt = round((sy + k * step) / H, 4)             # верх кубиков ряда (меньше y)
        yb = round((sy + (k + 1) * step) / H, 4)       # низ кубиков ряда (больше y)
        for i in range(ncols - 1):                     # слева направо до конца экрана
            idx = [i * ndown + k, i * ndown + k + 1, (i + 1) * ndown + k, (i + 1) * ndown + k + 1]
            key = predominant([colors_hex[j] for j in idx], predom)
            if key is None:
                continue
            blocks.setdefault(key, []).extend(points[j] for j in idx)
            cubes.append([round(xs[i] / W, 4), yt, round(xs[i + 1] / W, 4), yb])
            cube_fills.append(key_color(key))          # свой случайный цвет на каждый ключ
    return points, colors_hex, kinds, numbers, lines, cubes, blocks, cube_fills


def do_scan(step=STEP, predom=PREDOM, ndown=NDOWN):
    """Один проход: снимок -> детект -> публикация в _LATEST."""
    t0 = time.time()
    img = grab_screen()
    if img is None:
        return None
    points, colors_hex, kinds, numbers, lines, cubes, blocks, cube_fills = detect(
        img, step=step, predom=predom, ndown=ndown)
    with _COND:
        _LATEST.update(ts=round(time.time(), 3), count=len(points),
                       v=_LATEST["v"] + 1, ms=round((time.time() - t0) * 1000, 1),
                       points=points, colors=colors_hex, kinds=kinds,
                       numbers=numbers, lines=lines, cubes=cubes,
                       cube_fills=cube_fills, blocks=blocks)
        _COND.notify_all()
    print(f"[scan] {len(points)} точек, {len(cubes)} кубиков, {len(blocks)} цветов "
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
            predom_raw = qs.get("predom", [""])[0]
            predom = int(predom_raw) if predom_raw.isdigit() else PREDOM
            ndown_raw = qs.get("ndown", [""])[0]
            ndown = int(ndown_raw) if ndown_raw.isdigit() and int(ndown_raw) >= 2 else NDOWN
            result = do_scan(step=step, predom=predom, ndown=ndown)
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
