#!/usr/bin/env python3
"""Детекция по СЕТКЕ в левых 30% экрана. РУЧНОЙ триггер: снимок и детект
происходят только по запросу GET /scan (кнопка на оверлее), не по таймеру.

Сканируем ОДНУ строку (y = ROW_FRAC * H) с шагом STEP от X_OFFSET, ищем
одинаковые цвета подряд (блок), с допуском на одиночный "шумный" сэмпл
(текст внутри блока). Как только два сэмпла подряд дают другой цвет —
граница подтверждена, уточняем её бисекцией.

Старт (в фоне):
    python3 detect.py                 # слушает http://127.0.0.1:8132

Триггер скана (то же самое делает кнопка в оверлее):
    curl -s http://127.0.0.1:8132/scan

Результат каждого скана — в out/points.txt: "номер-цвет_hex" по одной точке
на строку (порядок точек = порядок сканирования слева направо).

Отладка (сканирует один раз и сохранит out/preview.png):
    python3 detect.py --snapshot
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
PORT = int(os.environ.get("SG_PORT", "8132"))
REGION_FRAC = float(os.environ.get("SG_REGION_FRAC", "0.30"))  # доля ширины экрана слева
STEP = int(os.environ.get("SG_STEP", "150"))                # шаг сканирования, пиксели
X_OFFSET = int(os.environ.get("SG_X_OFFSET", "10"))         # старт сканирования от левого края
ROW_FRAC = float(os.environ.get("SG_ROW_FRAC", "0.5"))       # какую строку сканируем (0..1 по Y)
PATCH = int(os.environ.get("SG_PATCH", "6"))                 # размер патча для цвета точки, px
COLOR_TOL = float(os.environ.get("SG_COLOR_TOL", "12"))      # допуск "тот же цвет" (на канал)
BISECT_ITERS = int(os.environ.get("SG_BISECT_ITERS", "8"))   # уточнение границы

_CAP_PATH = os.path.join(OUT_DIR, "screenshot.png")   # снимок экрана — в out/
_COND = threading.Condition()
_LATEST = {"ts": 0.0, "count": 0, "ms": 0.0, "points": [], "colors": [], "lines": [], "v": 0}


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
    """Средний цвет патча PATCH×PATCH вокруг (x, y), BGR."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, x0 + PATCH), min(H, y0 + PATCH)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    return roi.reshape(-1, 3).mean(axis=0)


def color_hex(c):
    """BGR-массив -> "rrggbb"."""
    b, g, r = c
    return f"{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def same_color(c1, c2):
    return bool(np.all(np.abs(c1 - c2) < COLOR_TOL))


def bisect_boundary(img, y, x_match, x_mismatch, run_color):
    """Уточняем границу блока бисекцией между x_match (ещё run_color)
    и x_mismatch (уже другой цвет)."""
    lo, hi = x_match, x_mismatch
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2.0
        c = color_at(img, mid, y)
        if same_color(c, run_color):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def scan_row(img, y, x_end):
    """Сканируем строку y слева направо с шагом STEP от X_OFFSET до x_end.

    Возвращает (xs, colors, boundaries): xs/colors — все сэмплы сканирования
    (для отладки и нумерации точек), boundaries — уточнённые x границ блоков."""
    xs = list(np.arange(X_OFFSET, x_end, STEP, dtype=np.float64))
    colors = [color_at(img, x, y) for x in xs]
    n = len(xs)
    boundaries = []
    i = 0
    while i < n - 1:
        if not same_color(colors[i], colors[i + 1]):
            i += 1
            continue
        # colors[i] == colors[i+1] -> начало блока, тянем дальше
        run_color = colors[i]
        last_match = i + 1
        j = i + 2
        while j < n:
            if same_color(colors[j], run_color):
                last_match = j
                j += 1
                continue
            # j отличается -> проверяем j+1: шум (текст) или реальная граница?
            if j + 1 < n and same_color(colors[j + 1], run_color):
                last_match = j + 1          # j был шумом, блок продолжается
                j += 2
                continue
            # j и j+1 (если есть) оба другого цвета -> граница подтверждена
            bx = bisect_boundary(img, y, xs[last_match], xs[j], run_color)
            boundaries.append(bx)
            break
        i = j + 1 if j < n else n
    return xs, colors, boundaries


def detect(img):
    """BGR-кадр -> (points, colors_hex, lines).

    points — сэмплы сканирования в НОРМАЛИЗОВАННЫХ коорд. (0..1, y сверху),
    порядок = порядок точки (номер точки = индекс в списке).
    colors_hex — цвет каждой точки в hex, тот же порядок.
    lines — нормализованный x найденных границ блоков."""
    H, W = img.shape[:2]
    y = H * ROW_FRAC
    x_end = W * REGION_FRAC
    xs, colors, boundaries = scan_row(img, y, x_end)

    points = [[round(x / W, 4), round(y / H, 4)] for x in xs]
    colors_hex = [color_hex(c) for c in colors]
    lines = [round(bx / W, 4) for bx in boundaries]
    return points, colors_hex, lines


def save_points_log(colors_hex):
    """Новый файл на каждый скан: out/points_<метка_времени>.txt,
    строки "номер-цвет_hex"."""
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    path = os.path.join(OUT_DIR, f"points_{ts}.txt")
    with open(path, "w") as f:
        for i, hexc in enumerate(colors_hex):
            f.write(f"{i}-{hexc}\n")
    return path


def do_scan():
    """Один проход: снимок экрана (out/screenshot.png) -> детект ->
    публикация в _LATEST -> лог out/points_<метка_времени>.txt."""
    t0 = time.time()
    img = grab_screen()
    if img is None:
        return None
    points, colors_hex, lines = detect(img)
    with _COND:
        _LATEST.update(ts=round(time.time(), 3), count=len(points),
                        v=_LATEST["v"] + 1, ms=round((time.time() - t0) * 1000, 1),
                        points=points, colors=colors_hex, lines=lines)
        _COND.notify_all()          # разбудить висящие long-poll запросы к /points
    log_path = save_points_log(colors_hex)
    print(f"[scan] {len(points)} точек, {len(lines)} границ за {_LATEST['ms']:.0f}мс "
          f"-> {os.path.basename(log_path)}", flush=True)
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
            return self._send(200, {"ok": True, "count": result["count"], "v": result["v"]})
        if parsed.path in ("/points", "/"):
            # long-poll: ?since=<v> — ждём, пока появится скан новее since,
            # и отвечаем сразу в момент его готовности (перерисовка без задержки).
            qs = urllib.parse.parse_qs(parsed.query)
            since = int(qs.get("since", ["0"])[0]) if qs.get("since", ["0"])[0].isdigit() else 0
            with _COND:
                if _LATEST["v"] <= since:
                    _COND.wait(timeout=25)
                return self._send(200, dict(_LATEST))
        return self._send(404, {"error": "use /scan, /points или /health"})

    def log_message(self, *args):
        pass


def snapshot():
    img = grab_screen()
    if img is None:
        print("не удалось снять экран", file=sys.stderr)
        sys.exit(1)
    points, colors_hex, lines = detect(img)
    canvas = img.copy()
    H, W = img.shape[:2]
    for i, (nx, ny) in enumerate(points):
        px, py = int(nx * W), int(ny * H)
        cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(canvas, str(i), (px + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    for nx in lines:
        cv2.line(canvas, (int(nx * W), 0), (int(nx * W), H), (0, 255, 0), 1)
    out = os.path.join(OUT_DIR, "preview.png")
    cv2.imwrite(out, canvas)
    log_path = save_points_log(colors_hex)
    print(f"{len(points)} точек, {len(lines)} границ; сохранил {out} и {log_path}")


def main():
    if "--snapshot" in sys.argv or "--once" in sys.argv:
        snapshot()
        return
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] слушаю http://{HOST}:{PORT} — детект ручной, по GET /scan "
          f"(Ctrl+C — стоп)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
