#!/usr/bin/env python3
"""Детекция по СЕТКЕ в левых 30% экрана. РУЧНОЙ триггер: снимок и детект
происходят только по запросу GET /scan (кнопка на оверлее), не по таймеру.

Сканируем ОДНУ строку (y = ROW_FRAC * H) с шагом STEP от X_OFFSET, читаем
цвет в каждой точке сетки. При несовпадении соседних точек проверяем ещё
MISMATCH_CONFIRM точек справа против исходного цвета — если все другие,
это не шум (текст), ставим ОДНУ точку посередине ("probe", другой цвет).

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

_CAP_PATH = os.path.join(OUT_DIR, "screenshot.png")   # снимок экрана — в out/
_COND = threading.Condition()
_LATEST = {"ts": 0.0, "count": 0, "ms": 0.0, "points": [], "colors": [], "kinds": [], "numbers": [], "lines": [], "v": 0}


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
    """Точное совпадение цвета (по hex-значению после округления), без допуска."""
    return bool(np.all(np.round(c1) == np.round(c2)))


MISMATCH_CONFIRM = int(os.environ.get("SG_MISMATCH_CONFIRM", "3"))   # сколько точек справа проверяем
BISECT_MAX = int(os.environ.get("SG_BISECT_MAX", "1"))               # макс. уточнений границы
BISECT_MIN_GAP = float(os.environ.get("SG_BISECT_MIN_GAP", "5"))     # px — дальше не мельчим


def scan_row(img, y, x_end):
    """Сканируем строку y слева направо с шагом STEP от X_OFFSET до x_end.

    Для каждой пары соседних точек сетки i, i+1 с несовпавшим цветом
    проверяем ЕЩЁ до MISMATCH_CONFIRM точек справа (i+1..i+MISMATCH_CONFIRM)
    против цвета точки i. Если ВСЕ они другого цвета — это не шум (текст),
    а реальная смена цвета. Если хоть одна из проверочных точек совпала
    с цветом i — это шум, ничего не делаем.

    Когда расхождение подтверждено — уточняем границу: ставим точку
    посередине между i и i+1. Если её цвет совпал с run_color (i) — блок
    продолжается дальше вправо, делим пополам новый отрезок (эта точка..
    i+1). Если цвет другой — граница левее, делим (i..эта точка). Повторяем
    до BISECT_MAX раз или пока отрезок не сузится до BISECT_MIN_GAP px.
    Последняя точка ЕЩЁ того же (run_color) цвета — граница блока, по ней
    рисуем вертикальную линию.

    Номер точки = порядок ЕЁ ОБНАРУЖЕНИЯ. Но в массиве (и в логе) точки
    стоят по своему МЕСТУ НА ЭКРАНЕ (geometric x), а не в конце.

    КАК ТОЛЬКО найдено первое расхождение — сосредотачиваемся только на
    его уточнении, дальше по сетке не идём."""
    xs_grid = list(np.arange(X_OFFSET, x_end, STEP, dtype=np.float64))
    colors_grid = [color_at(img, x, y) for x in xs_grid]
    n = len(xs_grid)

    entries = [{"num": i, "x": xs_grid[i], "color": colors_grid[i], "kind": "base"}
               for i in range(n)]

    next_num = n
    lines = []
    for i in range(n - 1):
        if same_color(colors_grid[i], colors_grid[i + 1]):
            continue
        lookahead = colors_grid[i + 1: i + 1 + MISMATCH_CONFIRM]
        if not all(not same_color(colors_grid[i], c) for c in lookahead):
            continue          # хоть одна справа совпала с исходным цветом -> шум

        run_color = colors_grid[i]
        lo_x, hi_x = xs_grid[i], xs_grid[i + 1]
        boundary_x = lo_x                     # последняя точка ЕЩЁ run_color
        for _ in range(BISECT_MAX):
            if hi_x - lo_x <= BISECT_MIN_GAP:
                break
            mid_x = (lo_x + hi_x) / 2.0
            mid_color = color_at(img, mid_x, y)
            entries.append({"num": next_num, "x": mid_x, "color": mid_color, "kind": "probe"})
            next_num += 1
            if same_color(mid_color, run_color):
                lo_x = mid_x
                boundary_x = mid_x            # блок продолжается дальше — сдвигаем границу
            else:
                hi_x = mid_x
        lines.append(boundary_x)
        break              # нашли расхождение -> дальше не идём, сосредоточены на нём

    entries.sort(key=lambda e: e["x"])          # по месту на экране, номер — как был присвоен
    xs = [e["x"] for e in entries]
    colors = [e["color"] for e in entries]
    kinds = [e["kind"] for e in entries]
    numbers = [e["num"] for e in entries]
    return xs, colors, kinds, numbers, lines


def detect(img):
    """BGR-кадр -> (points, colors_hex, kinds, numbers, lines).

    points — точки сканирования (сетка + проверочные) в НОРМАЛИЗОВАННЫХ
    коорд. (0..1, y сверху), отсортированы по месту на экране (слева
    направо). colors_hex — цвет каждой точки в hex, тот же порядок.
    kinds — "base" | "probe" (проверочная точка между несовпавшими цветами,
    красится иначе). numbers — присвоенный номер точки (порядок обнаружения,
    НЕ порядок в массиве) — им подписывается точка и пишется лог.
    lines — нормализованный x уточнённой границы блока (если нашли)."""
    H, W = img.shape[:2]
    y = H * ROW_FRAC
    x_end = W * REGION_FRAC
    xs, colors, kinds, numbers, boundary_xs = scan_row(img, y, x_end)

    points = [[round(x / W, 4), round(y / H, 4)] for x in xs]
    colors_hex = [color_hex(c) for c in colors]
    lines = [round(bx / W, 4) for bx in boundary_xs]
    return points, colors_hex, kinds, numbers, lines


def save_points_log(colors_hex, numbers):
    """Новый файл на каждый скан: out/points_<метка_времени>.txt,
    строки "номер-цвет_hex", по порядку МЕСТА НА ЭКРАНЕ (не по номеру)."""
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    path = os.path.join(OUT_DIR, f"points_{ts}.txt")
    with open(path, "w") as f:
        for num, hexc in zip(numbers, colors_hex):
            f.write(f"{num}-{hexc}\n")
    return path


def do_scan():
    """Один проход: снимок экрана (out/screenshot.png) -> детект ->
    публикация в _LATEST -> лог out/points_<метка_времени>.txt."""
    t0 = time.time()
    img = grab_screen()
    if img is None:
        return None
    points, colors_hex, kinds, numbers, lines = detect(img)
    with _COND:
        _LATEST.update(ts=round(time.time(), 3), count=len(points),
                        v=_LATEST["v"] + 1, ms=round((time.time() - t0) * 1000, 1),
                        points=points, colors=colors_hex, kinds=kinds, numbers=numbers, lines=lines)
        _COND.notify_all()          # разбудить висящие long-poll запросы к /points
    log_path = save_points_log(colors_hex, numbers)
    print(f"[scan] {len(points)} точек за {_LATEST['ms']:.0f}мс "
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
    points, colors_hex, kinds, numbers, lines = detect(img)
    canvas = img.copy()
    H, W = img.shape[:2]
    for num, (nx, ny), kind in zip(numbers, points, kinds):
        px, py = int(nx * W), int(ny * H)
        color = (0, 255, 0) if kind == "base" else (0, 140, 255)   # probe — оранжевый
        cv2.circle(canvas, (px, py), 5, color, -1)
        cv2.putText(canvas, str(num), (px + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    for nx in lines:
        cv2.line(canvas, (int(nx * W), 0), (int(nx * W), H), (0, 255, 0), 1)
    out = os.path.join(OUT_DIR, "preview.png")
    cv2.imwrite(out, canvas)
    log_path = save_points_log(colors_hex, numbers)
    print(f"{len(points)} точек; сохранил {out} и {log_path}")


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
