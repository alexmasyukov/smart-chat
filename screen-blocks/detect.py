#!/usr/bin/env python3
"""Детекция блоков по СЕТКЕ ЦВЕТОВ. Рисуем точки на границах блоков.

Идея: бьём экран на ячейки CELL×CELL пикселей. У каждой ячейки берём средний
цвет. Соседние ячейки одного (квантованного) цвета = один блок. Где цвет ячейки
отличается от соседа — там ГРАНИЦА блока. Эти граничные ячейки отдаём как точки,
оверлей рисует их на экране.

Старт (в фоне):
    python3 detect.py                 # слушает http://127.0.0.1:8131

Опрос:
    curl -s http://127.0.0.1:8131/points | python3 -m json.tool
    curl -s http://127.0.0.1:8131/health

Отладка (сохранит out/preview.png с точками):
    python3 detect.py --snapshot

Тюнинг — env: SE_CELL (размер ячейки px), SE_COLOR_Q (квант цвета),
SE_INTERVAL (период съёмки).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

HOST = os.environ.get("SB_HOST", "127.0.0.1")
PORT = int(os.environ.get("SB_PORT", "8131"))
INTERVAL = float(os.environ.get("SB_INTERVAL", "1.0"))   # период съёмки, сек
CELL = int(os.environ.get("SB_CELL", "100"))             # размер ячейки, пиксели экрана
COLOR_Q = int(os.environ.get("SB_COLOR_Q", "24"))        # квант цвета (объединять близкие)
FLAT_STD = float(os.environ.get("SB_FLAT_STD", "10"))    # макс. разброс ВНУТРИ ячейки = заливка
MIN_BLOCK_FRAC = float(os.environ.get("SB_MIN_BLOCK_FRAC", "0.01"))  # мин. площадь блока (доля экрана)

_CAP_PATH = os.path.join(tempfile.gettempdir(), "screen_blocks_frame.png")
# _COND охраняет _LATEST И будит long-poll: детект готов -> оверлей сразу рисует.
_COND = threading.Condition()
_LATEST = {"ts": 0.0, "cols": 0, "rows": 0, "count": 0, "ms": 0.0, "points": [], "v": 0}


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


def _iou(a, b):
    """IoU двух прямоугольников (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def detect(img):
    """BGR-кадр -> граничные точки блоков в НОРМАЛИЗОВАННЫХ коорд. (0..1, y сверху).

    Возвращает (points, cols, rows). points: [[x, y], ...] центры граничных ячеек."""
    H, W = img.shape[:2]
    cols, rows = W // CELL, H // CELL
    f = img.astype(np.float32)
    mean = cv2.resize(f, (cols, rows), interpolation=cv2.INTER_AREA)     # средний цвет ячейки
    var = cv2.resize(f * f, (cols, rows), interpolation=cv2.INTER_AREA) - mean * mean
    inner_std = np.sqrt(np.clip(var, 0, None)).mean(axis=2)              # разброс ВНУТРИ ячейки
    flat = inner_std < FLAT_STD                                          # ячейка = ровная заливка
    q = (mean.astype(np.int32) // COLOR_Q)                              # квантованный цвет ячейки
    key = q[..., 0] * 100000 + q[..., 1] * 1000 + q[..., 2]            # цвет ячейки одним числом

    # Собираем РОВНЫЕ ячейки одного цвета в связные области и берём только БОЛЬШИЕ.
    # У каждого большого блока — его прямоугольник (bbox) в ячейках.
    min_cells = max(6, int(MIN_BLOCK_FRAC * cols * rows))
    boxes = []
    for k in np.unique(key[flat]):
        m = ((key == k) & flat).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, 4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_cells:
                x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
                boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: -b[2] * b[3])          # крупные первыми
    kept = []
    for b in boxes:                                 # выкинуть почти-дубли (та же рамка)
        if all(_iou(b, o) < 0.8 for o in kept):
            kept.append(b)

    # Точки — по РАМКЕ (периметру bbox) каждого большого блока: ровный контур,
    # а не обводка вокруг каждого слова внутри.
    cells = set()
    for x, y, w, h in kept:
        for cx in range(x, x + w):
            cells.add((cx, y)); cells.add((cx, y + h - 1))
        for cy in range(y, y + h):
            cells.add((x, cy)); cells.add((x + w - 1, cy))
    points = [[round((cx + 0.5) / cols, 4), round((cy + 0.5) / rows, 4)]
              for cx, cy in cells]
    return points, cols, rows, len(kept)


def capture_loop():
    while True:
        t0 = time.time()
        img = grab_screen()
        if img is not None:
            try:
                points, cols, rows, nblocks = detect(img)
                with _COND:
                    _LATEST.update(ts=round(time.time(), 3), cols=cols, rows=rows,
                                   count=len(points), v=_LATEST["v"] + 1,
                                   ms=round((time.time() - t0) * 1000, 1), points=points)
                    _COND.notify_all()          # разбудить висящие long-poll запросы
                print(f"[det] {nblocks} блоков, {len(points)} точек границ "
                      f"({cols}x{rows} ячеек) за {_LATEST['ms']:.0f}мс", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[err] detect: {e}", flush=True)
        time.sleep(max(0.0, INTERVAL - (time.time() - t0)))


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
        if parsed.path in ("/points", "/"):
            # long-poll: ?since=<v> — ждём, пока появится детект новее since,
            # и отвечаем сразу в момент его готовности (перерисовка без задержки).
            qs = urllib.parse.parse_qs(parsed.query)
            since = int(qs.get("since", ["0"])[0]) if qs.get("since", ["0"])[0].isdigit() else 0
            with _COND:
                if _LATEST["v"] <= since:
                    _COND.wait(timeout=25)
                return self._send(200, dict(_LATEST))
        return self._send(404, {"error": "use /points or /health"})

    def log_message(self, *args):
        pass


def snapshot():
    img = grab_screen()
    if img is None:
        print("не удалось снять экран", file=sys.stderr)
        sys.exit(1)
    points, cols, rows, nblocks = detect(img)
    canvas = img.copy()
    H, W = img.shape[:2]
    for nx, ny in points:
        cv2.circle(canvas, (int(nx * W), int(ny * H)), 5, (0, 255, 0), -1)
    out = os.path.join(OUT_DIR, "preview.png")
    cv2.imwrite(out, canvas)
    print(f"{nblocks} блоков, {len(points)} точек ({cols}x{rows} ячеек); сохранил {out}")


def main():
    if "--snapshot" in sys.argv or "--once" in sys.argv:
        snapshot()
        return
    threading.Thread(target=capture_loop, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] слушаю http://{HOST}:{PORT}  (ячейка {CELL}px, Ctrl+C — стоп)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
