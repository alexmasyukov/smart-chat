#!/usr/bin/env python3
"""Детектор длинных ярких линий (граней) на экране — «полки и стены» для кота.

Каждые SE_INTERVAL секунд снимает экран нативным `screencapture`, ищет длинные
прямые контрастные грани (LSD из OpenCV), классифицирует их на горизонтальные
(H — по ним кот ходит) и вертикальные (V — по ним лезет), сливает коллинеарные
куски в цельные отрезки и отдаёт результат по HTTP.

Почему LSD, а не Hough: на UI-скриншотах LSD даёт меньше дублей, субпиксельно
точен и не требует подбора порогов Canny (ресёрч + замер на реальном экране:
весь цикл ~40мс, поэтому раз в 2с — почти бесплатно).

«Яркая линия» — это НЕ абсолютная светлота (тёмная тема даёт среднюю яркость ~40),
а высокий локальный КОНТРАСТ грани: линия заметно выделяется на фоне.

Координаты в выдаче — НОРМАЛИЗОВАННЫЕ (0..1, y сверху вниз), не в retina-пикселях.
Так потребитель (Swift-оверлей, кот) масштабирует их под свои point-размеры окна
без возни с backingScaleFactor.

Старт (в фоне):
    python3 detect.py                 # слушает http://127.0.0.1:8130

Опрос:
    curl -s http://127.0.0.1:8130/edges | python3 -m json.tool
    curl -s http://127.0.0.1:8130/health

Отладка без оверлея (сохранит out/preview.png с зелёными линиями):
    python3 detect.py --snapshot

Тюнинг — через переменные окружения (см. блок CONFIG).
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── CONFIG (всё переопределяется через env) ────────────────────────────────
HOST = os.environ.get("SE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SE_PORT", "8130"))
INTERVAL = float(os.environ.get("SE_INTERVAL", "2.0"))   # период съёмки, сек
WORK_WIDTH = int(os.environ.get("SE_WORK_WIDTH", "1600"))  # рабочая ширина кадра
MIN_LEN_FRAC = float(os.environ.get("SE_MIN_LEN_FRAC", "0.08"))  # мин. длина = доля ширины
ANGLE_TOL = float(os.environ.get("SE_ANGLE_TOL", "8"))    # ±градусов до оси = H/V
MIN_CONTRAST = float(os.environ.get("SE_MIN_CONTRAST", "12"))  # мин. контраст грани (0..255)
Y_TOL_FRAC = float(os.environ.get("SE_Y_TOL_FRAC", "0.004"))   # слияние: разброс поперёк
GAP_FRAC = float(os.environ.get("SE_GAP_FRAC", "0.012"))       # слияние: макс. разрыв вдоль
KEEP_DIAGONAL = os.environ.get("SE_KEEP_DIAGONAL", "0") == "1"  # оставлять ли диагонали

_CAP_PATH = os.path.join(tempfile.gettempdir(), "screen_edges_frame.png")

# Последний результат — общий между потоком-съёмщиком и HTTP-обработчиком.
_LOCK = threading.Lock()
_LATEST = {"ts": 0.0, "w": 0, "h": 0, "count": 0, "ms": 0.0, "segments": []}


def grab_screen():
    """Снимок основного дисплея через нативный screencapture -> BGR-массив.

    Требует разрешение Screen Recording у родительского приложения (терминала).
    Возвращает None, если снять не удалось."""
    try:
        subprocess.run(
            ["screencapture", "-x", "-t", "png", _CAP_PATH],
            check=True, capture_output=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[err] screencapture: {e}", flush=True)
        return None
    img = cv2.imread(_CAP_PATH)
    if img is None:
        print("[err] снимок не прочитался (нет прав Screen Recording?)", flush=True)
    return img


def _edge_contrast(gray, x1, y1, x2, y2, sw, sh):
    """Средняя |разница яркости| по обе стороны отрезка = насколько грань заметна."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L                       # единичная нормаль
    n = 24
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)

    def side(off):
        px = np.clip((xs + nx * off).astype(int), 0, sw - 1)
        py = np.clip((ys + ny * off).astype(int), 0, sh - 1)
        return gray[py, px].astype(float)

    return abs(side(3).mean() - side(-3).mean())


def _merge_axis(items, const_tol, gap):
    """Слить коллинеарные отрезки одной оси.

    items: список (const, a, b, contrast), где const — неизменная координата
    (y для H, x для V), [a,b] — интервал вдоль оси (a<b). Близкие по const
    группируются, внутри группы интервалы объединяются при разрыве <= gap.
    Возвращает список (const_avg, a, b, contrast_max)."""
    if not items:
        return []
    items = sorted(items, key=lambda t: (t[0], t[1]))
    clusters = []            # каждый: [consts[], intervals sorted]
    for const, a, b, con in items:
        placed = False
        for cl in clusters:
            if abs(const - cl["c"]) <= const_tol:
                cl["items"].append((a, b, con))
                cl["c"] = (cl["c"] * cl["n"] + const) / (cl["n"] + 1)
                cl["n"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"c": const, "n": 1, "items": [(a, b, con)]})

    out = []
    for cl in clusters:
        ivs = sorted(cl["items"], key=lambda t: t[0])
        ca, cb, ccon = ivs[0]
        for a, b, con in ivs[1:]:
            if a <= cb + gap:                      # перекрытие/малый разрыв — сливаем
                cb = max(cb, b)
                ccon = max(ccon, con)
            else:
                out.append((cl["c"], ca, cb, ccon))
                ca, cb, ccon = a, b, con
        out.append((cl["c"], ca, cb, ccon))
    return out


def detect(img):
    """BGR-кадр -> список сегментов в НОРМАЛИЗОВАННЫХ координатах (0..1)."""
    H, W = img.shape[:2]
    scale = WORK_WIDTH / W
    small = cv2.resize(img, (int(W * scale), int(H * scale)),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    min_len = MIN_LEN_FRAC * sw

    lsd = cv2.createLineSegmentDetector()
    raw, _, _, _ = lsd.detect(gray)

    horiz, vert, diag = [], [], []
    if raw is not None:
        for l in raw:
            x1, y1, x2, y2 = l[0]
            if math.hypot(x2 - x1, y2 - y1) < min_len:
                continue
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            is_h = ang < ANGLE_TOL or ang > 180 - ANGLE_TOL
            is_v = abs(ang - 90) < ANGLE_TOL
            if not (is_h or is_v) and not KEEP_DIAGONAL:
                continue
            con = _edge_contrast(gray, x1, y1, x2, y2, sw, sh)
            if con < MIN_CONTRAST:
                continue
            if is_h:
                yc = (y1 + y2) / 2
                horiz.append((yc, min(x1, x2), max(x1, x2), con))
            elif is_v:
                xc = (x1 + x2) / 2
                vert.append((xc, min(y1, y2), max(y1, y2), con))
            else:
                diag.append((x1, y1, x2, y2, con))

    y_tol = Y_TOL_FRAC * sh
    x_tol = Y_TOL_FRAC * sw
    gap = GAP_FRAC * sw
    merged_h = _merge_axis(horiz, y_tol, gap)
    merged_v = _merge_axis(vert, x_tol, gap)

    segs = []
    for yc, xa, xb, con in merged_h:
        if (xb - xa) < min_len:
            continue
        segs.append(_seg(xa, yc, xb, yc, "H", con, sw, sh))
    for xc, ya, yb, con in merged_v:
        if (yb - ya) < min_len:
            continue
        segs.append(_seg(xc, ya, xc, yb, "V", con, sw, sh))
    for x1, y1, x2, y2, con in diag:               # только если KEEP_DIAGONAL
        segs.append(_seg(x1, y1, x2, y2, "D", con, sw, sh))

    # длинные и контрастные — вперёд (полезнее для кота)
    segs.sort(key=lambda s: s["len"] * (1 + s["contrast"] / 255), reverse=True)
    return segs, W, H


def _seg(x1, y1, x2, y2, o, con, sw, sh):
    # приводим к обычному float: LSD/numpy отдаёт float32, а он не сериализуется в JSON
    nx1, ny1, nx2, ny2 = float(x1) / sw, float(y1) / sh, float(x2) / sw, float(y2) / sh
    return {
        "x1": round(nx1, 4), "y1": round(ny1, 4),
        "x2": round(nx2, 4), "y2": round(ny2, 4),
        "o": o,
        "len": round(math.hypot(nx2 - nx1, ny2 - ny1), 4),
        "contrast": round(float(con), 1),
    }


def capture_loop():
    """Фоновый поток: снимает и детектит каждые INTERVAL секунд."""
    while True:
        t0 = time.time()
        img = grab_screen()
        if img is not None:
            try:
                segs, w, h = detect(img)
                with _LOCK:
                    _LATEST.update(ts=round(time.time(), 3), w=w, h=h,
                                   count=len(segs), ms=round((time.time() - t0) * 1000, 1),
                                   segments=segs)
                nh = sum(1 for s in segs if s["o"] == "H")
                nv = sum(1 for s in segs if s["o"] == "V")
                print(f"[det] {len(segs)} линий (H={nh} V={nv}) за "
                      f"{_LATEST['ms']:.0f}мс", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[err] detect: {e}", flush=True)
        dt = time.time() - t0
        time.sleep(max(0.0, INTERVAL - dt))


def _json_default(o):
    # numpy-скаляры (float32/int64 из LSD) json не умеет — приводим к python-числам
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not serializable: {type(o).__name__}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False,
                          default=_json_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            with _LOCK:
                return self._send(200, {"ok": True, "age": round(time.time() - _LATEST["ts"], 1)
                                        if _LATEST["ts"] else None, "count": _LATEST["count"]})
        if path in ("/edges", "/"):
            with _LOCK:
                return self._send(200, dict(_LATEST))
        return self._send(404, {"error": "use /edges or /health"})

    def log_message(self, *args):
        pass


def snapshot():
    """Разовый прогон: снять, задетектить, отрисовать зелёные линии в out/preview.png."""
    img = grab_screen()
    if img is None:
        print("не удалось снять экран", file=sys.stderr)
        sys.exit(1)
    segs, W, H = detect(img)
    scale = WORK_WIDTH / W
    canvas = cv2.resize(img, (int(W * scale), int(H * scale)),
                        interpolation=cv2.INTER_AREA)
    sh, sw = canvas.shape[:2]
    for s in segs:
        p1 = (int(s["x1"] * sw), int(s["y1"] * sh))
        p2 = (int(s["x2"] * sw), int(s["y2"] * sh))
        col = (0, 255, 0) if s["o"] == "H" else (0, 200, 255)  # H зелёный, V жёлто-зелёный
        cv2.line(canvas, p1, p2, col, 2)
    out = os.path.join(OUT_DIR, "preview.png")
    cv2.imwrite(out, canvas)
    nh = sum(1 for s in segs if s["o"] == "H")
    nv = sum(1 for s in segs if s["o"] == "V")
    print(f"{len(segs)} линий (H={nh} V={nv}); сохранил {out}")


def main():
    if "--snapshot" in sys.argv or "--once" in sys.argv:
        snapshot()
        return
    threading.Thread(target=capture_loop, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] слушаю http://{HOST}:{PORT}  (интервал {INTERVAL}с, Ctrl+C — стоп)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] остановлен", flush=True)


if __name__ == "__main__":
    main()
