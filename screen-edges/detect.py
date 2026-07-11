#!/usr/bin/env python3
"""Детектор длинных линий интерфейса (горизонт/вертикаль) — «полки и стены» кота.

Каждые SE_INTERVAL секунд снимает экран нативным `screencapture`, находит длинные
ГОРИЗОНТАЛЬНЫЕ (H — по ним кот ходит) и ВЕРТИКАЛЬНЫЕ (V — по ним лезет) линии
интерфейса и отдаёт их по HTTP. Диагонали не ищем — только сетка интерфейса.

Пайплайн — морфология (надёжна на интерфейсах: панели, вкладки, карточки, меню):
    grayscale
      → adaptive threshold (обе полярности: и тёмные линии, и светлые)
      → морф-открытие горизонтальным / вертикальным ядром  (оставляет только H/V)
      → компоненты связности                                 (каждая = отрезок)
      → фильтры: длина ≥15%, вытянутость, ОДНОРОДНОСТЬ ВДОЛЬ (иначе это текст),
                 контраст ПОПЕРЁК (линия выделяется на фоне)
      → слияние коллинеарных кусков с малым разрывом

Ключевой фильтр текста: у сплошного разделителя яркость вдоль линии однородна
(низкий std), у строки текста — прыгает (буквы/пробелы). Так линии отделяются
от текста, из-за которого наивная морфология обводит каждую строку.

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
MIN_LEN_FRAC = float(os.environ.get("SE_MIN_LEN_FRAC", "0.40"))  # мин. длина = доля стороны
MIN_CONTRAST = float(os.environ.get("SE_MIN_CONTRAST", "8"))  # мин. контраст поперёк (0..255)
ALONG_STD_MAX = float(os.environ.get("SE_ALONG_STD", "22"))   # макс. разброс ВДОЛЬ (отсев текста)
MIN_ASPECT = float(os.environ.get("SE_MIN_ASPECT", "3"))      # вытянутость компонента (длина/толщина)
ADAPT_BLOCK = int(os.environ.get("SE_ADAPT_BLOCK", "15"))     # окно adaptive threshold (нечёт.)
MORPH_FRAC = float(os.environ.get("SE_MORPH_FRAC", "0.05"))   # длина морф-ядра = доля стороны
Y_TOL_FRAC = float(os.environ.get("SE_Y_TOL_FRAC", "0.004"))  # слияние: разброс поперёк
GAP_FRAC = float(os.environ.get("SE_GAP_FRAC", "0.005"))      # слияние: макс. разрыв вдоль

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
    """Заметность линии = max(step, ridge).

    Меряем яркость на самой линии (offset 0) и по обе стороны (±3px поперёк):
      • step  = |слева − справа|         — перепад на ГРАНИЦЕ двух областей;
      • ridge = |линия − среднее сторон|  — насколько сама ЛИНИЯ ярче/темнее фона.
    Берём максимум: step ловит контурные грани, ridge — тонкие яркие/тёмные
    полоски (у них фон с обеих сторон одинаков, поэтому step≈0 и один step их
    терял). Возвращаем одну величину 0..255."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L                       # единичная нормаль
    n = 24
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)

    def side(off):
        px = np.clip((xs + nx * off).astype(int), 0, sw - 1)
        py = np.clip((ys + ny * off).astype(int), 0, sh - 1)
        return gray[py, px].astype(float).mean()

    a, b, c = side(3), side(-3), side(0)
    step = abs(a - b)
    ridge = abs(c - (a + b) / 2)
    return max(step, ridge)


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


def _along_std(gray, a, b, const, horizontal, sw, sh):
    """Разброс яркости ВДОЛЬ линии. У сплошного разделителя он мал, у строки
    текста велик (буквы/пробелы чередуются) — так отсеиваем текст."""
    if horizontal:
        xs = np.clip(np.arange(int(a), int(b)), 0, sw - 1)
        y = int(np.clip(const, 0, sh - 1))
        vals = gray[y, xs]
    else:
        ys = np.clip(np.arange(int(a), int(b)), 0, sh - 1)
        x = int(np.clip(const, 0, sw - 1))
        vals = gray[ys, x]
    return float(vals.std()) if vals.size > 2 else 999.0


def _line_mask(bw, horizontal, sw, sh):
    """Морфологическое открытие: оставляет только длинные H- или V-структуры."""
    if horizontal:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(MORPH_FRAC * sw)), 1))
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(MORPH_FRAC * sh))))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)


def _mask_components(mask, horizontal):
    """Компоненты связности маски -> список (const, a, b) в рабочих пикселях.
    Для H: const=y, [a,b]=[x..x+w]; для V: const=x, [a,b]=[y..y+h].
    Отсекает невытянутые кляксы (не линии)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if horizontal:
            if w < MIN_ASPECT * max(h, 1):
                continue
            out.append((y + h / 2, float(x), float(x + w)))
        else:
            if h < MIN_ASPECT * max(w, 1):
                continue
            out.append((x + w / 2, float(y), float(y + h)))
    return out


def detect(img):
    """BGR-кадр -> список сегментов (только H/V) в НОРМАЛИЗОВАННЫХ коорд. (0..1).

    Пайплайн (морфология — надёжна на интерфейсах):
    grayscale → adaptive threshold (обе полярности) → морф-открытие H/V ядром →
    компоненты связности → фильтр (длина ≥15%, вытянутость, однородность вдоль=
    не текст, контраст поперёк) → слияние коллинеарных."""
    H, W = img.shape[:2]
    scale = WORK_WIDTH / W
    small = cv2.resize(img, (int(W * scale), int(H * scale)),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape

    # Бинаризация обеих полярностей: ловим и тёмные линии на светлом фоне, и
    # светлые на тёмном (работает и на светлой, и на тёмной теме).
    blk = ADAPT_BLOCK if ADAPT_BLOCK % 2 else ADAPT_BLOCK + 1
    bw = cv2.bitwise_or(
        cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, blk, -2),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, blk, -2))

    h_items = _mask_components(_line_mask(bw, True, sw, sh), True)
    v_items = _mask_components(_line_mask(bw, False, sw, sh), False)

    gap = GAP_FRAC * sw
    merged_h = _merge_axis([(c, a, b, 0) for c, a, b in h_items], Y_TOL_FRAC * sh, gap)
    merged_v = _merge_axis([(c, a, b, 0) for c, a, b in v_items], Y_TOL_FRAC * sw, gap)

    min_len_h = MIN_LEN_FRAC * sw
    min_len_v = MIN_LEN_FRAC * sh
    segs = []
    for yc, xa, xb, _ in merged_h:
        if (xb - xa) < min_len_h:
            continue
        if _along_std(gray, xa, xb, yc, True, sw, sh) > ALONG_STD_MAX:   # это текст
            continue
        con = _edge_contrast(gray, xa, yc, xb, yc, sw, sh)
        if con < MIN_CONTRAST:
            continue
        segs.append(_seg(xa, yc, xb, yc, "H", con, sw, sh))
    for xc, ya, yb, _ in merged_v:
        if (yb - ya) < min_len_v:
            continue
        if _along_std(gray, ya, yb, xc, False, sw, sh) > ALONG_STD_MAX:
            continue
        con = _edge_contrast(gray, xc, ya, xc, yb, sw, sh)
        if con < MIN_CONTRAST:
            continue
        segs.append(_seg(xc, ya, xc, yb, "V", con, sw, sh))

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
