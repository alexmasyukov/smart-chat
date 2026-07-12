#!/usr/bin/env python3
"""Детекция по СЕТКЕ в левых 30% экрана. РУЧНОЙ триггер: снимок и детект
происходят только по запросу GET /scan (кнопка на оверлее), не по таймеру.

Сканируем ОДНУ строку (y = ROW_FRAC * H) с шагом STEP от X_OFFSET, читаем
цвет в каждой точке сетки (медиана патча). БЛОК = прогон из >= MIN_RUN точек
подряд одного цвета; граница рисуется только МЕЖДУ двумя такими блоками, не
по первой сломанной точке (край/шум/узкая полоса игнорируются). Текст внутри
блока распознаём вертикальным столбиком проб (VPROBES сверху и снизу с тем же
шагом): большинство проб цвета блока — значит текст, лечим и не рвём прогон.
Границу уточняем бисекцией ("probe"). Пробы ("vprobe") тоже видны.

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
    """Срединный (медианный) цвет патча PATCH×PATCH вокруг (x, y), BGR.
    Медиана вместо среднего: одиночный выброс-пиксель (тонкая линия, край буквы)
    не утягивает цвет точки, как утягивало среднее."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, x0 + PATCH), min(H, y0 + PATCH)
    roi = img[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(roi, axis=0)


def color_hex(c):
    """BGR-массив -> "rrggbb"."""
    b, g, r = c
    return f"{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def same_color(c1, c2):
    """Точное совпадение цвета (по hex-значению после округления), без допуска."""
    return bool(np.all(np.round(c1) == np.round(c2)))


VPROBES = int(os.environ.get("SG_VPROBES", "2"))                     # проб сверху и столько же снизу от i+1
MIN_RUN = int(os.environ.get("SG_MIN_RUN", "3"))                     # мин. точек подряд одного цвета = блок
BISECT_MAX = int(os.environ.get("SG_BISECT_MAX", "3"))               # макс. уточнений границы
BISECT_MIN_GAP = float(os.environ.get("SG_BISECT_MIN_GAP", "5"))     # px — дальше не мельчим


def scan_row(img, y, x_end, bisect_max=BISECT_MAX, step=STEP):
    """Сканируем строку y слева направо с шагом step от X_OFFSET до x_end.

    Блок = ПРОГОН из >= MIN_RUN точек подряд одного цвета. Граница рисуется
    только МЕЖДУ двумя такими блоками — не по первой же сломанной точке. Если
    точка 0 отличается от 1,2,3,4, а 1,2,3,4 одного цвета, то блок начинается
    с точки 1, а точка 0 — край/шум, отбрасывается.

    Фаза 1. Идём слева направо, растим прогон одного цвета. Когда сосед j
    другого цвета — это может быть ТЕКСТ внутри блока: ставим у j ВЕРТИКАЛЬНЫЙ
    столбик проб (VPROBES вверх и вниз с тем же шагом step, далеко по вертикали
    — мимо строки текста). Если большинство проб — цвета текущего блока, значит
    j это текст: лечим и вливаем в прогон. Иначе прогон закрывается, начинается
    новый. Столбик проб — настоящие точки (kind="vprobe"), видны и в логе.

    Фаза 2. Граница = правый край ПЕРВОГО прогона длиной >= MIN_RUN, но только
    если дальше есть ЕЩЁ прогон >= MIN_RUN (граница между двумя настоящими
    блоками, а не сползание в шум/край региона). Уточняем бисекцией между этой
    точкой i (ещё цвет блока) и i+1: точка посередине, совпала с цветом блока —
    граница правее, делим (эта точка..i+1), иначе левее (i..эта точка). До
    BISECT_MAX раз или пока отрезок не сузится до BISECT_MIN_GAP px.

    Номер точки = порядок ЕЁ ОБНАРУЖЕНИЯ; в массиве и логе точки стоят по месту
    на экране (x, затем y)."""
    H = img.shape[0]
    xs_grid = list(np.arange(X_OFFSET, x_end, step, dtype=np.float64))
    colors_grid = [color_at(img, x, y) for x in xs_grid]
    n = len(xs_grid)

    entries = [{"num": i, "x": xs_grid[i], "y": y, "color": colors_grid[i], "kind": "base"}
               for i in range(n)]
    next_num = n
    lines = []

    def is_text(j, block_color):
        """Вертикальный столбик проб у точки j (VPROBES вверх и вниз, шаг step):
        True, если большинство проб — цвета block_color (j это текст в блоке)."""
        nonlocal next_num
        x_susp = xs_grid[j]
        same = total = 0
        for k in list(range(-VPROBES, 0)) + list(range(1, VPROBES + 1)):
            vy = y + k * step
            if vy < 0 or vy >= H:
                continue
            vcolor = color_at(img, x_susp, vy)
            entries.append({"num": next_num, "x": x_susp, "y": vy, "color": vcolor, "kind": "vprobe"})
            next_num += 1
            total += 1
            if same_color(vcolor, block_color):
                same += 1
        return total > 0 and same * 2 >= total

    # Фаза 1 + 2: растим прогоны, ищем правый край первого блока (>= MIN_RUN),
    # за которым идёт ещё один блок (>= MIN_RUN).
    run_color = colors_grid[0]
    start = 0
    first_block_end = None        # правый край первого подтверждённого блока
    boundary_i = None             # точка i, слева от которой граница
    j = 1
    while j < n:
        if same_color(colors_grid[j], run_color) or is_text(j, run_color):
            colors_grid[j] = run_color
            if first_block_end is not None and j - start + 1 >= MIN_RUN:
                boundary_i = first_block_end        # второй блок подтверждён -> стоп
                break
            j += 1
            continue
        if j - start >= MIN_RUN and first_block_end is None:
            first_block_end = j - 1                  # закрыли первый блок >= MIN_RUN
        run_color = colors_grid[j]                   # начинаем новый прогон
        start = j
        j += 1

    if boundary_i is not None:
        i = boundary_i
        run_color = colors_grid[i]
        lo_x, hi_x = xs_grid[i], xs_grid[i + 1]
        boundary_x = lo_x                     # последняя точка ЕЩЁ цвета блока
        for _ in range(bisect_max):
            if hi_x - lo_x <= BISECT_MIN_GAP:
                break
            mid_x = (lo_x + hi_x) / 2.0
            mid_color = color_at(img, mid_x, y)
            entries.append({"num": next_num, "x": mid_x, "y": y, "color": mid_color, "kind": "probe"})
            next_num += 1
            if same_color(mid_color, run_color):
                lo_x = mid_x
                boundary_x = mid_x            # блок продолжается дальше — сдвигаем границу
            else:
                hi_x = mid_x
        lines.append(boundary_x)

    entries.sort(key=lambda e: (e["x"], e["y"]))   # по месту на экране; номер — как присвоен
    xs = [e["x"] for e in entries]
    ys = [e["y"] for e in entries]
    colors = [e["color"] for e in entries]
    kinds = [e["kind"] for e in entries]
    numbers = [e["num"] for e in entries]
    return xs, ys, colors, kinds, numbers, lines


def detect(img, bisect_max=BISECT_MAX, step=STEP):
    """BGR-кадр -> (points, colors_hex, kinds, numbers, lines).

    points — точки сканирования (сетка + проверочные) в НОРМАЛИЗОВАННЫХ
    коорд. (0..1, y сверху), отсортированы по месту на экране (слева
    направо). colors_hex — цвет каждой точки в hex, тот же порядок.
    kinds — "base" | "probe" (точка бисекции границы) | "vprobe" (вертикальная
    проба сверху/снизу от i+1, красится иначе). numbers — номер точки (порядок обнаружения,
    НЕ порядок в массиве) — им подписывается точка и пишется лог.
    lines — нормализованный x уточнённой границы блока (если нашли)."""
    H, W = img.shape[:2]
    y = H * ROW_FRAC
    x_end = W * REGION_FRAC
    xs, ys, colors, kinds, numbers, boundary_xs = scan_row(img, y, x_end, bisect_max=bisect_max, step=step)

    points = [[round(px / W, 4), round(py / H, 4)] for px, py in zip(xs, ys)]
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


def do_scan(bisect_max=BISECT_MAX, step=STEP):
    """Один проход: снимок экрана (out/screenshot.png) -> детект ->
    публикация в _LATEST -> лог out/points_<метка_времени>.txt."""
    t0 = time.time()
    img = grab_screen()
    if img is None:
        return None
    points, colors_hex, kinds, numbers, lines = detect(img, bisect_max=bisect_max, step=step)
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
            qs = urllib.parse.parse_qs(parsed.query)
            bisect_raw = qs.get("bisect", [""])[0]
            bisect_max = int(bisect_raw) if bisect_raw.isdigit() else BISECT_MAX
            step_raw = qs.get("step", [""])[0]
            step = int(step_raw) if step_raw.isdigit() else STEP
            result = do_scan(bisect_max=bisect_max, step=step)
            if result is None:
                return self._send(500, {"error": "screencapture failed"})
            return self._send(200, {"ok": True, "count": result["count"], "v": result["v"],
                                     "numbers": result["numbers"], "colors": result["colors"]})
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
    kind_color = {"base": (0, 255, 0), "probe": (0, 140, 255), "vprobe": (255, 200, 0)}
    for num, (nx, ny), kind in zip(numbers, points, kinds):
        px, py = int(nx * W), int(ny * H)
        color = kind_color.get(kind, (0, 255, 0))   # base зелёный, probe оранж., vprobe голубой
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
