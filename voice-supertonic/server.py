#!/usr/bin/env python3
"""Persistent-сервер Supertonic-3: модель и пресеты голосов держатся в памяти.

Модель грузится ~0.2с, но при одиночных вызовах `say.py` эти 0.2с платятся
каждый раз. Сервер грузит модель ОДИН раз и предзагружает все 10 пресетов,
поэтому вызов = чистая генерация (steps=4 → ~0.4с вместо ~0.65с с загрузкой).

Старт (в фоне):
    python server.py            # слушает http://127.0.0.1:8126

Генерация:
    curl -s "http://127.0.0.1:8126/gen?out=out/x.wav" --data-urlencode \
        "text=Привет" -G
Или клиент:  python say.py "Текст" --server
"""
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from supertonic import TTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
HOST, PORT = "127.0.0.1", 8126
VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]

os.makedirs(OUT_DIR, exist_ok=True)

print("[server] загружаю модель ОДИН раз ...", flush=True)
_t = time.time()
TTS_MODEL = TTS(model="supertonic-3", auto_download=True)
# предзагружаем все пресеты, чтобы смена --voice была мгновенной
STYLES = {v: TTS_MODEL.get_voice_style(v) for v in VOICES}
print(f"[server] модель + {len(STYLES)} пресетов в памяти за {time.time() - _t:.1f}с",
      flush=True)


def style_for(voice: str, style_json: str | None):
    if style_json:
        return TTS_MODEL.get_voice_style_from_path(style_json)
    return STYLES.get(voice) or TTS_MODEL.get_voice_style(voice)


def synth(text: str, out: str, voice: str, style_json: str | None, lang: str,
          steps: int, speed: float) -> dict:
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    style = style_for(voice, style_json)
    t0 = time.time()
    wav, _ = TTS_MODEL.synthesize(text, voice_style=style, total_steps=steps,
                                  speed=speed, lang=lang)
    dt = time.time() - t0
    TTS_MODEL.save_audio(wav, out)
    dur = wav.size / 44100.0
    return {"out": out, "gen_sec": round(dt, 3), "audio_sec": round(dur, 2),
            "rtf": round(dt / dur, 3)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _params(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        length = int(self.headers.get("Content-Length", 0))
        if length:
            raw = self.rfile.read(length).decode("utf-8")
            ctype = self.headers.get("Content-Type", "")
            if "json" in ctype:
                q.update(json.loads(raw))
            else:
                q.update({k: v[0] for k, v in urllib.parse.parse_qs(raw).items()})
        return q

    def do_GET(self):
        self._route()

    def do_POST(self):
        self._route()

    def _route(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            return self._send(200, {"ok": True, "voices": VOICES})
        if parsed.path != "/gen":
            return self._send(404, {"error": "use /gen or /health"})
        p = self._params()
        text = p.get("text", "").strip()
        if not text:
            return self._send(400, {"error": "missing 'text'"})
        out = p.get("out") or os.path.join("out", f"supertonic_{int(time.time())}.wav")
        try:
            res = synth(
                text=text,
                out=out,
                voice=p.get("voice", "F1"),
                style_json=p.get("style_json"),
                lang=p.get("lang", "ru"),
                steps=int(p.get("steps", 6)),
                speed=float(p.get("speed", 1.05)),
            )
            print(f"[gen] «{text[:50]}» -> {res['out']} "
                  f"({res['gen_sec']}с, rtf={res['rtf']})", flush=True)
            self._send(200, res)
        except Exception as e:  # noqa: BLE001
            print(f"[err] {e}", flush=True)
            self._send(500, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] слушаю http://{HOST}:{PORT}  (Ctrl+C для остановки)", flush=True)
    srv.serve_forever()
