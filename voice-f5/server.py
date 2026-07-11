#!/usr/bin/env python3
"""Persistent-сервер F5-TTS_RUSSIAN: модель грузится ОДИН раз и живёт в памяти.

Ускорения без потери качества:
  - модель и вокодер не перезагружаются между вызовами (убирает ~load + прогрев)
  - ref_text (транскрипция референса) кешируется -> нет Whisper на каждый вызов
  - дефолт nfe_step=16 (одобренное качество, 2x быстрее дефолтных 32)

Старт:  python server.py            # http://127.0.0.1:8124
Клиент: python say.py "Текст"
"""
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
# Короткий референс 3с: F5 быстрее (короче диффузия-последовательность),
# качество/темп одобрены на слух. Полный 9с — ref/lily.wav.
REF = os.path.join(HERE, "ref", "lily_3s.wav")
CKPT = os.path.join(HERE, "ckpt", "model_212000.safetensors")
VOCAB = os.path.join(HERE, "ckpt", "vocab.txt")
OUT = os.path.join(HERE, "out")
HOST, PORT = "127.0.0.1", 8124
DEFAULT_NFE = 16
DEVICE = os.environ.get("F5_DEVICE", "mps")  # mps звучит чисто; cpu — надёжный фолбэк

os.makedirs(OUT, exist_ok=True)

from f5_tts.api import F5TTS

print(f"[server] device={DEVICE}  загружаю F5-TTS_RUSSIAN ОДИН раз ...", flush=True)
_t = time.time()
MODEL = F5TTS(model="F5TTS_v1_Base", ckpt_file=CKPT, vocab_file=VOCAB, device=DEVICE)
print(f"[server] модель в памяти за {time.time() - _t:.1f}с", flush=True)

# Кеш транскрипции референса: Whisper вызывается один раз на каждый ref
_REF_TEXT: dict[str, str] = {}


def ref_text_for(ref: str) -> str:
    ref = os.path.abspath(ref)
    if ref not in _REF_TEXT:
        t0 = time.time()
        _REF_TEXT[ref] = MODEL.transcribe(ref, language="ru")
        print(f"[ref] транскрибировал {os.path.basename(ref)} за {time.time()-t0:.1f}с "
              f"('{_REF_TEXT[ref][:50]}...')", flush=True)
    return _REF_TEXT[ref]


# прогрев: транскрипция дефолтного ref + один короткий синтез (инициализирует mps-ядра)
ref_text_for(REF)
print("[server] прогрев ...", flush=True)
MODEL.infer(ref_file=REF, ref_text=ref_text_for(REF), gen_text="Прогрев.",
            nfe_step=DEFAULT_NFE, show_info=lambda *a, **k: None)
print("[server] готов — референс закеширован, ядра прогреты", flush=True)


def synth(text: str, out: str, ref: str, nfe: int, speed: float) -> dict:
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    t0 = time.time()
    wav, sr, _ = MODEL.infer(
        ref_file=ref, ref_text=ref_text_for(ref), gen_text=text,
        nfe_step=nfe, speed=speed, file_wave=out, show_info=lambda *a, **k: None,
    )
    dt = time.time() - t0
    dur = len(wav) / sr
    return {"out": out, "gen_sec": round(dt, 2),
            "audio_sec": round(dur, 2), "rtf": round(dt / dur, 3)}


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
            if "json" in self.headers.get("Content-Type", ""):
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
            return self._send(200, {"ok": True, "device": DEVICE})
        if parsed.path != "/gen":
            return self._send(404, {"error": "use /gen or /health"})
        p = self._params()
        text = p.get("text", "").strip()
        if not text:
            return self._send(400, {"error": "missing 'text'"})
        out = p.get("out") or os.path.join("out", f"f5_{int(time.time())}.wav")
        try:
            res = synth(text=text, out=out, ref=p.get("ref", REF),
                        nfe=int(p.get("nfe", DEFAULT_NFE)),
                        speed=float(p.get("speed", 1.0)))
            print(f"[gen] «{text[:60]}» -> {res['out']} "
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
