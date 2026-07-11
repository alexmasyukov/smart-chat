#!/usr/bin/env python3
"""Persistent-сервер OmniVoice: модель грузится ОДИН раз и живёт в памяти.

Дальше любые тексты синтезируются без перезагрузки весов.

Старт (в фоне):
    python server.py            # слушает http://127.0.0.1:8123

Генерация:
    curl -s "http://127.0.0.1:8123/gen?out=out/x.wav" --data-urlencode \
        "text=Текст голосом Lily" -G && open out/x.wav

Или через клиент:  python say.py "Текст"
"""
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO = os.path.join(HERE, "ref", "lily.wav")
OUT_DIR = os.path.join(HERE, "out")
HOST, PORT = "127.0.0.1", 8123

os.makedirs(OUT_DIR, exist_ok=True)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


DEVICE = pick_device()
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

# Дефолт шагов диффузии: 16 — вдвое быстрее 32 при почти том же качестве
DEFAULT_NUM_STEP = 16

print(f"[server] device={DEVICE} dtype={DTYPE}", flush=True)
print("[server] загружаю модель ОДИН раз ...", flush=True)
_t = time.time()
MODEL = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEVICE, dtype=DTYPE)
print(f"[server] модель в памяти за {time.time() - _t:.1f}с", flush=True)

# Кешируем VoiceClonePrompt по пути референса: Whisper-транскрипция и
# кодирование референса делаются ОДИН раз, а не на каждый вызов generate().
_PROMPT_CACHE: dict[str, object] = {}


def get_prompt(ref: str):
    ref = os.path.abspath(ref)
    if ref not in _PROMPT_CACHE:
        t0 = time.time()
        _PROMPT_CACHE[ref] = MODEL.create_voice_clone_prompt(ref_audio=ref)
        print(f"[prompt] закешировал референс {os.path.basename(ref)} "
              f"за {time.time() - t0:.1f}с "
              f"(ref_text='{_PROMPT_CACHE[ref].ref_text[:60]}...')", flush=True)
    return _PROMPT_CACHE[ref]


# Прогреваем дефолтный референс сразу при старте
get_prompt(REF_AUDIO)
print("[server] референс закеширован — дальше без Whisper на каждый вызов", flush=True)


# Фикс обрезки коротких текстов (см. research):
#  - конец «съедается» постобработкой (fade_and_pad ~0.1с + remove_silence).
#    Лечим: fade_duration=0.02, terminal-пунктуация, speed<1 для коротких.
#  - на длинных текстах ничего не замедляем (speed как пришёл, guidance по умолч.)
SHORT_CHARS = 60          # порог «короткого» текста
SHORT_SPEED = 0.85        # медленнее -> последнее слово выходит из зоны fade
SHORT_GUIDANCE = 3.0      # сильнее держит текст на коротких


def synth(text: str, out: str, ref: str, num_step: int, speed: float) -> dict:
    prompt = get_prompt(ref)
    # terminal-пунктуация: без неё модель чаще роняет последнее слово
    gen_text = text if text.rstrip()[-1:] in ".!?…" else text.rstrip() + "."
    is_short = len(text) < SHORT_CHARS

    cfg = dict(num_step=num_step, fade_duration=0.02)
    gen_speed = speed
    if is_short:
        cfg["guidance_scale"] = SHORT_GUIDANCE
        gen_speed = min(speed, SHORT_SPEED)
    gc = OmniVoiceGenerationConfig(**cfg)

    t0 = time.time()
    audio = MODEL.generate(text=gen_text, voice_clone_prompt=prompt,
                           speed=gen_speed, generation_config=gc)
    dt = time.time() - t0
    wav = audio[0]
    dur = len(wav) / 24000
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    sf.write(out, wav, 24000)
    return {"out": out, "gen_sec": round(dt, 2), "audio_sec": round(dur, 2),
            "rtf": round(dt / dur, 3), "short": is_short}


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
            return self._send(200, {"ok": True, "device": DEVICE})
        if parsed.path != "/gen":
            return self._send(404, {"error": "use /gen or /health"})
        p = self._params()
        text = p.get("text", "").strip()
        if not text:
            return self._send(400, {"error": "missing 'text'"})
        out = p.get("out") or os.path.join("out", f"clone_{int(time.time())}.wav")
        try:
            res = synth(
                text=text,
                out=out,
                ref=p.get("ref", REF_AUDIO),
                num_step=int(p.get("num_step", DEFAULT_NUM_STEP)),
                speed=float(p.get("speed", 1.0)),
            )
            print(f"[gen] «{text[:60]}» -> {res['out']} "
                  f"({res['gen_sec']}с, rtf={res['rtf']})")
            self._send(200, res)
        except Exception as e:  # noqa: BLE001
            print(f"[err] {e}")
            self._send(500, {"error": str(e)})

    def log_message(self, *args):  # тише в консоли
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] слушаю http://{HOST}:{PORT}  (Ctrl+C для остановки)")
    srv.serve_forever()
