#!/usr/bin/env python3
"""Persistent-сервер Confucius4-TTS: модель грузится ОДИН раз и живёт в памяти.

Зачем сервер (в отличие от разового say.py):
  - веса 2.6 ГБ не перечитываются на каждый вызов (экономит ~0.5с);
  - ГЛАВНОЕ: Metal-кернелы компилируются на ПЕРВОЙ генерации (холодный RTF ~7),
    сервер прогревается при старте одним фиктивным вызовом, и все последующие
    запросы идут с тёплым RTF ~1.8 — без повторной компиляции.

Старт (в фоне):
    python server.py            # слушает http://127.0.0.1:8125

Генерация:
    curl -s "http://127.0.0.1:8125/gen?out=out/x.wav" --data-urlencode \
        "text=Привет, как дела?" -G
Или клиент:  python say.py "Текст" --server
"""
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.utils import load_audio

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "mlx-community/Confucius4-TTS-mlx-int8"
# 3-секундный референс: он ре-энкодится на КАЖДЫЙ вызов (w2v-bert по ref-аудио),
# поэтому короткий референс почти вдвое дешевле 9-секундного (rtf 1.25 vs 3.45).
DEFAULT_REF = os.path.join(HERE, "ref", "lily_3s.wav")
OUT_DIR = os.path.join(HERE, "out")
HOST, PORT = "127.0.0.1", 8125

os.makedirs(OUT_DIR, exist_ok=True)

# Порт mlx-audio знает только subset языков — дошиваем русский instruction-токен
# ДО загрузки модели.
from mlx_audio.tts.models.confucius4 import confucius4 as c4  # noqa: E402
c4.LANGUAGE_TOKEN.setdefault("ru", "请用俄语朗读接下来的文字")

from mlx_audio.tts.utils import load  # noqa: E402

# MLX по умолчанию кеширует освобождённые Metal-буферы без лимита — за несколько
# генераций это раздувает unified-память до десятков ГБ (в ps не видно, только в
# Activity Monitor/top). Ограничиваем кэш и чистим его после каждого вызова.
MLX_CACHE_LIMIT = int(os.environ.get("MLX_CACHE_GB", "2")) * 1024**3
mx.set_cache_limit(MLX_CACHE_LIMIT)

print("[server] загружаю модель ОДИН раз ...", flush=True)
_t = time.time()
MODEL = load(REPO)
print(f"[server] модель в памяти за {time.time() - _t:.1f}с", flush=True)


# Кеш признаков референса. cond_vec/style/ref_mel зависят ТОЛЬКО от ref-аудио
# (не от текста), поэтому w2v-bert + CAMPPlus + ref-mel гоняются ОДИН раз на
# референс, а не на каждый вызов — это и был главный оверхед (rtf 3.4 -> ~1).
_REF_CACHE: dict[str, tuple] = {}


def ref_features(ref: str) -> tuple:
    ref = os.path.abspath(ref)
    if ref not in _REF_CACHE:
        t0 = time.time()
        audio = np.asarray(load_audio(ref, sample_rate=16000))
        feats = MODEL._fbank(mx.array(audio), MODEL._mel, MODEL._win)
        h17 = np.array(MODEL.w2v.hidden17(feats))
        cond_vec = mx.array((h17 - MODEL.stats["mean"]) / MODEL.stats["std"])
        style = mx.array(np.array(MODEL.camp.inference(mx.array(audio))).reshape(1, 192))
        ref_mel = mx.array(c4._ref_mel(audio))
        _REF_CACHE[ref] = (cond_vec, style, ref_mel)
        print(f"[ref] закешировал {os.path.basename(ref)} за {time.time() - t0:.2f}с",
              flush=True)
    return _REF_CACHE[ref]


def synth(text: str, out: str, ref: str, lang: str, temperature: float,
          top_k: int, top_p: float, rep_pen: float, seed: int) -> dict:
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    t0 = time.time()
    cond_vec, style, ref_mel = ref_features(ref)  # из кэша (кроме первого раза)

    # текстовая часть — единственное, что считается на каждый вызов
    lt = c4.LANGUAGE_TOKEN.get(lang, c4.LANGUAGE_TOKEN["en"])
    ids = MODEL._tok.encode(f"You are a helpful assistant. {lt}:{text}").ids
    cond_emb = MODEL.prefix.cond_emb(cond_vec)
    text_emb = MODEL.prefix.text_emb(mx.array([ids]))
    codes, latent = MODEL.t2s.generate(cond_emb, text_emb, temperature=temperature,
                                       top_k=top_k, top_p=top_p, rep_pen=rep_pen,
                                       seed=seed)
    T_ref = ref_mel.shape[1]
    mu = MODEL.s2a.build_mu(mx.array(codes[None]), mx.array(latent), T_ref)
    mx.random.seed(seed)
    z = mx.random.normal((1, 80, mu.shape[1]))
    mel = MODEL.s2a.solve_euler(z, mx.transpose(ref_mel, (0, 2, 1)), mu, style,
                                mx.linspace(0, 1, 26), cfg=0.7)[:, :, T_ref:]
    wav = MODEL.voc(mel)
    mx.eval(wav)
    wav = np.array(wav).reshape(-1)
    sr = MODEL.sample_rate

    dt = time.time() - t0
    dur = wav.shape[0] / float(sr)
    sf.write(out, wav, sr)
    peak_gb = mx.get_peak_memory() / 1024**3
    mx.clear_cache()   # отдать закешированные Metal-буферы обратно системе
    return {"out": out, "gen_sec": round(dt, 2), "audio_sec": round(dur, 2),
            "rtf": round(dt / dur, 3), "sr": sr, "peak_gb": round(peak_gb, 2)}


# Прогрев: компилируем Metal-кернелы, чтобы первый реальный запрос был тёплым.
print("[server] прогрев (компиляция Metal-кернелов) ...", flush=True)
_t = time.time()
try:
    _w = synth("Прогрев.", os.path.join(OUT_DIR, "_warmup.wav"), DEFAULT_REF,
               "ru", 0.8, 30, 0.8, 10.0, 0)
    print(f"[server] прогрет за {time.time() - _t:.1f}с "
          f"(следующий rtf ~{_w['rtf']})", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[server] прогрев пропущен: {e}", flush=True)


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
            return self._send(200, {"ok": True, "backend": "mlx/int8"})
        if parsed.path != "/gen":
            return self._send(404, {"error": "use /gen or /health"})
        p = self._params()
        text = p.get("text", "").strip()
        if not text:
            return self._send(400, {"error": "missing 'text'"})
        out = p.get("out") or os.path.join("out", f"confucius_{int(time.time())}.wav")
        try:
            res = synth(
                text=text,
                out=out,
                ref=p.get("ref", DEFAULT_REF),
                lang=p.get("lang", "ru"),
                temperature=float(p.get("temperature", 0.8)),
                top_k=int(p.get("top_k", 30)),
                top_p=float(p.get("top_p", 0.8)),
                rep_pen=float(p.get("rep_pen", 10.0)),
                seed=int(p.get("seed", 0)),
            )
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
