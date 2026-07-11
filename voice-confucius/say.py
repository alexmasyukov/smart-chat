#!/usr/bin/env python3
"""Confucius4-TTS (netease-youdao) — многоязычный zero-shot TTS с клонированием
голоса, MLX-порт для Apple Silicon.

Веса: mlx-community/Confucius4-TTS-mlx-int8 (int8, ~2.8 ГБ). Пакет: mlx-audio.
14 языков, включая русский. Клонирование голоса — из референс-wav, без
транскрипта референса (unconstrained voice cloning). Выход 22050 Гц.

ГРАБЛИ: LANGUAGE_TOKEN в mlx-audio-порте — это ПОДМНОЖЕСТВО (zh/en/vi/ja/ko/th),
русского в нём нет, и lang="ru" молча падает на английский instruction-токен.
Ниже мы добавляем русский токен в словарь до генерации.

Запуск:
    python say.py "Привет, как дела?"
    python say.py "Текст" --ref ref/lily.wav --lang ru
"""
import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REPO = "mlx-community/Confucius4-TTS-mlx-int8"
DEFAULT_REF = HERE / "ref" / "lily_3s.wav"   # 3с дешевле 9с (ref ре-энкодится)
SERVER = "http://127.0.0.1:8125"


def via_server(text: str, out: Path, ref: str, lang: str) -> dict:
    data = urllib.parse.urlencode({"text": text, "out": str(out),
                                   "ref": ref, "lang": lang}).encode()
    with urllib.request.urlopen(f"{SERVER}/gen", data=data, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=str(DEFAULT_REF),
                    help="референс-wav для клонирования голоса")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--rep-pen", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server", action="store_true",
                    help="генерить через тёплый сервер (server.py, порт 8125)")
    ap.add_argument("--play", action="store_true",
                    help="проиграть сразу через afplay (без окна плеера)")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    out_path = Path(a.out) if a.out else OUT / f"confucius_{a.lang}_{int(time.time())}.wav"

    # Режим клиента: модель уже в памяти сервера, тёплый RTF ~1.8 без прогрева.
    if a.server:
        res = via_server(a.text, out_path, a.ref, a.lang)
        print(f"[server] gen={res['gen_sec']}s audio={res['audio_sec']}s "
              f"rtf={res['rtf']}")
        print(res["out"])
        if a.play:
            subprocess.run(["afplay", res["out"]], check=False)
        return

    # Порт mlx-audio знает только subset языков — дошиваем русский instruction-токен.
    from mlx_audio.tts.models.confucius4 import confucius4 as c4
    c4.LANGUAGE_TOKEN.setdefault("ru", "请用俄语朗读接下来的文字")

    from mlx_audio.tts.utils import load

    t0 = time.time()
    model = load(REPO)
    load_s = time.time() - t0

    t0 = time.time()
    wav = None
    sr = 22050
    for r in model.generate(
        a.text,
        ref_audio=a.ref,
        lang=a.lang,
        temperature=a.temperature,
        top_k=a.top_k,
        top_p=a.top_p,
        repetition_penalty=a.rep_pen,
        seed=a.seed,
    ):
        wav = np.array(r.audio).reshape(-1)
        sr = r.sample_rate
    gen_s = time.time() - t0

    sf.write(str(out_path), wav, sr)
    audio_s = wav.shape[0] / float(sr)

    print(f"[mlx/int8] ref={Path(a.ref).name} lang={a.lang} sr={sr}")
    print(f"[load] {load_s:.2f}s  [gen] {gen_s:.2f}s  audio={audio_s:.2f}s  "
          f"RTF={gen_s/audio_s:.3f}")
    print(out_path)
    if a.play:
        subprocess.run(["afplay", str(out_path)], check=False)


if __name__ == "__main__":
    main()
