#!/usr/bin/env python3
"""Piper TTS (ru_RU-irina-medium) — быстрый русский синтез на CPU, без сервера.

Piper держит модель в ONNX и грузится мгновенно, поэтому сервер не нужен —
каждый вызов самодостаточен и быстр.

Ударения: '+' перед ударной гласной работает только в espeak-фонемайзере;
у irina (RHVoice) ударения автоматические, спецсимволы не нужны.

Запуск:
    python say.py "Сделано, всего 24 компонента"
    python say.py "Текст" --out out/my.wav --length 1.0 --no-open
"""
import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

from piper import PiperVoice

HERE = Path(__file__).resolve().parent
MODEL = HERE / "voices" / "ru_RU-irina-medium.onnx"
OUT = HERE / "out"


def play(path: str) -> None:
    # afplay — системный проигрыватель, играет сразу через динамики, без окна
    subprocess.run(["afplay", path], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=int, choices=range(1, 11), default=7,
                    metavar="1-10",
                    help="скорость 1..10 (=length_scale 0.1..1.0; больше=медленнее). Дефолт 7")
    ap.add_argument("--length", type=float, default=None,
                    help="точный length_scale (переопределяет --speed)")
    ap.add_argument("--noise", type=float, default=0.667)
    ap.add_argument("--noise-w", type=float, default=0.8)
    ap.add_argument("--play", action="store_true",
                    help="проиграть сразу через afplay (без окна плеера)")
    a = ap.parse_args()

    if not MODEL.exists():
        sys.exit(f"Нет модели: {MODEL}\n"
                 f"Скачай: python -m piper.download_voices ru_RU-irina-medium --data-dir voices")

    OUT.mkdir(exist_ok=True)
    out_path = Path(a.out) if a.out else OUT / f"piper_{int(time.time())}.wav"

    t0 = time.time()
    voice = PiperVoice.load(str(MODEL))
    load_s = time.time() - t0

    length_scale = a.length if a.length is not None else a.speed / 10.0

    from piper import SynthesisConfig
    syn = SynthesisConfig(length_scale=length_scale, noise_scale=a.noise,
                          noise_w_scale=a.noise_w)

    t0 = time.time()
    with wave.open(str(out_path), "wb") as wf:
        voice.synthesize_wav(a.text, wf, syn_config=syn)
    gen_s = time.time() - t0

    with wave.open(str(out_path), "rb") as wf:
        dur = wf.getnframes() / wf.getframerate()
    print(f"[load] {load_s:.2f}s  [gen] {gen_s:.2f}s  audio={dur:.2f}s  "
          f"RTF={gen_s/dur:.3f}  len={length_scale:.2f}  -> {out_path}",
          file=sys.stderr)
    print(out_path)
    if a.play:
        play(str(out_path))


if __name__ == "__main__":
    main()
