#!/usr/bin/env python3
"""Supertone/supertonic-3 — быстрый on-device TTS (ONNX Runtime, CPU).

Supertonic-3 (~99M) поддерживает 31 язык, включая русский (lang="ru").
Работает на Apple Silicon через ONNX Runtime CPU (нет CUDA/MPS — чистый CPU).

Клонирование голоса: в open-source SDK НЕ входит. Доступны 10 пресетов
(M1..M5, F1..F5). Свой голос из референс-wav требует проприетарного
Supertone Voice Builder, который отдаёт .json со style-векторами; такой json
можно скормить через --style-json. Пресеты выбираются через --voice.

Этот скрипт грузит модель на каждый вызов (~0.2с). Для частых вызовов держи
поднятым server.py (порт 8126) и зови его через curl — см. say.sh и README.

Ударения: «+» перед ударной гласной ставит акут (за́мок ≠ замо́к) — нужно только
на спорных словах, обычные модель озвучивает сама. «ё» -> «е» (нет в наборе
символов); где слышится «э» — писать «э» (сэрвер, нэтворк), а не «е».

Запуск:
    python say.py "Привет, как дела?"
    python say.py "Текст" --voice M1 --speed 1.2 --steps 4 --play
    python say.py "Зам+ок открыт"                 # ударение на 2-й слог
    python say.py "Текст" --style-json path/to/voice.json
"""
import argparse
import re
import subprocess
import time
from pathlib import Path

from supertonic import TTS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# Ручные ударения: «+» перед ударной гласной -> комбинирующий акут U+0301
# (Supertonic его слушается: за́мок ≠ замо́к). «ё» не в наборе модели -> «е».
_VOWELS = "аеиоуыэюяАЕИОУЫЭЮЯ"


def apply_stress(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\+([" + _VOWELS + r"])", r"\1́", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--voice", default="F1",
                    help="пресет: M1..M5, F1..F5 (по умолчанию F1)")
    ap.add_argument("--style-json", default=None,
                    help="путь к voice-style .json (из Voice Builder)")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--steps", type=int, default=6, help="шаги диффузии (меньше=быстрее)")
    ap.add_argument("--speed", type=float, default=1.05,
                    help="множитель темпа: больше = быстрее (0.8 медленно, 1.4 быстро)")
    ap.add_argument("--play", action="store_true",
                    help="проиграть сразу через afplay (без окна плеера)")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tag = Path(a.style_json).stem if a.style_json else a.voice
    out_path = Path(a.out) if a.out else OUT / f"supertonic_{tag}_{int(time.time())}.wav"

    t0 = time.time()
    tts = TTS(model="supertonic-3", auto_download=True)
    if a.style_json:
        style = tts.get_voice_style_from_path(a.style_json)
    else:
        style = tts.get_voice_style(a.voice)
    load_s = time.time() - t0

    t0 = time.time()
    wav, dur = tts.synthesize(apply_stress(a.text), voice_style=style,
                              total_steps=a.steps, speed=a.speed, lang=a.lang)
    gen_s = time.time() - t0

    tts.save_audio(wav, str(out_path))
    audio_s = wav.size / 44100.0  # wav имеет форму (1, N)

    print(f"[cpu/onnx] voice={tag} lang={a.lang} steps={a.steps} speed={a.speed}")
    print(f"[load] {load_s:.2f}s  [gen] {gen_s:.2f}s  audio={audio_s:.2f}s  "
          f"RTF={gen_s/audio_s:.3f}")
    print(out_path)
    if a.play:
        subprocess.run(["afplay", str(out_path)], check=False)


if __name__ == "__main__":
    main()
