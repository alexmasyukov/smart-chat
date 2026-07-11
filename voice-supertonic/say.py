#!/usr/bin/env python3
"""Supertone/supertonic-3 — быстрый on-device TTS (ONNX Runtime, CPU).

Supertonic-3 (~99M) поддерживает 31 язык, включая русский (lang="ru").
Работает на Apple Silicon через ONNX Runtime CPU (нет CUDA/MPS — чистый CPU).

Клонирование голоса: в open-source SDK НЕ входит. Доступны 10 пресетов
(M1..M5, F1..F5). Свой голос из референс-wav требует проприетарного
Supertone Voice Builder, который отдаёт .json со style-векторами; такой json
можно скормить через --style-json. Пресеты выбираются через --voice.

Standalone: модель грузится ~0.2с на каждый вызов. Для частых вызовов есть
сервер (server.py, порт 8126) — модель и пресеты в памяти, вызов = чистая
генерация без 0.2с загрузки: python say.py "Текст" --server

Запуск:
    python say.py "Привет, как дела?"
    python say.py "Текст" --voice M1 --speed 1.2 --steps 4 --play
    python say.py "Текст" --server            # через тёплый сервер
    python say.py "Текст" --style-json path/to/voice.json
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from supertonic import TTS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SERVER = "http://127.0.0.1:8126"


def via_server(text: str, out: Path, voice: str, style_json: str | None,
               lang: str, steps: int, speed: float) -> dict:
    q = {"text": text, "out": str(out), "voice": voice, "lang": lang,
         "steps": steps, "speed": speed}
    if style_json:
        q["style_json"] = style_json
    data = urllib.parse.urlencode(q).encode()
    with urllib.request.urlopen(f"{SERVER}/gen", data=data, timeout=60) as r:
        return json.loads(r.read().decode())


def repl(a) -> None:
    """Интерактивный режим: процесс живёт, читает строки из stdin и озвучивает
    каждую через сервер — без перезапуска Python на каждую фразу."""
    print("REPL: вводи текст, Enter — озвучить. Ctrl-D/Ctrl-C — выход.",
          file=sys.stderr)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        out_path = OUT / f"supertonic_{a.voice}_{int(time.time()*1000)}.wav"
        t0 = time.time()
        res = via_server(text, out_path, a.voice, a.style_json, a.lang,
                         a.steps, a.speed)
        print(f"[{time.time()-t0:.2f}s wall, gen={res['gen_sec']}s] {res['out']}",
              file=sys.stderr)
        if a.play:
            subprocess.run(["afplay", res["out"]], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="текст (не нужен в --repl)")
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
    ap.add_argument("--server", action="store_true",
                    help="генерить через тёплый сервер (server.py, порт 8126)")
    ap.add_argument("--repl", action="store_true",
                    help="интерактивный режим: одна сессия, строки из stdin "
                         "(всегда через сервер, без перезапуска Python)")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)

    if a.repl:
        repl(a)
        return
    if not a.text:
        ap.error("нужен text (или --repl)")
    tag = Path(a.style_json).stem if a.style_json else a.voice
    out_path = Path(a.out) if a.out else OUT / f"supertonic_{tag}_{int(time.time())}.wav"

    if a.server:
        res = via_server(a.text, out_path, a.voice, a.style_json, a.lang,
                         a.steps, a.speed)
        print(f"[server] gen={res['gen_sec']}s audio={res['audio_sec']}s "
              f"rtf={res['rtf']}")
        print(res["out"])
        if a.play:
            subprocess.run(["afplay", res["out"]], check=False)
        return

    t0 = time.time()
    tts = TTS(model="supertonic-3", auto_download=True)
    if a.style_json:
        style = tts.get_voice_style_from_path(a.style_json)
    else:
        style = tts.get_voice_style(a.voice)
    load_s = time.time() - t0

    t0 = time.time()
    wav, dur = tts.synthesize(a.text, voice_style=style, total_steps=a.steps,
                              speed=a.speed, lang=a.lang)
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
