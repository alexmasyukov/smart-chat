#!/usr/bin/env python3
"""Zero-shot клонирование голоса через OmniVoice (k2-fsa/OmniVoice).

Референс — голос из ElevenLabs (Lily). ref_text не задаём: модель сама
транскрибирует референс через Whisper ASR.

Запуск:
    python generate.py "Текст, который нужно озвучить голосом Lily"
"""
import argparse
import os
import sys
import time

import soundfile as sf
import torch

from omnivoice import OmniVoice

REF_AUDIO = os.path.join(os.path.dirname(__file__), "ref", "lily.wav")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

DEFAULT_TEXT = (
    "Hello there. This is a zero-shot voice clone, generated locally on "
    "Apple Silicon using OmniVoice. The velvety tone should feel familiar."
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniVoice voice cloning")
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT,
                        help="Текст для озвучки")
    parser.add_argument("--ref", default=REF_AUDIO, help="Референс-аудио")
    parser.add_argument("--ref-text", default=None,
                        help="Транскрипция референса (по умолчанию — авто-Whisper)")
    parser.add_argument("--out", default=None, help="Путь для результата .wav")
    parser.add_argument("--num-step", type=int, default=32,
                        help="Шаги диффузии (16 — быстрее, 32 — качественнее)")
    parser.add_argument("--speed", type=float, default=1.0, help="Скорость речи")
    args = parser.parse_args()

    device = pick_device()
    # mps не поддерживает float16 стабильно для всех операций — берём float32 на mac
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"[device] {device}  [dtype] {dtype}", file=sys.stderr)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, f"clone_{int(time.time())}.wav"
    )

    print("[load] загружаю модель k2-fsa/OmniVoice ...", file=sys.stderr)
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
    )
    print(f"[load] готово за {time.time() - t0:.1f}с", file=sys.stderr)

    gen_kwargs = dict(
        text=args.text,
        ref_audio=args.ref,
        num_step=args.num_step,
        speed=args.speed,
    )
    if args.ref_text:
        gen_kwargs["ref_text"] = args.ref_text

    print(f"[gen] «{args.text}»", file=sys.stderr)
    t0 = time.time()
    audio = model.generate(**gen_kwargs)
    dt = time.time() - t0
    wav = audio[0]
    dur = len(wav) / 24000
    print(f"[gen] {dur:.1f}с аудио за {dt:.1f}с  (RTF={dt / dur:.3f})",
          file=sys.stderr)

    sf.write(out_path, wav, 24000)
    print(out_path)


if __name__ == "__main__":
    main()
