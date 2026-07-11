#!/usr/bin/env python3
"""Клонирование голоса через F5-TTS_RUSSIAN (Misha24-10, файнтюн F5-TTS).

Русская модель, кастомный vocab (кириллица). Референс — тот же Lily.

ВАЖНО (из обсуждения на HF #52): на mps F5-TTS часто выдаёт «китайскую» кашу.
Поэтому дефолт — CPU (медленнее, но корректный русский). mps — через --device mps.

Ударение: ставь '+' перед ударной гласной (молок+о). Без разметки тоже работает.

Запуск:
    python generate.py "Текст для озвучки"
    python generate.py "Текст" --device mps        # эксперимент
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref", "lily_3s.wav")  # короткий референс — быстрее
CKPT = os.path.join(HERE, "ckpt", "model_212000.safetensors")
VOCAB = os.path.join(HERE, "ckpt", "vocab.txt")
OUT = os.path.join(HERE, "out")

# Пусто -> F5 сам транскрибирует референс через Whisper (важно при смене
# длины референса: текст всегда совпадает с аудио).
REF_TEXT = ""

DEFAULT_TEXT = ("При этом пользователи могли запретить генерировать изображения с собой "
                "в настройках или разрешить создавать их только избранным людям "
                "в приложении Meta AI.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="cpu — надёжно; mps — быстрее, но бывает мусор")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--ref-text", default=REF_TEXT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nfe", type=int, default=32, help="шаги диффузии")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    from f5_tts.api import F5TTS

    os.makedirs(OUT, exist_ok=True)
    out_path = args.out or os.path.join(OUT, f"f5_{args.device}_{int(time.time())}.wav")

    print(f"[device] {args.device}", file=sys.stderr)
    print("[load] загружаю F5-TTS_RUSSIAN (v4_winter) ...", file=sys.stderr)
    t0 = time.time()
    model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=CKPT,
        vocab_file=VOCAB,
        device=args.device,
    )
    print(f"[load] готово за {time.time() - t0:.1f}с", file=sys.stderr)

    print(f"[gen] «{args.text[:60]}...»", file=sys.stderr)
    t0 = time.time()
    wav, sr, _ = model.infer(
        ref_file=args.ref,
        ref_text=args.ref_text,
        gen_text=args.text,
        nfe_step=args.nfe,
        speed=args.speed,
        file_wave=out_path,
    )
    dt = time.time() - t0
    dur = len(wav) / sr
    print(f"[gen] {dur:.1f}с аудио за {dt:.1f}с  (RTF={dt / dur:.3f}), sr={sr}",
          file=sys.stderr)
    print(out_path)


if __name__ == "__main__":
    main()
