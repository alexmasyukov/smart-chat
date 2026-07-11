#!/usr/bin/env python3
"""Бенчмарк рычагов ускорения OmniVoice.

Меряет вклад:
  1) кеш референса (create_voice_clone_prompt) vs авто-Whisper каждый вызов
  2) num_step  (32 -> 16 -> 8)
  3) guidance_scale (2.0 -> 1.0, отключение CFG = один проход на шаг)
  4) dtype (float32 vs bfloat16) — грузим модель дважды

Все результаты пишет в out/bench_*.wav для сравнения качества на слух.
"""
import os
import time

import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref", "lily.wav")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

TEXT = ("При этом пользователи могли запретить генерировать изображения с собой "
        "в настройках или разрешить создавать их только избранным людям в приложении Meta AI.")


def run(model, tag, prompt=None, ref_audio=None, ref_text=None, **cfg):
    gc = OmniVoiceGenerationConfig(**cfg)
    kw = dict(text=TEXT, generation_config=gc)
    if prompt is not None:
        kw["voice_clone_prompt"] = prompt
    else:
        kw["ref_audio"] = ref_audio
        if ref_text:
            kw["ref_text"] = ref_text
    t0 = time.time()
    audio = model.generate(**kw)
    dt = time.time() - t0
    wav = audio[0]
    dur = len(wav) / 24000
    path = os.path.join(OUT, f"bench_{tag}.wav")
    sf.write(path, wav, 24000)
    print(f"{tag:28s} gen={dt:6.2f}s  audio={dur:5.2f}s  RTF={dt/dur:5.3f}", flush=True)
    return dt / dur


def bench_dtype(dtype, dtag):
    print(f"\n===== dtype={dtag} =====", flush=True)
    t0 = time.time()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=dtype)
    print(f"[load {dtag}] {time.time()-t0:.1f}s", flush=True)

    # кеш референса — один раз транскрибируем + кодируем
    t0 = time.time()
    prompt = model.create_voice_clone_prompt(ref_audio=REF)
    print(f"[create_voice_clone_prompt] {time.time()-t0:.1f}s  ref_text='{prompt.ref_text}'", flush=True)

    # baseline: авто-whisper каждый раз (как сейчас в server.py), num_step=32
    run(model, f"{dtag}_auto_step32", ref_audio=REF, num_step=32)
    # с кешем префикса, num_step=32
    run(model, f"{dtag}_cached_step32", prompt=prompt, num_step=32)
    # с кешем, num_step=16
    run(model, f"{dtag}_cached_step16", prompt=prompt, num_step=16)
    # с кешем, num_step=8
    run(model, f"{dtag}_cached_step8", prompt=prompt, num_step=8)
    # с кешем, step16 + CFG off (guidance 1.0 -> один проход на шаг)
    run(model, f"{dtag}_cached_step16_noCFG", prompt=prompt, num_step=16, guidance_scale=1.0)
    del model
    if hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    bench_dtype(torch.float32, "fp32")
    bench_dtype(torch.bfloat16, "bf16")
    print("\nГотово. Слушай out/bench_*.wav", flush=True)
