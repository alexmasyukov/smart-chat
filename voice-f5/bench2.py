#!/usr/bin/env python3
"""Бенчмарк ускорения F5-TTS при фиксированном nfe=16 (качество-база).

Проверяем рычаги БЕЗ снижения качества выхода:
  1) длина референса (9с/5с/3с) — короче префикс => короче диффузия-последовательность
  2) fp16 на mps

ref_text для каждого референса получаем через model.transcribe ОДИН раз,
чтобы замер = чистая диффузия (без Whisper внутри infer).
"""
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt", "model_212000.safetensors")
VOCAB = os.path.join(HERE, "ckpt", "vocab.txt")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

REFS = {
    "ref9": os.path.join(HERE, "ref", "lily.wav"),
    "ref5": os.path.join(HERE, "ref", "lily_5s.wav"),
    "ref3": os.path.join(HERE, "ref", "lily_3s.wav"),
}
TEXT = ("При этом пользователи могли запретить генерировать изображения с собой "
        "в настройках или разрешить создавать их только избранным людям в приложении Meta AI.")
NFE = 16

from f5_tts.api import F5TTS

print("[load] F5-TTS mps ...", flush=True)
model = F5TTS(model="F5TTS_v1_Base", ckpt_file=CKPT, vocab_file=VOCAB, device="mps")

# транскрибируем каждый референс один раз
ref_texts = {}
for k, p in REFS.items():
    ref_texts[k] = model.transcribe(p, language="ru")
    print(f"[asr] {k}: {ref_texts[k][:70]}", flush=True)


def run(tag, ref_key, half=False):
    out = os.path.join(OUT, f"fbench2_{tag}.wav")
    t0 = time.time()
    wav, sr, _ = model.infer(ref_file=REFS[ref_key], ref_text=ref_texts[ref_key],
                             gen_text=TEXT, nfe_step=NFE, file_wave=out,
                             show_info=lambda *a, **k: None)
    dt = time.time() - t0
    dur = len(wav) / sr
    print(f"{tag:16s} ref={ref_key} half={half} gen={dt:6.2f}s audio={dur:5.2f}s RTF={dt/dur:5.3f}",
          flush=True)


# прогрев
run("warmup", "ref9")
print("--- замеры (nfe=16) ---", flush=True)
run("ref9", "ref9")
run("ref5", "ref5")
run("ref3", "ref3")

# fp16: переводим DiT в half (вокодер оставляем fp32)
print("--- пробуем fp16 (DiT.half) ---", flush=True)
try:
    model.ema_model = model.ema_model.half()
    run("ref5_fp16", "ref5", half=True)
except Exception as e:
    print(f"[fp16] не вышло: {e}", flush=True)

print("Готово. Слушай out/fbench2_*.wav", flush=True)
