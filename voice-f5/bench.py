#!/usr/bin/env python3
"""Бенчмарк ускорения F5-TTS_RUSSIAN на mps.

Модель грузится ОДИН раз, гоняем разные nfe_step / cfg_strength.
Все сэмплы -> out/fbench_*.wav для сравнения качества на слух.
"""
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref", "lily.wav")
CKPT = os.path.join(HERE, "ckpt", "model_212000.safetensors")
VOCAB = os.path.join(HERE, "ckpt", "vocab.txt")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

REF_TEXT = ("Инструмент позволял создавать изображение на основе фотографий "
            "из публичных аккаунтов Instagram. Функция появилась в июле 2021.")
TEXT = ("При этом пользователи могли запретить генерировать изображения с собой "
        "в настройках или разрешить создавать их только избранным людям в приложении Meta AI.")

from f5_tts.api import F5TTS

print("[load] F5-TTS на mps ...", flush=True)
t0 = time.time()
model = F5TTS(model="F5TTS_v1_Base", ckpt_file=CKPT, vocab_file=VOCAB, device="mps")
print(f"[load] {time.time()-t0:.1f}s", flush=True)


def run(tag, nfe, cfg=2.0):
    out = os.path.join(OUT, f"fbench_{tag}.wav")
    t0 = time.time()
    wav, sr, _ = model.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=TEXT,
                             nfe_step=nfe, cfg_strength=cfg, file_wave=out,
                             show_info=lambda *a, **k: None)
    dt = time.time() - t0
    dur = len(wav) / sr
    print(f"{tag:20s} nfe={nfe:2d} cfg={cfg:<3} gen={dt:6.2f}s audio={dur:5.2f}s RTF={dt/dur:5.3f}",
          flush=True)


# прогрев (первый вызов инициализирует mps-ядра)
run("warmup", 16)
print("--- замеры ---", flush=True)
run("nfe32", 32)
run("nfe16", 16)
run("nfe8", 8)
run("nfe4", 4)
run("nfe16_cfg1", 16, cfg=1.0)
print("Готово. Слушай out/fbench_*.wav", flush=True)
