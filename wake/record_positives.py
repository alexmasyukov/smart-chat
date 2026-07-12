#!/usr/bin/env python3
"""
Запись твоего голоса «Кот, слушай» для дообучения детектора.

Запуск:  .venv/bin/python record_positives.py           # 40 записей
         N=60 .venv/bin/python record_positives.py       # сколько хочешь

Каждый раунд: обратный отсчёт -> запись 2 сек -> сохранение в out/user_pos/.
Говори по-разному: обычно, быстрее/медленнее, громче/тише, с разной интонацией,
чуть отвернувшись от микрофона — чем разнообразнее, тем лучше детектор.
Ctrl+C — закончить досрочно (уже записанное сохранится).
"""
import os, time, glob
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "user_pos")
SR = 16000
DUR = 2.0
N = int(os.environ.get("N", "40"))

os.makedirs(OUT, exist_ok=True)
dev = sd.query_devices(kind="input")
nsr = int(dev["default_samplerate"])
start_idx = len(glob.glob(os.path.join(OUT, "*.wav")))

print("=" * 60)
print("  Запись голоса для «Кот, слушай»")
print(f"  Микрофон: {dev['name']} ({nsr} Гц)")
print(f"  Записей: {N}  (уже есть: {start_idx})")
print("  Говори РАЗНО: темп, громкость, интонация, расстояние.")
print("  Ctrl+C — закончить досрочно.")
print("=" * 60)


def record():
    n = int(DUR * nsr)
    a = sd.rec(n, samplerate=nsr, channels=1, dtype="int16")
    sd.wait()
    a = a[:, 0].astype(np.float32)
    if nsr != SR:
        a = resample_poly(a, SR, nsr)
    return a.astype(np.int16)


try:
    for k in range(N):
        idx = start_idx + k
        for c in ("3", "2", "1", "ГОВОРИ!"):
            print(f"\r  запись {k+1}/{N}:  {c}      ", end="", flush=True)
            time.sleep(0.55)
        a = record()
        peak = int(np.abs(a).max())
        wav.write(os.path.join(OUT, f"user_{idx:03d}.wav"), SR, a)
        warn = "  ⚠️ тихо — говори громче/ближе" if peak < 1500 else "  ✓"
        print(f"\r  запись {k+1}/{N}: сохранено (пик {peak}){warn}        ")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass

total = len(glob.glob(os.path.join(OUT, "*.wav")))
print(f"\nГотово. Всего твоих записей: {total} в {OUT}")
print("Теперь я дообучу модель на них.")
