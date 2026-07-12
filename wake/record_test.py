#!/usr/bin/env python3
"""
Запись ТЕСТОВОГО набора твоего голоса (НЕ для обучения — только для проверки модели).

Позитивы (должны срабатывать):
    LABEL=pos N=15 .venv/bin/python record_test.py
    -> говори «Кот, слушай» по-разному.

Негативы (НЕ должны срабатывать — та «хуйня», на которую сейчас палит):
    LABEL=neg N=15 .venv/bin/python record_test.py
    -> говори левые слова, шуми, кашляй, стучи, «кот»/«слушай» по отдельности и т.п.

Пишет в out/test_pos/ или out/test_neg/. Ctrl+C — закончить досрочно.
"""
import os, time, glob
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = os.environ.get("LABEL", "pos")
assert LABEL in ("pos", "neg"), "LABEL должен быть pos или neg"
OUT = os.path.join(HERE, "out", f"test_{LABEL}")
SR = 16000
DUR = float(os.environ.get("DUR", "2.5" if LABEL == "pos" else "4.0"))
N = int(os.environ.get("N", "15"))

os.makedirs(OUT, exist_ok=True)
dev = sd.query_devices(kind="input")
nsr = int(dev["default_samplerate"])
start = len(glob.glob(os.path.join(OUT, "*.wav")))

hint = ("говори «Кот, слушай» по-разному" if LABEL == "pos"
        else "говори ЛЕВОЕ / шуми / стучи / «кот» и «слушай» по отдельности")
print("=" * 60)
print(f"  ТЕСТ-набор: {LABEL.upper()}  ({'должны срабатывать' if LABEL=='pos' else 'НЕ должны срабатывать'})")
print(f"  Микрофон: {dev['name']} ({nsr} Гц) | длит. {DUR}с | записей: {N}")
print(f"  {hint}")
print("  Ctrl+C — закончить досрочно.")
print("=" * 60)


def record():
    n = int(DUR * nsr)
    a = sd.rec(n, samplerate=nsr, channels=1, dtype="int16"); sd.wait()
    a = a[:, 0].astype(np.float32)
    if nsr != SR:
        a = resample_poly(a, SR, nsr)
    return a.astype(np.int16)


try:
    for k in range(N):
        for c in ("3", "2", "1", "ГОВОРИ!"):
            print(f"\r  {k+1}/{N}:  {c}       ", end="", flush=True)
            time.sleep(0.5)
        a = record()
        wav.write(os.path.join(OUT, f"{LABEL}_{start+k:03d}.wav"), SR, a)
        print(f"\r  {k+1}/{N}: сохранено (пик {int(np.abs(a).max())})        ")
        time.sleep(0.15)
except KeyboardInterrupt:
    pass
print(f"\nГотово. Всего {LABEL}-тестов: {len(glob.glob(os.path.join(OUT, '*.wav')))} в {OUT}")
