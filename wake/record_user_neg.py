#!/usr/bin/env python3
"""
Запись ТВОИХ обучающих НЕГАТИВОВ — твой голос/звуки БЕЗ «кот слушай».
Это ключ к персональному wake word: модель должна знать, что твой голос,
говорящий ДРУГОЕ, — это НЕ срабатывание.

Запуск:  N=25 .venv/bin/python record_user_neg.py

Каждая запись ~5 сек — говори РАЗНОЕ и шуми:
  - обычная речь на любые темы, случайные слова, счёт, чтение;
  - «кот» и «слушай» ПО ОТДЕЛЬНОСТИ, похожие слова (кто, компот, слушать);
  - кашель, смех, хмыканье, стук по столу, шуршание, хлопки.
Пишет в out/user_neg/ (обучение это подхватит). Ctrl+C — досрочно.
"""
import os, time, glob
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "user_neg")
SR = 16000
DUR = float(os.environ.get("DUR", "5.0"))
N = int(os.environ.get("N", "25"))

os.makedirs(OUT, exist_ok=True)
dev = sd.query_devices(kind="input")
nsr = int(dev["default_samplerate"])
start = len(glob.glob(os.path.join(OUT, "*.wav")))

print("=" * 60)
print("  Запись ТВОИХ негативов (речь/шум БЕЗ «кот слушай»)")
print(f"  Микрофон: {dev['name']} ({nsr} Гц) | {DUR}с × {N} записей")
print("  Говори РАЗНОЕ, шуми, НЕ говори «кот слушай» целиком.")
print("  Ctrl+C — закончить досрочно.")
print("=" * 60)


def record():
    a = sd.rec(int(DUR * nsr), samplerate=nsr, channels=1, dtype="int16"); sd.wait()
    a = a[:, 0].astype(np.float32)
    if nsr != SR:
        a = resample_poly(a, SR, nsr)
    return a.astype(np.int16)


try:
    for k in range(N):
        for c in ("3", "2", "1", "ГОВОРИ РАЗНОЕ!"):
            print(f"\r  {k+1}/{N}:  {c}        ", end="", flush=True)
            time.sleep(0.5)
        a = record()
        wav.write(os.path.join(OUT, f"uneg_{start+k:03d}.wav"), SR, a)
        print(f"\r  {k+1}/{N}: сохранено (пик {int(np.abs(a).max())})            ")
        time.sleep(0.15)
except KeyboardInterrupt:
    pass
print(f"\nГотово. Всего твоих негативов: {len(glob.glob(os.path.join(OUT, '*.wav')))} в {OUT}")
