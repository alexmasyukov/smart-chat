#!/usr/bin/env python3
"""
Прогон модели по ТЕСТОВОМУ набору твоего голоса (out/test_pos, out/test_neg).
Считает как детектор: скользящее 2-сек окно + энергетический гейт (как detect.py).

Запуск:  .venv/bin/python test_model.py
"""
import os, glob
import numpy as np
import scipy.io.wavfile as wav
import onnxruntime as ort
import openwakeword.utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
WIN = 2 * SR
HOP = 4000
GATE = float(os.environ.get("GATE", "1200"))

F = U.AudioFeatures(device="cpu")
sess = ort.InferenceSession(os.path.join(HERE, "kot_slushai.onnx"))
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name


def stream_max(path):
    sr, d = wav.read(path)
    d = (d[:, 0] if d.ndim > 1 else d).astype(np.int16)
    d = np.concatenate([np.zeros(6000, np.int16), d, np.zeros(6000, np.int16)])
    best = 0.0
    for st in range(0, len(d) - WIN + 1, HOP):
        w = d[st:st + WIN]
        if np.abs(w).max() < GATE:
            continue
        f = F.embed_clips(w[None, :], batch_size=1).astype(np.float32)
        best = max(best, float(sess.run([oname], {iname: f})[0].flatten()[0]))
    return best


def load(label):
    return sorted(glob.glob(os.path.join(HERE, "out", f"test_{label}", "*.wav")))


pos = {os.path.basename(p): stream_max(p) for p in load("pos")}
neg = {os.path.basename(p): stream_max(p) for p in load("neg")}

print(f"\nПОЗИТИВЫ (твой «Кот, слушай», должны срабатывать) — {len(pos)} шт:")
for k, v in pos.items():
    print(f"   {v:0.2f}  {k}")
print(f"\nНЕГАТИВЫ (левое/шум, НЕ должны) — {len(neg)} шт:")
for k, v in sorted(neg.items(), key=lambda x: -x[1]):
    flag = "  <-- ЛОЖНЯК" if v >= 0.85 else ""
    print(f"   {v:0.2f}  {k}{flag}")

pv = np.array(list(pos.values())) if pos else np.array([])
nv = np.array(list(neg.values())) if neg else np.array([])
print("\n=== СВОДКА ===")
for th in [0.7, 0.85, 0.9, 0.95]:
    rec = (pv >= th).mean() * 100 if len(pv) else float("nan")
    fp = (nv >= th).mean() * 100 if len(nv) else float("nan")
    print(f"порог {th}:  recall(твой)={rec:5.1f}%   ложняки(левое)={fp:5.1f}%")
