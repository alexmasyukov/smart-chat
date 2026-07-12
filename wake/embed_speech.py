#!/usr/bin/env python3
"""
Качает русский речевой корпус (потоково) и считает из него фичи-негативы —
реальная человеческая речь, главный убийца ложняков (как LibriSpeech в hey-gaia).
Сохраняет out/neg_speech_feats.npy (готовые фичи, train_local их подхватит).

Запуск:  TARGET=10000 .venv/bin/python embed_speech.py
"""
import os, sys, io
import numpy as np
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from scipy.signal import resample_poly
import soundfile as sf
from datasets import load_dataset, Audio
import wake_common as C            # детерминированный embed_all

SR = 16000
WIN = 2 * SR
STRIDE = SR                      # окно каждую 1 сек
TARGET = int(os.environ.get("TARGET", "10000"))
DATASET = os.environ.get("DATASET", "bond005/sberdevices_golos_10h_crowd")
CONFIG = os.environ.get("CONFIG", "") or None
SPLIT = os.environ.get("SPLIT", "train")
AUDIO_KEY = os.environ.get("AUDIO_KEY", "audio")
print(f"качаю {DATASET} [{CONFIG}] потоково, цель {TARGET} окон...", flush=True)
ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
ds = ds.cast_column(AUDIO_KEY, Audio(decode=False))   # сами декодируем через soundfile

feats = []
n_utt = 0
for ex in ds:
    a = ex[AUDIO_KEY]
    try:
        raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
    except Exception:
        continue
    if sr != SR:
        arr = resample_poly(arr, SR, sr)
    x = np.clip(arr * 32767, -32768, 32767).astype(np.int16)
    if len(x) < WIN:
        b = np.zeros(WIN, np.int16); b[:len(x)] = x
        wins = [b]
    else:
        wins = [x[s:s + WIN] for s in range(0, len(x) - WIN + 1, STRIDE)]
    if wins:
        feats.append(C.embed_all(wins))
    n_utt += 1
    total = sum(f.shape[0] for f in feats)
    if n_utt % 25 == 0:
        print(f"  реплик {n_utt}, окон {total}", flush=True)
    if total >= TARGET:
        break

out = np.vstack(feats).astype(np.float32)
np.save(os.path.join(os.path.dirname(__file__), "out", "neg_speech_feats.npy"), out)
print("СОХРАНЕНО out/neg_speech_feats.npy", out.shape, flush=True)
