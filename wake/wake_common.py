#!/usr/bin/env python3
"""Общие функции признаков для build_cache.py и train_local.py."""
import os, glob
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
import openwakeword.utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
WIN = 2 * SR                      # окно 2 сек = (16,96)
POS_AUG = 25                      # аугментаций на позитивный клип
STRIDE = SR // 2                  # шаг скользящего окна по негативам
NCPU = int(os.environ.get("NCPU", "6"))
RNG = np.random.RandomState(0)

CACHE = os.path.join(HERE, "out", "cache")
os.makedirs(CACHE, exist_ok=True)

# голоса >= HOLD_VOICES не идут в обучение (честная проверка на невиданных голосах)
HOLD_FROM = int(os.environ.get("HOLD_FROM", "60"))

F = U.AudioFeatures(device="cpu")
n_frames, n_feat = F.get_embedding_shape(2.0)

try:
    from gen_negatives import SENTENCES
    HARD_START = len(SENTENCES)
except Exception:
    HARD_START = 30


def d(sub):
    return sorted(glob.glob(os.path.join(HERE, sub)))


def read16(p):
    sr, x = wav.read(p)
    return (x[:, 0] if x.ndim > 1 else x).astype(np.int16)


def voice_of(path):
    return int(os.path.basename(path).split("_")[1])


def reverb(x):
    L = int(RNG.uniform(0.05, 0.25) * SR)
    ir = np.zeros(L, np.float32); ir[0] = 1.0
    for _ in range(RNG.randint(2, 6)):
        ir[RNG.randint(1, L)] += RNG.uniform(0.1, 0.5)
    ir *= np.exp(-np.arange(L) / (SR * RNG.uniform(0.03, 0.12)))
    y = np.convolve(x, ir)[:len(x)]
    return y / (np.abs(y).max() + 1e-6) * (np.abs(x).max() + 1e-6)


def augment_positive(clip):
    clip = clip.astype(np.float32)
    if RNG.random() < 0.7:
        clip = resample_poly(clip, 10, RNG.randint(9, 12))     # темп+высота ±~10%
    if RNG.random() < 0.4:
        clip = reverb(clip)                                     # комната/микрофон
    clip = clip[:WIN]
    buf = np.zeros(WIN, dtype=np.float32)
    off = RNG.randint(0, max(1, WIN - len(clip)))
    buf[off:off + len(clip)] = clip
    buf *= RNG.uniform(0.6, 1.1)
    if RNG.random() < 0.8:
        rms = np.sqrt((buf ** 2).mean()) + 1e-6
        buf += RNG.randn(WIN).astype(np.float32) * rms / (10 ** (RNG.uniform(8, 30) / 20))
    return np.clip(buf, -32768, 32767).astype(np.int16)


def neg_windows(clip):
    if len(clip) <= WIN:
        b = np.zeros(WIN, dtype=np.int16)
        off = RNG.randint(0, max(1, WIN - len(clip)))
        b[off:off + len(clip)] = clip
        return [b]
    return [clip[s:s + WIN].astype(np.int16) for s in range(0, len(clip) - WIN + 1, STRIDE)]


def synth_noise(k):
    outs, t = [], np.arange(WIN) / SR
    for _ in range(k):
        typ, amp = RNG.randint(6), RNG.uniform(300, 9000)
        if typ == 0: x = RNG.randn(WIN)
        elif typ == 1: x = np.cumsum(RNG.randn(WIN))
        elif typ == 2: x = np.convolve(RNG.randn(WIN), np.ones(24) / 24, "same")
        elif typ == 3: x = np.sin(2 * np.pi * RNG.uniform(80, 4000) * t)
        elif typ == 4:
            f0, f1 = RNG.uniform(80, 900), RNG.uniform(1200, 6000)
            x = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / (WIN / SR)) * t)
        else:
            x = np.zeros(WIN); idx = RNG.randint(0, WIN, RNG.randint(3, 50)); x[idx] = RNG.randn(len(idx))
        outs.append(np.clip(x / (np.abs(x).max() + 1e-6) * amp, -32768, 32767).astype(np.int16))
    return outs


def embed_all(bufs):
    if not bufs:
        return np.zeros((0, n_frames, n_feat), np.float32)
    arr = np.stack(bufs).astype(np.int16)
    return np.vstack([F.embed_clips(arr[i:i + 1024], batch_size=1024, ncpu=NCPU)
                      for i in range(0, len(arr), 1024)]).astype(np.float32)
