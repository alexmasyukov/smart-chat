#!/usr/bin/env python3
"""
Шаг 1/2: считает эмбеддинги всех групп и КЭШИРУЕТ их в out/cache/feat_*.npy.
Это долгий проход (признаки через ONNX). После него train_local.py — секунды.

Запуск:  .venv/bin/python build_cache.py
Реальная речь (Golos) кэшируется отдельно скриптом embed_speech.py.
"""
import os
import numpy as np
import wake_common as C

print(f"geometry: {(C.n_frames, C.n_feat)} | NCPU={C.NCPU} | held-out голоса >= {C.HOLD_FROM}", flush=True)


def cache(name, bufs):
    a = C.embed_all(bufs)
    np.save(os.path.join(C.CACHE, f"feat_{name}.npy"), a)
    print(f"[saved] {name}: {a.shape}", flush=True)


# ---------- ПОЗИТИВЫ (эй кот), голоса < HOLD_FROM ----------
pos_clips = C.d("out/positives/*.wav")
train_pos = [p for p in pos_clips if C.voice_of(p) < C.HOLD_FROM]
print(f"позитив-клипов train: {len(train_pos)} (всего {len(pos_clips)})", flush=True)
cache("pos", [C.augment_positive(C.read16(p)) for p in train_pos for _ in range(C.POS_AUG)])

# ---------- НЕГАТИВЫ: чужая речь (рандом) и похожие слова (хард) ----------
rand, hard = [], []
for p in C.d("out/negatives/*.wav"):
    (hard if C.voice_of(p) >= C.HARD_START else rand).extend(C.neg_windows(C.read16(p)))
cache("neg_rand", rand)
cache("neg_hard", hard)

# ---------- НЕГАТИВЫ: синтетический шум ----------
quiet = [(C.RNG.randn(C.WIN) * C.RNG.uniform(50, 800)).clip(-32768, 32767).astype(np.int16) for _ in range(200)]
cache("noise", C.synth_noise(2000) + quiet)

# ---------- НЕГАТИВЫ: эфир микрофона на разных уровнях ----------
amb = []
for p in C.d("out/ambient/*.wav"):
    a = C.read16(p).astype(np.float32)
    for g in (0.7, 1.0, 1.5, 2.5, 4.0):
        gg = np.clip(a * g, -32768, 32767).astype(np.int16)
        amb += [gg[s:s + C.WIN] for s in range(0, len(gg) - C.WIN + 1, C.WIN // 4)]
cache("amb", amb)

print("КЭШ ГОТОВ. Теперь: .venv/bin/python train_local.py", flush=True)
