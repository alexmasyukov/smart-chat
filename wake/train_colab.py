#!/usr/bin/env python3
"""
Обучение wake word «Кот, слушай» на Colab-VM (GPU) по рецепту из
openWakeWord training_models.ipynb.

Геометрия: окно 2 сек -> (16, 96) — канонический формат openWakeWord.
Негативы — готовые предпосчитанные фичи (validation_set_features), нарезанные
окнами. Позитивы — наши 80 ElevenLabs-клипов, прогнанные через augment_clips
(×PASSES проходов с аугментацией). Модель — маленький FCN, экспорт в ONNX.
"""
import os, glob, collections
import numpy as np
import torch, torch.nn as nn
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import openwakeword.data as D
import openwakeword.utils as U
from huggingface_hub import hf_hub_download

SR = 16000
TOTAL_S = 2.0
PASSES = 25                      # сколько раз прогнать 80 клипов через аугментацию
NEG_WINDOWS = 40000             # сколько негативных окон взять
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

F = U.AudioFeatures(device=device)
n_frames, n_feat = F.get_embedding_shape(TOTAL_S)
print("geometry:", (n_frames, n_feat))

# ---------- НЕГАТИВЫ: плоские фичи -> окна (n_frames, 96) ----------
neg_path = hf_hub_download(repo_id="davidscripka/openwakeword_features",
                           filename="validation_set_features.npy", repo_type="dataset")
neg_flat = np.load(neg_path, mmap_mode="r")
M = neg_flat.shape[0] // n_frames
neg = np.asarray(neg_flat[:M * n_frames]).reshape(M, n_frames, n_feat).astype(np.float32)
if neg.shape[0] > NEG_WINDOWS:
    idx = np.random.RandomState(0).choice(neg.shape[0], NEG_WINDOWS, replace=False)
    neg = neg[idx]
print("negatives:", neg.shape)

# ---------- ПОЗИТИВЫ: наши клипы -> augment_clips -> фичи ----------
clip_paths = sorted(glob.glob("/content/pos/positives/*.wav"))
print("наших клипов:", len(clip_paths))
paths = clip_paths * PASSES
total_length = int(TOTAL_S * SR)

def to_rows(item):
    """augment_clips может вернуть один клип (1-D) или батч (2-D)."""
    a = np.asarray(item[0])
    if a.ndim == 1:
        return [a]
    return [a[i] for i in range(a.shape[0])]

audios = []
for item in D.augment_clips(paths, total_length=total_length, sr=SR, batch_size=128,
                            background_clip_paths=[], RIR_paths=[]):
    for a in to_rows(item):
        a = np.asarray(a).flatten()
        if a.shape[0] < total_length:
            a = np.pad(a, (0, total_length - a.shape[0]))
        audios.append(a[:total_length])
audios = np.stack(audios).astype(np.float32)
# в int16 (embed_clips ждёт int16-диапазон)
if np.abs(audios).max() <= 1.5:
    audios = (audios * 32767).clip(-32768, 32767)
audios = audios.astype(np.int16)
print("augmented audios:", audios.shape)

pos_list = []
for i in range(0, audios.shape[0], 512):
    pos_list.append(F.embed_clips(audios[i:i + 512], batch_size=256))
pos = np.vstack(pos_list).astype(np.float32)
print("positives:", pos.shape)

# ---------- ДАННЫЕ ----------
X = np.vstack((neg, pos))
y = np.array([0] * len(neg) + [1] * len(pos), dtype=np.float32)[..., None]
Xt = torch.from_numpy(X).to(device)
yt = torch.from_numpy(y).to(device)
ds = torch.utils.data.TensorDataset(Xt, yt)
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

# ---------- МОДЕЛЬ (FCN, как в туториале) ----------
layer_dim = 32
fcn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(n_frames * n_feat, layer_dim), nn.LayerNorm(layer_dim), nn.ReLU(),
    nn.Linear(layer_dim, layer_dim), nn.LayerNorm(layer_dim), nn.ReLU(),
    nn.Linear(layer_dim, 1), nn.Sigmoid(),
).to(device)
opt = torch.optim.Adam(fcn.parameters(), lr=1e-3)
bce = torch.nn.functional.binary_cross_entropy

n_epochs = 30
for ep in range(n_epochs):
    hist = collections.defaultdict(list)
    for xb, yb in dl:
        w = torch.ones(yb.shape[0], device=device)
        w[yb.flatten() == 1] = 0.1                 # негативам больше веса -> меньше ложняков
        opt.zero_grad()
        p = fcn(xb)
        loss = bce(p, yb, w[..., None])
        loss.backward(); opt.step()
        hist["loss"].append(float(loss))
        tp = float((p.flatten()[yb.flatten() == 1] >= 0.5).sum())
        pos_n = float((yb.flatten() == 1).sum())
        hist["recall"].append(tp / max(pos_n, 1))
    if ep % 5 == 0 or ep == n_epochs - 1:
        print(f"epoch {ep:2d}  loss={np.mean(hist['loss']):.4f}  recall={np.mean(hist['recall']):.3f}")

# ---------- ОЦЕНКА ----------
fcn.eval()
with torch.no_grad():
    pp = fcn(torch.from_numpy(pos).to(device)).cpu().numpy().flatten()
    nn_ = fcn(torch.from_numpy(neg[:20000]).to(device)).cpu().numpy().flatten()
print(f"\nположительных >0.5: {(pp>=0.5).mean()*100:.1f}%  | средний скор pos={pp.mean():.3f}")
print(f"негативных  >0.5 (ложняки): {(nn_>=0.5).mean()*100:.3f}%  | средний скор neg={nn_.mean():.4f}")

# ---------- ЭКСПОРТ ONNX ----------
out = "/content/kot_slushai.onnx"
torch.onnx.export(fcn.cpu(), torch.zeros((1, n_frames, n_feat)), out, opset_version=13)
print("\nЭкспортировано:", out)
