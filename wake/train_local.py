#!/usr/bin/env python3
"""
Локальное обучение wake word «Кот, слушай» — фичи И обучение на маке (ARM CPU),
чтобы train/inference совпадали (кросс-окруженческий рассинхрон фич устранён).

Позитивы: наши 80 ElevenLabs-клипов, размноженные numpy-аугментацией (позиция в
2-сек окне, шум, громкость). Негативы: наши длинные ru-фразы + хард-негативы,
нарезанные скользящим окном 2 сек. Фичи — openwakeword.AudioFeatures (те же, что
в детекторе). Модель — FCN, экспорт ONNX.
"""
import glob, os, collections
import numpy as np
import scipy.io.wavfile as wav
import torch, torch.nn as nn
import openwakeword.utils as U

SR = 16000
WIN = 2 * SR                 # 32000 сэмплов = окно (16,96)
POS_AUG = 30                 # аугментаций на позитивный клип
STRIDE = SR // 2             # шаг скользящего окна по негативам (0.5с)
RNG = np.random.RandomState(0)

F = U.AudioFeatures(device="cpu")
n_frames, n_feat = F.get_embedding_shape(2.0)
print("geometry:", (n_frames, n_feat))


def read16(path):
    sr, d = wav.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    return d.astype(np.int16)


def augment_positive(clip):
    """Клип -> 2-сек int16 окно: случайная позиция, шум, громкость."""
    clip = clip[:WIN]
    buf = np.zeros(WIN, dtype=np.float32)
    off = RNG.randint(0, max(1, WIN - len(clip)))
    buf[off:off + len(clip)] = clip.astype(np.float32)
    # громкость
    buf *= RNG.uniform(0.6, 1.1)
    # шум со случайным SNR
    if RNG.random() < 0.8:
        rms = np.sqrt((buf ** 2).mean()) + 1e-6
        snr = RNG.uniform(8, 30)
        noise_rms = rms / (10 ** (snr / 20))
        buf += RNG.randn(WIN).astype(np.float32) * noise_rms
    return np.clip(buf, -32768, 32767).astype(np.int16)


def neg_windows(clip):
    """Длинный клип -> список 2-сек окон (скользящим окном)."""
    outs = []
    if len(clip) <= WIN:
        b = np.zeros(WIN, dtype=np.int16)
        off = RNG.randint(0, max(1, WIN - len(clip)))
        b[off:off + len(clip)] = clip
        outs.append(b)
    else:
        for s in range(0, len(clip) - WIN + 1, STRIDE):
            outs.append(clip[s:s + WIN].astype(np.int16))
    return outs


def embed_all(buffers):
    """Список 2-сек int16 -> (N,16,96) фичи, батчами."""
    arr = np.stack(buffers).astype(np.int16)
    feats = []
    for i in range(0, len(arr), 256):
        feats.append(F.embed_clips(arr[i:i + 256], batch_size=256))
    return np.vstack(feats).astype(np.float32)


# ---------- ПОЗИТИВЫ ----------
# SPEAKER-SPECIFIC: позитивы — ТОЛЬКО твой голос. Чужие «кот слушай» -> хард-негативы.
user_clips = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "out/user_pos/*.wav")))
synth_clips = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "out/positives/*.wav")))
assert user_clips, "нет записей твоего голоса в out/user_pos/ — запусти record_positives.py"
# held-out: последние 6 твоих записей не для обучения (честная проверка)
train_user, holdout_pos_clips = user_clips[:-6], user_clips[-6:]
USER_AUG = 50
pos_bufs = []
for p in train_user:
    c = read16(p)
    for _ in range(USER_AUG):
        pos_bufs.append(augment_positive(c))
pos = embed_all(pos_bufs)
print("positives (ТОЛЬКО твой голос):", pos.shape, f"из {len(train_user)} записей")

# ---------- НЕГАТИВЫ ----------
neg_clips = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "out/negatives/*.wav")))
neg_bufs = []
for p in neg_clips:
    neg_bufs.extend(neg_windows(read16(p)))
# ЧУЖИЕ голоса, говорящие «кот слушай» -> ХАРД-негативы (нужен только твой голос)
for p in synth_clips:
    c = read16(p)
    for _ in range(5):
        neg_bufs.append(augment_positive(c))
print("хард-негативов (чужой «кот слушай»):", len(synth_clips) * 5)
# богатый синтетический шум (против ложняков на любой шум/тон/стук)
def synth_noise(k):
    outs = []
    t = np.arange(WIN) / SR
    for _ in range(k):
        typ = RNG.randint(6)
        amp = RNG.uniform(300, 9000)
        if typ == 0:                              # белый
            x = RNG.randn(WIN)
        elif typ == 1:                            # коричневый (интеграл белого)
            x = np.cumsum(RNG.randn(WIN))
        elif typ == 2:                            # розовый-ish (сглаженный белый)
            x = np.convolve(RNG.randn(WIN), np.ones(24) / 24, "same")
        elif typ == 3:                            # чистый тон
            x = np.sin(2 * np.pi * RNG.uniform(80, 4000) * t)
        elif typ == 4:                            # чирп (свип частоты)
            f0, f1 = RNG.uniform(80, 900), RNG.uniform(1200, 6000)
            x = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / (WIN / SR)) * t)
        else:                                     # клики/стуки
            x = np.zeros(WIN)
            idx = RNG.randint(0, WIN, RNG.randint(3, 50))
            x[idx] = RNG.randn(len(idx))
        x = x / (np.abs(x).max() + 1e-6) * amp
        outs.append(np.clip(x, -32768, 32767).astype(np.int16))
    return outs

neg_bufs.extend(synth_noise(2500))
# немного тихого шума/тишины
for _ in range(200):
    neg_bufs.append((RNG.randn(WIN) * RNG.uniform(50, 800)).clip(-32768, 32767).astype(np.int16))

# РЕАЛЬНЫЙ эфир микрофона (главное против ложняков на тишине) — на разных уровнях
amb_clips = glob.glob(os.path.join(os.path.dirname(__file__), "out/ambient/*.wav"))
amb_bufs = []
for p in amb_clips:
    a = read16(p).astype(np.float32)
    for gain in (0.7, 1.0, 1.5, 2.5, 4.0):
        g = np.clip(a * gain, -32768, 32767).astype(np.int16)
        for s in range(0, len(g) - WIN + 1, WIN // 4):
            amb_bufs.append(g[s:s + WIN])
neg_bufs.extend(amb_bufs)
print("реального эфира-негативов:", len(amb_bufs))
neg = embed_all(neg_bufs)
print("negatives:", neg.shape, "(речь + шум + эфир микрофона)")

# ---------- ОБУЧЕНИЕ ----------
X = np.vstack((neg, pos))
y = np.array([0] * len(neg) + [1] * len(pos), dtype=np.float32)[..., None]
Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt), batch_size=512, shuffle=True)

ld = 32
fcn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(n_frames * n_feat, ld), nn.LayerNorm(ld), nn.ReLU(),
    nn.Linear(ld, ld), nn.LayerNorm(ld), nn.ReLU(),
    nn.Linear(ld, 1), nn.Sigmoid(),
)
opt = torch.optim.Adam(fcn.parameters(), lr=1e-3)
bce = torch.nn.functional.binary_cross_entropy
wpos = len(neg) / len(pos)
for ep in range(40):
    for xb, yb in dl:
        w = torch.ones(yb.shape[0]); w[yb.flatten() == 1] = wpos
        opt.zero_grad(); loss = bce(fcn(xb), yb, w[..., None]); loss.backward(); opt.step()

# ---------- ПРОВЕРКА на held-out (честная) ----------
fcn.eval()
s = torch.nn.Sigmoid()
with torch.no_grad():
    # held-out позитивы: центрированный клип в 2с окне
    ho = []
    for p in holdout_pos_clips:
        c = read16(p); b = np.zeros(WIN, dtype=np.int16)
        off = (WIN - min(len(c), WIN)) // 2; b[off:off + min(len(c), WIN)] = c[:WIN]
        ho.append(b)
    ho_f = embed_all(ho)
    ho_sc = fcn(torch.from_numpy(ho_f)).numpy().flatten()
    neg_sc = fcn(torch.from_numpy(neg)).numpy().flatten()
print("\n=== held-out проверка ===")
print("held-out ПОЗИТИВЫ (8 клипов, не в обучении):", [round(float(x), 3) for x in ho_sc])
for th in [0.5, 0.7, 0.9]:
    print(f"th={th}: held-out recall={(ho_sc>=th).mean()*100:5.1f}%  ложняки(neg)={(neg_sc>=th).mean()*100:.3f}%")

# ---------- ЭКСПОРТ ----------
out = os.path.join(os.path.dirname(__file__), "kot_slushai.onnx")
torch.onnx.export(fcn, torch.zeros((1, n_frames, n_feat)), out, opset_version=13,
                  dynamo=False, input_names=["features"], output_names=["score"],
                  dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}})
print("\nЭкспортировано:", out)
