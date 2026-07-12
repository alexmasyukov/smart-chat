#!/usr/bin/env python3
"""
Обучение wake word «эй кот» — детектор ФРАЗЫ (любой голос), рецепт из hey-gaia.
Всё локально на маке (фичи и обучение в одном окружении).

Позитивы: ElevenLabs «эй кот» (много голосов) + аугментация (темп/высота/позиция/шум).
Негативы:
  - РЕАЛЬНАЯ русская речь (Golos) -> out/neg_speech_feats.npy (главный убийца ложняков);
  - ElevenLabs случайные фразы;
  - ElevenLabs похожие слова («эй код», «кот», ...) -> ХАРД-негативы, вес ×4;
  - синтетический шум + эфир микрофона.
Модель: Conv1D-голова. Веса классов сбалансированы, хард-негативы усилены.
Честный held-out: невиданные голоса позитивов + отложенная реальная речь.
"""
import glob, os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
import torch, torch.nn as nn
import openwakeword.utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
WIN = 2 * SR
POS_AUG = 25
STRIDE = SR // 2
RNG = np.random.RandomState(0)

F = U.AudioFeatures(device="cpu")
n_frames, n_feat = F.get_embedding_shape(2.0)
print("geometry:", (n_frames, n_feat))


def d(sub):
    return sorted(glob.glob(os.path.join(HERE, sub)))


def read16(p):
    sr, x = wav.read(p)
    return (x[:, 0] if x.ndim > 1 else x).astype(np.int16)


def augment_positive(clip):
    clip = clip.astype(np.float32)
    if RNG.random() < 0.7:                       # темп+высота ±~10%
        clip = resample_poly(clip, 10, RNG.randint(9, 12))
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


def embed_all(bufs):
    if not bufs:
        return np.zeros((0, n_frames, n_feat), np.float32)
    arr = np.stack(bufs).astype(np.int16)
    return np.vstack([F.embed_clips(arr[i:i + 256], batch_size=256)
                      for i in range(0, len(arr), 256)]).astype(np.float32)


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


def voice_of(path):
    return int(os.path.basename(path).split("_")[1])


# ---------- ПОЗИТИВЫ (эй кот) с held-out по голосам ----------
pos_clips = d("out/positives/*.wav")
HOLD_VOICES = {36, 37, 38, 39}
train_pos = [p for p in pos_clips if voice_of(p) not in HOLD_VOICES]
hold_pos = [p for p in pos_clips if voice_of(p) in HOLD_VOICES]
pos = embed_all([augment_positive(read16(p)) for p in train_pos for _ in range(POS_AUG)])
print(f"positives: {pos.shape} (train голосов-клипов {len(train_pos)}, held-out {len(hold_pos)})")

# ---------- НЕГАТИВЫ ----------
try:
    from gen_negatives import SENTENCES
    HARD_START = len(SENTENCES)
except Exception:
    HARD_START = 30

neg_rand_bufs, neg_hard_bufs = [], []
for p in d("out/negatives/*.wav"):
    (neg_hard_bufs if voice_of(p) >= HARD_START else neg_rand_bufs).extend(neg_windows(read16(p)))
neg_rand = embed_all(neg_rand_bufs)
neg_hard = embed_all(neg_hard_bufs)

# синтетический шум + тихий шум
noise = embed_all(synth_noise(2000) +
                  [(RNG.randn(WIN) * RNG.uniform(50, 800)).clip(-32768, 32767).astype(np.int16) for _ in range(200)])
# эфир микрофона на разных уровнях
amb_bufs = []
for p in d("out/ambient/*.wav"):
    a = read16(p).astype(np.float32)
    for g in (0.7, 1.0, 1.5, 2.5, 4.0):
        gg = np.clip(a * g, -32768, 32767).astype(np.int16)
        amb_bufs += [gg[s:s + WIN] for s in range(0, len(gg) - WIN + 1, WIN // 4)]
amb = embed_all(amb_bufs)

# реальная речь (Golos) — готовые фичи
sp_path = os.path.join(HERE, "out", "neg_speech_feats.npy")
speech = np.load(sp_path).astype(np.float32) if os.path.exists(sp_path) else np.zeros((0, n_frames, n_feat), np.float32)
sp_hold, sp_train = speech[:1500], speech[1500:]
print(f"negatives: речь={sp_train.shape[0]} рандом={len(neg_rand)} хард={len(neg_hard)} шум={len(noise)} эфир={len(amb)}")

# ---------- сборка с пофакторными весами ----------
groups = [  # (features, weight)
    (sp_train, 1.0), (neg_rand, 1.0), (noise, 1.0), (amb, 1.0),
    (neg_hard, 4.0),                       # похожие слова — усиленный вес
]
neg = np.vstack([g[0] for g in groups])
neg_w = np.concatenate([np.full(len(g[0]), g[1], np.float32) for g in groups])
w_pos = float(neg_w.sum()) / max(len(pos), 1)   # баланс классов
X = np.vstack((neg, pos))
y = np.array([0] * len(neg) + [1] * len(pos), np.float32)[..., None]
sw = np.concatenate([neg_w, np.full(len(pos), w_pos, np.float32)])[..., None]
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(sw)),
    batch_size=512, shuffle=True)


class ConvHead(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.c1 = nn.Conv1d(nf, 64, 3, padding=1); self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 64, 3, padding=1); self.b2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.3); self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.drop(torch.relu(self.b1(self.c1(x))))
        x = torch.relu(self.b2(self.c2(x)))
        return torch.sigmoid(self.fc(self.pool(x).squeeze(-1)))


fcn = ConvHead(n_feat)
opt = torch.optim.Adam(fcn.parameters(), lr=8e-4)
bce = torch.nn.functional.binary_cross_entropy
for ep in range(60):
    fcn.train()
    for xb, yb, wb in dl:
        opt.zero_grad(); loss = bce(fcn(xb), yb, wb); loss.backward(); opt.step()

# ---------- ЧЕСТНАЯ ПРОВЕРКА (held-out) ----------
fcn.eval()
with torch.no_grad():
    hp = embed_all([augment_positive(read16(p)) for p in hold_pos for _ in range(4)]) if hold_pos else pos[:0]
    hp_sc = fcn(torch.from_numpy(hp)).numpy().flatten() if len(hp) else np.array([1.0])
    hard_sc = fcn(torch.from_numpy(neg_hard)).numpy().flatten() if len(neg_hard) else np.array([0.0])
    sp_sc = fcn(torch.from_numpy(sp_hold)).numpy().flatten() if len(sp_hold) else np.array([0.0])
print("\n=== held-out ===")
for th in [0.5, 0.7, 0.85, 0.9]:
    print(f"th={th}: recall(невид.голоса)={(hp_sc>=th).mean()*100:5.1f}%  "
          f"FP-хард={(hard_sc>=th).mean()*100:5.1f}%  FP-речь={(sp_sc>=th).mean()*100:.2f}%")

out = os.path.join(HERE, "kot_slushai.onnx")
torch.onnx.export(fcn, torch.zeros((1, n_frames, n_feat)), out, opset_version=13, dynamo=False,
                  input_names=["features"], output_names=["score"],
                  dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}})
print("\nЭкспортировано:", out)
