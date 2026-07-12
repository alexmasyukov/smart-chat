#!/usr/bin/env python3
"""
Шаг 2/2: обучение головы на КЭШИРОВАННЫХ признаках (out/cache/ + neg_speech_feats.npy).
Быстро (секунды-минута) — признаки не пересчитываются. Сначала запусти build_cache.py.

Запуск:  .venv/bin/python train_local.py
"""
import os
import numpy as np
import torch, torch.nn as nn
import wake_common as C

n_frames, n_feat = C.n_frames, C.n_feat


def load(name):
    p = os.path.join(C.CACHE, f"feat_{name}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"нет кэша {p} — сначала запусти build_cache.py")
    return np.load(p).astype(np.float32)


pos = load("pos")
neg_rand = load("neg_rand")
neg_hard = load("neg_hard")
noise = load("noise")
amb = load("amb")
sp_path = os.path.join(C.HERE, "out", "neg_speech_feats.npy")
speech = np.load(sp_path).astype(np.float32) if os.path.exists(sp_path) else np.zeros((0, n_frames, n_feat), np.float32)
sp_hold, sp_train = speech[:1500], speech[1500:]
print(f"pos={len(pos)} | речь={len(sp_train)} рандом={len(neg_rand)} хард={len(neg_hard)} шум={len(noise)} эфир={len(amb)}")

# ---------- сборка с пофакторными весами ----------
groups = [(sp_train, 1.0), (neg_rand, 1.0), (noise, 1.0), (amb, 1.0), (neg_hard, 4.0)]
neg = np.vstack([g[0] for g in groups])
neg_w = np.concatenate([np.full(len(g[0]), g[1], np.float32) for g in groups])
w_pos = float(neg_w.sum()) / max(len(pos), 1)
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


def spec_augment(xb):
    """Векторизованный SpecAugment: маска по времени и каналам, без питон-цикла."""
    B, T, Cc = xb.shape
    xb = xb.clone()
    dev = xb.device
    # маска по времени (у половины примеров)
    tlen = torch.randint(0, 4, (B, 1), device=dev)
    tstart = (torch.rand(B, 1, device=dev) * (T - tlen).clamp(min=1)).long()
    ar_t = torch.arange(T, device=dev).view(1, T)
    tmask = (ar_t >= tstart) & (ar_t < tstart + tlen) & (torch.rand(B, 1, device=dev) < 0.5)
    xb[tmask.unsqueeze(-1).expand(-1, -1, Cc)] = 0
    # маска по каналам
    clen = torch.randint(0, 12, (B, 1), device=dev)
    cstart = (torch.rand(B, 1, device=dev) * (Cc - clen).clamp(min=1)).long()
    ar_c = torch.arange(Cc, device=dev).view(1, Cc)
    cmask = (ar_c >= cstart) & (ar_c < cstart + clen) & (torch.rand(B, 1, device=dev) < 0.5)
    xb[cmask.unsqueeze(1).expand(-1, T, -1)] = 0
    return xb


dev = "mps" if torch.backends.mps.is_available() else "cpu"
fcn = ConvHead(n_feat).to(dev)
opt = torch.optim.Adam(fcn.parameters(), lr=8e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70)
bce = torch.nn.functional.binary_cross_entropy
import time as _t; _t0 = _t.time()
for ep in range(70):
    fcn.train()
    for xb, yb, wb in dl:
        xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
        opt.zero_grad(); loss = bce(fcn(spec_augment(xb)), yb, wb); loss.backward(); opt.step()
    sched.step()
fcn = fcn.to("cpu")
print(f"обучение головы: {_t.time()-_t0:.1f}с (device={dev})")

# ---------- ЧЕСТНАЯ ПРОВЕРКА (held-out: невиданные голоса + отложенная речь) ----------
fcn.eval()
hold_pos = [p for p in C.d("out/positives/*.wav") if C.voice_of(p) >= C.HOLD_FROM]
with torch.no_grad():
    hp = C.embed_all([C.augment_positive(C.read16(p)) for p in hold_pos for _ in range(4)]) if hold_pos else pos[:0]
    hp_sc = fcn(torch.from_numpy(hp)).numpy().flatten() if len(hp) else np.array([1.0])
    hard_sc = fcn(torch.from_numpy(neg_hard)).numpy().flatten() if len(neg_hard) else np.array([0.0])
    sp_sc = fcn(torch.from_numpy(sp_hold)).numpy().flatten() if len(sp_hold) else np.array([0.0])
print(f"\n=== held-out (голосов {len(hold_pos)}) ===")
for th in [0.5, 0.7, 0.85, 0.9]:
    print(f"th={th}: recall(невид.голоса)={(hp_sc>=th).mean()*100:5.1f}%  "
          f"FP-хард={(hard_sc>=th).mean()*100:5.1f}%  FP-речь={(sp_sc>=th).mean()*100:.2f}%")

out = os.path.join(C.HERE, "kot_slushai.onnx")
torch.onnx.export(fcn, torch.zeros((1, n_frames, n_feat)), out, opset_version=13, dynamo=False,
                  input_names=["features"], output_names=["score"],
                  dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}})
print("\nЭкспортировано:", out)
