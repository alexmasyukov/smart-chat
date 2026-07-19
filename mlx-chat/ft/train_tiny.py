#!/usr/bin/env python3
"""Обучение модели с нуля (tiny_model_def) на тех же данных, что joint-модель.

Никакого предобучения: веса стартуют случайными, всё знание — из наших 5900
примеров. Held-out тот же, что у ruBERT, поэтому сравнение честное.

    cd mlx-chat && .venv/bin/python ft/train_tiny.py
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gen_slots as gs
import test_joint
import test_whisper
import tiny_model_def as tm

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
torch.manual_seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "tiny_model"))

# MPS быстрее CPU в 7 раз (36 с против 4 мин 16 с на 40 эпохах) — замерено,
# а не предположено: интуиция «LSTM на MPS плохо оптимизирован, модель мелкая,
# накладные расходы съедят выигрыш» оказалась неверной.
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = 64
LR = 2e-3
SLOT_WEIGHT = 4.0

IT2I = {t: i for i, t in enumerate(tm.INTENT_TAGS)}
T2I = {t: i for i, t in enumerate(tm.TAGS)}


def build_data():
    rows = gs.build()
    holdout = {t.lower().strip() for t, _, _ in test_joint.CASES}
    holdout |= {t.lower().strip() for t, _ in test_joint.MULTI_CASES}
    holdout |= {t.lower().strip() for t, _, _, _ in test_whisper.CASES}
    return [r for r in rows if " ".join(r["words"]).lower().strip() not in holdout]


def main():
    rows = build_data()
    max_words = max(len(r["words"]) for r in rows)
    c2i, w2i = tm.build_vocabs(rows)
    print(f"примеров: {len(rows)}  |  алфавит: {len(c2i)}  |  словарь: {len(w2i)}"
          f"  |  макс слов: {max_words}")

    chars, words, masks, y_int, y_tag = [], [], [], [], []
    for r in rows:
        ch, wi, mask = tm.encode(r["words"], c2i, w2i, max_words)
        chars.append(ch)
        words.append(wi)
        masks.append(mask)
        # обе метки потокенные; -100 на паддинге, чтобы он не влиял на лосс
        tags = torch.full((max_words,), -100, dtype=torch.long)
        itags = torch.full((max_words,), -100, dtype=torch.long)
        for i, t in enumerate(r["tags"][:max_words]):
            tags[i] = T2I[t]
        for i, t in enumerate(r["itags"][:max_words]):
            itags[i] = IT2I[t]
        y_tag.append(tags)
        y_int.append(itags)

    ds = TensorDataset(torch.stack(chars), torch.stack(words), torch.stack(masks),
                       torch.stack(y_int), torch.stack(y_tag))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = tm.TinyTagger(len(c2i) + 2, len(w2i) + 2).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n/1000:.0f}K  ({n*4/1024/1024:.1f} МБ fp32)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce_i = nn.CrossEntropyLoss(ignore_index=-100)
    ce_t = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for ep in range(EPOCHS):
        li = lt = 0.0
        for ch, wi, mask, bi, bt in dl:
            opt.zero_grad()
            il, sl = model(ch.to(DEVICE), wi.to(DEVICE), mask.to(DEVICE))
            loss_i = ce_i(il.reshape(-1, len(tm.INTENT_TAGS)),
                          bi.to(DEVICE).reshape(-1))
            loss_t = ce_t(sl.reshape(-1, len(tm.TAGS)), bt.to(DEVICE).reshape(-1))
            (loss_i + SLOT_WEIGHT * loss_t).backward()
            opt.step()
            li += loss_i.item()
            lt += loss_t.item()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  эпоха {ep+1}/{EPOCHS}  intent {li/len(dl):.4f}  slots {lt/len(dl):.4f}")

    tm.save(model, {"chars": c2i, "words": w2i, "max_words": max_words}, OUT)
    print(f"\nМодель сохранена: {OUT}")
    print("Проверка: .venv/bin/python ft/test_tiny.py")


if __name__ == "__main__":
    main()
