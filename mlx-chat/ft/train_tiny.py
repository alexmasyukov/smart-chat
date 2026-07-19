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

    # Группировка по длине: средняя фраза 9.4 слова, но 5% составных тянутся
    # до 42, и паддинг всех до глобального максимума удваивал стоимость шага
    # (char-RNN обрабатывает батч×ширину слов). Сортируем по длине, режем на
    # батчи, каждый паддим по СВОЕЙ ширине. Упаковка в модели делает это
    # безопасным: паддинг на результат не влияет, сколько его ни будь.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["words"]))
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]

    def make_batch(idx):
        w = max(len(rows[i]["words"]) for i in idx)
        ch, wi, mk, yi, yt = [], [], [], [], []
        for i in idx:
            r = rows[i]
            c, x, m = tm.encode(r["words"], c2i, w2i, w)
            ch.append(c)
            wi.append(x)
            mk.append(m)
            # обе метки потокенные; -100 на паддинге, чтобы он не влиял на лосс
            tags = torch.full((w,), -100, dtype=torch.long)
            itags = torch.full((w,), -100, dtype=torch.long)
            for j, t in enumerate(r["tags"][:w]):
                tags[j] = T2I[t]
            for j, t in enumerate(r["itags"][:w]):
                itags[j] = IT2I[t]
            yt.append(tags)
            yi.append(itags)
        return (torch.stack(ch), torch.stack(wi), torch.stack(mk),
                torch.stack(yi), torch.stack(yt))

    dl = [make_batch(b) for b in batches]
    print(f"батчей: {len(dl)}  |  средняя ширина: "
          f"{sum(b[0].shape[1] for b in dl)/len(dl):.1f} слов (было бы {max_words})")

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
        random.shuffle(dl)          # порядок батчей случайный, состав фиксирован
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
