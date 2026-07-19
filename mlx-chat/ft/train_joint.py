#!/usr/bin/env python3
"""Обучение joint-модели: интент + слоты (branch, target) одной ruBERT-tiny2.

Данные — gen_slots.py (templates.txt + phrases.txt с авторазметкой target).
Held-out — кейсы из test_joint.py, из обучения вычитаются.

    cd mlx-chat && .venv/bin/python ft/train_joint.py
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gen_slots as gs
import joint_model_def as jm
import test_joint
import test_whisper

random.seed(42)
torch.manual_seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(HERE, "joint_model")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 12
BATCH = 32
LR = 3e-4
MAXLEN = 48
# Слот-голова учится дольше: её сигнал реже, почти все токены — O. После того
# как пунктуация стала отдельными токенами, доля O выросла ещё, и веса 2.0
# перестало хватать — номер после префикса начал теряться.
SLOT_WEIGHT = 4.0

IT2I = {t: i for i, t in enumerate(jm.INTENT_TAGS)}
T2I = {t: i for i, t in enumerate(jm.TAGS)}


def build_data():
    rows = gs.build()
    holdout = {t.lower().strip() for t, _, _ in test_joint.CASES}
    holdout |= {t.lower().strip() for t, _ in test_joint.MULTI_CASES}
    holdout |= {t.lower().strip() for t, _, _, _ in test_whisper.CASES}
    rows = [r for r in rows if " ".join(r["words"]).lower().strip() not in holdout]
    return rows


def encode(tok, rows):
    """Слова+теги → тензоры. Обе метки на первый subword слова, хвосту -100."""
    enc = tok([r["words"] for r in rows], is_split_into_words=True,
              padding="max_length", truncation=True, max_length=MAXLEN,
              return_tensors="pt")
    slot_labels, intent_labels = [], []
    for i, r in enumerate(rows):
        word_ids = enc.word_ids(i)
        prev, sseq, iseq = None, [], []
        for wid in word_ids:
            if wid is None or wid == prev:
                sseq.append(-100)                      # спецтокены, паддинг, хвост
                iseq.append(-100)
            else:
                sseq.append(T2I[r["tags"][wid]])       # первый subword слова
                iseq.append(IT2I[r["itags"][wid]])
            prev = wid
        slot_labels.append(sseq)
        intent_labels.append(iseq)
    return (enc["input_ids"], enc["attention_mask"],
            torch.tensor(intent_labels), torch.tensor(slot_labels))


def main():
    rows = build_data()
    print(f"Обучающих примеров: {len(rows)}  |  устройство: {DEVICE}")

    tok = jm.AutoTokenizer.from_pretrained(jm.BASE)
    model = jm.JointClassifier().to(DEVICE)

    ids, mask, y_int, y_tag = encode(tok, rows)
    dl = DataLoader(TensorDataset(ids, mask, y_int, y_tag), batch_size=BATCH, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce_int = nn.CrossEntropyLoss(ignore_index=-100)
    ce_tag = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for ep in range(EPOCHS):
        li = lt = 0.0
        for b_ids, b_mask, b_int, b_tag in dl:
            b_ids, b_mask = b_ids.to(DEVICE), b_mask.to(DEVICE)
            b_int, b_tag = b_int.to(DEVICE), b_tag.to(DEVICE)
            opt.zero_grad()
            intent_logits, slot_logits = model(b_ids, b_mask)
            loss_i = ce_int(intent_logits.reshape(-1, len(jm.INTENT_TAGS)),
                            b_int.reshape(-1))
            loss_t = ce_tag(slot_logits.reshape(-1, len(jm.TAGS)), b_tag.reshape(-1))
            (loss_i + SLOT_WEIGHT * loss_t).backward()
            opt.step()
            li += loss_i.item()
            lt += loss_t.item()
        print(f"  эпоха {ep+1}/{EPOCHS}  intent {li/len(dl):.4f}  slots {lt/len(dl):.4f}")

    jm.save(model, tok, OUT)
    print(f"\nМодель сохранена: {OUT}")
    print("Проверка: .venv/bin/python ft/test_joint.py")


if __name__ == "__main__":
    main()
