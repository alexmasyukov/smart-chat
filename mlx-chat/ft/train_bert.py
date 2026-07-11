#!/usr/bin/env python3
"""Дообучение ruBERT-tiny2 (энкодер, 29M) как классификатора команд.

Тот же банк фраз (phrases.txt) и та же аугментация, что у 350M — сравнение честное.
Энкодер + классификационная голова: 5 классов. Обучается на CPU/MPS за минуту.

    python train_bert.py          # обучить + оценить на held-out (test80)
"""
import os
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import gen_dataset as gd
import test80

random.seed(42)
torch.manual_seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))

MODEL = "cointegrated/rubert-tiny2"
LABELS = ["open_adsw", "open_network", "open_components", "open_projects", "none"]
L2I = {l: i for i, l in enumerate(LABELS)}
OUT = os.path.join(HERE, "bert_model")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 6
BATCH = 32
LR = 3e-4  # голова + энкодер целиком (модель мелкая, можно полностью)


def build_data():
    core = gd.load_phrases(os.path.join(HERE, "phrases.txt"))
    rows = gd.build(core)                       # аугментированные (text, label)
    seen, uniq = set(), []
    for t, l in rows:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append((t, l))
    holdout = {t.lower() for t, _ in test80.CASES}   # честный held-out
    uniq = [(t, l) for t, l in uniq if t.lower() not in holdout]
    random.shuffle(uniq)
    return uniq


def encode(tok, texts):
    return tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")


def main():
    data = build_data()
    texts = [t for t, _ in data]
    labels = [L2I[l] for _, l in data]
    print(f"Обучающих примеров: {len(data)}  |  устройство: {DEVICE}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS),
        id2label={i: l for l, i in L2I.items()}, label2id=L2I,
    ).to(DEVICE)

    enc = encode(tok, texts)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for ids, mask, y in dl:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        print(f"  эпоха {ep+1}/{EPOCHS}  loss {tot/len(dl):.4f}")

    # eval на held-out test80
    model.eval()
    ok = 0
    fails = []
    import time
    t0 = time.time()
    with torch.no_grad():
        for text, exp in test80.CASES:
            e = encode(tok, [text])
            logits = model(input_ids=e["input_ids"].to(DEVICE),
                           attention_mask=e["attention_mask"].to(DEVICE)).logits
            got = LABELS[int(logits.argmax(-1))]
            ok += got == exp
            if got != exp:
                fails.append((text, exp, got))
    dt = (time.time() - t0) / len(test80.CASES)
    print(f"\n=== ruBERT-tiny2: {ok}/{len(test80.CASES)}  "
          f"({100*ok//len(test80.CASES)}%)  |  {dt*1000:.1f} мс/запрос ({DEVICE}) ===")
    if fails:
        print("Ошибки:")
        for text, exp, got in fails:
            print(f"  «{text}»: {got}, ждали {exp}")

    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"\nМодель сохранена: {OUT}")


if __name__ == "__main__":
    main()
