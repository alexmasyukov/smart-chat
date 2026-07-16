#!/usr/bin/env python3
"""Полный тест ruBERT-tiny2 по всем наборам сразу — с проверкой честности.

Перед замером строит ровно тот набор, на котором учится train_bert.py, и
пересекает его с каждым тестовым набором: если фраза просочилась в обучение,
её результат не считается обобщением, и это печатается явно.

Наборы: test80.CASES (80), eval.CASES (26), compare.NOVEL (20).

    cd mlx-chat && .venv/bin/python ft/test_bert_full.py
"""
import os
import time
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import compare
import eval as ev
import test80
import train_bert as tb

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = tb.LABELS

SUITES = [
    ("test80", test80.CASES),
    ("eval.CASES", ev.CASES),
    ("compare.NOVEL", compare.NOVEL),
]


def main():
    train_keys = {t.lower().strip() for t, _ in tb.build_data()}
    print(f"Обучающих примеров у ruBERT: {len(train_keys)}\n")

    print("=== Чистота наборов (утечка в train) ===")
    leaks = {}
    for name, cases in SUITES:
        hit = [t for t, _ in cases if t.lower().strip() in train_keys]
        leaks[name] = hit
        status = "чист" if not hit else f"УТЕЧКА {len(hit)}/{len(cases)}"
        print(f"  {name:16s} {len(cases):3d} кейсов  → {status}")
        for t in hit:
            print(f"      просочилось: «{t}»")

    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "bert_model"))
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(HERE, "bert_model")).to("cpu").eval()

    def clf(text):
        e = tok([text], padding=True, truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            return LABELS[int(model(**e).logits.argmax(-1))]

    clf("прогрев")

    print()
    fails, tot, good = [], Counter(), Counter()
    for name, cases in SUITES:
        t0 = time.time()
        got = [clf(t) for t, _ in cases]
        dt = (time.time() - t0) / len(cases)
        ok = 0
        for (text, exp), g in zip(cases, got):
            ok += g == exp
            tot[exp] += 1
            good[exp] += g == exp
            if g != exp:
                fails.append((name, text, exp, g))
        note = "held-out" if not leaks[name] else f"загрязнён на {len(leaks[name])}"
        print(f"=== {name}: {ok}/{len(cases)} ({100*ok//len(cases)}%)  "
              f"|  {dt*1000:.2f} мс/запрос (CPU)  |  {note} ===")

    print()
    if fails:
        print("=== Ошибки ===")
        for suite, text, exp, got_ in fails:
            print(f"  [{suite}] «{text}» → {got_}, ждали {exp}")
    else:
        print("Ошибок нет ни в одном наборе.")

    print("\n=== По классам (все наборы вместе) ===")
    for l in LABELS:
        if tot[l]:
            print(f"  {l:18s} {good[l]:3d}/{tot[l]:3d}")


if __name__ == "__main__":
    main()
