#!/usr/bin/env python3
"""Честное сравнение ruBERT-tiny2 (29M) vs дообученной LFM2.5-350M на
СВЕЖИХ формулировках (нет ни в обучении, ни в test80) + замер скорости.
"""
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import mlx_lm

LABELS = ["open_adsw", "open_network", "open_components", "open_projects", "none"]
SYS = "Ты — классификатор. По запросу пользователя верни ровно одно имя инструмента из набора или none. Только имя, без пояснений."

# Свежие, нарочно «живые» формулировки — ни в phrases.txt, ни в test80.
NOVEL = [
    ("слушай а где там наши компоненты лежат", "open_components"),
    ("мне бы адсв папочку", "open_adsw"),
    ("нужен нетворк каталог", "open_network"),
    ("проектики мои открой", "open_projects"),
    ("покажи-ка библиотеку нашу ui", "open_components"),
    ("закинь меня в адсв", "open_adsw"),
    ("хочу глянуть network директорию", "open_network"),
    ("где лежит my-pro", "open_projects"),
    ("аренадата адсв", "open_adsw"),
    ("открой пожалуйста папочку с компонентами библиотеки", "open_components"),
    ("чё по проектам, открой", "open_projects"),
    ("открой пж нетворк", "open_network"),
    ("дай попасть в папку адсв", "open_adsw"),
    ("врубай нетворк", "open_network"),
    ("хочу увидеть свои проекты в майпро", "open_projects"),
    ("нужны наши юай компоненты", "open_components"),
    ("открой мне глаза", "none"),
    ("открой окно пошире", "none"),
    ("сколько будет дважды два", "none"),
    ("открой папку с котиками", "none"),
]


def run_bert():
    tok = AutoTokenizer.from_pretrained("ft/bert_model")
    m = AutoModelForSequenceClassification.from_pretrained("ft/bert_model").to("cpu").eval()
    def clf(t):
        e = tok([t], padding=True, truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            return LABELS[int(m(**e).logits.argmax(-1))]
    clf("прогрев")
    res, t0 = [], time.time()
    for text, exp in NOVEL:
        res.append((text, exp, clf(text)))
    return res, (time.time() - t0) / len(NOVEL)


def run_mlx():
    base, tok = mlx_lm.load("ft/fused")
    def clf(t):
        p = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": t}],
            add_generation_prompt=True)
        return mlx_lm.generate(base, tok, p, max_tokens=8, verbose=False).strip().split()[0]
    clf("прогрев")
    res, t0 = [], time.time()
    for text, exp in NOVEL:
        res.append((text, exp, clf(text)))
    return res, (time.time() - t0) / len(NOVEL)


def score(res):
    return sum(1 for _, e, g in res if g == e)


def main():
    print("Гружу модели …")
    br, bt = run_bert()
    mr, mt = run_mlx()
    print(f"\n{'запрос':42s} {'ждали':16s} {'ruBERT-29M':16s} {'LFM-350M':16s}")
    print("-" * 92)
    for (text, exp, gb), (_, _, gm) in zip(br, mr):
        mb = "✓" if gb == exp else "✗"
        mm = "✓" if gm == exp else "✗"
        print(f"{text[:40]:42s} {exp:16s} {mb+' '+gb:16s} {mm+' '+gm:16s}")
    print("-" * 92)
    print(f"{'ИТОГ':42s} {'':16s} {str(score(br))+'/'+str(len(NOVEL)):16s} "
          f"{str(score(mr))+'/'+str(len(NOVEL)):16s}")
    print(f"{'скорость (мс/запрос, прогретые)':42s} {'':16s} "
          f"{bt*1000:.1f} мс (CPU){'':6s} {mt*1000:.1f} мс")


if __name__ == "__main__":
    main()
