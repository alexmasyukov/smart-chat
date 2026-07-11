#!/usr/bin/env python3
"""80 тестов дообученной модели — свободная генерация (greedy), как в LM Studio
без Structured Output. Меряем точность и скорость на разнообразных запросах.
"""
import time
import mlx_lm

SYS = "Ты — классификатор. По запросу пользователя верни ровно одно имя инструмента из набора или none. Только имя, без пояснений."
NAMES = ["open_adsw", "open_network", "open_components", "open_projects", "none"]

CASES = [
    # --- open_adsw ---
    ("открой папку adsw", "open_adsw"),
    ("открой адсв", "open_adsw"),
    ("зайди в адсв", "open_adsw"),
    ("adsw открой", "open_adsw"),
    ("перейди в папку adsw", "open_adsw"),
    ("открой каталог адсв", "open_adsw"),
    ("мне нужна папка adsw", "open_adsw"),
    ("адсв папку покажи", "open_adsw"),
    ("открой arenadata-adsw", "open_adsw"),
    ("оскрой папку адсв", "open_adsw"),
    ("ОТКРОЙ АДСВ", "open_adsw"),
    ("слушай открой adsw пожалуйста", "open_adsw"),
    ("давай adsw", "open_adsw"),
    ("хочу adsw открыть", "open_adsw"),
    ("открой директорию adsw", "open_adsw"),
    ("адсв", "open_adsw"),
    # --- open_network ---
    ("открой папку network", "open_network"),
    ("открой нетворк", "open_network"),
    ("зайди в нетворк", "open_network"),
    ("network открой", "open_network"),
    ("перейди в папку network", "open_network"),
    ("открой каталог нетворк", "open_network"),
    ("мне нужна папка нетворк", "open_network"),
    ("нетворк папку покажи", "open_network"),
    ("открой arenadata-network", "open_network"),
    ("открой нетворкк", "open_network"),
    ("ОТКРОЙ NETWORK", "open_network"),
    ("слушай открой network пожалуйста", "open_network"),
    ("давай нетворк", "open_network"),
    ("хочу открыть network", "open_network"),
    ("открой директорию нетворк", "open_network"),
    ("нетворк", "open_network"),
    # --- open_components ---
    ("открой компоненты", "open_components"),
    ("открой папку с компонентами", "open_components"),
    ("открой библиотеку компонентов", "open_components"),
    ("покажи компоненты", "open_components"),
    ("открой ui", "open_components"),
    ("открой папку ui", "open_components"),
    ("наши компоненты покажи", "open_components"),
    ("открой arenadata-ui", "open_components"),
    ("открой компаненты", "open_components"),
    ("ОТКРОЙ КОМПОНЕНТЫ", "open_components"),
    ("открой нашу библиотеку компонентов", "open_components"),
    ("мне нужны компоненты", "open_components"),
    ("зайди в папку компонентов", "open_components"),
    ("давай компоненты", "open_components"),
    ("открой ui кит", "open_components"),
    ("компоненты открой быстро", "open_components"),
    # --- open_projects ---
    ("открой проекты", "open_projects"),
    ("открой мои проекты", "open_projects"),
    ("открой my-pro", "open_projects"),
    ("открой папку с проектами", "open_projects"),
    ("покажи мои проекты", "open_projects"),
    ("открой папку my-pro", "open_projects"),
    ("открой майпро", "open_projects"),
    ("открой папку проектов", "open_projects"),
    ("открой праекты", "open_projects"),
    ("ОТКРОЙ ПРОЕКТЫ", "open_projects"),
    ("где мои проекты", "open_projects"),
    ("зайди в папку с проектами", "open_projects"),
    ("открой рабочие проекты", "open_projects"),
    ("давай проекты", "open_projects"),
    ("папку с проектами открой", "open_projects"),
    ("открой папку с проектами минина", "open_projects"),
    # --- none ---
    ("привет", "none"),
    ("как дела", "none"),
    ("спасибо", "none"),
    ("расскажи анекдот", "none"),
    ("который час", "none"),
    ("открой youtube", "none"),
    ("открой браузер", "none"),
    ("открой документы", "none"),
    ("открой загрузки", "none"),
    ("2+2", "none"),
    ("пока", "none"),
    ("что ты умеешь", "none"),
    ("включи музыку", "none"),
    ("открой terminal", "none"),
    ("погода на завтра", "none"),
    ("открой корзину", "none"),
]


def parse(out):
    s = out.strip().lower()
    for n in NAMES:
        if s.startswith(n):
            return n
    for n in NAMES:
        if n in s:
            return n
    return s[:20]


def main():
    base, tok = mlx_lm.load("ft/fused")
    ok = 0
    total = 0.0
    fails = []
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for text, exp in CASES:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text}]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True)
        t = time.time()
        out = mlx_lm.generate(base, tok, p, max_tokens=12, verbose=False)
        total += time.time() - t
        got = parse(out)
        good = got == exp
        ok += good
        per[exp][0] += good
        per[exp][1] += 1
        mark = "✓" if good else "✗"
        line = f"  {mark} «{text}» → {got}" + ("" if good else f"  (ждали {exp})")
        print(line)
        if not good:
            fails.append((text, exp, got))
    print(f"\n=== ИТОГ: {ok}/{len(CASES)}  ({100*ok//len(CASES)}%)  |  {total/len(CASES):.3f}s/запрос ===")
    print("По классам:")
    for cls in NAMES:
        g, n = per[cls]
        print(f"  {cls:16s} {g}/{n}")
    if fails:
        print("\nОшибки:")
        for text, exp, got in fails:
            print(f"  «{text}»: получили {got}, ждали {exp}")


if __name__ == "__main__":
    main()
