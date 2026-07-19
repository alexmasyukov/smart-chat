#!/usr/bin/env python3
"""Held-out тест joint-модели: интент и слоты меряются ОТДЕЛЬНО.

Две метрики, потому что одной мало: модель может угадать папку и потерять хвост
номера («ард 1120» → «ард»). Итог «всё верно» = интент И все слоты совпали.

Кейсы писались руками и в train не попадают (train_joint.py их вычитает).

Ожидаемые слоты `None` = «не проверяем»: при интенте none команда не
исполняется, поэтому слоты в контракт не входят и потребитель их не читает.

    cd mlx-chat && .venv/bin/python ft/test_joint.py
"""
import os
import time

import joint_model_def as jm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "joint_model")

# (запрос, интент, {слот: значение})
CASES = [
    # --- боевой пример: оба слота сразу, тикет разобран на части ---
    ("открой в браузере адсв на фиче ветке ард 1120", "open_adsw",
     {"target": "в браузере", "ticket": "ард", "num": "1120"}),

    # --- тикеты: ticket и num РАЗДЕЛЬНО, номера НЕ из обучения ---
    ("открой адсв на ветке ард 7777", "open_adsw", {"ticket": "ард", "num": "7777"}),
    ("открой нетворк на ветке ARD-4242", "open_network", {"ticket": "ARD", "num": "4242"}),
    ("компоненты на фиче ард 315", "open_components", {"ticket": "ард", "num": "315"}),
    ("переключи проекты на ard-8080", "open_projects", {"ticket": "ard", "num": "8080"}),
    ("открой адсв на фиче 999", "open_adsw", {"num": "999"}),

    # --- branch: постоянные ветки ---
    ("открой нетворк на ветке main", "open_network", {"branch": "main"}),
    ("открой проекты на мастере", "open_projects", {"branch": "мастере"}),
    ("компоненты на dev", "open_components", {"branch": "dev"}),

    # --- branch: произвольные, регуляркой не взять ---
    ("открой адсв на ветке про логин", "open_adsw", {"branch": "про логин"}),
    ("открой нетворк на фиче про темную тему", "open_network", {"branch": "про темную тему"}),
    ("открой компоненты на ветке feature/new-header", "open_components",
     {"branch": "feature/new-header"}),
    ("открой адсв на ветке васи", "open_adsw", {"branch": "васи"}),
    ("проекты на тестовой ветке", "open_projects", {"branch": "тестовой"}),
    ("открой нетворк на fix/crash-on-start", "open_network", {"branch": "fix/crash-on-start"}),

    # --- target без branch ---
    ("открой в финдере адсв", "open_adsw", {"target": "в финдере"}),
    ("покажи компоненты в вскоде", "open_components", {"target": "в вскоде"}),
    ("открой проекты в идее", "open_projects", {"target": "в идее"}),
    ("открой нетворк в терминале", "open_network", {"target": "в терминале"}),

    # --- оба слота ---
    ("открой в идее компоненты на ветке ард 55", "open_components",
     {"target": "в идее", "ticket": "ард", "num": "55"}),
    ("открой в терминале нетворк на ветке main", "open_network",
     {"target": "в терминале", "branch": "main"}),
    ("покажи проекты в хроме на фиче про хедер", "open_projects",
     {"target": "в хроме", "branch": "про хедер"}),

    # --- без слотов вообще: старое поведение не сломалось ---
    ("открой адсв", "open_adsw", {}),
    ("открой папку нетворк", "open_network", {}),
    ("покажи компоненты", "open_components", {}),
    ("открой мои проекты", "open_projects", {}),
    ("зайди в адсв", "open_adsw", {}),

    # --- none: near-miss, слот-слова есть, папки из набора нет ---
    ("открой ветку ард 500", "none", None),
    # branch при none подавляется: на посторонней речи это не имя ветки
    ("создай ветку про оплату", "none", {}),
    ("какая ветка сейчас", "none", {}),
    ("открой браузер", "none", {}),
    ("открой в браузере ютуб", "none", {"target": "в браузере"}),
    ("смержи ард 42", "none", None),
    ("привет", "none", {}),
    ("2+2", "none", {}),
    ("открой youtube", "none", {}),
]


def norm(s):
    return s.lower().strip()


def main():
    model, tok, cfg = jm.load(MODEL_DIR)

    jm.predict(model, tok, cfg, "прогрев")

    t0 = time.time()
    preds = [jm.predict(model, tok, cfg, text)[:2] for text, _, _ in CASES]
    dt = (time.time() - t0) / len(CASES)

    intent_ok = slot_ok = both_ok = 0
    fails = []
    for (text, exp_i, exp_s), (got_i, got_s) in zip(CASES, preds):
        i_ok = got_i == exp_i
        s_ok = exp_s is None or ({k: norm(v) for k, v in got_s.items()} ==
                                 {k: norm(v) for k, v in exp_s.items()})
        intent_ok += i_ok
        slot_ok += s_ok
        both_ok += i_ok and s_ok
        if not (i_ok and s_ok):
            fails.append((text, exp_i, exp_s, got_i, got_s, i_ok, s_ok))

    n = len(CASES)
    print(f"=== Joint-модель на {n} held-out кейсах ===")
    print(f"  интент:        {intent_ok}/{n} ({100*intent_ok//n}%)")
    print(f"  слоты (точно): {slot_ok}/{n} ({100*slot_ok//n}%)")
    print(f"  всё верно:     {both_ok}/{n} ({100*both_ok//n}%)")
    print(f"  скорость:      {dt*1000:.2f} мс/запрос (CPU)")

    if fails:
        print("\n=== Ошибки ===")
        for text, ei, es, gi, gs_, i_ok, s_ok in fails:
            what = "слот" if i_ok else ("интент" if s_ok else "интент+слот")
            print(f"  [{what}] «{text}»")
            print(f"        ждали: {ei}  {es}")
            print(f"        вышло: {gi}  {gs_}")
    else:
        print("\nОшибок нет.")


if __name__ == "__main__":
    main()
