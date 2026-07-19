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

# Составные: несколько команд в одной фразе. Ожидание — список по порядку.
MULTI_CASES = [
    ("Открой нетворк на ветке 2070 и ADSV на ветке 3511",
     [("open_network", {"num": "2070"}), ("open_adsw", {"num": "3511"})]),
    ("открой адсв на ветке ард 100 и нетворк на ветке main",
     [("open_adsw", {"ticket": "ард", "num": "100"}),
      ("open_network", {"branch": "main"})]),
    # две команды с ОДИНАКОВЫМ интентом — их разделяет только B-
    ("открой адсв на ветке 111 и адсв на ветке 222",
     [("open_adsw", {"num": "111"}), ("open_adsw", {"num": "222"})]),
    ("покажи компоненты в идее а также проекты на ветке dev",
     [("open_components", {"target": "в идее"}),
      ("open_projects", {"branch": "dev"})]),
    ("открой нетворк на ITDEV 555, потом адсв на фиче про логин",
     [("open_network", {"ticket": "ITDEV", "num": "555"}),
      ("open_adsw", {"branch": "про логин"})]),
    ("открой адсв, нетворк и компоненты",
     [("open_adsw", {}), ("open_network", {}), ("open_components", {})]),
]


def norm(s):
    return s.lower().strip()


def as_expected(exp_i, exp_s):
    """Старый формат (интент, слоты) → новый контракт: список команд."""
    return [] if exp_i == "none" else [(exp_i, exp_s)]


def cmp_commands(got, expected):
    """Сравнивает список команд. expected элемент со слотами None — не проверяем."""
    if len(got) != len(expected):
        return False
    for g, (ei, es) in zip(got, expected):
        if g["intent"] != ei:
            return False
        if es is not None and ({k: norm(v) for k, v in g["slots"].items()} !=
                               {k: norm(v) for k, v in es.items()}):
            return False
    return True


def main():
    model, tok, cfg = jm.load(MODEL_DIR)

    jm.predict(model, tok, cfg, "прогрев")

    t0 = time.time()
    preds = [jm.predict(model, tok, cfg, text) for text, _, _ in CASES]
    dt = (time.time() - t0) / len(CASES)

    ok, fails = 0, []
    for (text, exp_i, exp_s), got in zip(CASES, preds):
        expected = as_expected(exp_i, exp_s)
        if cmp_commands(got, expected):
            ok += 1
        else:
            fails.append((text, expected, got))

    n = len(CASES)
    print(f"=== Одиночные команды: {ok}/{n} ({100*ok//n}%)  "
          f"|  {dt*1000:.2f} мс/запрос (CPU) ===")

    mok, mfails = 0, []
    for text, expected in MULTI_CASES:
        got = jm.predict(model, tok, cfg, text)
        if cmp_commands(got, expected):
            mok += 1
        else:
            mfails.append((text, expected, got))
    print(f"=== Составные команды: {mok}/{len(MULTI_CASES)} ===")

    if fails or mfails:
        print("\n=== Ошибки ===")
        for text, expected, got in fails + mfails:
            print(f"  «{text}»")
            print(f"        ждали: {expected}")
            print(f"        вышло: {[(g['intent'], g['slots']) for g in got]}")
    else:
        print("\nОшибок нет.")


if __name__ == "__main__":
    main()
