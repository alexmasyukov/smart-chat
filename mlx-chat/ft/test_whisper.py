#!/usr/bin/env python3
"""Регрессия на ЖИВЫХ расшифровках Whisper из голосового ассистента.

Это не придуманные фразы: они сняты с боевого лога
`~/Library/Logs/Assistant/commands.jsonl` и приложены потребителем модели
(`my-pro/assistant/MODEL_REQUESTS.md`). Отсюда всё, чего нет в других тестах:

  - точка в конце почти всегда, запятые внутри («Открой ADSV ARD 1111.»);
  - Whisper коверкает названия: adsw → «ADSV», «АДСВ», «Адрес свой»;
  - латиница и кириллица мешаются в одной фразе;
  - слова склеиваются и рвутся: «фичеветку», «Fitch-оветку», «ит дев»;
  - порядок слов свободный: «Фича ветку Network открой, ARD 1911».

Проверяем интент и пару (ticket, num) — имя стенда собирается из них.
Регистр и пунктуация в значениях не важны, сравниваем нормализованно.

    cd mlx-chat && .venv/bin/python ft/test_whisper.py
"""
import os
import time

import joint_model_def as jm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "joint_model")

# (расшифровка, интент, ticket, num) — ticket/num None, если их быть не должно
CASES = [
    ("Открой фича ветку нетворка на ард 1863", "open_network", "ард", "1863"),
    ("Открой адсв ветку, фича-ветку на ARD 2020.", "open_adsw", "ARD", "2020"),
    ("Открой нетворк на ветке ARD2020.", "open_network", "ARD", "2020"),
    ("На нетворке открой фича-ветку ARD-2000.", "open_network", "ARD", "2000"),
    ("АДСВ, открой фича-ветку ARD-1919.", "open_adsw", "ARD", "1919"),
    ("Адрес свой фичеветку открой, ARD 1980.", "open_adsw", "ARD", "1980"),
    ("Открой Network на ITDEV 1897.", "open_network", "ITDEV", "1897"),
    ("Открой АДСВ на ветке ITDEV 1981.", "open_adsw", "ITDEV", "1981"),
    ("Натворк открой на ветке 1987 ITDEV.", "open_network", "ITDEV", "1987"),
    ("открой ветку нетворк ит дев 204", "open_network", "ит дев", "204"),
    ("Фича ветку Network открой, ARD 1911.", "open_network", "ARD", "1911"),
    ("Открой Network Fitch-оветку ARD-2020.", "open_network", "ARD", "2020"),
    ("Открой ADSV ARD 1111.", "open_adsw", "ARD", "1111"),
    ("Плохо распознает название номер ветки, ты режешь что-то с конца",
     "none", None, None),
    ("Чё ты как вообще? Жизнь-то у тебя?", "none", None, None),
    ("Создай MD-файл.", "none", None, None),
    ("Закоммить. Закоммить приложение.", "none", None, None),
]


def norm(s):
    """Нормализация для сравнения: регистр и пунктуация не важны."""
    if s is None:
        return None
    return s.lower().replace("-", " ").replace(".", "").strip()


def main():
    model, tok, cfg = jm.load(MODEL_DIR)
    jm.predict(model, tok, cfg, "прогрев")

    t0 = time.time()
    preds = [jm.predict(model, tok, cfg, text) for text, _, _, _ in CASES]
    dt = (time.time() - t0) / len(CASES)

    ok = 0
    fails = []
    for (text, exp_i, exp_t, exp_n), (got_i, slots, score) in zip(CASES, preds):
        got_t, got_n = slots.get("ticket"), slots.get("num")
        good = (got_i == exp_i and norm(got_t) == norm(exp_t)
                and norm(got_n) == norm(exp_n))
        ok += good
        if not good:
            fails.append((text, exp_i, exp_t, exp_n, got_i, got_t, got_n, slots))

    n = len(CASES)
    print(f"=== Живые расшифровки Whisper: {ok}/{n} ({100*ok//n}%) ===")
    print(f"    {dt*1000:.2f} мс/запрос (CPU)")

    if fails:
        print("\n=== Ошибки ===")
        for text, ei, et, en, gi, gt, gn, slots in fails:
            print(f"  «{text}»")
            print(f"        ждали: {ei:14s} ticket={et}  num={en}")
            print(f"        вышло: {gi:14s} ticket={gt}  num={gn}   (все слоты: {slots})")
    else:
        print("\nОшибок нет.")

    # уверенность: потребитель применяет порог 0.7
    print("\n=== Уверенность интента ===")
    low = [(t, i, s) for (t, _, _, _), (i, _, s) in zip(CASES, preds) if s < 0.7]
    if low:
        print("  ниже порога 0.7:")
        for t, i, s in low:
            print(f"    {s:.3f}  {i:14s} «{t}»")
    else:
        print("  все выше порога 0.7")


if __name__ == "__main__":
    main()
