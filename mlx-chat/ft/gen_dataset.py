#!/usr/bin/env python3
"""Генератор датасета для LoRA-файнтюна классификатора команд «открыть папку».

Читает живые формулировки из phrases.txt (человекочитаемый банк фраз) и к каждой
добавляет вариации: опечатки, разный регистр, лишние слова, перестановку слов.

Пишет:
  data/train.jsonl, data/valid.jsonl — chat-формат для mlx-lm;
  data/preview.md                    — весь датасет по-человечески, по классам.

Класс none обязателен: учит модель отвечать «ничего не подходит» на болтовню и
посторонние команды, а не открывать случайную папку.

Held-out фразы из eval.py в train не попадают.
"""
import json
import os
import random

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = "Ты — классификатор. По запросу пользователя верни ровно одно имя инструмента из набора или none. Только имя, без пояснений."

FILLERS = ["пожалуйста", "плиз", "слушай", "давай", "ну", "мне", "можешь", "быстро",
           "срочно", "эй", "короче", "так", "будь добр", "го", "щас", "надо", "пож"]

CYR = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz-"


def load_phrases(path):
    """Парсит phrases.txt → {label: [фразы]}. Секция начинается с '# <label> ...'."""
    core, label = {}, None
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        if s.strip().startswith("#"):
            # заголовок секции: "# open_adsw  -> ..."
            head = s.strip("# ").split("->")[0].strip()
            token = head.split()[0] if head else ""
            if token and (token == "none" or token.startswith("open_")):
                label = token
                core.setdefault(label, [])
            continue
        if not s.strip():
            continue
        if label:
            core[label].append(s.strip())
    return core


def typo(word):
    if len(word) < 4:
        return word
    i = random.randint(1, len(word) - 2)
    op = random.choice(["swap", "drop", "dup", "sub"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    return word[:i] + random.choice(CYR) + word[i + 1:]


def recase(s):
    r = random.random()
    if r < 0.4:
        return s
    if r < 0.58:
        return s.capitalize()
    if r < 0.7:
        return s.upper()
    if r < 0.85:
        return s.lower()
    return " ".join(w.capitalize() if random.random() < 0.5 else w for w in s.split())


def augment(phrase):
    words = phrase.split()
    if len(words) > 2 and random.random() < 0.3:
        random.shuffle(words)
    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
            words.insert(random.randint(0, len(words)), random.choice(FILLERS))
    if random.random() < 0.4 and words:
        j = random.randrange(len(words))
        words[j] = typo(words[j])
    return recase(" ".join(words))


def build(core):
    rows = []
    for label, phrases in core.items():
        # none аугментируем слабее (фраз и так много, разнообразие естественное)
        per = 4 if label == "none" else 22
        for ph in phrases:
            rows.append((ph, label))
            for _ in range(per):
                rows.append((augment(ph), label))
    return rows


def main():
    core = load_phrases(os.path.join(HERE, "phrases.txt"))
    rows = build(core)

    # дедуп
    seen, uniq = set(), []
    for text, label in rows:
        k = text.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append((text, label))
    rows = uniq
    random.shuffle(rows)

    # held-out из eval убираем
    try:
        import eval as ev  # noqa
        holdout = {t.lower() for t, _ in ev.CASES}
    except Exception:
        holdout = set()
    rows = [(t, l) for t, l in rows if t.lower() not in holdout]

    n = len(rows)
    nval = max(30, n // 10)
    valid, train = rows[:nval], rows[nval:]

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for name, part in [("train", train), ("valid", valid)]:
        with open(os.path.join(HERE, "data", f"{name}.jsonl"), "w") as f:
            for text, label in part:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": label},
                ]}, ensure_ascii=False) + "\n")

    # человекочитаемый предпросмотр всего датасета
    from collections import Counter, defaultdict
    dist = Counter(l for _, l in rows)
    by_label = defaultdict(list)
    for t, l in rows:
        by_label[l].append(t)
    with open(os.path.join(HERE, "data", "preview.md"), "w", encoding="utf-8") as f:
        f.write(f"# Датасет классификатора — {n} примеров\n\n")
        f.write("Сгенерировано из `phrases.txt` с аугментацией (опечатки, регистр, "
                "лишние слова, перестановка). Класс `none` = «ничего из набора».\n\n")
        for label in sorted(by_label):
            items = sorted(by_label[label], key=str.lower)
            f.write(f"## {label} — {len(items)}\n\n")
            for t in items:
                f.write(f"- {t}\n")
            f.write("\n")

    print(f"Всего: {n}  (train {len(train)}, valid {len(valid)})")
    for k, v in sorted(dist.items()):
        print(f"  {k:18s} {v}")
    print("\nПредпросмотр для глаз: ft/data/preview.md")


if __name__ == "__main__":
    main()
