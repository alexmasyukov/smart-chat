#!/usr/bin/env python3
"""Генератор размеченных данных для joint-модели (интент + слоты).

Два источника:
  templates.txt — шаблоны со слотами; значения подставляет сам генератор,
                  поэтому границы слотов известны точно, руками не размечаем;
  phrases.txt   — старый банк фраз без параметров; идёт как есть, но с
                  авторазметкой target (там уже встречается «в finder»).

Отдаёт список примеров: {"words": [...], "tags": [...], "intent": "open_adsw"}
Теги BIO: O, B-branch, I-branch, B-target, I-target.

    python gen_slots.py            # показать статистику и примеры
    python gen_slots.py --preview  # + записать data/slots_preview.md
"""
import argparse
import os
import random
import re
from collections import Counter

import gen_dataset as gd

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))

INTENTS = ["open_adsw", "open_network", "open_components", "open_projects", "none"]
# ticket и num раздельно: «ард 7777» → ticket=ард, num=7777, чтобы приложение
# собрало «ARD-7777» по своей конвенции. branch — произвольное имя целиком.
SLOTS = ["ticket", "num", "branch", "target"]
TAGS = ["O"] + [f"{p}-{s}" for s in SLOTS for p in ("B", "I")]
T2I = {t: i for i, t in enumerate(TAGS)}

# Сколько примеров на шаблон: разные значения слотов дают разнообразие,
# поэтому меньше, чем аугментаций на плоскую фразу в gen_dataset.
PER_TEMPLATE = 34
PER_PHRASE = 8


def load_templates(path):
    """Парсит templates.txt → (templates {intent: [шаблоны]}, slots {name: [значения]})."""
    templates, slots = {}, {}
    bucket = None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith("#"):
            head = s.strip("# ").split("->")[0].strip()
            token = head.split()[0] if head else ""
            if token.startswith("slot:"):
                bucket = slots.setdefault(token[5:], [])
            elif token in INTENTS:
                bucket = templates.setdefault(token, [])
            continue
        if not s or bucket is None:
            continue
        bucket.append(s)
    return templates, slots


# Пул для {WORD}: реальные имена и слова + псевдослова из слогов.
# Псевдослова нужны специально: они не несут смысла, и выучить их нельзя —
# только правило «после „на ветке“ стоит branch, каким бы словом он ни был».
# Без них модель запоминает конкретные значения и на «ветке коли» молчит.
NAMES = ["васи", "пети", "коли", "димы", "саши", "лены", "миши", "юли", "олега",
         "игоря", "кати", "антона", "макса", "вовы", "жени", "ромы", "стаса"]
WORDS = ["логина", "оплаты", "поиска", "хедера", "футера", "таблицы", "формы",
         "корзины", "профиля", "настроек", "графиков", "экспорта", "импорта",
         "фильтров", "модалки", "дашборда", "авторизации", "уведомлений"]
SYLL = ["зю", "ка", "ме", "ро", "ти", "ла", "ну", "фи", "ша", "бо", "гу", "де"]
LAT = ["login", "auth", "table", "chart", "modal", "cache", "sync", "theme",
       "search", "export", "grid", "toast", "menu", "badge", "input"]


def rand_word():
    r = random.random()
    if r < 0.3:
        return random.choice(NAMES)
    if r < 0.55:
        return random.choice(WORDS)
    if r < 0.75:
        return random.choice(LAT)
    return "".join(random.choice(SYLL) for _ in range(random.randint(2, 4)))


def rand_slug():
    return "-".join(random.choice(LAT) for _ in range(random.randint(1, 3)))


def make_value(raw):
    """Значение слота: {NNNN} → случайный номер, {WORD}/{SLUG} → случайное имя.

    Все три подставляются случайно, а не берутся из списка: иначе модель учит
    сами значения вместо позиции и промахивается на невиданном слове.
    """
    raw = re.sub(r"\{NNNN\}", lambda _: str(random.randint(1, 9999)), raw)
    raw = re.sub(r"\{WORD\}", lambda _: rand_word(), raw)
    raw = re.sub(r"\{SLUG\}", lambda _: rand_slug(), raw)
    return raw


def tag_words(words, slot):
    """Слова значения слота → BIO-теги."""
    return [f"{'B' if i == 0 else 'I'}-{slot}" for i in range(len(words))]


def ticket_value(slots):
    """Тикет → (words, tags) с РАЗДЕЛЬНЫМИ ticket и num.

    Живые формы записи одного и того же тикета:
      «ард 7777»  → два слова: ticket + num  (основной случай, речь)
      «ард-7777»  → одно слово: разделить на уровне слов нельзя, тег ticket,
                    приложение доразберёт (это канонический ID, не текст)
      «7777»      → голый номер без префикса
    """
    prefix = random.choice(slots["ticket"])
    num = str(random.randint(1, 9999))
    mode = random.random()
    if mode < 0.60:                       # «ард 7777» — то, что говорят вслух
        return [prefix, num], ["B-ticket", "B-num"]
    if mode < 0.85:                       # «ARD-7777» / «ard7777» — одно слово
        return [prefix + random.choice(["-", "-", "_", ""]) + num], ["B-ticket"]
    return [num], ["B-num"]               # «на фиче 315»


def expand_slot(slot, slots):
    """Значение слота → (words, tags). branch умеет два режима."""
    if slot == "branch":
        # тикет или произвольное имя в одной и той же позиции. Имён чуть
        # больше: они разнообразнее (любое слово), тикеты же однотипны.
        if random.random() < 0.42:
            return ticket_value(slots)
        value = make_value(random.choice(slots["branch"])).split()
        return value, tag_words(value, "branch")
    value = make_value(random.choice(slots[slot])).split()
    return value, tag_words(value, slot)


def fill(template, slots):
    """Шаблон → (words, tags). Слоты подставляются вместе с разметкой."""
    words, tags = [], []
    for part in re.split(r"(\{\w+\})", template):
        if not part:
            continue
        m = re.fullmatch(r"\{(\w+)\}", part)
        if m:
            value, vtags = expand_slot(m.group(1), slots)
            words += value
            tags += vtags
        else:
            plain = part.split()
            words += plain
            tags += ["O"] * len(plain)
    return words, tags


def autotag_phrase(phrase, targets):
    """Плоская фраза из phrases.txt → (words, tags) с авторазметкой target.

    Ищет вхождения известных target-выражений («в finder»). Только предложные
    формы, поэтому none-фраза «открой браузер» остаётся чистой (там нет «в»).
    """
    words = phrase.split()
    tags = ["O"] * len(words)
    low = [w.lower() for w in words]
    for t in sorted(targets, key=lambda x: -len(x.split())):
        tw = t.lower().split()
        n = len(tw)
        for i in range(len(low) - n + 1):
            if low[i:i + n] == tw and all(x == "O" for x in tags[i:i + n]):
                tags[i:i + n] = tag_words(tw, "target")
    return words, tags


def augment(words, tags):
    """Аугментация с уважением к слотам.

    Филлеры вставляем только в позиции вне слот-спанов, опечатки — только по
    словам с тегом O. Опечатка в номере тикета сделала бы значение невалидным,
    а перестановку слов (shuffle из gen_dataset) не делаем вовсе: живой человек
    не говорит «ветке на ард открой 1120».
    """
    words, tags = list(words), list(tags)

    # филлеры — только на границах слотов, и НЕ между ticket и num:
    # «ард пожалуйста 574» — не то, что говорят живые люди, а модель на таком
    # учится рвать пару префикс-номер.
    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
            spots = [i for i in range(len(words) + 1)
                     if (i == len(words) or not tags[i].startswith("I-"))
                     and not (0 < i < len(tags)
                              and tags[i - 1] == "B-ticket" and tags[i] == "B-num")]
            i = random.choice(spots)
            words.insert(i, random.choice(gd.FILLERS))
            tags.insert(i, "O")

    # опечатка — только по не-слотовым словам
    if random.random() < 0.4:
        plain = [i for i, t in enumerate(tags) if t == "O" and len(words[i]) >= 4]
        if plain:
            i = random.choice(plain)
            words[i] = gd.typo(words[i])

    # регистр — по всей фразе (границы слов не меняются, теги валидны)
    text = gd.recase(" ".join(words))
    recased = text.split()
    if len(recased) == len(words):
        words = recased
    return words, tags


def build():
    templates, slots = load_templates(os.path.join(HERE, "templates.txt"))
    core = gd.load_phrases(os.path.join(HERE, "phrases.txt"))
    targets = slots["target"]

    rows = []
    for intent, tpls in templates.items():
        for tpl in tpls:
            for _ in range(PER_TEMPLATE):
                w, t = fill(tpl, slots)
                rows.append(augment(w, t))
                rows[-1] = (rows[-1][0], rows[-1][1], intent)

    for intent, phrases in core.items():
        for ph in phrases:
            w, t = autotag_phrase(ph, targets)
            rows.append((w, t, intent))
            for _ in range(PER_PHRASE):
                aw, at = augment(w, t)
                rows.append((aw, at, intent))

    # дедуп по тексту
    seen, uniq = set(), []
    for w, t, i in rows:
        k = " ".join(w).lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append({"words": w, "tags": t, "intent": i})
    random.shuffle(uniq)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="записать data/slots_preview.md")
    args = ap.parse_args()

    rows = build()
    print(f"Всего примеров: {len(rows)}")
    by_intent = Counter(r["intent"] for r in rows)
    for k in INTENTS:
        with_slot = sum(1 for r in rows if r["intent"] == k and set(r["tags"]) != {"O"})
        print(f"  {k:18s} {by_intent[k]:5d}   из них со слотами: {with_slot}")
    tag_dist = Counter(t for r in rows for t in r["tags"])
    print("\nТеги:")
    for t in TAGS:
        print(f"  {t:10s} {tag_dist[t]}")

    print("\nПримеры:")
    for r in rows[:6]:
        pairs = "  ".join(f"{w}/{t}" if t != "O" else w for w, t in zip(r["words"], r["tags"]))
        print(f"  [{r['intent']:16s}] {pairs}")

    if args.preview:
        path = os.path.join(HERE, "data", "slots_preview.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Датасет joint-модели — {len(rows)} примеров\n\n")
            f.write("Слоты размечены генератором при подстановке значений "
                    "(`templates.txt`) и авторазметкой target (`phrases.txt`).\n\n")
            for intent in INTENTS:
                items = [r for r in rows if r["intent"] == intent]
                f.write(f"## {intent} — {len(items)}\n\n")
                for r in sorted(items, key=lambda r: " ".join(r["words"]).lower()):
                    pairs = " ".join(f"**{w}**`{t}`" if t != "O" else w
                                     for w, t in zip(r["words"], r["tags"]))
                    f.write(f"- {pairs}\n")
                f.write("\n")
        print(f"\nПредпросмотр: {path}")


if __name__ == "__main__":
    main()
