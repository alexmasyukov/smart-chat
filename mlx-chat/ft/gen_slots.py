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
    """Парсит templates.txt → (templates, slots, aliases).

    Секции: '# <intent>' — шаблоны, '# slot:<name>' — значения слота,
    '# alias:<intent>' — написания названия проекта, '# service' — служебные слова.
    """
    templates, slots, aliases = {}, {}, {}
    bucket = None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith("#"):
            head = s.strip("# ").split("->")[0].strip()
            token = head.split()[0] if head else ""
            if token.startswith("slot:"):
                bucket = slots.setdefault(token[5:], [])
            elif token.startswith("alias:"):
                bucket = aliases.setdefault(token[6:], [])
            elif token == "service":
                bucket = slots.setdefault("service", [])
            elif token in INTENTS:
                bucket = templates.setdefault(token, [])
            # прочие '#' — пояснения внутри секции, bucket не трогаем
            continue
        if not s or bucket is None:
            continue
        bucket.append(s)
    return templates, slots, aliases


# Пул для {WORD}: реальные имена и слова + псевдослова из слогов.
# Псевдослова нужны специально: они не несут смысла, и выучить их нельзя —
# только правило «после „на ветке“ стоит branch, каким бы словом он ни был».
# Без них модель запоминает конкретные значения и на «ветке коли» молчит.
NAMES = ["васи", "пети", "коли", "димы", "саши", "лены", "миши", "юли", "олега",
         "игоря", "кати", "антона", "макса", "вовы", "жени", "ромы", "стаса",
         "паши", "гриши", "толи", "серёги", "артёма", "оли", "иры", "нади",
         "феди", "бори", "гены", "клима", "яны", "тимура", "марка", "аси",
         "глеба", "зины", "кирилла", "лёхи", "матвея", "нины", "остапа",
         "платона", "риты", "севы", "тани", "ульяны", "фимы", "хари", "цезаря",
         "чарли", "шуры", "эдика", "юры", "яши", "агаты", "богдана", "вали"]
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


# Режем на буквы / цифры / знаки препинания по отдельности. Это принципиально:
# пока «ARD-2020.» был одним словом, тег у него мог быть только один, и части
# приходилось доставать регуляркой уже после модели. С такой сегментацией
# «ARD», «-», «2020», «.» — четыре токена, и модель сама метит их
# B-ticket / O / B-num / O. Никакой постобработки не нужно.
#
# Граница буква/цифра тоже режет: Whisper пишет «ARD2020» слитно, и без этого
# правила номер не отделить — ровно на этом тест и падал.
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")


def seg(s):
    """Строка → токены (слова и знаки препинания раздельно)."""
    return TOKEN_RE.findall(s)


def tag_value(s, slot):
    """Значение слота → (токены, BIO-теги) с той же сегментацией."""
    toks = seg(s)
    return toks, tag_words(toks, slot)


def ticket_value(slots):
    """Тикет → (words, tags) с РАЗДЕЛЬНЫМИ ticket и num.

    Живые формы записи одного и того же тикета:
      «ард 7777»   → ticket + num  (основной случай, так говорят вслух)
      «ит дев 204» → многословный префикс: B-ticket I-ticket, потом num
      «ард-7777»   → одно слово: разделить на уровне слов нельзя, тег ticket,
                     приложение доразберёт (это канонический ID, не текст)
      «7777»       → голый номер без префикса
    """
    prefix = seg(random.choice(slots["ticket"]))
    num = str(random.randint(1, 9999))
    mode = random.random()
    if mode < 0.58:                       # «ард 7777», «ит дев 204»
        return prefix + [num], tag_words(prefix, "ticket") + ["B-num"]
    if mode < 0.80:
        # «ARD-7777» и слитное «ARD2020»: разделитель — отдельный токен с O,
        # а слитную форму режет сама сегментация по границе буква/цифра.
        # В обоих случаях ticket и num размечает модель, разбора после неё нет.
        sep = random.choice(["-", "-", "_", ""])
        if sep:
            return prefix + [sep, num], tag_words(prefix, "ticket") + ["O", "B-num"]
        return prefix + [num], tag_words(prefix, "ticket") + ["B-num"]
    if mode < 0.92:
        # «на ветке 1987 ITDEV» — номер впереди префикса. Форма редкая, но
        # реальная (есть в живых расшифровках), а на малой доле модель её
        # теряла: номер уходил в branch вместо num.
        return [num] + prefix, ["B-num"] + tag_words(prefix, "ticket")
    return [num], ["B-num"]               # «на фиче 315»


def expand_slot(slot, slots):
    """Значение слота → (words, tags). branch умеет два режима."""
    if slot == "service":
        # служебное слово («фича-ветку») — тип ветки, а не имя: всегда O
        value = seg(random.choice(slots["service"]))
        return value, ["O"] * len(value)
    if slot == "branch":
        # тикет или произвольное имя в одной и той же позиции. Имён чуть
        # больше: они разнообразнее (любое слово), тикеты же однотипны.
        if random.random() < 0.42:
            return ticket_value(slots)
        return tag_value(make_value(random.choice(slots["branch"])), "branch")
    return tag_value(make_value(random.choice(slots[slot])), slot)


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
            plain = seg(part)
            words += plain
            tags += ["O"] * len(plain)
    return words, tags


def autotag_phrase(phrase, targets):
    """Плоская фраза из phrases.txt → (words, tags) с авторазметкой target.

    Ищет вхождения известных target-выражений («в finder»). Только предложные
    формы, поэтому none-фраза «открой браузер» остаётся чистой (там нет «в»).
    """
    words = seg(phrase)
    tags = ["O"] * len(words)
    low = [w.lower() for w in words]
    for t in sorted(targets, key=lambda x: -len(seg(x))):
        tw = seg(t.lower())
        n = len(tw)
        for i in range(len(low) - n + 1):
            if low[i:i + n] == tw and all(x == "O" for x in tags[i:i + n]):
                tags[i:i + n] = tag_words(tw, "target")
    return words, tags


def swap_alias(words, tags, alias_bank):
    """Заменяет название проекта на другое живое написание.

    Вход — расшифровка Whisper, а он коверкает короткие незнакомые слова
    устойчиво: adsw → «ADSV», «АДСВ», «Адрес свой». Подстановка даёт каждому
    шаблону все написания, не размножая сам банк шаблонов.
    Меняем только слова с тегом O — название проекта слотом не является.
    """
    if not alias_bank:
        return words, tags
    low = [w.lower() for w in words]
    # ищем самое длинное вхождение любого алиаса
    for alias in sorted(alias_bank, key=lambda a: -len(seg(a))):
        aw = seg(alias.lower())
        n = len(aw)
        for i in range(len(low) - n + 1):
            if low[i:i + n] == aw and all(t == "O" for t in tags[i:i + n]):
                new = seg(random.choice(alias_bank))
                return (words[:i] + new + words[i + n:],
                        tags[:i] + ["O"] * len(new) + tags[i + n:])
    return words, tags


def punctuate(words, tags):
    """Пунктуация как в расшифровке Whisper: точка в конце, запятые внутри.

    На вход модели текст приходит только от Whisper, а он почти всегда ставит
    точку в конце и запятые внутри. Без этого в обучении модель видит
    «ARD 2020», а в бою получает «ARD 2020.» — и номер отваливается.
    """
    words, tags = list(words), list(tags)
    # знак — отдельный токен с тегом O: тогда он не прилипает к значению слота
    # и «1887.» не превращается в мусорный номер. Снимать точку постфактум не
    # нужно — модель просто не включает её в span.
    if len(words) > 3 and random.random() < 0.30:
        i = random.randrange(1, len(words) - 1)
        if not (tags[i] == "B-ticket" and i + 1 < len(tags) and tags[i + 1] == "B-num"):
            words.insert(i + 1, ",")
            tags.insert(i + 1, "O")
    if random.random() < 0.65:
        words.append(random.choice([".", ".", ".", "?", "!"]))
        tags.append("O")
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

    # опечатка — только по не-слотовым словам, и только если она не рвёт
    # токен надвое (gd.typo может подставить дефис, а он теперь разделитель)
    if random.random() < 0.4:
        plain = [i for i, t in enumerate(tags) if t == "O" and len(words[i]) >= 4]
        if plain:
            i = random.choice(plain)
            cand = gd.typo(words[i])
            if len(seg(cand)) == 1:
                words[i] = cand

    # регистр — по всей фразе (границы слов не меняются, теги валидны)
    text = gd.recase(" ".join(words))
    recased = text.split()
    if len(recased) == len(words):
        words = recased
    return words, tags


def build():
    templates, slots, aliases = load_templates(os.path.join(HERE, "templates.txt"))
    core = gd.load_phrases(os.path.join(HERE, "phrases.txt"))
    targets = slots["target"]

    rows = []
    for intent, tpls in templates.items():
        bank = aliases.get(intent, [])
        for tpl in tpls:
            for _ in range(PER_TEMPLATE):
                w, t = fill(tpl, slots)
                w, t = swap_alias(w, t, bank)
                w, t = augment(w, t)
                w, t = punctuate(w, t)
                rows.append((w, t, intent))

    for intent, phrases in core.items():
        bank = aliases.get(intent, [])
        for ph in phrases:
            w, t = autotag_phrase(ph, targets)
            rows.append((w, t, intent))
            for _ in range(PER_PHRASE):
                aw, at = swap_alias(w, t, bank)
                aw, at = augment(aw, at)
                aw, at = punctuate(aw, at)
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
