#!/usr/bin/env python3
"""Joint-модель: один энкодер ruBERT-tiny2, две головы — интент и слоты.

Общий для train_joint.py и test_joint.py, чтобы определение модели и код
предсказания жили в одном месте.

Головы читают один и тот же last_hidden_state, поэтому форвард один и
скорость остаётся на уровне обычного классификатора (~1 мс на CPU):
  intent — по [CLS], 5 классов;
  slots  — по каждому токену, BIO-теги.

Слот из предсказания достаём по символьным offsets ИСХОДНОЙ строки, а не
склейкой токенов: «ард 9999» бьётся на ['ар','##д','999','##9'], и декодировать
это обратно — путь к мусору вида «ар ##д 999 ##9».
"""
import json
import os
import re

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BASE = "cointegrated/rubert-tiny2"
INTENTS = ["open_adsw", "open_network", "open_components", "open_projects", "none"]
# ticket и num — раздельные слоты: «ард 7777» → ticket=ард, num=7777.
# Собрать из них «ARD-7777» — дело приложения: конвенцию именования модель
# знать не может, она видела только текст запроса.
SLOTS = ["ticket", "num", "branch", "target"]
TAGS = ["O"] + [f"{p}-{s}" for s in SLOTS for p in ("B", "I")]

# Режем текст на слова И знаки препинания по отдельности — ровно так же, как
# при обучении (gen_slots.seg). Благодаря этому разбор целиком делает модель:
# «ARD-2020.» → токены «ARD», «-», «2020», «.», которым она сама ставит
# B-ticket / O / B-num / O. Ни регулярки для разбора ID, ни снятия точки после
# модели не нужно — их тут раньше и не было бы, режь мы текст так с самого
# начала.
# Граница буква/цифра тоже режет: Whisper пишет «ARD2020» слитно.
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")


def segment(text):
    """Строка → [(токен, начало, конец)] с позициями в исходной строке.

    Позиции нужны, чтобы собрать значение слота срезом ИСХОДНОГО текста:
    «feature/new-header» разбирается на пять токенов, а склеивать их обратно
    вручную — значит гадать, где были пробелы. Срез по offsets точен всегда.
    """
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


class JointClassifier(nn.Module):
    def __init__(self, base=BASE, n_intents=len(INTENTS), n_tags=len(TAGS)):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(h, n_intents)
        self.slot_head = nn.Linear(h, n_tags)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        return self.intent_head(h[:, 0]), self.slot_head(h)


def save(model, tok, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "joint.pt"))
    tok.save_pretrained(path)
    with open(os.path.join(path, "joint_config.json"), "w") as f:
        json.dump({"base": BASE, "intents": INTENTS, "tags": TAGS}, f,
                  ensure_ascii=False, indent=2)


def load(path, device="cpu"):
    with open(os.path.join(path, "joint_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(path)
    model = JointClassifier(cfg["base"], len(cfg["intents"]), len(cfg["tags"]))
    model.load_state_dict(torch.load(os.path.join(path, "joint.pt"), map_location=device))
    return model.to(device).eval(), tok, cfg


@torch.no_grad()
def predict(model, tok, cfg, text, device="cpu"):
    """Строка → (intent, {slot: значение}, score).

    score — вероятность интента (softmax по голове интента). Нужна потребителю:
    у ассистента есть порог уверенности, и без числа его не к чему применять.

    Слова режем по пробелам и подаём is_split_into_words=True — ровно так же,
    как при обучении. Это важно: обычная токенизация рвёт «ARD-4242» на три
    слова по пунктуации, а обучалась модель на нём как на одном, и теги у «-»
    и «4242» оказались бы необученным мусором.

    Тег слова берём с первого subword: у хвоста при обучении стоит -100.
    """
    segs = segment(text)
    if not segs:
        return cfg["intents"][-1], {}, 1.0
    words = [s[0] for s in segs]
    enc = tok([words], is_split_into_words=True, truncation=True,
              max_length=48, return_tensors="pt")
    word_ids = enc.word_ids(0)
    feed = {k: v.to(device) for k, v in enc.items()}
    intent_logits, slot_logits = model(feed["input_ids"], feed["attention_mask"])

    probs = torch.softmax(intent_logits[0], -1)
    best = int(probs.argmax())
    intent, score = cfg["intents"][best], float(probs[best])
    tag_ids = slot_logits[0].argmax(-1).tolist()

    # тег на слово — с первого subword, хвост игнорируем
    word_tags, prev = {}, None
    for tid, wid in zip(tag_ids, word_ids):
        if wid is not None and wid != prev:
            word_tags[wid] = cfg["tags"][tid]
        prev = wid

    # BIO-склейка: значение слота — срез исходного текста от начала первого
    # токена до конца последнего, поэтому «feature/new-header» собирается
    # ровно как написано, без догадок о пробелах.
    slots, cur, start, end = {}, None, 0, 0
    for i, (_, s, e) in enumerate(segs):
        tag = word_tags.get(i, "O")
        if tag.startswith("B-"):
            if cur:
                slots.setdefault(cur, text[start:end])
            cur, start, end = tag[2:], s, e
        elif tag.startswith("I-") and cur == tag[2:]:
            end = e
        else:
            if cur:
                slots.setdefault(cur, text[start:end])
            cur = None
    if cur:
        slots.setdefault(cur, text[start:end])

    # branch на none — это не имя ветки, а слово из посторонней речи
    # («плохо распознаёт номер ветки» → branch: номер). Потребителю такой слот
    # только мешает: он читает его как реальное имя.
    if intent == "none":
        slots.pop("branch", None)

    slots = {k: v for k, v in slots.items() if v}
    return intent, slots, score
