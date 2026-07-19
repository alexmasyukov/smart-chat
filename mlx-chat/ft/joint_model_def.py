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

# «ARD-7777» пишется одним словом — на уровне слов его не разделить, поэтому
# модель метит его как ticket, а здесь разбираем на части. Это разбор
# канонического идентификатора, а не извлечение параметра из живой речи.
#
# Хвостовая пунктуация обязательна: на вход идёт расшифровка Whisper, а он
# почти всегда ставит точку в конце — «ARD2020.» должен разбираться так же.
# Внутренний дефис («ит-дев-204») тоже допускаем: префикс может быть составным.
PUNCT = ".,!?;:…"
ID_RE = re.compile(r"^([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]*?)[-_ ]?(\d+)$")


def strip_punct(s):
    """Убирает пунктуацию с краёв значения слота: «1887.» → «1887»."""
    return s.strip(PUNCT + " ")


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
    words = text.split()
    if not words:
        return cfg["intents"][-1], {}, 1.0
    enc = tok([words], is_split_into_words=True, truncation=True,
              max_length=32, return_tensors="pt")
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

    # BIO-склейка по словам
    slots, cur, buf = {}, None, []
    for i in range(len(words)):
        tag = word_tags.get(i, "O")
        if tag.startswith("B-"):
            if cur:
                slots.setdefault(cur, " ".join(buf))
            cur, buf = tag[2:], [words[i]]
        elif tag.startswith("I-") and cur == tag[2:]:
            buf.append(words[i])
        else:
            if cur:
                slots.setdefault(cur, " ".join(buf))
            cur, buf = None, []
    if cur:
        slots.setdefault(cur, " ".join(buf))

    # пунктуация Whisper липнет к значениям: «1887.» → «1887»
    slots = {k: strip_punct(v) for k, v in slots.items()}

    # «ARD-7777» одним словом → ticket=ARD, num=7777
    if "ticket" in slots and "num" not in slots:
        m = ID_RE.match(slots["ticket"])
        if m:
            slots["ticket"], slots["num"] = m.group(1), m.group(2)

    # branch на none — это не имя ветки, а слово из посторонней речи
    # («плохо распознаёт номер ветки» → branch: номер). Потребителю такой слот
    # только мешает: он читает его как реальное имя.
    if intent == "none":
        slots.pop("branch", None)

    slots = {k: v for k, v in slots.items() if v}
    return intent, slots, score
