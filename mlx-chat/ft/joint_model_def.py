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
OPEN_INTENTS = [i for i in INTENTS if i != "none"]
# ticket и num — раздельные слоты: «ард 7777» → ticket=ард, num=7777.
# Собрать из них «ARD-7777» — дело приложения: конвенцию именования модель
# знать не может, она видела только текст запроса.
SLOTS = ["ticket", "num", "branch", "target"]
TAGS = ["O"] + [f"{p}-{s}" for s in SLOTS for p in ("B", "I")]

# Интент размечается ПО ТОКЕНАМ, тем же BIO. Так фраза «открой нетворк на 2070
# и адсв на 3511» распадается на две команды, и каждый слот достаётся своей:
# при интенте на всю фразу привязать номер к папке нечем. B- отделяет и две
# соседние команды с одинаковым интентом.
INTENT_TAGS = ["O"] + [f"{p}-{i}" for i in OPEN_INTENTS for p in ("B", "I")]

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
    """Обе головы потокенные: интент и слоты размечаются на каждом слове."""

    def __init__(self, base=BASE, n_intents=len(INTENT_TAGS), n_tags=len(TAGS)):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(h, n_intents)
        self.slot_head = nn.Linear(h, n_tags)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        return self.intent_head(h), self.slot_head(h)


def save(model, tok, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "joint.pt"))
    tok.save_pretrained(path)
    with open(os.path.join(path, "joint_config.json"), "w") as f:
        json.dump({"base": BASE, "intents": INTENTS, "tags": TAGS,
                   "intent_tags": INTENT_TAGS}, f, ensure_ascii=False, indent=2)


def load(path, device="cpu"):
    with open(os.path.join(path, "joint_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(path)
    model = JointClassifier(cfg["base"], len(cfg["intent_tags"]), len(cfg["tags"]))
    model.load_state_dict(torch.load(os.path.join(path, "joint.pt"), map_location=device))
    return model.to(device).eval(), tok, cfg


def _spans(tags, n):
    """BIO-теги → [(метка, первый токен, последний токен)]."""
    out, cur, start = [], None, 0
    for i in range(n):
        tag = tags.get(i, "O")
        if tag.startswith("B-"):
            if cur:
                out.append((cur, start, i - 1))
            cur, start = tag[2:], i
        elif tag.startswith("I-") and cur == tag[2:]:
            continue
        else:
            if cur:
                out.append((cur, start, i - 1))
            cur = None
    if cur:
        out.append((cur, start, n - 1))
    return out


@torch.no_grad()
def predict(model, tok, cfg, text, device="cpu"):
    """Строка → список команд: [{"intent", "slots", "score"}, ...].

    Список, а не одна команда: во фразе «открой нетворк на 2070 и адсв на 3511»
    их две, и каждый слот принадлежит своей. Пустой список = ничего не
    распознано (бывшее none).

    score — уверенность интента, усреднённая по токенам команды. Потребитель
    применяет к ней свой порог.

    Текст режем на буквы/цифры/знаки — ровно так же, как при обучении; любой
    рассинхрон сегментации бьёт по разметке. Тег слова берём с первого subword:
    у хвоста при обучении стоит -100.
    """
    segs = segment(text)
    if not segs:
        return []
    words = [s[0] for s in segs]
    enc = tok([words], is_split_into_words=True, truncation=True,
              max_length=64, return_tensors="pt")
    word_ids = enc.word_ids(0)
    feed = {k: v.to(device) for k, v in enc.items()}
    intent_logits, slot_logits = model(feed["input_ids"], feed["attention_mask"])

    iprobs = torch.softmax(intent_logits[0], -1)
    itag_ids = iprobs.argmax(-1).tolist()
    stag_ids = slot_logits[0].argmax(-1).tolist()

    # теги на слово — с первого subword, хвост игнорируем
    wi, ws, wp, prev = {}, {}, {}, None
    for k, wid in enumerate(word_ids):
        if wid is not None and wid != prev:
            wi[wid] = cfg["intent_tags"][itag_ids[k]]
            ws[wid] = cfg["tags"][stag_ids[k]]
            wp[wid] = float(iprobs[k][itag_ids[k]])
        prev = wid

    n = len(segs)
    slot_spans = _spans(ws, n)
    commands = []
    for intent, ci, cj in _spans(wi, n):
        slots = {}
        for name, si, sj in slot_spans:
            if ci <= si <= cj:                       # слот внутри этой команды
                slots.setdefault(name, text[segs[si][1]:segs[sj][2]])
        conf = [wp[i] for i in range(ci, cj + 1) if i in wp]
        commands.append({
            "intent": intent,
            "slots": {k: v for k, v in slots.items() if v},
            "score": round(sum(conf) / len(conf), 4) if conf else 0.0,
        })
    return commands
