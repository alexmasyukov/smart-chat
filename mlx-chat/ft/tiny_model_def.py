#!/usr/bin/env python3
"""Модель с нуля: те же интент + слоты, но ~300K параметров вместо 29M.

Зачем: задача узкая — 3032 уникальных слова, 78 символов, до 18 слов в запросе.
ruBERT-tiny2 предобучен на всём русском интернете, и почти всё это знание тут
простаивает. Проверяем, хватит ли модели, обученной только на наших данных.

Устройство (классический char-CNN/BiLSTM тэггер):

    символы слова → char-BiLSTM ──┐
                                  ├─→ word-BiLSTM ─┬─→ голова интента (по пулу)
    слово → словарный эмбеддинг ──┘                └─→ голова слотов (по слову)

Вход посимвольный не для экономии, а по существу: на вход идут расшифровки
Whisper, где слова коверкаются («Натворк» вместо «нетворк»). Модель, которая
смотрит на буквы, видит их похожими; словарная — как два разных токена.
Словарный эмбеддинг оставлен для частых слов («открой», «ветке»), где важна
не форма, а само слово.

Контракт predict() тот же, что у joint_model_def: (intent, slots, score).
Токенизатор не нужен вовсе — алфавит строится из данных.
"""
import json
import os
import re

import torch
import torch.nn as nn

INTENTS = ["open_adsw", "open_network", "open_components", "open_projects",
           # Окружения — отдельные интенты: Whisper склеивает окружение с
           # проектом в одно слово («Девнетворк», «Протодсв»), а слот на
           # половину токена не поставить. У каждого свой URL на стороне
           # приложения.
           "open_network_prod", "open_adsw_prod",
           "open_network_dev", "open_adsw_dev",
           # Папка в Finder — не стенд: «адсв на ветке 2511» это стенд,
           # «папку адсв» это каталог на диске.
           "open_adsw_folder", "open_network_folder",
           # Джира: задача по номеру (со слотами ticket+num) и доска целиком.
           # Различие ровно в наличии номера в фразе.
           "open_jira_task", "open_jira",
           "none"]
OPEN_INTENTS = [i for i in INTENTS if i != "none"]
SLOTS = ["ticket", "num", "branch", "target"]
TAGS = ["O"] + [f"{p}-{s}" for s in SLOTS for p in ("B", "I")]

# Интент размечается ПО ТОКЕНАМ, тем же BIO, что и слоты: иначе фраза «открой
# нетворк на 2070 и адсв на 3511» неразрешима — интент один на всю фразу, и
# привязать номер к своей папке нечем. B- отделяет и две соседние команды с
# одинаковым интентом.
INTENT_TAGS = ["O"] + [f"{p}-{i}" for i in OPEN_INTENTS for p in ("B", "I")]

# Та же сегментация, что у joint-модели: буквы / цифры / знаки по отдельности.
# Именно она избавляет от разбора «ARD-2020.» регуляркой после модели.
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")

MAX_WORD = 18          # символов в слове; длиннее — обрезаем
PAD, UNK = 0, 1

CHAR_EMB = 32
CHAR_HID = 32          # на каждое направление → вектор слова 64
WORD_EMB = 48
WORD_HID = 96          # на каждое направление → признак слова 192


def segment(text):
    """Строка → [(токен, начало, конец)] с позициями в исходной строке."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


class TinyTagger(nn.Module):
    def __init__(self, n_chars, n_words, n_intents=len(INTENT_TAGS), n_tags=len(TAGS)):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, CHAR_EMB, padding_idx=PAD)
        self.char_rnn = nn.LSTM(CHAR_EMB, CHAR_HID, batch_first=True, bidirectional=True)
        self.word_emb = nn.Embedding(n_words, WORD_EMB, padding_idx=PAD)
        self.drop = nn.Dropout(0.2)
        self.word_rnn = nn.LSTM(2 * CHAR_HID + WORD_EMB, WORD_HID, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=0.2)
        self.slot_head = nn.Linear(2 * WORD_HID, n_tags)
        self.intent_head = nn.Linear(2 * WORD_HID, n_intents)

    def forward(self, chars, words, mask):
        """chars (B,W,C) · words (B,W) · mask (B,W) → потокенные (интент, слоты)."""
        B, W, C = chars.shape
        ce = self.char_emb(chars.reshape(B * W, C))
        _, (h, _) = self.char_rnn(ce)                      # h: (2, B*W, CHAR_HID)
        cw = torch.cat([h[0], h[1]], -1).reshape(B, W, -1)  # вектор слова из букв

        x = self.drop(torch.cat([cw, self.word_emb(words)], -1))

        # Упаковываем по реальной длине: без этого обратный проход BiLSTM
        # стартует с паддинга и тащит его состояние в настоящие слова. Тогда
        # результат зависит от того, до какой длины добит батч, — то есть
        # обучение и применение обязаны паддить одинаково. С упаковкой паддинг
        # не влияет вовсе, и на инференсе можно считать ровно по длине фразы.
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.word_rnn(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.shape[1])
        x = self.drop(x)
        # обе головы потокенные: интент размечает каждое слово, как и слоты
        return self.intent_head(x), self.slot_head(x)


def build_vocabs(rows, min_word_count=2):
    """Алфавит и словарь из обучающих данных.

    В словарь берём слова, встреченные хотя бы дважды: 2244 из 3032 слов —
    это случайные значения слотов и опечатки, каждое встречается один раз, и
    запоминать их бессмысленно. Такие слова пойдут как UNK, а распознаются по
    буквам — ради этого char-ветка и нужна.
    """
    from collections import Counter
    wc = Counter(w.lower() for r in rows for w in r["words"])
    chars = sorted({c for w in wc for c in w})
    words = sorted(w for w, c in wc.items() if c >= min_word_count)
    c2i = {c: i + 2 for i, c in enumerate(chars)}
    w2i = {w: i + 2 for i, w in enumerate(words)}
    return c2i, w2i


def encode(words, c2i, w2i, max_words):
    """Слова → (chars, words, mask) для одного примера."""
    words = words[:max_words]
    ch = torch.zeros(max_words, MAX_WORD, dtype=torch.long)
    wi = torch.zeros(max_words, dtype=torch.long)
    mask = torch.zeros(max_words, dtype=torch.bool)
    for i, w in enumerate(words):
        low = w.lower()
        for j, c in enumerate(low[:MAX_WORD]):
            ch[i, j] = c2i.get(c, UNK)
        wi[i] = w2i.get(low, UNK)
        mask[i] = True
    return ch, wi, mask


def save(model, cfg, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "tiny.pt"))
    with open(os.path.join(path, "tiny_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)


def load(path, device="cpu"):
    with open(os.path.join(path, "tiny_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = TinyTagger(len(cfg["chars"]) + 2, len(cfg["words"]) + 2)
    model.load_state_dict(torch.load(os.path.join(path, "tiny.pt"), map_location=device))
    return model.to(device).eval(), cfg


def _spans(tags, n):
    """BIO-теги → [(метка, первый токен, последний токен)]."""
    out, cur, start = [], None, 0
    for i in range(n):
        tag = tags[i] if i < len(tags) else "O"
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
def predict(model, cfg, text, device="cpu"):
    """Строка → список команд: [{"intent", "slots", "score"}, ...].

    Список, а не одна команда: во фразе «открой нетворк на 2070 и адсв на 3511»
    их две, и каждый слот принадлежит своей. Пустой список = ничего не
    распознано (бывшее none). Контракт совпадает с joint-моделью.
    """
    segs = segment(text)
    if not segs:
        return []
    words = [s[0] for s in segs]
    # паддим по фактической длине, а не по максимуму обучающего набора: тот
    # вырос до 42 из-за составных фраз, и короткий запрос считался бы впустую
    ch, wi, mask = encode(words, cfg["chars"], cfg["words"],
                          min(len(words), cfg["max_words"]))
    il, sl = model(ch.unsqueeze(0).to(device), wi.unsqueeze(0).to(device),
                   mask.unsqueeze(0).to(device))

    iprobs = torch.softmax(il[0], -1)
    itags = [INTENT_TAGS[i] for i in iprobs.argmax(-1).tolist()]
    stags = [TAGS[i] for i in sl[0].argmax(-1).tolist()]

    n = min(len(segs), cfg["max_words"])
    slot_spans = _spans(stags, n)
    commands = []
    for intent, ci, cj in _spans(itags, n):
        slots = {}
        for name, si, sj in slot_spans:
            if ci <= si <= cj:                       # слот внутри этой команды
                slots.setdefault(name, text[segs[si][1]:segs[sj][2]])
        conf = [float(iprobs[i].max()) for i in range(ci, cj + 1)]
        commands.append({
            "intent": intent,
            "slots": {k: v for k, v in slots.items() if v},
            "score": round(sum(conf) / len(conf), 4) if conf else 0.0,
        })
    return commands
