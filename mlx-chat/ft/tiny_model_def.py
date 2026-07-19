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

INTENTS = ["open_adsw", "open_network", "open_components", "open_projects", "none"]
SLOTS = ["ticket", "num", "branch", "target"]
TAGS = ["O"] + [f"{p}-{s}" for s in SLOTS for p in ("B", "I")]

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
    def __init__(self, n_chars, n_words, n_intents=len(INTENTS), n_tags=len(TAGS)):
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
        """chars (B,W,C) · words (B,W) · mask (B,W) → (интент, теги слов)."""
        B, W, C = chars.shape
        ce = self.char_emb(chars.reshape(B * W, C))
        _, (h, _) = self.char_rnn(ce)                      # h: (2, B*W, CHAR_HID)
        cw = torch.cat([h[0], h[1]], -1).reshape(B, W, -1)  # вектор слова из букв

        x = torch.cat([cw, self.word_emb(words)], -1)
        x, _ = self.word_rnn(self.drop(x))
        x = self.drop(x)

        slot_logits = self.slot_head(x)
        # интент — по максимуму вдоль фразы: одного решающего слова достаточно
        masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        intent_logits = self.intent_head(masked.max(dim=1).values)
        return intent_logits, slot_logits


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


@torch.no_grad()
def predict(model, cfg, text, device="cpu"):
    """Строка → (intent, {slot: значение}, score). Контракт как у joint-модели."""
    segs = segment(text)
    if not segs:
        return INTENTS[-1], {}, 1.0
    words = [s[0] for s in segs]
    c2i, w2i = cfg["chars"], cfg["words"]
    ch, wi, mask = encode(words, c2i, w2i, cfg["max_words"])
    il, sl = model(ch.unsqueeze(0).to(device), wi.unsqueeze(0).to(device),
                   mask.unsqueeze(0).to(device))

    probs = torch.softmax(il[0], -1)
    best = int(probs.argmax())
    intent, score = INTENTS[best], float(probs[best])
    tags = [TAGS[i] for i in sl[0].argmax(-1).tolist()]

    # BIO-склейка: значение — срез исходной строки по offsets
    slots, cur, start, end = {}, None, 0, 0
    for i, (_, s, e) in enumerate(segs[:cfg["max_words"]]):
        tag = tags[i]
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

    if intent == "none":
        slots.pop("branch", None)
    return intent, {k: v for k, v in slots.items() if v}, score
