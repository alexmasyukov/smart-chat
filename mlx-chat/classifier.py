#!/usr/bin/env python3
"""Локальный классификатор инструментов на mlx-lm + Outlines.

Модель — чистый классификатор: по запросу пользователя возвращает имя ОДНОГО
инструмента из enum или "none". Ничего не запускает. Грамматика (constrained
decoding, как в LM Studio) гарантирует формат на 100%; greedy → детерминизм.

Два режима модели:
  • по умолчанию — дообученная LoRA-модель LFM2.5-350M (ft/fused), короткий
    промпт БЕЗ few-shot (маппинг зашит обучением). Быстро (~0.14с), мелко (~0.4ГБ).
    Переобучение под другой набор инструментов: см. ft/README.md.
  • --base — базовая LFM2.5-1.2B Instruct + few-shot промпт (работает zero-shot
    на любом наборе инструментов, без обучения; ~0.33с, ~1.3ГБ).

Инструменты — в TOOLS/FEWSHOT ниже. При смене набора: правишь их, для 350M
дополнительно переобучаешь (ft/), для --base достаточно правки TOOLS/FEWSHOT.

Использование:
    python classifier.py "покажи проекты"        # разовая классификация (350M)
    python classifier.py --base "покажи проекты"  # на базовой 1.2B
    python classifier.py --serve [--port 8090]    # HTTP: POST /classify {"text": "..."}
    python classifier.py --selftest               # прогнать встроенные кейсы
"""
import argparse
import json
import os
import sys
import time
from typing import Literal

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Модели -----------------------------------------------------------------
FINETUNED = os.path.join(HERE, "ft", "fused")         # дообученная 350M (по умолчанию)
BASE = "mlx-community/LFM2.5-1.2B-Instruct-8bit"       # базовая, zero-shot + few-shot
# CLF_MODEL=<repo|путь> переопределяет модель принудительно.
REPO = os.environ.get("CLF_MODEL")

# Короткий промпт для дообученной модели (тот же, что в обучении — ft/gen_dataset.py).
SHORT_SYSTEM = "Ты — классификатор. По запросу пользователя верни ровно одно имя инструмента из набора или none. Только имя, без пояснений."

# --- Инструменты (тестовые; имя -> краткое описание для промпта) -------------
TOOLS = [
    ("show_components",      "показать компоненты."),
    ("show_projects",        "показать проекты (общие, без уточнения ADSW/Network)."),
    ("show_rag",             "показать RAG-систему (в запросе «раг», «рак», RAG)."),
    ("show_adsw_projects",   "показать проекты ADSW (в запросе «адсв»/ADSW)."),
    ("show_network_projects","показать проекты Network (в запросе «нетворк»/Network)."),
    ("open_adsw",            "открыть ПАПКУ ADSW в Finder (только если явно про папку/folder/Finder)."),
]
TOOL_NAMES = [name for name, _ in TOOLS]

# Few-shot: формулировки ОТЛИЧНЫ от реальных запросов (учим паттернам, не подгонка).
FEWSHOT = [
    ("выведи список компонентов",   "show_components"),
    ("какие есть проекты",          "show_projects"),
    ("проекты в адсв",              "show_adsw_projects"),
    ("network проекты покажи",      "show_network_projects"),
    ("открой rag",                  "show_rag"),
    ("зайди в папку adsw",          "open_adsw"),
    ("здравствуй",                  "none"),
    ("доброе утро",                 "none"),
    ("ок, спасибо большое",         "none"),
    ("как твои дела сегодня",       "none"),
    ("что нового",                  "none"),
    ("расскажи что-нибудь смешное", "none"),
    ("который сейчас час",          "none"),
    ("пока",                        "none"),
]


def build_system():
    tools_block = "\n".join(f"- {name} — {desc}" for name, desc in TOOLS)
    examples = "\n".join(f"{q} → {t}" for q, t in FEWSHOT)
    return f"""Ты — классификатор. По запросу пользователя верни ровно ОДНО имя инструмента из списка, либо none. Ты ничего не запускаешь, только называешь подходящее.

Сначала проверь: это явная команда показать/открыть что-то из списка? Если запрос — приветствие, вопрос о тебе, болтовня, благодарность, шутка или что угодно не из списка — сразу none.

Инструменты:
{tools_block}

Правила:
- Глаголы «открой», «запусти», «покажи» означают запуск раздела и сами по себе НЕ значат open_adsw.
- Уточнение «адсв»/ADSW рядом с «проект» = show_adsw_projects; «нетворк»/Network рядом с «проект» = show_network_projects.
- Приветствие, благодарность, болтовня, вопрос не по теме → none.

Примеры (другие формулировки):
{examples}"""


# Ограничение вывода грамматикой: ровно одно из имён или "none".
ToolChoice = Literal[tuple(TOOL_NAMES + ["none"])]


class Classifier:
    """Грузит модель один раз, держит в памяти, классифицирует запросы.

    use_base=False → дообученная 350M (FINETUNED) + короткий промпт.
    use_base=True  → базовая 1.2B Instruct (BASE) + few-shot промпт.
    CLF_MODEL переопределяет путь к модели (промпт выбирается по use_base).
    """

    def __init__(self, use_base=False):
        import mlx_lm
        import outlines
        self._mlx = mlx_lm
        repo = REPO or (BASE if use_base else FINETUNED)
        self.repo = repo
        base, self.tok = mlx_lm.load(repo)
        self.model = outlines.from_mlxlm(base, self.tok)
        self.system = build_system() if use_base else SHORT_SYSTEM

    def classify(self, text):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": text}]
        prompt = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        return self.model(prompt, ToolChoice, max_tokens=32)


# --- HTTP-сервер ------------------------------------------------------------
def serve(clf, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": clf.repo, "tools": TOOL_NAMES})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/classify":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                text = (data.get("text") or "").strip()
            except Exception:
                return self._send(400, {"error": "bad json"})
            if not text:
                return self._send(400, {"error": "empty text"})
            t = time.time()
            tool = clf.classify(text)
            self._send(200, {"tool": tool, "ms": round((time.time() - t) * 1000)})

        def log_message(self, *_):
            pass  # тихо

    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Классификатор на http://127.0.0.1:{port}  (POST /classify, GET /health)")
    srv.serve_forever()


# --- Встроенный самотест ----------------------------------------------------
SELFTEST = [
    ("покажи компоненты", "show_components"),
    ("открой проекты", "show_projects"),
    ("покажи проекты адсв", "show_adsw_projects"),
    ("открой адсв проекты", "show_adsw_projects"),
    ("проекты адсв", "show_adsw_projects"),
    ("проекты нетворк", "show_network_projects"),
    ("запусти раг", "show_rag"),
    ("покажи рак систему", "show_rag"),
    ("открой папку адсв", "open_adsw"),
    ("открой адсв в finder", "open_adsw"),
    ("привет", "none"),
    ("как дела", "none"),
    ("расскажи анекдот", "none"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="запрос для разовой классификации")
    ap.add_argument("--base", action="store_true",
                    help="базовая 1.2B Instruct + few-shot (вместо дообученной 350M)")
    ap.add_argument("--serve", action="store_true", help="запустить HTTP-сервер")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    clf = Classifier(use_base=args.base)
    print(f"Загружено: {clf.repo}", file=sys.stderr)

    if args.serve:
        serve(clf, args.port)
    elif args.selftest:
        ok = 0
        for text, exp in SELFTEST:
            got = clf.classify(text)
            mark = "✓" if got == exp else "✗"
            ok += got == exp
            print(f"  {mark} «{text}» → {got}  (ждали {exp})")
        print(f"\nИтог: {ok}/{len(SELFTEST)}")
    elif args.text:
        print(clf.classify(args.text))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
