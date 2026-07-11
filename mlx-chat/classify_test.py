#!/usr/bin/env python3
"""Classifier: ОДИН constrained-вызов, голый enum (mlx-lm + Outlines).

Вывод ограничен грамматикой = ровно одно имя инструмента или none. 100%
детерминированный формат, без рассуждения → без лупов, мгновенно.
Few-shot учит паттернам (формулировки примеров ОТЛИЧНЫ от тестовых).

Запуск:  classify_test.py instruct   |   classify_test.py thinking
"""
import sys
import time
from enum import Enum
from typing import Literal

import mlx_lm
import outlines
from pydantic import BaseModel, Field

REPOS = {
    "instruct": "mlx-community/LFM2.5-1.2B-Instruct-8bit",
    "thinking": "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit",
}

ToolChoice = Literal[
    "show_components", "show_projects", "show_rag",
    "show_adsw_projects", "show_network_projects", "open_adsw", "none",
]


class Tool(str, Enum):
    show_components = "show_components"
    show_projects = "show_projects"
    show_rag = "show_rag"
    show_adsw_projects = "show_adsw_projects"
    show_network_projects = "show_network_projects"
    open_adsw = "open_adsw"
    none = "none"


class Result(BaseModel):
    reasoning: str = Field(max_length=300)
    tool: Tool

SYSTEM = """Ты — классификатор. По запросу пользователя верни ровно ОДНО имя инструмента из списка, либо none. Ты ничего не запускаешь, только называешь подходящее.

Сначала проверь: это явная команда показать/открыть что-то из списка? Если запрос — приветствие, вопрос о тебе, болтовня, благодарность, шутка или что угодно не из списка — сразу none.

Инструменты:
- show_components — показать компоненты.
- show_projects — показать проекты (общие, без уточнения ADSW/Network).
- show_rag — показать RAG-систему (в запросе «раг», «рак», RAG).
- show_adsw_projects — показать проекты ADSW (в запросе «адсв»/ADSW).
- show_network_projects — показать проекты Network (в запросе «нетворк»/Network).
- open_adsw — открыть ПАПКУ ADSW в Finder (только если явно про папку/folder/Finder).

Правила:
- Глаголы «открой», «запусти», «покажи» означают запуск раздела и сами по себе НЕ значат open_adsw.
- Уточнение «адсв»/ADSW рядом с «проект» = show_adsw_projects; «нетворк»/Network рядом с «проект» = show_network_projects.
- Приветствие, благодарность, болтовня, вопрос не по теме → none.

Примеры (другие формулировки):
выведи список компонентов → show_components
какие есть проекты → show_projects
проекты в адсв → show_adsw_projects
network проекты покажи → show_network_projects
открой rag → show_rag
зайди в папку adsw → open_adsw
здравствуй → none
доброе утро → none
ок, спасибо большое → none
как твои дела сегодня → none
что нового → none
расскажи что-нибудь смешное → none
который сейчас час → none
пока → none"""

CASES = [
    ("покажи компоненты", "show_components"),
    ("что по компонентам", "show_components"),
    ("открой проекты", "show_projects"),
    ("покажи проекты", "show_projects"),
    ("покажи проекты адсв", "show_adsw_projects"),
    ("открой адсв проекты", "show_adsw_projects"),
    ("проекты адсв", "show_adsw_projects"),
    ("проекты нетворк", "show_network_projects"),
    ("покажи нетворк проекты", "show_network_projects"),
    ("запусти раг", "show_rag"),
    ("покажи рак систему", "show_rag"),
    ("открой папку адсв", "open_adsw"),
    ("открой адсв в finder", "open_adsw"),
    ("привет", "none"),
    ("спасибо", "none"),
    ("как дела", "none"),
    ("расскажи анекдот", "none"),
]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "instruct"
    reason = len(sys.argv) > 2 and sys.argv[2] == "reason"
    repo = REPOS[which]
    print(f"Гружу [{which}{' +reason' if reason else ''}] {repo} …")
    base, tok = mlx_lm.load(repo)
    model = outlines.from_mlxlm(base, tok)
    print("Готово.\n")

    ok = 0
    total = 0.0
    for text, expected in CASES:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t = time.time()
        if reason:
            raw = model(prompt, Result, max_tokens=400)
            got = Result.model_validate_json(raw).tool.value
        else:
            got = model(prompt, ToolChoice, max_tokens=32)
        dt = time.time() - t
        total += dt
        mark = "✓" if got == expected else "✗"
        if got == expected:
            ok += 1
        print(f"  {mark} «{text}»  →  {got}   (ждали {expected})  [{dt:.2f}s]")
    print(f"\n[{which}{' +reason' if reason else ''}] Итог: {ok}/{len(CASES)}   среднее {total/len(CASES):.2f}s/запрос")


if __name__ == "__main__":
    main()
