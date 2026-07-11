#!/usr/bin/env python3
"""Оценка классификатора (базовая или с LoRA-адаптером) на held-out фразах.

Короткий системный промпт (как в обучении) + грамматика-enum (Outlines) —
формат гарантирован, меряем только точность выбора.

    python eval.py                       # базовая 350M (без адаптера)
    python eval.py --adapter ft/adapters # 350M + LoRA
    python eval.py --model <repo|путь> [--adapter ...]
"""
import argparse
import sys
import time
from typing import Literal

import mlx_lm
import outlines

DEFAULT_MODEL = "/Users/alex/.cache/huggingface/hub/LiquidAI/LFM2.5-350M-MLX-8bit"

SYSTEM = "Ты — классификатор. По запросу пользователя верни ровно одно имя инструмента из набора или none. Только имя, без пояснений."

TOOL_NAMES = ["show_components", "show_projects", "show_rag",
              "show_adsw_projects", "show_network_projects", "open_adsw"]
ToolChoice = Literal[tuple(TOOL_NAMES + ["none"])]

# Held-out тестовые фразы (в train не попадают — исключаются gen_dataset.py)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    tag = f"{args.model.split('/')[-1]}" + (f" +adapter" if args.adapter else " (base)")
    print(f"Гружу {tag} …", file=sys.stderr)
    load_kw = {"adapter_path": args.adapter} if args.adapter else {}
    base, tok = mlx_lm.load(args.model, **load_kw)
    model = outlines.from_mlxlm(base, tok)

    ok = 0
    total = 0.0
    for text, exp in CASES:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t = time.time()
        got = model(prompt, ToolChoice, max_tokens=32)
        total += time.time() - t
        mark = "✓" if got == exp else "✗"
        ok += got == exp
        print(f"  {mark} «{text}» → {got}  (ждали {exp})")
    print(f"\n[{tag}] Итог: {ok}/{len(CASES)}   среднее {total/len(CASES):.2f}s")


if __name__ == "__main__":
    main()
