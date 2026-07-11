#!/usr/bin/env python3
"""Локальный чат с MLX-моделями LFM2.5 (Instruct / Thinking).

Запуск моделей напрямую через mlx-lm (Apple Silicon). Выбираешь модель и общаешься;
история диалога сохраняется в рамках сессии.
"""

import sys

MODELS = {
    "1": ("Instruct — быстрый, отвечает сразу", "mlx-community/LFM2.5-1.2B-Instruct-8bit"),
    "2": ("Thinking — сначала рассуждает, потом отвечает", "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit"),
}

GREEN = "\033[32m"
CYAN = "\033[36m"
GREY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


ALIASES = {"1": "1", "instruct": "1", "2": "2", "thinking": "2"}


def resolve_preset(arg):
    """Преобразует аргумент (1/2/instruct/thinking) в repo модели, иначе None."""
    key = ALIASES.get((arg or "").lower())
    return MODELS[key][1] if key else None


def pick_model():
    print(f"{BOLD}Выберите модель:{RESET}")
    for key, (label, repo) in MODELS.items():
        print(f"  {key}) {label}  {GREY}[{repo}]{RESET}")
    while True:
        choice = input("Номер модели: ").strip()
        if choice in MODELS:
            return MODELS[choice][1]
        print("Введите 1 или 2.")


def main():
    from mlx_lm import load, stream_generate

    # Модель можно задать аргументом (1/2/instruct/thinking), иначе спросим.
    preset = resolve_preset(sys.argv[1]) if len(sys.argv) > 1 else None
    repo = preset or pick_model()
    print(f"\nЗагружаю модель {CYAN}{repo}{RESET} … (первый раз — скачается с HuggingFace)")
    model, tokenizer = load(repo)
    print("Готово. Команды: /model — сменить модель, /clear — очистить историю, /exit — выход.\n")

    messages = []

    while True:
        try:
            user_text = input(f"{BOLD}Вы:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/clear":
            messages = []
            print("История очищена.\n")
            continue
        if user_text == "/model":
            repo = pick_model()
            print(f"\nЗагружаю {CYAN}{repo}{RESET} …")
            model, tokenizer = load(repo)
            messages = []
            print("Готово.\n")
            continue

        messages.append({"role": "user", "content": user_text})

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        print(f"{GREEN}Бот:{RESET} ", end="", flush=True)
        answer = ""
        for response in stream_generate(model, tokenizer, prompt, max_tokens=2048):
            print(response.text, end="", flush=True)
            answer += response.text
        print("\n")

        messages.append({"role": "assistant", "content": answer})

    print("Пока!")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Не найден mlx-lm. Установите зависимости: см. README.md (./run.sh всё сделает сам).")
        sys.exit(1)
