# mlx-chat

Локальный CLI-чат с моделями **LFM2.5-1.2B** через [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon).
Можно выбрать модель и поговорить с ней; история диалога хранится в рамках сессии.

## Модели

| № | Вариант | Репозиторий |
|---|---------|-------------|
| 1 | Instruct — быстрый, отвечает сразу | `mlx-community/LFM2.5-1.2B-Instruct-8bit` |
| 2 | Thinking — сначала рассуждает | `LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit` |

## Запуск

```bash
cd mlx-chat
./run.sh
```

При первом запуске создаётся `.venv` (Python 3.12) и ставится `mlx-lm`, затем модель
скачивается с HuggingFace (~1.3 ГБ на 8-bit). Дальше — чат.

Команды в чате: `/model` — сменить модель, `/clear` — очистить историю, `/exit` — выход.

## Требования

- Apple Silicon (MLX),
- `uv` и Python 3.12 (ставится автоматически в venv).
