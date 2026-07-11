#!/usr/bin/env bash
# Локальный чат с MLX-моделями. Поднимает venv и ставит mlx-lm при первом запуске.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Создаю окружение (.venv) и ставлю mlx-lm…" >&2
  uv venv --python 3.12 .venv
  uv pip install --python .venv -r requirements.txt
fi

exec .venv/bin/python chat.py
