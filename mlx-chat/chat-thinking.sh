#!/usr/bin/env bash
# Чат только с Thinking-моделью (для запуска в отдельном терминале).
set -euo pipefail
cd "$(dirname "$0")"
[[ -d .venv ]] || { uv venv --python 3.12 .venv; uv pip install --python .venv -r requirements.txt; }
exec .venv/bin/python chat.py thinking
