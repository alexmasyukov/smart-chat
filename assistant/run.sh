#!/usr/bin/env bash
# Собирает и запускает Assistant. По умолчанию — прямой запуск бинаря, чтобы
# видеть логи в терминале (Ctrl+C — закрыть). При первом старте macOS спросит
# доступ к микрофону.
#   ./run.sh          — запустить с логами
#   ./run.sh open     — запустить как обычное приложение (open Assistant.app)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [[ ! -x Assistant.app/Contents/MacOS/Assistant || main.swift -nt Assistant.app/Contents/MacOS/Assistant ]]; then
  ./build.sh
fi

if [[ "${1:-}" == "open" ]]; then
  open Assistant.app
else
  exec ./Assistant.app/Contents/MacOS/Assistant
fi
