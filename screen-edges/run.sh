#!/usr/bin/env bash
# Запуск детектора граней экрана + прозрачного оверлея.
# Поднимает Python-сервер detect.py (если не запущен) и собирает/запускает
# нативный Swift-оверлей, который рисует зелёные линии поверх экрана.
# Сервер НЕ убивается при выходе оверлея — он общий (его же будет читать кот).
#   Остановить сервер:  pkill -f 'screen-edges/detect.py'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${SE_PORT:-8130}"
HEALTH="http://127.0.0.1:${PORT}/health"

cd "$HERE"
mkdir -p out

# Пересобираем оверлей, если исходник новее бинаря (или бинаря нет).
if [[ ! -x overlay || overlay.swift -nt overlay ]]; then
  echo "Собираю оверлей…" >&2
  swiftc -swift-version 5 -O overlay.swift -o overlay -framework AppKit -framework Foundation
fi

# Поднимаем детектор в фоне (detached), если ещё не отвечает.
if ! curl -sf "$HEALTH" >/dev/null 2>&1; then
  echo "Запускаю детектор на :${PORT}…" >&2
  SE_PORT="$PORT" nohup python3 detect.py > out/server.log 2>&1 &
  disown || true
  for _ in $(seq 1 40); do
    curl -sf "$HEALTH" >/dev/null 2>&1 && break
    sleep 0.25
  done
fi

echo "Оверлей запущен. Линии обновляются ~раз в 2с. Ctrl+C — закрыть оверлей." >&2
EDGES_URL="http://127.0.0.1:${PORT}/edges" exec ./overlay
