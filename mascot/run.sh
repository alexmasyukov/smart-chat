#!/usr/bin/env bash
# Запуск облака-пета: поднимает общий API-сервер (если не запущен) и нативную оболочку.
# Сервер НЕ убивается при выходе пета — он общий для CLI и облака. Остановить: pnpm stop.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8787}"
API_URL="http://127.0.0.1:${PORT}"

cd "$DIR"

# Собираем нативный бинарь, если исходник новее (или бинаря нет).
if [[ ! -x mascot/pet || mascot/main.swift -nt mascot/pet ]]; then
  echo "Собираю нативную оболочку…" >&2
  swiftc -swift-version 5 -O mascot/main.swift -o mascot/pet -framework AppKit
fi

# Поднимаем сервер в фоне (detached), если он ещё не отвечает.
if ! curl -s "${API_URL}/api/health" >/dev/null 2>&1; then
  echo "Запускаю API-сервер на ${API_URL}…" >&2
  PORT="$PORT" nohup node src/server.js >/dev/null 2>&1 &
  disown || true
  for _ in $(seq 1 60); do
    curl -s "${API_URL}/api/health" >/dev/null 2>&1 && break
    sleep 0.3
  done
fi

API_URL="$API_URL" exec ./mascot/pet
