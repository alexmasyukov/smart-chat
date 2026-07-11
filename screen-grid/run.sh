#!/usr/bin/env bash
# Детектор (ручной триггер) + два отдельных приложения: button (кнопка Scan)
# и overlay (рисует точки/номера/границы).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${SG_PORT:-8132}"
cd "$HERE"; mkdir -p out

if [[ ! -x overlay || overlay.swift -nt overlay ]]; then
  echo "Собираю overlay…" >&2
  swiftc -swift-version 5 -O overlay.swift -o overlay -framework AppKit -framework Foundation
fi
if [[ ! -x button || button.swift -nt button ]]; then
  echo "Собираю button…" >&2
  swiftc -swift-version 5 -O button.swift -o button -framework AppKit -framework Foundation
fi

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Запускаю детектор на :${PORT}…" >&2
  SG_PORT="$PORT" nohup python3 detect.py > out/server.log 2>&1 &
  disown || true
  for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 0.25; done
fi

POINTS_URL="http://127.0.0.1:${PORT}/points" nohup ./overlay > out/overlay.log 2>&1 &
disown || true
SCAN_URL="http://127.0.0.1:${PORT}/scan" nohup ./button > out/button.log 2>&1 &
disown || true

echo "overlay и button запущены (детектор :${PORT})." >&2
