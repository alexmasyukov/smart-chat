#!/usr/bin/env bash
# Детектор блоков по сетке цветов + оверлей с точками.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${SB_PORT:-8131}"
cd "$HERE"; mkdir -p out

if [[ ! -x overlay || overlay.swift -nt overlay ]]; then
  echo "Собираю оверлей…" >&2
  swiftc -swift-version 5 -O overlay.swift -o overlay -framework AppKit -framework Foundation
fi

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Запускаю детектор на :${PORT}…" >&2
  SB_PORT="$PORT" nohup python3 detect.py > out/server.log 2>&1 &
  disown || true
  for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 0.25; done
fi

echo "Оверлей запущен. Ctrl+C — закрыть." >&2
POINTS_URL="http://127.0.0.1:${PORT}/points" exec ./overlay
