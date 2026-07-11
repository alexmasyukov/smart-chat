#!/usr/bin/env bash
# Быстрый клиент Supertonic через curl — БЕЗ запуска Python на каждый вызов.
# Требует поднятый server.py (порт 8126). Фиксированный out => сразу afplay,
# без парсинга JSON. ~0.27с на короткую фразу.
#
#   ./say.sh "Обновление: сервер перезапущен"
#   VOICE=M1 SPEED=1.4 STEPS=4 ./say.sh "Быстрее и мужским голосом"
#   NOPLAY=1 ./say.sh "Только сгенерировать, не проигрывать"
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8126}"
VOICE="${VOICE:-F1}"
SPEED="${SPEED:-1.2}"
STEPS="${STEPS:-6}"
LANG_="${LANG_:-ru}"
OUT="${OUT:-$HERE/out/say.wav}"

TEXT="$*"
[ -z "$TEXT" ] && { echo "usage: ./say.sh \"текст\"" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
curl -sf "http://127.0.0.1:$PORT/gen" -G \
  --data-urlencode "text=$TEXT" \
  -d "voice=$VOICE" -d "speed=$SPEED" -d "steps=$STEPS" \
  -d "lang=$LANG_" -d "out=$OUT" >/dev/null \
  || { echo "сервер на :$PORT не отвечает — запусти server.py" >&2; exit 1; }

[ -n "${NOPLAY:-}" ] || afplay "$OUT"
