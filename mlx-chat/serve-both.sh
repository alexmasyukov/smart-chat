#!/usr/bin/env bash
# Поднимает ОБЕ модели LFM2.5 в память одновременно как локальные MLX-серверы.
# Instruct → :8081, Thinking → :8082 (OpenAI-совместимый API на /v1).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv -r requirements.txt
fi
PY=.venv/bin/python

INSTRUCT="mlx-community/LFM2.5-1.2B-Instruct-8bit"
THINKING="LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit"

start() { # name repo port
  local name=$1 repo=$2 port=$3
  if curl -s "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "${name} уже запущен на :${port}"; return
  fi
  echo "Поднимаю ${name} на :${port} ..."
  nohup "$PY" -m mlx_lm server --model "$repo" --port "${port}" >"/tmp/mlx_${name}.log" 2>&1 &
  echo $! > "/tmp/mlx_${name}.pid"
}

warm() { # port repo  — форсируем загрузку весов в память одним токеном
  curl -s -X POST "http://127.0.0.1:$1/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" >/dev/null 2>&1 || true
}

start instruct "$INSTRUCT" 8081
start thinking "$THINKING" 8082

for port in 8081 8082; do
  for _ in $(seq 1 120); do curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break; sleep 0.5; done
done

echo "Прогреваю (загружаю веса в RAM) ..."
warm 8081 "$INSTRUCT"
warm 8082 "$THINKING"

echo
echo "Обе модели в памяти:"
echo "  Instruct → http://127.0.0.1:8081/v1   (PID $(cat /tmp/mlx_instruct.pid 2>/dev/null))"
echo "  Thinking → http://127.0.0.1:8082/v1   (PID $(cat /tmp/mlx_thinking.pid 2>/dev/null))"
echo "Логи: /tmp/mlx_instruct.log, /tmp/mlx_thinking.log"
echo "Остановить: ./stop-both.sh"
