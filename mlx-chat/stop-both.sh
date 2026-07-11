#!/usr/bin/env bash
# Останавливает обе модели (mlx-серверы), освобождает память.
cd "$(dirname "$0")"
for name in instruct thinking; do
  pid=$(cat "/tmp/mlx_$name.pid" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null; then
    echo "остановлен $name ($pid)"
  fi
  rm -f "/tmp/mlx_$name.pid"
done
pkill -f "mlx_lm server" 2>/dev/null || true
echo "Готово."
