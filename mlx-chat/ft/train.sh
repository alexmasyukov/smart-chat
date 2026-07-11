#!/usr/bin/env bash
# Полный пайплайн дообучения классификатора: датасет → LoRA → fuse → eval.
# Запуск:  cd mlx-chat && ./ft/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # → mlx-chat
PY=.venv/bin/python
BASE="/Users/alex/.cache/huggingface/hub/LiquidAI/LFM2.5-350M-MLX-8bit"

echo "==> 1/4  Генерация датасета"
$PY ft/gen_dataset.py

echo "==> 2/4  LoRA-обучение (~2 мин)"
$PY -m mlx_lm lora \
  --model "$BASE" --train --data ft/data \
  --iters 500 --batch-size 8 --num-layers 8 --learning-rate 1e-4 \
  --steps-per-report 50 --steps-per-eval 100 \
  --adapter-path ft/adapters 2>&1 | grep -vE "PyTorch was not found|Calculating loss"

echo "==> 3/4  Fuse адаптера в самостоятельную модель ft/fused"
$PY -m mlx_lm fuse --model "$BASE" --adapter-path ft/adapters --save-path ft/fused \
  2>&1 | grep -v "PyTorch was not found"

echo "==> 4/4  Eval на held-out"
$PY ft/eval.py --adapter ft/adapters 2>&1 | grep -vE "PyTorch was not found|Fetching" | grep -E "✓|✗|Итог"

echo "Готово. Дообученная модель: ft/fused (classifier.py использует её по умолчанию)."
