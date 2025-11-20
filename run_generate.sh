#!/bin/bash

# 以訓練完成的 DDPM 權重產生題目要求的 10,000 張 28x28 RGB PNG。
# 使用方式：
#   bash run_generate.sh [checkpoint_path] [output_dir]
# 若未提供參數，預設使用 trained_model/mnist-ddpm-baseline/checkpoints/best.pt
# 並將影像寫入 generated/。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT="${PROJECT_ROOT}/trained_model/mnist-ddpm-baseline/checkpoints/best.pt"
CHECKPOINT_PATH="${1:-$DEFAULT_CKPT}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/generated}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "找不到 checkpoint：$CHECKPOINT_PATH"
  echo "請提供有效的 checkpoint 路徑，例如 ckpt_epoch_xx.pt 或 ckpt_xxxxxx.pt"
  exit 1
fi

echo "使用 checkpoint：$CHECKPOINT_PATH"
echo "輸出資料夾：$OUTPUT_DIR"

python "${PROJECT_ROOT}/src/generate_images.py" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-images 10000 \
  --batch-size 4080 \
  --model-image-size 28 \
  --output-size 28 \
  --timesteps 1000 \
  --beta-start 1e-4 \
  --beta-end 0.02 \
  --base-channels 128 \
  --residual-dropout 0.0

