#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/build/train_multimodal"
OUT_DIR="$SCRIPT_DIR/outputs"

LR="${LR:-1e-5}"
MAX_STEPS="${MAX_STEPS:-50}"
PRECISION="${PRECISION:-fp32}"
BENCHMARK="${BENCHMARK:-0}"

if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$BIN" ]; then
  echo ">>> Building train_multimodal in $SCRIPT_DIR/build..."
  cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build"
  cmake --build "$SCRIPT_DIR/build" --target train_multimodal -j
fi

mkdir -p "$OUT_DIR"

if [ "$BENCHMARK" = "1" ]; then
  echo ">>> Running empirical precision benchmark (FP32 vs BF16 vs FP16)..."
  "$BIN" \
    --benchmark_precisions \
    --lr "$LR" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUT_DIR"
else
  echo ">>> Running Qwen Multimodal fine-tuning with lr=$LR, precision=$PRECISION, steps=$MAX_STEPS..."
  "$BIN" \
    --lr "$LR" \
    --precision "$PRECISION" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUT_DIR"
fi
