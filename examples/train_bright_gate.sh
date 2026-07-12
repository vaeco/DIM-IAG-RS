#!/usr/bin/env bash
set -euo pipefail

python scripts/train_bright_gate.py \
  --input_dir /path/to/paired/input \
  --target_dir /path/to/paired/target \
  --ckpt weights/dim_iag_rs_best.pth \
  --load_mode iag \
  --save_dir weights/bright_gate_training \
  --epochs 8 \
  --batch_size 4 \
  --crop_size 384 \
  --amp \
  --print_freq 20
