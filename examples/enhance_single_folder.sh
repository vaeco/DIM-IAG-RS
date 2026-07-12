#!/usr/bin/env bash
set -euo pipefail

python scripts/enhance_dim_iag_rs.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --conservative_ckpt weights/dim_iag_rs_best.pth \
  --bright_ckpt weights/dim_iag_rs_bright_gate_best.pth \
  --base_alpha 0.6 \
  --router_center 0.81 \
  --router_width 0.02 \
  --rho_max 0.95 \
  --device cuda:0 \
  --print_freq 100
