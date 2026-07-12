# DIM-IAG-RS

Release implementation for **Deep Intensity Mapping with Risk-Aware Residual Suppression for Low-Light Image Enhancement in High-Level Vision Tasks**.

This release uses a **shared DIM-WRR backbone**, two complementary illumination-aware gate experts, and a continuous image-level router:

- **Conservative gate expert**: limits unnecessary enhancement for moderately illuminated or exposure-sensitive images.
- **Brightness-oriented gate expert**: increases enhancement strength for severely under-exposed images.
- **Risk-aware dual-gate router**: mixes the two experts according to image-level under-exposure severity, over-exposure risk, and noise risk.

The default release parameters are the final validated router candidate:

```text
base_alpha   = 0.60
router_center = 0.81
router_width  = 0.02
rho_max       = 0.95
```

## Final Formulation

The shared DIM-WRR backbone first predicts a base enhanced image:

```text
Y_base = F_DIM-WRR(X)
R = Y_base - X
```

The conservative and brightness-oriented experts produce effective residual gates:

```text
E_c = G_c * S_c
E_b = G_b * S_b
```

The image-level router predicts a continuous coefficient `rho`:

```text
rho = rho_max * sigmoid((u - router_center) / router_width) * risk_safety
```

where `u` is the image-level under-exposure severity. The final effective gate and enhanced image are:

```text
E = (1 - rho) * E_c + rho * E_b
Y = clamp(X + E * R, 0, 1)
```

No dataset name is used by the router. The same model, checkpoints, and parameters are applied to CODaN, LOD, LIS, and other inputs.

## Repository Structure

```text
DIM-IAG-RS/
├── configs/
├── datasets/
├── dim_iag_rs/
├── docs/
├── examples/
├── scripts/
│   ├── enhance_dim_iag_rs.py      # final Dual-Gate Router inference
│   ├── enhance_single_gate.py     # legacy single-gate inference
│   ├── train_bright_gate.py       # brightness-oriented expert training
│   └── validate_release.py        # checkpoint integrity validation
├── tests/
├── tools/
├── weights/
├── README.md
├── LICENSE
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The code is intended for a CUDA-enabled PyTorch environment. CPU inference is supported but slower.

## Model Weights

Model weights are not distributed in this source archive. Place the two checkpoints at:

```text
weights/dim_iag_rs_best.pth
weights/dim_iag_rs_bright_gate_best.pth
```

The first file is the conservative Release checkpoint. The second file is the retrained brightness-oriented gate checkpoint.

Run the release check after copying the weights:

```bash
python scripts/validate_release.py
```

## Final Image Enhancement

```bash
python scripts/enhance_dim_iag_rs.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --conservative_ckpt weights/dim_iag_rs_best.pth \
  --bright_ckpt weights/dim_iag_rs_bright_gate_best.pth \
  --device cuda:0 \
  --print_freq 100
```

The final router parameters are already the script defaults. They can also be written explicitly:

```bash
python scripts/enhance_dim_iag_rs.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --conservative_ckpt weights/dim_iag_rs_best.pth \
  --bright_ckpt weights/dim_iag_rs_bright_gate_best.pth \
  --base_alpha 0.6 \
  --router_center 0.81 \
  --router_width 0.02 \
  --rho_max 0.95 \
  --device cuda:0
```

The script preserves nested class folders and prints:

```text
num_images
FINAL STATS
rho mean avg
effective gate mean avg
Done.
```

These fields should be checked before downstream evaluation.

## Bright-Gate Training

The brightness-oriented expert is trained from the conservative full IAG checkpoint while the DIM-WRR backbone remains frozen by default:

```bash
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
```

After training, copy the best checkpoint to:

```bash
cp weights/bright_gate_training/best.pth \
   weights/dim_iag_rs_bright_gate_best.pth
```

## Legacy Single-Gate Inference

The original single-gate Release inference is retained for reproducibility:

```bash
python scripts/enhance_single_gate.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --ckpt weights/dim_iag_rs_best.pth \
  --base_alpha 0.6 \
  --preset medium \
  --device cuda:0
```

## Evaluation

Evaluation utilities are provided under `tools/eval/`.

The downstream tasks are:

| Dataset | Task | Metrics |
|---|---|---|
| CODaN | Image classification | Top-1, Top-5 Accuracy |
| ACDC | Semantic segmentation | mIoU, mAcc, aAcc |
| LOD | Object detection | Precision, Recall, mAP50, mAP50-95 |
| LIS | Instance segmentation | Box and Mask metrics |

See `docs/results_summary.md` for the current validated records and `docs/server_deployment.md` for the minimal replacement procedure used by the server release.

## Citation

```bibtex
@article{dim_iag_rs,
  title   = {Deep Intensity Mapping with Risk-Aware Residual Suppression for Low-Light Image Enhancement in High-Level Vision Tasks},
  author  = {Anonymous},
  journal = {To be updated},
  year    = {2026}
}
```

## License

This project is released under the license in `LICENSE`.
