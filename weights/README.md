# Model Weights

Model weights are not included in the source archive and are ignored by `.gitignore`.

The final Dual-Gate Router expects two full DIM-IAG-RS checkpoints:

```text
weights/dim_iag_rs_best.pth
weights/dim_iag_rs_bright_gate_best.pth
```

- `dim_iag_rs_best.pth`: conservative Release expert.
- `dim_iag_rs_bright_gate_best.pth`: brightness-oriented expert retrained with `scripts/train_bright_gate.py`.

Check both files and load them into the model:

```bash
python scripts/validate_release.py
```

Run final enhancement:

```bash
python scripts/enhance_dim_iag_rs.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --conservative_ckpt weights/dim_iag_rs_best.pth \
  --bright_ckpt weights/dim_iag_rs_bright_gate_best.pth \
  --device cuda:0
```
