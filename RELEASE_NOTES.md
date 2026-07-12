# Release Notes

## Final clean Release

This package keeps the original repository structure and changes only the files needed for the final Dual-Gate Router workflow.

### Added

- `scripts/enhance_dim_iag_rs.py`: final Dual-Gate Router inference entry.
- `scripts/train_bright_gate.py`: brightness-oriented gate training entry.
- `scripts/validate_release.py`: checkpoint existence and load check.
- `configs/dual_gate_router.example.yaml`: validated router parameters.
- `docs/server_deployment.md`: minimal replacement procedure.
- Compatibility aliases for the earlier Dual-Gate Router script names.

### Preserved

- All files under `dim_iag_rs/` are unchanged.
- The original single-gate inference is retained as `scripts/enhance_single_gate.py`.
- Existing training, dataset, and evaluation utilities are retained.

### Final Router Defaults

```text
base_alpha    = 0.60
router_center = 0.81
router_width  = 0.02
rho_max       = 0.95
```

### Current validated record

```text
CODaN Dual-Gate Router v1
Top-1: 55.00
Top-5: 91.08
Images: 2500
```
