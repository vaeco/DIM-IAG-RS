# Results Summary

This file separates validated results by the exact inference version used. Metrics from different checkpoints or routing variants must not be merged into one row.

## CODaN Image Classification

| Inference version | Top-1 Accuracy | Top-5 Accuracy | Total images |
|---|---:|---:|---:|
| Conservative Release + medium RS | 55.96 | 91.24 | 2500 |
| Bright-gate expert | 51.84 | 89.80 | 2500 |
| **Dual-Gate Router v1** (`0.81 / 0.02 / 0.95`) | **55.00** | **91.08** | **2500** |

The Dual-Gate Router recovers most of the conservative expert's classification performance while retaining a stronger route for severely under-exposed inputs.

## LOD Object Detection

| Inference version | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Bright-gate expert | 0.772 | 0.598 | 0.698 | 0.501 |
| Dual-Gate Router v1 | Pending full downstream evaluation | Pending | Pending | Pending |

## LIS Instance Segmentation

| Inference version | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bright-gate expert | 0.761 | 0.558 | 0.657 | 0.480 | 0.750 | 0.527 | 0.616 | 0.383 |
| Dual-Gate Router v1 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## ACDC Semantic Segmentation

The current release retains the existing ACDC evaluation utility. Final Dual-Gate Router metrics should be added only after a complete, version-controlled evaluation.
