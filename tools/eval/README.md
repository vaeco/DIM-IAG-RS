# Evaluation Tools

This directory contains evaluation scripts for downstream tasks.

## CODaN

`eval_codan_resnet18_official_order.py` is used for image classification evaluation on CODaN.

Metrics:

- Top-1 Accuracy
- Top-5 Accuracy

## ACDC

`eval_acdc_mmseg.py` is used for semantic segmentation evaluation on ACDC when the required MMSegmentation environment and checkpoints are available.

Metrics:

- mIoU
- mAcc
- aAcc

## LOD and LIS

LOD and LIS are evaluated with Ultralytics YOLO using the corresponding dataset configuration files.

LOD uses object detection metrics:

- Precision
- Recall
- mAP50
- mAP50-95

LIS uses instance segmentation metrics:

- Box Precision
- Box Recall
- Box mAP50
- Box mAP50-95
- Mask Precision
- Mask Recall
- Mask mAP50
- Mask mAP50-95
