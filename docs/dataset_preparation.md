# Dataset Preparation

This document describes the dataset organization used for DIM-IAG-RS evaluation.

The final experiments are conducted on four datasets:

- CODaN for image classification
- ACDC for semantic segmentation
- LOD for object detection
- LIS for instance segmentation

## CODaN

CODaN is used for image classification evaluation.

Expected structure:

CODaN/
  test_night/
    Bicycle/
    Boat/
    Bottle/
    Bus/
    Car/
    Cat/
    Chair/
    Cup/
    Dog/
    Motorbike/
  resnet18_baseline.pth.tar

The enhanced images should preserve the same class-folder structure as the original test set.

## ACDC

ACDC is used for semantic segmentation evaluation.

Expected structure:

ACDC/
  images/
    val/
  labels/
    val/

The enhanced validation images should preserve the original image names and directory structure.

## LOD

LOD is used for object detection evaluation in YOLO detection format.

Expected structure:

LOD_exp/
  images/
    raw/
    dim_iag_rs/
  labels/
    raw/
    dim_iag_rs/

The enhanced images should keep the same file names as the raw images. Since enhancement does not change object locations, the detection labels are shared with the raw images.

## LIS

LIS is used for instance segmentation evaluation in YOLO segmentation format.

Expected structure:

LIS/
  yolo_lis_coco80_fixed/
    images/
      val/
      val_dim_iag_rs/
    labels/
      val/
      val_dim_iag_rs/

The enhanced images should keep the same file names as the raw images. Since enhancement does not change instance masks, the segmentation labels are shared with the raw images.

## Evaluation Protocol

For fair comparison, all enhanced images should be generated from the same raw inputs and evaluated using the same downstream models, metrics, and dataset splits.
