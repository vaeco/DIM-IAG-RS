import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def list_images(img_dir):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    files = []
    for p in Path(img_dir).rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def find_label_for_image(img_path, img_root, label_root):
    rel = img_path.relative_to(img_root)
    candidates = []

    # 1. Same relative path
    candidates.append(Path(label_root) / rel)

    # 2. Common Cityscapes / ACDC naming rules
    name = rel.name
    candidates.append(Path(label_root) / rel.with_name(name.replace("_rgb_anon", "_gt_labelTrainIds")))
    candidates.append(Path(label_root) / rel.with_name(name.replace("_leftImg8bit", "_gtFine_labelTrainIds")))
    candidates.append(Path(label_root) / rel.with_name(name.replace("_rgb", "_gt_labelTrainIds")))
    candidates.append(Path(label_root) / rel.with_name(name.replace("rgb_anon", "gt_labelTrainIds")))

    # 3. Same stem search in same subfolder
    same_folder = Path(label_root) / rel.parent
    if same_folder.exists():
        stem = img_path.stem
        for p in same_folder.glob("*.png"):
            if stem in p.stem or p.stem in stem:
                candidates.append(p)

    for c in candidates:
        if c.exists():
            return c

    return None


def get_pred_from_result(result):
    # mmseg 1.x
    if hasattr(result, "pred_sem_seg"):
        pred = result.pred_sem_seg.data
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0]
        return pred.astype(np.int64)

    # old mmseg style
    if isinstance(result, (list, tuple)):
        pred = result[0]
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        return np.asarray(pred).astype(np.int64)

    raise RuntimeError("Unsupported mmseg inference result format.")


def fast_hist(label, pred, num_classes=19, ignore_index=255):
    mask = (label != ignore_index)
    mask &= (label >= 0)
    mask &= (label < num_classes)
    hist = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    from mmseg.apis import init_model, inference_model

    img_root = Path(args.img_dir)
    label_root = Path(args.label_dir)

    img_files = list_images(img_root)
    print("num images:", len(img_files))
    print("img dir:", img_root)
    print("label dir:", label_root)

    if len(img_files) == 0:
        raise RuntimeError("No images found.")

    model = init_model(args.config, args.checkpoint, device=args.device)

    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    missing = 0

    for img_path in tqdm(img_files):
        label_path = find_label_for_image(img_path, img_root, label_root)
        if label_path is None:
            missing += 1
            if missing <= 10:
                print("missing label for:", img_path)
            continue

        result = inference_model(model, str(img_path))
        pred = get_pred_from_result(result)

        label = np.array(Image.open(label_path))
        if label.ndim == 3:
            label = label[:, :, 0]
        label = label.astype(np.int64)

        if pred.shape != label.shape:
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_img = pred_img.resize((label.shape[1], label.shape[0]), resample=Image.NEAREST)
            pred = np.array(pred_img).astype(np.int64)

        hist += fast_hist(label, pred, args.num_classes, args.ignore_index)

    if missing > 0:
        print("missing labels:", missing)

    tp = np.diag(hist)
    pos_gt = hist.sum(axis=1)
    pos_pred = hist.sum(axis=0)

    iou = tp / np.maximum(pos_gt + pos_pred - tp, 1)
    acc_cls = tp / np.maximum(pos_gt, 1)

    valid_classes = pos_gt > 0

    miou = np.mean(iou[valid_classes])
    macc = np.mean(acc_cls[valid_classes])
    acc = tp.sum() / np.maximum(hist.sum(), 1)

    print("====================================")
    print(f"mIoU: {miou:.4f}")
    print(f"mAcc: {macc:.4f}")
    print(f"Acc.: {acc:.4f}")
    print("====================================")
    print("per-class IoU:")
    for i, v in enumerate(iou):
        print(f"class {i:02d}: {v:.4f}")


if __name__ == "__main__":
    main()
