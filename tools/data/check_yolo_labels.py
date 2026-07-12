import random
from pathlib import Path
import cv2


def draw_yolo_boxes(image_path, label_path, save_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] failed to read image: {image_path}")
        return False

    h, w = img.shape[:2]

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        lines = []

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        _, xc, yc, bw, bh = parts
        xc, yc, bw, bh = map(float, [xc, yc, bw, bh])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
    return True


def check_split(split="train", num_samples=5):
    image_root = Path(f"./widerface/images/{split}")
    label_root = Path(f"./widerface/labels/{split}")
    save_root = Path(f"./vis_check/{split}")
    save_root.mkdir(parents=True, exist_ok=True)

    image_files = list(image_root.rglob("*.jpg"))
    print(f"[INFO] split={split}, found {len(image_files)} images")

    if len(image_files) == 0:
        print(f"[ERROR] no images found in {image_root}")
        return

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    success = 0
    for image_path in samples:
        rel_path = image_path.relative_to(image_root)
        label_path = (label_root / rel_path).with_suffix(".txt")
        save_path = save_root / rel_path.name

        ok = draw_yolo_boxes(image_path, label_path, save_path)
        if ok:
            print(f"[OK] saved: {save_path}")
            success += 1

    print(f"[INFO] done, saved {success} visualizations to {save_root}")


if __name__ == "__main__":
    random.seed(42)

    # 检查训练集
    check_split(split="train", num_samples=5)

    # 检查验证集
    check_split(split="val", num_samples=5)