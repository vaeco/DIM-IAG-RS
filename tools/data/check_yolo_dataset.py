import argparse
from pathlib import Path
from collections import Counter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path):
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_labels(label_dir: Path):
    return sorted(
        p for p in label_dir.rglob("*.txt")
        if p.is_file()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check YOLO detection or segmentation dataset structure."
    )
    parser.add_argument("--images", required=True, help="Path to image directory.")
    parser.add_argument("--labels", required=True, help="Path to label directory.")
    parser.add_argument("--nc", type=int, default=None, help="Number of classes.")
    parser.add_argument("--seg", action="store_true", help="Check YOLO segmentation labels.")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)

    images = collect_images(image_dir)
    labels = collect_labels(label_dir)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)

    class_counter = Counter()
    bad_lines = []
    empty_labels = 0

    for label_path in labels:
        lines = [
            line.strip()
            for line in label_path.read_text(errors="ignore").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        for line_id, line in enumerate(lines, start=1):
            parts = line.split()

            try:
                class_id = int(float(parts[0]))
            except Exception:
                bad_lines.append((str(label_path), line_id, "invalid class id", line))
                continue

            class_counter[class_id] += 1

            if args.nc is not None and not (0 <= class_id < args.nc):
                bad_lines.append(
                    (str(label_path), line_id, f"class id out of range [0, {args.nc - 1}]", line)
                )

            if args.seg:
                if len(parts) < 7:
                    bad_lines.append((str(label_path), line_id, "segmentation label too short", line))
                elif (len(parts) - 1) % 2 != 0:
                    bad_lines.append((str(label_path), line_id, "segmentation coordinates should be pairs", line))
            else:
                if len(parts) != 5:
                    bad_lines.append((str(label_path), line_id, "detection label should have 5 columns", line))

    print("Image directory:", image_dir)
    print("Label directory:", label_dir)
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(extra_labels))
    print("Empty label files:", empty_labels)
    print("Class distribution:", dict(sorted(class_counter.items())))
    print("Bad lines:", len(bad_lines))

    if missing_labels[:10]:
        print("Sample images without labels:", missing_labels[:10])

    if extra_labels[:10]:
        print("Sample labels without images:", extra_labels[:10])

    for item in bad_lines[:20]:
        print("BAD:", item)

    if len(images) == 0:
        print("ERROR: no images found.")

    if len(labels) == 0:
        print("ERROR: no labels found.")


if __name__ == "__main__":
    main()
