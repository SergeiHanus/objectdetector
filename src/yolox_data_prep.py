#!/usr/bin/env python3
"""
YOLOX data preparation (the only supported mode).

This script prepares the `train_yolox/data/` layout consumed by our training pipeline:
- data/train/images
- data/val/images
- data/test/images
- data/annotations/{train,val,test}_labels.json

It converts YOLO-format `.txt` labels into COCO JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _is_image(fname: str) -> bool:
    return any(fname.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def _check_train_yolox_root(root: Path) -> None:
    if not (root / "data").is_dir() or not (root / "scripts").is_dir():
        print(f"Error: {root} does not look like train_yolox root (missing data/ or scripts/).", file=sys.stderr)
        sys.exit(1)


def cleanup_train_yolox_dataset(root: Path, raw_dir: str) -> None:
    print(f"Cleaning up dataset dirs under {root} (keeping {raw_dir})...")
    for rel in ["data/train", "data/val", "data/test", "data/annotations"]:
        p = root / rel
        if p.exists():
            shutil.rmtree(p)
            print(f"   Removed: {rel}")
    print("Cleanup done.")


def get_image_files(raw_dir: Path) -> list[str]:
    return sorted([f for f in os.listdir(raw_dir) if _is_image(f)])


def get_matching_labels(image_files: list[str], raw_dir: Path) -> set[str]:
    out: set[str] = set()
    for img_file in image_files:
        label_file = Path(img_file).stem + ".txt"
        if (raw_dir / label_file).exists():
            out.add(label_file)
        else:
            print(f"Warning: No label for {img_file}", file=sys.stderr)
    return out


def convert_yolo_to_coco(yolo_labels_dir: Path, images_dir: Path, output_file: Path, class_name: str) -> None:
    if Image is None:
        print("Error: Pillow required. Install with: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    coco_data = {
        "info": {"description": "YOLOX dataset", "version": "1.0"},
        "licenses": [{"id": 1, "name": "MIT", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": class_name, "supercategory": "object"}],
    }
    annotation_id = 0

    label_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith(".txt")]
    available_images = {f for f in os.listdir(images_dir) if _is_image(f)}
    stem_to_image = {Path(f).stem: f for f in available_images}

    for label_file in label_files:
        stem = Path(label_file).stem
        image_name = stem_to_image.get(stem)
        if not image_name:
            continue

        image_path = images_dir / image_name
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"Warning: Could not read {image_path}: {e}", file=sys.stderr)
            continue

        image_id = len(coco_data["images"])
        coco_data["images"].append({"id": image_id, "file_name": image_name, "width": width, "height": height})

        try:
            lines = (yolo_labels_dir / label_file).read_text(encoding="utf-8").splitlines()
        except Exception as e:
            print(f"Warning: Could not read {(yolo_labels_dir / label_file)}: {e}", file=sys.stderr)
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                lw = float(parts[3])
                lh = float(parts[4])
            except Exception:
                continue

            if class_id != 0:
                continue

            x_center_abs = x_center * width
            y_center_abs = y_center * height
            width_abs = lw * width
            height_abs = lh * height

            x = max(0.0, x_center_abs - width_abs / 2)
            y = max(0.0, y_center_abs - height_abs / 2)
            width_abs = min(width_abs, width - x)
            height_abs = min(height_abs, height - y)
            if width_abs <= 0 or height_abs <= 0:
                continue

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x, y, width_abs, height_abs],
                    "area": width_abs * height_abs,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(coco_data, indent=2), encoding="utf-8")
    print(f"Wrote {output_file} ({len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations).")


def prepare_train_yolox_dataset(*, train_yolox_root: str, raw_dir: str, dataset_name: str, class_name: str) -> None:
    root = Path(train_yolox_root).expanduser().resolve()
    _check_train_yolox_root(root)

    raw_path = Path(raw_dir).expanduser()
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not raw_path.is_dir():
        print(f"Error: raw data directory not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    image_files = get_image_files(raw_path)
    if not image_files:
        print(f"Error: No image files found in {raw_path}", file=sys.stderr)
        sys.exit(1)

    label_files = get_matching_labels(image_files, raw_path)
    pairs: list[tuple[str, str]] = []
    for img_file in image_files:
        label_file = Path(img_file).stem + ".txt"
        if label_file in label_files:
            pairs.append((img_file, label_file))
    if not pairs:
        print("Error: No image-label pairs found", file=sys.stderr)
        sys.exit(1)

    cleanup_train_yolox_dataset(root, raw_dir)

    total = len(pairs)
    # Ensure val has at least 1 image when total >= 2 so YOLOX evaluation produces non-zero AP
    # (with 70/20/10, total=2 gives val_count=0 -> empty val -> AP 0.00 and 0.00 ms inference).
    val_count = max(1, int(total * 0.2)) if total >= 2 else 0
    test_count = int(total * 0.1)
    train_count = total - val_count - test_count
    if train_count < 1:
        test_count = 0
        train_count = total - val_count
    random.seed(42)
    random.shuffle(pairs)
    train_files = pairs[:train_count]
    val_files = pairs[train_count : train_count + val_count]
    test_files = pairs[train_count + val_count :]

    def _copy_images(file_list: list[tuple[str, str]], dst_dir: Path, split_name: str) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_file, _ in file_list:
            shutil.copy2(raw_path / img_file, dst_dir / img_file)
        print(f"{split_name}: {len(file_list)} images -> {dst_dir.relative_to(root)}")

    _copy_images(train_files, root / "data/train/images", "train")
    _copy_images(val_files, root / "data/val/images", "val")
    _copy_images(test_files, root / "data/test/images", "test")

    ann_dir = root / "data/annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    convert_yolo_to_coco(raw_path, root / "data/train/images", ann_dir / "train_labels.json", class_name)
    convert_yolo_to_coco(raw_path, root / "data/val/images", ann_dir / "val_labels.json", class_name)
    convert_yolo_to_coco(raw_path, root / "data/test/images", ann_dir / "test_labels.json", class_name)

    print("Data preparation done.")
    print(f"- train_yolox root: {root}")
    print(f"- raw dir: {raw_path}")
    print(f"- annotations: {ann_dir}")
    print(f"- expected symlink: YOLOX/datasets/{dataset_name} -> data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLOX training data (train_yolox layout)")
    parser.add_argument("--train-yolox-root", required=True, help="Path to train_yolox root")
    parser.add_argument("--raw-dir", required=True, help="Raw data dir (relative to train_yolox root unless absolute)")
    parser.add_argument("--dataset-name", default="marker_dataset", help="Dataset name for symlink docs")
    parser.add_argument("--class-name", default="circle", help="COCO category name")
    args = parser.parse_args()

    prepare_train_yolox_dataset(
        train_yolox_root=args.train_yolox_root,
        raw_dir=args.raw_dir,
        dataset_name=args.dataset_name,
        class_name=args.class_name,
    )


if __name__ == "__main__":
    main()
