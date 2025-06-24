import argparse
import os
import random
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/validation folders")
    parser.add_argument('--datapath', required=True, help='Path to dataset root (contains images/ and labels/)')
    parser.add_argument('--train_pct', type=float, default=0.8, help='Proportion of data to use for training (0.01 - 0.99)')
    return parser.parse_args()

def create_dirs(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def split_dataset(images, train_pct):
    total = len(images)
    train_count = int(total * train_pct)
    random.shuffle(images)
    return images[:train_count], images[train_count:]

def copy_files(image_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_path in image_list:
        base_name = img_path.stem
        label_path = src_lbl_dir / f"{base_name}.txt"

        shutil.copy2(img_path, dst_img_dir / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)

def main():
    args = parse_args()

    root = Path(args.datapath).resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Error: 'images/' or 'labels/' folders not found in {root}")
        return

    if not (0.01 <= args.train_pct <= 0.99):
        print("❌ Error: --train_pct must be between 0.01 and 0.99")
        return

    image_paths = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png"))
    if not image_paths:
        print("❌ No image files found in 'images/' folder.")
        return

    train_imgs, val_imgs = split_dataset(image_paths, args.train_pct)

    # Final structure under ./data/
    output_base = root 
    train_img_dir = output_base / "train" / "images"
    train_lbl_dir = output_base / "train" / "labels"
    val_img_dir   = output_base / "validation" / "images"
    val_lbl_dir   = output_base / "validation" / "labels"

    create_dirs(train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir)

    print(f"📦 Copying {len(train_imgs)} images to data/train/...")
    copy_files(train_imgs, images_dir, labels_dir, train_img_dir, train_lbl_dir)

    print(f"📦 Copying {len(val_imgs)} images to data/validation/...")
    copy_files(val_imgs, images_dir, labels_dir, val_img_dir, val_lbl_dir)

    print("✅ Dataset split complete.")

if __name__ == "__main__":
    main()
