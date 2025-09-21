#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLOv9
Helps convert and organize datasets into YOLO format
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import random

def create_yolo_structure(dataset_path):
    """Create YOLO dataset directory structure"""
    dataset_path = Path(dataset_path)

    # Create directories
    dirs = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val",
        dataset_path / "images" / "test",
        dataset_path / "labels" / "train",
        dataset_path / "labels" / "val",
        dataset_path / "labels" / "test"
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.15):
    """Split dataset into train/val/test sets"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Get all image files
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in images_dir.iterdir()
                   if f.suffix.lower() in image_exts]

    # Shuffle for random split
    random.shuffle(image_files)

    # Calculate split indices
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # Create output structure
    create_yolo_structure(output_dir)

    # Copy files to appropriate splits
    for split_name, files in splits.items():
        print(f"\n{split_name.upper()} split: {len(files)} images")

        for img_file in files:
            # Copy image
            dst_img = output_dir / "images" / split_name / img_file.name
            shutil.copy2(img_file, dst_img)

            # Copy corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_dir / "labels" / split_name / label_file.name
                shutil.copy2(label_file, dst_label)
            else:
                print(f"Warning: No label file for {img_file.name}")

    print(f"\nDataset split completed!")
    print(f"Train: {len(splits['train'])} images")
    print(f"Val: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")

def convert_coco_to_yolo(coco_json, images_dir, output_dir, class_mapping=None):
    """Convert COCO format annotations to YOLO format"""
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Create output structure
    create_yolo_structure(output_dir)

    # Build image lookup
    images = {img['id']: img for img in coco_data['images']}

    # Build category mapping
    if class_mapping is None:
        # Use all categories, remap to 0-based index
        categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    else:
        # Use provided mapping
        categories = class_mapping

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Convert annotations
    converted_count = 0
    for img_id, img_info in images.items():
        # Copy image file
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # Determine split (simple heuristic - you might want to improve this)
        if random.random() < 0.8:
            split = 'train'
        elif random.random() < 0.9:
            split = 'val'
        else:
            split = 'test'

        # Copy image
        dst_img = output_dir / "images" / split / img_info['file_name']
        shutil.copy2(img_path, dst_img)

        # Convert annotations
        img_width = img_info['width']
        img_height = img_info['height']

        label_lines = []
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                cat_id = ann['category_id']
                if cat_id not in categories:
                    continue  # Skip unmapped categories

                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                center_x = (x + w/2) / img_width
                center_y = (y + h/2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                yolo_class = categories[cat_id]
                label_lines.append(f"{yolo_class} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Save label file
        label_path = output_dir / "labels" / split / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

        converted_count += 1
        if converted_count % 100 == 0:
            print(f"Converted {converted_count} images...")

    print(f"COCO to YOLO conversion completed! Converted {converted_count} images.")

def validate_dataset(dataset_dir):
    """Validate YOLO dataset structure and format"""
    dataset_dir = Path(dataset_dir)

    print("🔍 Validating dataset structure...")

    # Check directory structure
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False

    # Count files
    for split in ['train', 'val']:
        img_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split

        img_files = list(img_dir.glob("*.*"))
        label_files = list(label_dir.glob("*.txt"))

        print(f"{split.upper()}: {len(img_files)} images, {len(label_files)} labels")

        # Check for missing labels
        missing_labels = 0
        for img_file in img_files:
            label_file = label_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                missing_labels += 1

        if missing_labels > 0:
            print(f"⚠️  {missing_labels} images in {split} missing label files")

    print("✅ Dataset validation completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLOv9 training')
    parser.add_argument('action', choices=['create', 'split', 'coco2yolo', 'validate'],
                       help='Action to perform')

    # Common arguments
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory path')

    # For create action
    parser.add_argument('--name', type=str, default='custom_dataset',
                       help='Dataset name (for create action)')

    # For split action
    parser.add_argument('--images', type=str,
                       help='Input images directory (for split action)')
    parser.add_argument('--labels', type=str,
                       help='Input labels directory (for split action)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')

    # For COCO conversion
    parser.add_argument('--coco-json', type=str,
                       help='COCO annotations JSON file')
    parser.add_argument('--coco-images', type=str,
                       help='COCO images directory')

    args = parser.parse_args()

    if args.action == 'create':
        print(f"Creating YOLO dataset structure: {args.output}")
        create_yolo_structure(args.output)

    elif args.action == 'split':
        if not args.images or not args.labels:
            print("Error: --images and --labels required for split action")
            return 1
        print(f"Splitting dataset from {args.images} to {args.output}")
        split_dataset(args.images, args.labels, args.output,
                     args.train_ratio, args.val_ratio)

    elif args.action == 'coco2yolo':
        if not args.coco_json or not args.coco_images:
            print("Error: --coco-json and --coco-images required for coco2yolo action")
            return 1
        print(f"Converting COCO dataset to YOLO format")
        convert_coco_to_yolo(args.coco_json, args.coco_images, args.output)

    elif args.action == 'validate':
        validate_dataset(args.output)

    return 0

if __name__ == "__main__":
    exit(main())