#!/usr/bin/env python3
"""
Custom YOLOv9 Training Script
Simplified training script with common configurations for custom datasets
"""

import argparse
import os
import sys
from pathlib import Path

# Add training directory to path
training_dir = Path(__file__).parent / "training"
sys.path.append(str(training_dir))

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv9 on custom dataset')

    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset yaml file (e.g., datasets/concrete_cracks.yaml)')

    # Model configuration
    parser.add_argument('--model', type=str, default='yolov9s',
                       choices=['yolov9s', 'yolov9m', 'yolov9c', 'yolov9e', 'gelan-s', 'gelan-m', 'gelan-c', 'gelan-e'],
                       help='Model variant to use')
    parser.add_argument('--weights', type=str, default='',
                       help='Initial weights path (leave empty for random initialization)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (adjust based on GPU memory)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (0,1,2,3 for GPU, cpu for CPU)')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='SGD',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')

    # Data augmentation
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Enable data augmentation')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup augmentation probability')

    # Validation and saving
    parser.add_argument('--val-interval', type=int, default=1,
                       help='Validation interval (epochs)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Model save interval (epochs)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='custom',
                       help='Experiment name')

    # Resume training
    parser.add_argument('--resume', type=str, default='',
                       help='Resume training from checkpoint')

    # Advanced options
    parser.add_argument('--freeze', type=int, default=0,
                       help='Number of layers to freeze (0 = train all)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.data):
        print(f"Error: Dataset file '{args.data}' not found!")
        return

    # Determine weights path
    if not args.weights:
        # Use pre-trained weights from models directory
        weights_map = {
            'yolov9s': 'models/gelan-s2.pt',
            'yolov9m': 'models/yolov9m.pt',
            'yolov9c': 'models/yolov9c.pt',
            'yolov9e': 'models/yolov9e.pt',
            'gelan-s': 'models/gelan-s2.pt',
            'gelan-m': 'models/gelan-m.pt',
            'gelan-c': 'models/gelan-c.pt',
            'gelan-e': 'models/gelan-e.pt',
        }
        args.weights = weights_map.get(args.model, 'models/gelan-s2.pt')

    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"Warning: Weights file '{args.weights}' not found. Training from scratch.")
        args.weights = ''

    # Build training command
    train_script = training_dir / "train.py"
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        return

    cmd = [
        "python", str(train_script),
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
        "--optimizer", args.optimizer,
        "--lr0", str(args.lr),
        "--momentum", str(args.momentum),
        "--weight-decay", str(args.weight_decay),
        "--val", str(args.val_interval),
        "--save-period", str(args.save_interval),
        "--project", args.project,
        "--name", args.name,
        "--workers", str(args.workers),
        "--patience", str(args.patience),
    ]

    # Add optional arguments
    if args.weights:
        cmd.extend(["--weights", args.weights])

    if args.device:
        cmd.extend(["--device", args.device])

    if args.resume:
        cmd.extend(["--resume", args.resume])

    if args.freeze > 0:
        cmd.extend(["--freeze", str(args.freeze)])

    if not args.augment:
        cmd.append("--no-augment")

    cmd.extend(["--mosaic", str(args.mosaic)])
    cmd.extend(["--mixup", str(args.mixup)])

    # Print configuration
    print("🚀 Starting YOLOv9 Training")
    print("=" * 50)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print("=" * 50)

    # Execute training
    import subprocess
    try:
        subprocess.run(cmd, cwd=training_dir, check=True)
        print("\n✅ Training completed successfully!")
        print(f"Results saved in: {args.project}/{args.name}")
        print("Best model: weights/best.pt")
        print("Last model: weights/last.pt")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())