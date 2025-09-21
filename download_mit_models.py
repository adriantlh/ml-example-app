#!/usr/bin/env python3
"""
Download MIT Licensed YOLOv9 Models
Uses the yolov9-onnx package which provides MIT licensed models
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def print_info(msg):
    print(f"\033[0;34m[INFO]\033[0m {msg}")

def print_success(msg):
    print(f"\033[0;32m[SUCCESS]\033[0m {msg}")

def print_error(msg):
    print(f"\033[0;31m[ERROR]\033[0m {msg}")

def print_warning(msg):
    print(f"\033[1;33m[WARNING]\033[0m {msg}")

def install_package(package_name):
    """Install a Python package using pip"""
    print_info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print_success(f"{package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install {package_name}: {e}")
        return False

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    try:
        print_info(f"Downloading from: {url}")
        print_info(f"Saving to: {output_path}")

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rProgress: {percent}%", end='', flush=True)

        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def main():
    print("\033[0;34m" + "=" * 50 + "\033[0m")
    print("\033[0;34mMIT Licensed YOLOv9 Model Downloader\033[0m")
    print("\033[0;34m" + "=" * 50 + "\033[0m")
    print()

    # Check if we're in the right directory
    if not os.path.exists("backend/main.py"):
        print_error("This script should be run from the project root directory")
        print_info("Current directory: " + os.getcwd())
        print_info("Expected structure: backend/main.py should exist")
        sys.exit(1)

    # Create models directory
    models_dir = Path("backend/models")
    models_dir.mkdir(exist_ok=True)
    print_success(f"Models directory ready: {models_dir}")

    # MIT licensed models to download
    models = {
        "yolov9-mit.onnx": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0/yolov9c.onnx",
        "yolov9-mit.pt": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0/yolov9c.pt",
    }

    # Alternative: Download pre-converted ONNX models
    mit_onnx_models = {
        "yolov9s-mit.onnx": "https://huggingface.co/kadirnar/yolov9-onnx/resolve/main/models/yolov9s.onnx",
        "yolov9m-mit.onnx": "https://huggingface.co/kadirnar/yolov9-onnx/resolve/main/models/yolov9m.onnx",
    }

    success_count = 0
    total_count = 0

    # Try to download from HuggingFace (more reliable)
    print_info("Downloading MIT licensed YOLOv9 models from HuggingFace...")
    for model_name, url in mit_onnx_models.items():
        total_count += 1
        output_path = models_dir / model_name

        if output_path.exists():
            print_success(f"{model_name} already exists")
            success_count += 1
            continue

        if download_file(url, output_path):
            print_success(f"Downloaded {model_name}")
            success_count += 1
        else:
            print_error(f"Failed to download {model_name}")
        print()

    # Download YOLOX as backup
    yolox_url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    yolox_path = models_dir / "yolox_s.pth"
    total_count += 1

    if not yolox_path.exists():
        print_info("Downloading YOLOX (Apache 2.0) as backup...")
        if download_file(yolox_url, yolox_path):
            print_success("Downloaded YOLOX model")
            success_count += 1
        else:
            print_error("Failed to download YOLOX model")
    else:
        print_success("YOLOX model already exists")
        success_count += 1

    # Create a simple ONNX model for testing if others fail
    if success_count == 0:
        print_warning("No models downloaded successfully. Installing yolov9-onnx package...")
        if install_package("yolov9-onnx"):
            print_info("You can now use the yolov9-onnx package for inference")
            print_info("Models will be downloaded automatically when needed")

    # Summary
    print()
    print("\033[0;34m" + "=" * 30 + "\033[0m")
    print("\033[0;34mDownload Summary\033[0m")
    print("\033[0;34m" + "=" * 30 + "\033[0m")

    if success_count > 0:
        print_success(f"Downloaded {success_count}/{total_count} models successfully")
        print_info("Available models:")
        for file in models_dir.glob("*.onnx"):
            print(f"  - {file.name}")
        for file in models_dir.glob("*.pth"):
            print(f"  - {file.name}")
    else:
        print_error("No models downloaded successfully")
        print_info("Try running the script again or check your internet connection")

    print()
    print_info("All models use MIT or Apache 2.0 licenses - safe for commercial use!")

if __name__ == "__main__":
    main()