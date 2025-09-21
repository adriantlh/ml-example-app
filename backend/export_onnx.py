#!/usr/bin/env python3
"""
Export YOLOv9 PyTorch model to ONNX format
For deployment in FastAPI inference app
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path
import onnx
import onnxruntime

# Add training directory to path
training_dir = Path(__file__).parent / "training"
sys.path.append(str(training_dir))

def export_to_onnx(weights_path, output_path, img_size=640, batch_size=1,
                   opset_version=11, simplify=True, check=True):
    """
    Export YOLOv9 model to ONNX format

    Args:
        weights_path: Path to trained .pt model
        output_path: Output .onnx file path
        img_size: Input image size
        batch_size: Batch size (1 for single image inference)
        opset_version: ONNX opset version
        simplify: Whether to simplify the model
        check: Whether to check the exported model
    """

    print(f"🚀 Exporting YOLOv9 model to ONNX")
    print(f"Model: {weights_path}")
    print(f"Output: {output_path}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print("=" * 50)

    # Load model
    device = torch.device('cpu')
    model = torch.load(weights_path, map_location=device)

    # Handle different model formats
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'ema' in model:
            model = model['ema']
        else:
            # Assume it's a state dict
            raise ValueError("Cannot extract model from checkpoint. Please check the file format.")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # Dynamic axes for flexible input sizes (optional)
    dynamic_axes = {
        'images': {0: 'batch_size'},  # batch dimension
        'output': {0: 'batch_size'}   # batch dimension
    } if batch_size == 1 else None

    print("📦 Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print("✅ ONNX export successful!")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

    # Simplify model (optional)
    if simplify:
        try:
            import onnxsim
            print("🔧 Simplifying ONNX model...")
            model_onnx = onnx.load(output_path)
            model_onnx, check_ok = onnxsim.simplify(model_onnx)
            onnx.save(model_onnx, output_path)
            print("✅ Model simplified!")
        except ImportError:
            print("⚠️  onnx-simplifier not installed. Skipping simplification.")
            print("   Install with: pip install onnx-simplifier")

    # Verify exported model
    if check:
        print("🔍 Verifying exported model...")
        try:
            # Check ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model structure is valid!")

            # Test inference with ONNX Runtime
            ort_session = onnxruntime.InferenceSession(output_path)

            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]

            print(f"📊 Model Information:")
            print(f"   Input: {input_info.name} {input_info.shape} ({input_info.type})")
            print(f"   Output: {output_info.name} {output_info.shape} ({output_info.type})")

            # Test inference
            test_input = dummy_input.numpy()
            outputs = ort_session.run(None, {input_info.name: test_input})

            print(f"✅ ONNX Runtime inference test passed!")
            print(f"   Output shape: {outputs[0].shape}")

        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return False

    # Print file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"📁 Exported model size: {file_size:.1f} MB")

    return True

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv9 model to ONNX')

    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained .pt model file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output .onnx file path')

    # Model configuration
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1 for single image inference)')

    # Export options
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable model simplification')
    parser.add_argument('--no-check', action='store_true',
                       help='Disable model verification')

    # Device options
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for export (cpu/cuda)')

    args = parser.parse_args()

    # Validate input file
    if not Path(args.weights).exists():
        print(f"❌ Error: Model file '{args.weights}' not found!")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set device
    if args.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Export model
    success = export_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        check=not args.no_check
    )

    if success:
        print("\n🎉 Export completed successfully!")
        print(f"ONNX model saved to: {args.output}")
        print("\n💡 Next steps:")
        print("1. Copy the ONNX model to your FastAPI app's models/ directory")
        print("2. Update the model path in main.py")
        print("3. Update class names in main.py if using custom classes")
        return 0
    else:
        print("\n❌ Export failed!")
        return 1

if __name__ == "__main__":
    exit(main())