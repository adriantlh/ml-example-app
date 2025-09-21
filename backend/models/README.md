# YOLOv9 Models Directory (MIT Licensed)

This directory contains MIT/Apache 2.0 licensed machine learning models for commercial-friendly object detection.

## Required Models (MIT/Apache 2.0 Licensed)

The application uses the following commercially-friendly model files:

1. **yolov9-mit.onnx** (ONNX format, MIT License)
   - Pre-trained YOLOv9 model in ONNX format for inference
   - MIT Licensed version from MultimediaTechLab/YOLO
   - Used by the FastAPI backend for object detection

2. **yolov9-mit.pt** (PyTorch format, MIT License)
   - YOLOv9 model in PyTorch format
   - Alternative model option for different inference frameworks

3. **yolox_s.pth** (PyTorch format, Apache 2.0 License)
   - YOLOX-S model as an alternative to YOLOv9
   - Apache 2.0 licensed for commercial use

## Downloading Models

### Automatic Download (Recommended)

Run the download script from the project root directory:

```bash
# From the project root directory
./download_models.sh
```

The script will:
- Check if models already exist
- Download missing models from official sources
- Verify file integrity
- Show download progress and summary

### Manual Download

If the automatic download fails, you can manually download the models:

#### YOLOv9 ONNX Model
- **Primary source**: [YOLOv9 ONNX Repository](https://github.com/danielsyahputra/yolov9-onnx/releases)
- **Alternative**: Convert from PyTorch using the official YOLOv9 repository

#### GELAN Models
- **Official source**: [YOLOv9 Official Repository](https://github.com/WongKinYiu/yolov9/releases)
- Download `gelan-s2.pt` from the releases page

### Converting PyTorch to ONNX (Advanced)

If you need to convert PyTorch models to ONNX format:

```python
import torch
from yolov9 import YOLOv9  # Assuming you have the YOLOv9 implementation

# Load PyTorch model
model = YOLOv9('gelan-s2.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640)
```

## Model Information

### YOLOv9 ONNX Model Details
- **Input Shape**: [1, 3, 480, 640] (batch_size, channels, height, width)
- **Input Type**: float32
- **Preprocessing**:
  - Resize to 480x640
  - Normalize to [0, 1]
  - Convert BGR to RGB
  - Transpose to CHW format

### COCO Classes
The models are trained on COCO dataset with 80 object classes:
- person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
- traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat
- dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack
- umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball
- kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple
- sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- chair, couch, potted plant, bed, dining table, toilet, tv, laptop
- mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink
- refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## File Structure

```
backend/models/
├── README.md           # This file
├── yolov9.onnx        # ONNX model for inference (excluded from git)
└── gelan-s2.pt        # PyTorch model (excluded from git)
```

## Git Ignore

Model files are automatically excluded from git commits due to their large size. The `.gitignore` file contains:

```
# ML Models - exclude large model files
backend/models/*.onnx
backend/models/*.pt
backend/models/*.pth
backend/models/*.h5
backend/models/*.pkl
backend/models/*.joblib
```

## Troubleshooting

### Download Issues
- Ensure you have `curl` or `wget` installed
- Check your internet connection
- Try running the download script with verbose output
- Manually download from the provided URLs if automatic download fails

### Model Loading Issues
- Verify file integrity (check file sizes)
- Ensure ONNX Runtime is installed: `pip install onnxruntime`
- Check that the model file is not corrupted

### Performance Issues
- For faster inference, consider using `onnxruntime-gpu` if you have a compatible GPU
- Adjust batch size and input resolution based on your hardware capabilities

## License

The models are subject to their respective licenses:
- YOLOv9: [Official YOLOv9 License](https://github.com/WongKinYiu/yolov9)
- ONNX models may have different licensing terms

Please check the original repositories for detailed license information.