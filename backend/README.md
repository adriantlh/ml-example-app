# FastAPI YOLOv9 Image Inference App

A FastAPI application that provides image inference using YOLOv9 ONNX model for CPU-based object detection. This application also supports custom model training and dataset preparation.

## Setup

### 1. Virtual Environment
The project uses Python 3.12 with Poetry for dependency management:

```bash
cd backend
# Create virtual environment if it doesn't exist
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Dependencies are managed with Poetry:

```bash
# Install all dependencies including training dependencies
poetry install
```

Main dependencies include:
- FastAPI
- Uvicorn
- ONNX Runtime (CPU)
- OpenCV
- Pillow
- NumPy
- PyTorch (for training)
- Additional training utilities

### 3. Get YOLOv9 ONNX Model

You need to download a YOLOv9 ONNX model and place it in the `models/` directory as `yolov9.onnx`.

**Option 1: Download pre-converted ONNX model**
```bash
# Example URLs (you'll need to find actual YOLOv9 ONNX models)
wget -O models/yolov9.onnx [URL_TO_YOLOV9_ONNX_MODEL]
```

**Option 2: Convert PyTorch model to ONNX**
If you have a YOLOv9 PyTorch model (`.pt` file), you can convert it:

```python
import torch
import onnx

# Load your YOLOv9 PyTorch model
model = torch.load('yolov9.pt', map_location='cpu')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/yolov9.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={
        'images': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Running the Application

### Start the FastAPI server:
```bash
cd backend
source .venv/bin/activate
python main.py
```

Or using uvicorn directly:
```bash
cd backend
source .venv/bin/activate
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Health Check
- **GET** `/` - Basic API info
- **GET** `/health` - Health check and model status

### 2. Image Inference
- **POST** `/infer` - Single image inference
  - Upload an image file
  - Returns detected objects with bounding boxes, confidence scores, and class names

- **POST** `/infer/batch` - Batch image inference
  - Upload multiple image files
  - Returns detection results for all images

### Example Usage

#### Single Image Inference
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

#### Response Format
```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox": [100.0, 150.0, 300.0, 400.0]
    }
  ],
  "count": 1
}
```

## Interactive API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Features

- **CPU-only inference** using ONNX Runtime
- **Single and batch image processing**
- **COCO dataset class names** (80 classes)
- **Configurable confidence and IoU thresholds**
- **Proper image preprocessing** (resize, pad, normalize)
- **Bounding box coordinate conversion**
- **FastAPI with automatic API documentation**

## Model Configuration

You can modify the detector parameters in `main.py`:

```python
self.conf_threshold = 0.25  # Confidence threshold
self.iou_threshold = 0.45   # IoU threshold for NMS
self.input_size = 640       # Model input size
```

## Custom Model Training

This application includes tools for training custom YOLOv9 models on your own datasets.

### Training Files

- `prepare_dataset.py` - Prepares and splits datasets for training
- `train_custom.py` - Custom training script for YOLOv9
- `export_onnx.py` - Exports trained PyTorch models to ONNX format
- `TRAINING_GUIDE.md` - Detailed training guide and instructions

### Quick Training Setup

1. **Prepare your dataset:**
   ```bash
   cd backend
   python prepare_dataset.py --help
   ```

2. **Train a custom model:**
   ```bash
   python train_custom.py --help
   ```

3. **Export to ONNX:**
   ```bash
   python export_onnx.py --model path/to/your/model.pt
   ```

For detailed training instructions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── prepare_dataset.py   # Dataset preparation script
├── train_custom.py      # Training script
├── export_onnx.py       # ONNX export script
├── models/              # Directory for ONNX models
├── training/            # Training data and configurations
├── pyproject.toml       # Poetry dependencies
└── TRAINING_GUIDE.md    # Detailed training instructions
```

## License

This project uses YOLOv9 which is released under the MIT License.