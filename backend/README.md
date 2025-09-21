# FastAPI YOLOv9 Image Inference App

A FastAPI application that provides image inference using YOLOv9 ONNX model for CPU-based object detection. This application also supports custom model training and dataset preparation.

## Setup

### 1. Virtual Environment Setup
The project uses Python 3.12 with Poetry for dependency management. Create the virtual environment from the project root:

```bash
# From project root directory
python3.12 -m venv .venv
source .venv/bin/activate
pip install poetry
```

### 2. Install Dependencies
Navigate to the backend directory and install dependencies with Poetry:

```bash
cd backend
poetry install
```

This installs all required dependencies including:
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running the FastAPI application
- **ONNX Runtime (CPU)**: Optimized inference engine for ONNX models
- **OpenCV**: Computer vision library for image processing
- **Pillow**: Python imaging library for image manipulation
- **NumPy**: Numerical computing library for array operations
- **PyTorch**: Machine learning framework (for training and model conversion)
- **Additional training utilities**: For custom model training workflows

### 3. Download Models

From the project root directory, use the provided script to download the required YOLOv9 ONNX model:

```bash
# From project root directory
./download_models.sh
```

This script automatically downloads and places the YOLOv9 ONNX model in the correct location (`backend/models/yolov9.onnx`).

**Manual Model Setup (Alternative)**
If you prefer to set up models manually or use custom models:

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
uvicorn main:app
```

**Alternative Options:**
```bash
# With reload for development
uvicorn main:app --reload

# Custom host and port
uvicorn main:app --host 0.0.0.0 --port 8000

# Using poetry run
poetry run uvicorn main:app
```

The API will be available at: `http://localhost:8000`

## Technical Architecture

### Core Components

**FastAPI Application (`main.py`)**
- RESTful API with automatic OpenAPI documentation
- Asynchronous request handling with uvicorn ASGI server
- Multipart file upload support for image processing
- CORS enabled for frontend integration

**YOLOv9 ONNX Inference Engine**
- CPU-optimized inference using ONNX Runtime
- Efficient memory management for batch processing
- Image preprocessing pipeline (resize, normalize, format conversion)
- Post-processing with Non-Maximum Suppression (NMS)

**Image Processing Pipeline**
1. **Input Validation**: File format and size validation
2. **Preprocessing**: Resize to model input dimensions (640x640), normalize pixel values (0-1), convert BGR→RGB
3. **Inference**: Forward pass through ONNX model
4. **Post-processing**: Apply confidence thresholds, NMS filtering, coordinate conversion
5. **Output Formatting**: JSON response with bounding boxes, class names, confidence scores

### Performance Considerations

- **Memory Efficiency**: Streaming file uploads to handle large images
- **CPU Optimization**: ONNX Runtime with CPU optimizations
- **Batch Processing**: Support for multiple image inference
- **Caching**: Model loaded once at startup for faster inference

### Model Specifications

- **Input Shape**: `[1, 3, 640, 640]` (batch_size, channels, height, width)
- **Input Type**: `float32` normalized to [0, 1]
- **Output Format**: Detection tensors with class probabilities and bounding box coordinates
- **Classes**: 80 COCO dataset classes (person, car, dog, etc.)

### Inference Route Workflow

```
POST /infer - Single Image Object Detection Workflow
═══════════════════════════════════════════════════════

┌─────────────────┐
│   Client        │
│   (Frontend)    │ ──── HTTP POST with image file ────┐
└─────────────────┘                                    │
                                                       ▼
┌────────────────────────────────────────────────────────────────┐
│                    FastAPI Server                             │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │              1. File Upload Handler                      │   │
│ │  • Validate file format (JPEG, PNG, etc.)              │   │
│ │  • Check file size limits                              │   │
│ │  • Read image bytes from multipart form               │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                ▼                               │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │              2. Image Preprocessing                      │   │
│ │  • Load image with OpenCV (BGR format)                 │   │
│ │  • Store original dimensions: (orig_h, orig_w)         │   │
│ │  • Resize to model input: 640x640                      │   │
│ │  • Convert BGR → RGB                                   │   │
│ │  • Normalize pixel values: [0-255] → [0-1]            │   │
│ │  • Add batch dimension: (3,640,640) → (1,3,640,640)   │   │
│ │  • Convert to float32 numpy array                      │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                ▼                               │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │              3. ONNX Model Inference                     │   │
│ │  • Create ONNX input dict: {'images': preprocessed}    │   │
│ │  • Run inference: session.run(None, input_dict)        │   │
│ │  • Get raw output tensor: [num_detections, 6]          │   │
│ │    Format: [x1, y1, x2, y2, confidence, class_id]     │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                ▼                               │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │              4. Post-Processing                          │   │
│ │  • Filter by confidence threshold (> 0.25)             │   │
│ │  • Apply Non-Maximum Suppression (IoU < 0.45)          │   │
│ │  • Scale coordinates back to original image size:       │   │
│ │    - scale_x = orig_w / 640                            │   │
│ │    - scale_y = orig_h / 640                            │   │
│ │    - bbox = [x1*scale_x, y1*scale_y, x2*scale_x, ...]  │   │
│ │  • Map class_id to COCO class names                    │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                ▼                               │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │              5. Response Formatting                      │   │
│ │  • Create JSON response:                               │   │
│ │    {                                                   │   │
│ │      "filename": "image.jpg",                          │   │
│ │      "detections": [                                   │   │
│ │        {                                               │   │
│ │          "class_id": 0,                               │   │
│ │          "class_name": "person",                      │   │
│ │          "confidence": 0.85,                          │   │
│ │          "bbox": [100.0, 150.0, 300.0, 400.0]        │   │
│ │        }                                               │   │
│ │      ],                                                │   │
│ │      "count": 1                                        │   │
│ │    }                                                   │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                                ▼
┌─────────────────┐
│   Client        │ ◄──── HTTP 200 OK with JSON response ────────┘
│   (Frontend)    │
└─────────────────┘

Error Handling:
• File validation errors → HTTP 400 Bad Request
• Model inference errors → HTTP 500 Internal Server Error
• Missing model file → HTTP 500 Internal Server Error
```

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