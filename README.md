# ML Example App

A full-stack machine learning application featuring YOLOv9 object detection with a FastAPI backend and React frontend.

## 🌟 Features

- **Real-time Object Detection**: YOLOv9 ONNX model for accurate object detection
- **Smart Image Scaling**: Automatic scaling to model requirements with coordinate mapping
- **RESTful API**: FastAPI backend with comprehensive Swagger documentation
- **React Frontend**: Modern UI for image upload and result visualization
- **Batch Processing**: Support for multiple image inference
- **80 COCO Classes**: Detects persons, vehicles, animals, furniture, and everyday objects

## 🏗️ Architecture

```
ml-example-app/
├── backend/           # FastAPI server
│   ├── main.py       # API routes and YOLOv9 inference
│   ├── models/       # ONNX model files
│   └── tests/        # Test images
├── frontend/         # React application
│   ├── src/
│   │   ├── components/
│   │   └── ...
└── .venv/           # Python virtual environment
```

## 🚀 Quick Start (Standalone Localhost for Homelabs)

### Prerequisites

- Python 3.12
- Node.js 16+
- Poetry

### Setup Instructions

1. **Create Python virtual environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install poetry
   ```

2. **Install Python dependencies**:
   ```bash
   cd backend
   poetry install
   cd ..
   ```

3. **Download models**:
   ```bash
   ./download_models.sh
   ```

4. **Start the backend server**:
   ```bash
   cd backend
   uvicorn main:app
   ```

   The API will be available at:
   - **API**: http://localhost:8000
   - **Swagger UI**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

5. **Start the frontend**:
   ```bash
   cd ../frontend
   npm run start-standalone
   ```

   The frontend will be available at: http://localhost:3000

## 📡 API Documentation

### Endpoints

#### `GET /health`
Check API and model status
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `POST /infer`
Single image object detection
- **Input**: Image file (JPEG, PNG, etc.)
- **Output**: Detection results with bounding boxes

Example response:
```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox": [100, 50, 200, 300]
    }
  ],
  "count": 1
}
```

#### `POST /infer/batch`
Multiple image processing
- **Input**: Array of image files
- **Output**: Array of detection results

## 🔧 Model Requirements

The YOLOv9 ONNX model must have:
- **Input shape**: `[1, 3, 480, 640]` (batch, channels, height, width)
- **Input type**: `float32` (normalized 0-1)
- **Output format**: `[num_detections, 6]` where each detection is `[x1, y1, x2, y2, confidence, class_id]`

## 🖼️ Image Processing

### Automatic Scaling
The backend automatically handles images of any size:

1. **Input Scaling**: Images are scaled independently in X and Y directions to fit 480×640
2. **Model Inference**: Processes the scaled image
3. **Output Scaling**: Bounding boxes are scaled back to original image coordinates

### Coordinate System
- Input: Original image coordinates
- Processing: Model coordinates (480×640)
- Output: Original image coordinates

## 🧪 Testing

Test the API with the provided sample image:
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@backend/tests/download.jpeg"
```

## 📁 COCO Classes

The model detects 80 COCO classes including:
- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: backpack, umbrella, handbag, tie, suitcase, frisbee, skis, sports ball
- **Furniture**: chair, couch, bed, dining table, toilet
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave
- And many more...

## 🛠️ Development

### Backend Development
- FastAPI with automatic OpenAPI documentation
- ONNX Runtime for efficient inference
- OpenCV for image processing
- Error handling and validation

### Frontend Development
- React with TypeScript
- Component-based architecture
- Image upload and preview
- Detection result visualization

## 🔍 Debugging

### Check Model Loading
```bash
curl http://localhost:8000/health
```

### View API Documentation
Visit http://localhost:8000/docs for interactive Swagger UI

### Backend Logs
The server prints model input/output shapes and processing information

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the Swagger documentation at `/docs`
2. Verify model file placement and format
3. Check backend logs for error details
4. Ensure all dependencies are installed correctly