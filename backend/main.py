"""
FastAPI Object Detection API using YOLO models with ONNX Runtime.

This application provides REST endpoints for object detection using YOLO models
(YOLOv9 and YOLOX) converted to ONNX format. It supports single image inference,
batch processing, and provides health check endpoints.

The API automatically loads models on startup and provides CORS support for
frontend communication.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import cv2
import numpy as np
from PIL import Image
import io
from typing import List

# Import detector classes from detectors module
from detectors import YOLOv9Detector, YOLOXDetector

# Initialize detector instances with model paths
yolov9_detector = YOLOv9Detector("models/yolov9-mit.onnx")
yolox_detector = YOLOXDetector("models/yolox_s.onnx")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events - load models on startup."""
    # Try to load both models on startup
    yolov9_success = yolov9_detector.load_model()
    if not yolov9_success:
        print("Warning: Could not load YOLOv9 MIT model. Make sure 'models/yolov9-mit.onnx' exists.")

    yolox_success = yolox_detector.load_model()
    if not yolox_success:
        print("Warning: Could not load YOLOX model. Make sure 'models/yolox_s.onnx' exists.")

    if not yolov9_success and not yolox_success:
        print("Warning: No models loaded successfully!")

    yield  # App runs here
    # Cleanup would go here if needed

app = FastAPI(title="YOLOX Object Detection API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint providing API information.

    Returns:
        dict: API status information including name and readiness status
    """
    return {"message": "YOLOv9 Image Inference API", "status": "ready"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and model status.

    Returns:
        dict: Health status containing:
            - status: API health status ("healthy")
            - yolov9_loaded: Boolean indicating if YOLOv9 MIT model is loaded
            - yolox_loaded: Boolean indicating if YOLOX model is loaded
    """
    return {
        "status": "healthy",
        "yolov9_loaded": yolov9_detector.session is not None,
        "yolox_loaded": yolox_detector.session is not None,
        "models_available": ["yolov9-mit", "yolox"]
    }

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    """
    Perform object detection on a single uploaded image using YOLOv9.

    The image is automatically scaled to fit the model's 480x640 input requirement,
    and bounding boxes are scaled back to match the original image dimensions.

    Args:
        file: Image file (JPEG, PNG, etc.) to analyze

    Returns:
        dict: Detection results containing:
            - filename: Original filename of uploaded image
            - detections: List of detected objects, each containing:
                - class_id: Integer ID of detected object class (0-79 for COCO classes)
                - class_name: Human-readable name of detected class (e.g., "person", "car")
                - confidence: Detection confidence score (0.0 to 1.0)
                - bbox: Bounding box coordinates [x1, y1, x2, y2] in original image pixels
            - count: Total number of detections found

    Raises:
        400: If uploaded file is not an image
        500: If image processing fails or model inference error occurs
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run inference with YOLOv9 MIT model
        detections = yolov9_detector.detect(image_cv)

        return JSONResponse(content={
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/infer-yolox")
async def infer_image_yolox(file: UploadFile = File(...)):
    """
    Perform object detection on a single uploaded image using YOLOX.

    The image is automatically scaled to fit the model's input requirement,
    and bounding boxes are scaled back to match the original image dimensions.

    Args:
        file: Image file (JPEG, PNG, etc.) to analyze

    Returns:
        dict: Detection results containing:
            - filename: Original filename of uploaded image
            - detections: List of detected objects, each containing:
                - class_id: Integer ID of detected object class (0-79 for COCO classes)
                - class_name: Human-readable name of detected class (e.g., "person", "car")
                - confidence: Detection confidence score (0.0 to 1.0)
                - bbox: Bounding box coordinates [x1, y1, x2, y2] in original image pixels
            - count: Total number of detections found
            - model: "yolox" to indicate which model was used

    Raises:
        400: If uploaded file is not an image
        500: If image processing fails or model inference error occurs
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run inference with YOLOX model
        detections = yolox_detector.detect(image_cv)

        return JSONResponse(content={
            "filename": file.filename,
            "detections": detections,
            "count": len(detections),
            "model": "yolox"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image with YOLOX: {str(e)}")

@app.post("/infer/batch")
async def infer_batch(files: List[UploadFile] = File(...)):
    """
    Perform object detection on multiple uploaded images using YOLOv9.

    Each image is processed independently with the same scaling and detection
    logic as the single image endpoint.

    Args:
        files: List of image files (JPEG, PNG, etc.) to analyze

    Returns:
        dict: Batch processing results containing:
            - results: List of results for each image, each containing either:
                Success case:
                - filename: Original filename of uploaded image
                - detections: List of detected objects (same format as /infer endpoint)
                - count: Total number of detections found
                Error case:
                - filename: Original filename of uploaded image
                - error: Error message describing what went wrong

    Note:
        - Invalid image files are skipped with error messages
        - Processing continues even if individual images fail
        - Results are returned in the same order as input files
    """
    results = []

    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue

        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            detections = yolov9_detector.detect(image_cv)

            results.append({
                "filename": file.filename,
                "detections": detections,
                "count": len(detections)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Error processing image: {str(e)}"
            })

    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)