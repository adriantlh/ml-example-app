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
import onnxruntime as ort
from PIL import Image
import io
from typing import List, Dict, Any


class YOLOv9Detector:
    """
    YOLO Object Detection class that handles ONNX model inference.

    This class provides a complete pipeline for object detection using YOLO models
    in ONNX format, including preprocessing, inference, and postprocessing.

    Attributes:
        model_path (str): Path to the ONNX model file
        session (ort.InferenceSession): ONNX Runtime inference session
        input_height (int): Model input height in pixels (default: 480)
        input_width (int): Model input width in pixels (default: 640)
        conf_threshold (float): Confidence threshold for detections (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.45)
        class_names (list): COCO dataset class names (80 classes)
    """

    def __init__(self, model_path: str):
        """
        Initialize the YOLOv9Detector with model path and default parameters.

        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_height = 480
        self.input_width = 640
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        # COCO dataset class names (80 classes total)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def load_model(self):
        """
        Load the ONNX model using ONNX Runtime.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for YOLO model inference.

        Resizes image to model input dimensions, converts BGR to RGB,
        normalizes pixel values, and formats for ONNX inference.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C)

        Returns:
            tuple: Processed image array (1, C, H, W), scale_x, scale_y
        """
        # Get original image dimensions
        original_height, original_width = image.shape[:2]

        # Calculate scaling factors for width and height separately
        scale_x = self.input_width / original_width
        scale_y = self.input_height / original_height

        # Resize image to model input dimensions (no padding needed)
        resized_image = cv2.resize(image, (self.input_width, self.input_height))

        # Convert BGR to RGB and normalize pixel values to [0, 1]
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension: (H, W, C) -> (1, C, H, W)
        processed_image = np.transpose(resized_image, (2, 0, 1))
        processed_image = np.expand_dims(processed_image, axis=0)

        return processed_image, scale_x, scale_y

    def postprocess_detections(self, outputs: np.ndarray, scale_x: float, scale_y: float,
                             original_width: int, original_height: int) -> List[Dict[str, Any]]:
        """
        Post-process YOLO model outputs to extract final detections.

        Converts model outputs to bounding boxes in original image coordinates,
        filters by confidence threshold, and formats results.

        Args:
            outputs (np.ndarray): Raw model outputs [N, 6] where 6 = [x1, y1, x2, y2, conf, class_id]
            scale_x (float): Horizontal scaling factor from preprocessing
            scale_y (float): Vertical scaling factor from preprocessing
            original_width (int): Original image width
            original_height (int): Original image height

        Returns:
            List[Dict[str, Any]]: List of detections with class_id, class_name, confidence, and bbox
        """
        detections = []

        # Process each detection from model output

        # Model outputs format: [x1, y1, x2, y2, confidence, class_id]
        for detection in outputs:
            x1, y1, x2, y2, confidence, class_id = detection

            # Filter out low-confidence detections
            if confidence < self.conf_threshold:
                continue

            # Convert from model coordinates back to original image coordinates
            x1 = x1 / scale_x
            y1 = y1 / scale_y
            x2 = x2 / scale_x
            y2 = y2 / scale_y

            # Clamp coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width, x2)
            y2 = min(original_height, y2)

            # Get class name (ensure class_id is within valid range)
            class_id = int(class_id)
            if 0 <= class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"unknown_{class_id}"

            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(confidence),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

        return detections

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run complete object detection pipeline on an image.

        Preprocesses the image, runs ONNX model inference, and postprocesses
        the results to return final detections.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C)

        Returns:
            List[Dict[str, Any]]: List of detections with class_id, class_name, confidence, and bbox

        Raises:
            ValueError: If model is not loaded
        """
        if self.session is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Get original image dimensions for coordinate scaling
        original_height, original_width = image.shape[:2]

        # Preprocess image for model input
        processed_image, scale_x, scale_y = self.preprocess_image(image)

        # Run ONNX model inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: processed_image})


        # Post-process model outputs to get final detections
        detections = self.postprocess_detections(
            outputs[0], scale_x, scale_y, original_width, original_height
        )

        return detections

# Initialize detector instances with model paths
yolov9_detector = YOLOv9Detector("models/yolov9-mit.onnx")
yolox_detector = YOLOv9Detector("models/yolox_s.onnx")

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