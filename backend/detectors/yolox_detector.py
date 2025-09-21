"""
YOLOX Object Detection Module

This module contains the YOLOXDetector class for object detection using
YOLOX models in ONNX format with CPU inference.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any


class YOLOXDetector:
    """
    YOLOX Object Detection class that handles ONNX model inference.

    This class provides a complete pipeline for object detection using YOLOX models
    in ONNX format, including preprocessing, inference, and postprocessing.

    Attributes:
        model_path (str): Path to the ONNX model file
        session (ort.InferenceSession): ONNX Runtime inference session
        input_size (int): Model input size in pixels (640x640 for YOLOX)
        conf_threshold (float): Confidence threshold for detections (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.45)
        class_names (list): COCO dataset class names (80 classes)
    """

    def __init__(self, model_path: str):
        """
        Initialize the YOLOXDetector with model path and default parameters.

        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_width = 640   # YOLOX uses 640x640 input
        self.input_height = 640
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
            print(f"Error loading YOLOX model: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for YOLOX model inference.

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

        # Resize image to model input dimensions
        resized_image = cv2.resize(image, (self.input_width, self.input_height))

        # Convert BGR to RGB and normalize pixel values to [0, 1]
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension: (H, W, C) -> (1, C, H, W)
        processed_image = np.transpose(resized_image, (2, 0, 1))
        processed_image = np.expand_dims(processed_image, axis=0)

        return processed_image, scale_x, scale_y

    def nms(self, boxes, scores, score_threshold, nms_threshold):
        """
        Apply Non-Maximum Suppression to filter overlapping boxes.

        Args:
            boxes (np.ndarray): Bounding boxes [N, 4] format [x1, y1, x2, y2]
            scores (np.ndarray): Confidence scores [N]
            score_threshold (float): Minimum score threshold
            nms_threshold (float): IoU threshold for NMS

        Returns:
            np.ndarray: Indices of boxes to keep
        """
        # Filter by score threshold
        valid_indices = scores > score_threshold
        if not np.any(valid_indices):
            return np.array([])

        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        original_indices = np.where(valid_indices)[0]

        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]

        keep = []
        while len(sorted_indices) > 0:
            # Keep the box with highest score
            current = sorted_indices[0]
            keep.append(original_indices[current])

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]

            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            # Calculate areas
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                            (remaining_boxes[:, 3] - remaining_boxes[:, 1])

            # Calculate IoU
            union = current_area + remaining_areas - intersection
            iou = intersection / (union + 1e-6)

            # Keep boxes with IoU below threshold
            keep_mask = iou < nms_threshold
            sorted_indices = sorted_indices[1:][keep_mask]

        return np.array(keep)

    def postprocess_detections(self, outputs: np.ndarray, scale_x: float, scale_y: float,
                             original_width: int, original_height: int) -> List[Dict[str, Any]]:
        """
        Post-process YOLOX model outputs to extract final detections.

        YOLOX output format: [1, 8400, 85] where 85 = [4 bbox coords + 1 objectness + 80 classes]

        Args:
            outputs (np.ndarray): Raw model outputs [1, 8400, 85]
            scale_x (float): Horizontal scaling factor from preprocessing
            scale_y (float): Vertical scaling factor from preprocessing
            original_width (int): Original image width
            original_height (int): Original image height

        Returns:
            List[Dict[str, Any]]: List of detections with class_id, class_name, confidence, and bbox
        """
        detections = []

        # YOLOX output shape: [1, 8400, 85]
        # 85 = [cx, cy, w, h, obj_conf, class_0_conf, class_1_conf, ..., class_79_conf]
        predictions = outputs[0]  # Remove batch dimension: [8400, 85]

        # Extract box coordinates and confidence
        boxes = predictions[:, :4]  # [cx, cy, w, h]
        obj_conf = predictions[:, 4]  # objectness confidence
        class_confs = predictions[:, 5:]  # class confidences [8400, 80]

        # Convert center format to corner format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        corner_boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Get class predictions
        class_ids = np.argmax(class_confs, axis=1)
        class_scores = np.max(class_confs, axis=1)

        # Try using just class confidence instead of objectness * class
        # Some YOLOX implementations don't multiply by objectness
        final_scores = class_scores  # Try without objectness multiplication


        # Apply NMS
        keep_indices = self.nms(corner_boxes, final_scores, self.conf_threshold, self.iou_threshold)

        for idx in keep_indices:
            x1, y1, x2, y2 = corner_boxes[idx]
            confidence = final_scores[idx]
            class_id = int(class_ids[idx])

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

            # Get class name
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
            raise ValueError("YOLOX model not loaded. Call load_model() first.")

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