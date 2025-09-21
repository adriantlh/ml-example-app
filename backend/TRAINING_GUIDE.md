# YOLOv9 Training Guide

Complete guide for training custom YOLOv9 models and deploying them in your FastAPI app.

## 📁 Project Structure

```
ml-example-app/
├── main.py                    # FastAPI inference app
├── models/                    # Trained models
│   ├── yolov9.onnx           # Current ONNX model
│   └── gelan-s2.pt           # PyTorch base model
├── training/                  # YOLOv9 repository (cloned)
├── train_custom.py            # Custom training script
├── prepare_dataset.py         # Dataset preparation tools
├── export_onnx.py            # PyTorch → ONNX conversion
└── training/datasets/         # Dataset configurations
    ├── concrete_cracks.yaml
    ├── custom_template.yaml
    └── coco_subset.yaml
```

## 🚀 Quick Start

### 1. Prepare Your Dataset

#### Option A: Create YOLO Dataset Structure
```bash
python prepare_dataset.py create --output datasets/my_dataset
```

#### Option B: Split Existing Dataset
```bash
python prepare_dataset.py split \
    --images /path/to/images \
    --labels /path/to/labels \
    --output datasets/my_dataset \
    --train-ratio 0.8 --val-ratio 0.15
```

#### Option C: Convert COCO to YOLO
```bash
python prepare_dataset.py coco2yolo \
    --coco-json annotations/instances_train2017.json \
    --coco-images train2017/ \
    --output datasets/coco_converted
```

### 2. Create Dataset Configuration

Copy and modify a template:
```bash
cp training/datasets/custom_template.yaml training/datasets/my_dataset.yaml
```

Edit `my_dataset.yaml`:
```yaml
path: datasets/my_dataset
train: images/train
val: images/val

nc: 3  # Number of classes
names:
  0: class_1
  1: class_2
  2: class_3
```

### 3. Train the Model

#### Basic Training
```bash
python train_custom.py --data training/datasets/my_dataset.yaml
```

#### Advanced Training
```bash
python train_custom.py \
    --data training/datasets/concrete_cracks.yaml \
    --model gelan-s \
    --epochs 200 \
    --batch-size 16 \
    --img-size 640 \
    --lr 0.01 \
    --name concrete_detection
```

### 4. Export to ONNX

```bash
python export_onnx.py \
    --weights runs/train/concrete_detection/weights/best.pt \
    --output models/my_custom_model.onnx \
    --img-size 640
```

### 5. Update FastAPI App

Update `main.py`:
```python
# Change model path
detector = YOLOv9Detector("models/my_custom_model.onnx")

# Update class names
self.class_names = [
    'class_1',
    'class_2',
    'class_3'
]
```

## 📊 Training Options

### Model Sizes
- `gelan-s`: Smallest, fastest (14MB)
- `gelan-m`: Medium (38MB)
- `gelan-c`: Large (49MB)
- `gelan-e`: Largest, most accurate (112MB)

### Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--epochs` | Training epochs | 100 | 50-300 |
| `--batch-size` | Batch size | 16 | 8-32 |
| `--img-size` | Input size | 640 | 320-1280 |
| `--lr` | Learning rate | 0.01 | 0.001-0.1 |
| `--optimizer` | Optimizer | SGD | SGD/Adam/AdamW |

### Data Augmentation
```bash
python train_custom.py \
    --data my_dataset.yaml \
    --mosaic 1.0 \      # Mosaic augmentation
    --mixup 0.1 \       # Mixup augmentation
    --augment           # Enable augmentation
```

## 🗂️ Dataset Format

### YOLO Label Format
Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```

All coordinates are normalized (0-1):
```
0 0.5 0.3 0.2 0.4
1 0.7 0.8 0.15 0.25
```

### Directory Structure
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

## 🔄 Complete Workflow Examples

### Example 1: Concrete Crack Detection

1. **Prepare dataset**:
```bash
python prepare_dataset.py create --output datasets/concrete_cracks
# Manually organize your crack images and labels
```

2. **Configure dataset** (`training/datasets/concrete_cracks.yaml`):
```yaml
path: datasets/concrete_cracks
train: images/train
val: images/val

nc: 5
names:
  0: no_crack
  1: longitudinal_crack
  2: transverse_crack
  3: alligator_crack
  4: pothole
```

3. **Train model**:
```bash
python train_custom.py \
    --data training/datasets/concrete_cracks.yaml \
    --model gelan-s \
    --epochs 150 \
    --batch-size 16 \
    --name crack_detection
```

4. **Export to ONNX**:
```bash
python export_onnx.py \
    --weights runs/train/crack_detection/weights/best.pt \
    --output models/crack_detector.onnx
```

5. **Update FastAPI**:
```python
# In main.py
detector = YOLOv9Detector("models/crack_detector.onnx")

# Update class names
self.class_names = [
    'no_crack',
    'longitudinal_crack',
    'transverse_crack',
    'alligator_crack',
    'pothole'
]
```

### Example 2: Person Detection Only

1. **Use COCO subset**:
```yaml
# training/datasets/person_only.yaml
path: datasets/person_detection
train: images/train
val: images/val

nc: 1
names:
  0: person
```

2. **Train with pre-trained weights**:
```bash
python train_custom.py \
    --data training/datasets/person_only.yaml \
    --model gelan-s \
    --epochs 100 \
    --freeze 10 \    # Freeze first 10 layers
    --name person_detector
```

## 🛠️ Utilities

### Validate Dataset
```bash
python prepare_dataset.py validate --output datasets/my_dataset
```

### Resume Training
```bash
python train_custom.py \
    --data my_dataset.yaml \
    --resume runs/train/exp/weights/last.pt
```

### Monitor Training
Training outputs are saved to `runs/train/experiment_name/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Latest checkpoint
- `results.png` - Training curves
- `val_batch*.jpg` - Validation predictions

## 🎯 Tips for Better Results

1. **Dataset Quality**:
   - Minimum 100 images per class
   - Balanced class distribution
   - High-quality annotations
   - Diverse lighting/backgrounds

2. **Training Tips**:
   - Start with pre-trained weights
   - Use appropriate image size (640 recommended)
   - Monitor validation loss for overfitting
   - Adjust learning rate if training stalls

3. **Hardware Requirements**:
   - GPU: 4GB+ VRAM recommended
   - RAM: 8GB+ for large datasets
   - CPU training: Slower but possible

4. **Deployment**:
   - Test ONNX model before deployment
   - Verify class names match training config
   - Check inference speed with your hardware

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `--batch-size`
   - Use smaller `--img-size`
   - Add `--device cpu` for CPU training

2. **No improvement in training**:
   - Check dataset quality
   - Reduce learning rate
   - Increase training epochs
   - Verify class labels are correct

3. **ONNX export fails**:
   - Update PyTorch version
   - Use `--opset 11` or lower
   - Check model compatibility

4. **Poor inference results**:
   - Verify class name order matches training
   - Check image preprocessing
   - Test with training resolution

## 📚 Additional Resources

- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- [Original Repository](https://github.com/WongKinYiu/yolov9)
- [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [Data Augmentation Tips](https://blog.roboflow.com/yolo-augmentation/)

Happy training! 🎯