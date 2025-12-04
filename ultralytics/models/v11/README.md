# YOLOv11 Model Configuration

This directory contains the model configurations for YOLOv11, the next generation of the YOLO object detection and image processing framework.

## Model Variants

- `yolov11n.yaml` - YOLOv11 Nano (smallest and fastest)
- `yolov11s.yaml` - YOLOv11 Small 
- `yolov11m.yaml` - YOLOv11 Medium
- `yolov11l.yaml` - YOLOv11 Large
- `yolov11x.yaml` - YOLOv11 Extra Large (largest and most accurate)

## Features

YOLOv11 introduces several improvements over previous versions:

- Enhanced backbone architecture
- Improved neck design for better feature fusion
- Optimized head for accurate detection
- Better performance across all model sizes

## Usage

```python
from ultralytics import YOLO

# Load a YOLOv11 model
model = YOLO('yolov11n.yaml')  # build a new model from YAML
# or
model = YOLO('yolov11n.pt')    # load a pretrained model

# Use the model
results = model.predict('image.jpg')
```

## Future Support

This framework is designed to support future YOLO versions (v12, v13, etc.) with minimal changes. The modular architecture allows for easy integration of new models while maintaining backward compatibility.