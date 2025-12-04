# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.detect.val import DetectionValidator as V8DetectionValidator

# YOLOv11 uses the same validator as YOLOv8 with potential future modifications
DetectionValidator = V8DetectionValidator

def val(cfg):
    """
    Validates YOLOv11 model.

    Args:
        cfg (dict): Configuration dictionary containing model and validation settings.

    Returns:
        Results: Validation results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    validator = V8DetectionValidator(cfg)
    validator()
    return validator