# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.classify.val import ClassificationValidator as V8ClassificationValidator

# YOLOv11 uses the same validator as YOLOv8 with potential future modifications
ClassificationValidator = V8ClassificationValidator

def val(cfg):
    """
    Validates YOLOv11 classification model.

    Args:
        cfg (dict): Configuration dictionary containing model and validation settings.

    Returns:
        Results: Validation results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    validator = V8ClassificationValidator(cfg)
    validator()
    return validator