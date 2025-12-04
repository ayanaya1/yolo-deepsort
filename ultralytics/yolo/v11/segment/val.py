# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.segment.val import SegmentationValidator as V8SegmentationValidator

# YOLOv11 uses the same validator as YOLOv8 with potential future modifications
SegmentationValidator = V8SegmentationValidator

def val(cfg):
    """
    Validates YOLOv11 segmentation model.

    Args:
        cfg (dict): Configuration dictionary containing model and validation settings.

    Returns:
        Results: Validation results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    validator = V8SegmentationValidator(cfg)
    validator()
    return validator