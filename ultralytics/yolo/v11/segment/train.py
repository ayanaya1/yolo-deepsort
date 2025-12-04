# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.segment.train import SegmentationTrainer as V8SegmentationTrainer

# YOLOv11 uses the same trainer as YOLOv8 with potential future modifications
SegmentationTrainer = V8SegmentationTrainer

def train(cfg):
    """
    Trains YOLOv11 segmentation model.

    Args:
        cfg (dict): Configuration dictionary containing model and training settings.

    Returns:
        Results: Training results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    trainer = V8SegmentationTrainer(cfg)
    trainer.train()
    return trainer