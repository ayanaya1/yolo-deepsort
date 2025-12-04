# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.detect.train import DetectionTrainer as V8DetectionTrainer

# YOLOv11 uses the same trainer as YOLOv8 with potential future modifications
DetectionTrainer = V8DetectionTrainer

def train(cfg):
    """
    Trains YOLOv11 model.

    Args:
        cfg (dict): Configuration dictionary containing model and training settings.

    Returns:
        Results: Training results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    trainer = V8DetectionTrainer(cfg)
    trainer.train()
    return trainer