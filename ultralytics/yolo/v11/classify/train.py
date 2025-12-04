# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.classify.train import ClassificationTrainer as V8ClassificationTrainer

# YOLOv11 uses the same trainer as YOLOv8 with potential future modifications
ClassificationTrainer = V8ClassificationTrainer

def train(cfg):
    """
    Trains YOLOv11 classification model.

    Args:
        cfg (dict): Configuration dictionary containing model and training settings.

    Returns:
        Results: Training results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    trainer = V8ClassificationTrainer(cfg)
    trainer.train()
    return trainer