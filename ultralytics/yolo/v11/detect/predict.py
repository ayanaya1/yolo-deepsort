# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.v8.detect.predict import DetectionPredictor as V8DetectionPredictor

# YOLOv11 uses the same predictor as YOLOv8 with potential future modifications
DetectionPredictor = V8DetectionPredictor

def predict(cfg, use_python=False):
    """
    Performs prediction using YOLOv11 model.

    Args:
        cfg (dict): Configuration dictionary containing model and prediction settings.
        use_python (bool): Flag to use Python-based prediction instead of CLI.

    Returns:
        Results: Prediction results.
    """
    # This function can be customized for YOLOv11-specific features in the future
    return V8DetectionPredictor(cfg).predict_cli() if not use_python else V8DetectionPredictor(cfg)