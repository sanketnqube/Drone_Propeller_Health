import os
import cv2
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model configuration
MODEL_CONFIG = {
    'model_path': os.path.join(BASE_DIR, 'models', '/home/ai-nxtqube/Desktop/Rpi/propehealth_yolov8x.pt'),
    'healthy_threshold': 0.2,
    'faulty_threshold': 0.2,
    'max_per_quadrant': 2,
    'min_per_quadrant': 2,
    'expected_total_count': 8,
    'device': 'cpu'  # Use '0' if you have a Coral TPU
}

# Camera configuration
CAMERA_CONFIG = {
    'preview_config': {
        'size': (1280, 720),
        'format': 'BGR888'
    },
    'capture_delay': 0.1  # seconds
}

# Detection visualization
VISUALIZATION_CONFIG = {
    'healthy_color': (0, 255, 0),
    'faulty_color': (0, 0, 255),
    'quadrant_line_color': (255, 0, 0),
    'text_color': (255, 255, 255),
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'font_scale': 0.5,
    'line_thickness': 2
}
