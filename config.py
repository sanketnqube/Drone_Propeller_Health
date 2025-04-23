import os
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple

class Config:
    """Centralized configuration for the Propeller Health Detection system."""
    
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Model configuration
    MODEL: Dict[str, Any] = {
        'model_path': str(BASE_DIR / 'models' / '/home/ai-nxtqube/Desktop/Rpi/propehealth_yolov8x.pt'),
        'healthy_threshold': 0.2,
        'faulty_threshold': 0.2,
        'max_per_quadrant': 2,
        'min_per_quadrant': 2,
        'expected_total_count': 8,
        'device': 'cpu',  # Use '0' for GPU or 'cpu' for CPU
        'nms_threshold': 0.4  # Non-Maximum Suppression threshold
    }
    
    # Camera configuration
    CAMERA: Dict[str, Any] = {
        'preview_config': {
            'size': (1280, 720),
            'format': 'BGR888',  # Options: 'BGR888', 'RGB888'
            'fps': 30
        },
        'capture_delay': 0.1,  # seconds
        'rotation': 0,  # Image rotation (0, 90, 180, 270)
        'flip': None  # 'horizontal', 'vertical', or None
    }
    
    # Detection visualization
    VISUALIZATION: Dict[str, Any] = {
        'healthy_color': (0, 255, 0),  # BGR format (green)
        'faulty_color': (0, 0, 255),   # BGR format (red)
        'quadrant_line_color': (255, 0, 0),  # BGR format (blue)
        'text_color': (255, 255, 255),  # BGR format (white)
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale': 0.5,
        'line_thickness': 2,
        'text_thickness': 2,
        'status_font_scale': 1.0
    }
    
    # Output configuration
    OUTPUT: Dict[str, Any] = {
        'results_directory': str(BASE_DIR / '/home/ai-nxtqube/Desktop/Rpi/detection_results'),
        'save_images': True,
        'image_format': 'jpg',  # Options: 'jpg', 'png'
        'image_quality': 95,    # For JPEG (1-100)
        'save_json': True,
        'json_indent': 4,       # Pretty-print JSON with indentation
        'timestamp_format': '%Y%m%d_%H%M%S'  # For filenames
    }
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate configuration values."""
        # Check model path exists
        if not os.path.exists(cls.MODEL['model_path']):
            raise FileNotFoundError(f"Model file not found at {cls.MODEL['model_path']}")
        
        # Validate image format
        if cls.OUTPUT['image_format'] not in ['jpg', 'png']:
            raise ValueError("Output image format must be 'jpg' or 'png'")
        
        # Validate camera rotation
        if cls.CAMERA['rotation'] not in [0, 90, 180, 270]:
            raise ValueError("Camera rotation must be 0, 90, 180, or 270 degrees")

# Validate configuration on import
try:
    Config.validate()
except Exception as e:
    print(f"Configuration error: {e}")
    raise

# Aliases for backward compatibility
MODEL_CONFIG = Config.MODEL
CAMERA_CONFIG = Config.CAMERA
VISUALIZATION_CONFIG = Config.VISUALIZATION
OUTPUT_CONFIG = Config.OUTPUT
