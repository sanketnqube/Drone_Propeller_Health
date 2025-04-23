from picamera2 import Picamera2
import cv2
from typing import Optional, Tuple
import numpy as np

class PiCameraHandler:
    def __init__(self, config: dict):
        self.camera = Picamera2()
        self.config = config
        self._setup_camera()

    def _setup_camera(self):
        """Configure the camera with preview settings."""
        preview_config = self.camera.create_preview_configuration(
            main={"size": self.config['preview_config']['size'], 
                  "format": self.config['preview_config']['format']}
        )
        self.camera.configure(preview_config)

    def start(self):
        """Start the camera stream."""
        self.camera.start()

    def stop(self):
        """Stop the camera stream."""
        self.camera.stop()

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Optional[np.ndarray]: Captured frame in BGR format or None if capture fails
        """
        try:
            frame = self.camera.capture_array()
            if frame is not None and frame.size > 0:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return None
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def get_frame_size(self) -> Tuple[int, int]:
        """Get the current frame size (width, height)."""
        return self.config['preview_config']['size']