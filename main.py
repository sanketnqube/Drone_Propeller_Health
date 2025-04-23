import cv2
import sys
from typing import Optional, Dict, Any
from dataclasses import asdict
import numpy as np

from config import MODEL_CONFIG, CAMERA_CONFIG, VISUALIZATION_CONFIG
from camera.picamera_handler import PiCameraHandler
from detection.detector import PropellerDetector, DetectionResult
from utils.visualization import display_image, setup_mouse_callback, destroy_all_windows

class PropellerHealthApp:
    def __init__(self):
        """Initialize the application with configuration."""
        self.camera = PiCameraHandler(CAMERA_CONFIG)
        self.detector = PropellerDetector(MODEL_CONFIG['model_path'], {
            **MODEL_CONFIG,
            **VISUALIZATION_CONFIG
        })
        self.captured_image = None
        self.window_name = 'Propeller Health Detection'

    def run(self):
        """Run the main application loop."""
        try:
            self.camera.start()
            cv2.namedWindow(self.window_name)
            setup_mouse_callback(
                self.window_name, 
                self._capture_image_callback, 
                {'frame': None}
            )

            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                # Update the frame in the mouse callback
                setup_mouse_callback(
                    self.window_name, 
                    self._capture_image_callback, 
                    {'frame': frame}
                )

                display_image(self.window_name, frame, 1)

                if self.captured_image is not None:
                    self._process_captured_image()
                    self.captured_image = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nApplication stopped by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)
        finally:
            self._cleanup()

    def _capture_image_callback(self, event, x, y, flags, param):
        """Mouse callback function to capture an image on right-click."""
        if event == cv2.EVENT_RBUTTONDOWN:
            if param and 'frame' in param and param['frame'] is not None:
                self.captured_image = param['frame'].copy()
                print("Image captured for processing!")

    def _process_captured_image(self):
        """Process the captured image and display results."""
        if self.captured_image is None:
            return

        print("\nProcessing captured image...")
        result = self.detector.detect(self.captured_image)
        
        if result is not None:
            self._display_results(result)

    def _display_results(self, result: DetectionResult):
        """Display the detection results."""
        # Print results to console
        print("\nDetection Results:")
        print(f"Healthy propellers: {result.healthy_count}")
        print(f"Faulty propellers: {result.faulty_count}")
        print(f"Total propellers: {result.total_count}")
        print(f"Quadrant counts: {result.quadrant_counts}")
        print(f"Status: {result.status}")
        
        if result.damaged_quadrants:
            print(f"Damaged Quadrants: {', '.join(result.damaged_quadrants)}")
        if result.missing_quadrants:
            print(f"Missing Quadrants: {', '.join(result.missing_quadrants)}")
        if result.excess_quadrants:
            print(f"Excess Quadrants: {', '.join(result.excess_quadrants)}")

        # Display processed image
        display_image('Processed Image', result.processed_image)
        cv2.waitKey(0)
        destroy_all_windows()

    def _cleanup(self):
        """Clean up resources."""
        self.camera.stop()
        destroy_all_windows()
        print("Application shutdown complete")

if __name__ == "__main__":
    app = PropellerHealthApp()
    app.run()