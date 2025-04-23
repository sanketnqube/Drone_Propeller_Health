import cv2
import numpy as np
from typing import Optional

def display_image(window_name: str, image: np.ndarray, wait_time: int = 0) -> None:
    """
    Display an image in a window.
    
    Args:
        window_name: Name of the window
        image: Image to display
        wait_time: Time to wait for key press (0 = indefinitely)
    """
    if image is not None and isinstance(image, np.ndarray):
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)

def setup_mouse_callback(window_name: str, callback_func, params: dict) -> None:
    """
    Set up a mouse callback for a window.
    
    Args:
        window_name: Name of the window to attach callback to
        callback_func: Function to call on mouse events
        params: Parameters to pass to the callback function
    """
    cv2.setMouseCallback(window_name, callback_func, params)

def destroy_all_windows() -> None:
    """Destroy all OpenCV windows."""
    cv2.destroyAllWindows()