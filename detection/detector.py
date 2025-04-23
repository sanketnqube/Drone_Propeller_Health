from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DetectionResult:
    healthy_count: int
    faulty_count: int
    total_count: int
    quadrant_counts: Dict[str, int]
    status: str
    damaged_quadrants: List[str]
    missing_quadrants: List[str]
    excess_quadrants: List[str]
    processed_image: np.ndarray

class PropellerDetector:
    def __init__(self, model_path: str, config: dict):
        self.model = YOLO(model_path)
        self.config = config
        self.class_names = self.model.names

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Process an image to detect and classify propellers.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            DetectionResult: Contains all detection information
            None: If processing fails
        """
        try:
            # Validate input image
            if not self._validate_image(image):
                return None

            # Preprocess image (edge detection)
            edges_bgr = self._preprocess_image(image)
            if edges_bgr is None:
                return None

            # Get image dimensions and quadrants
            height, width = edges_bgr.shape[:2]
            quadrants = self._get_quadrants(width, height)

            # Perform object detection
            detection_results = self.model(edges_bgr)
            
            # Process detections
            result = self._process_detections(
                detection_results, 
                edges_bgr, 
                quadrants, 
                width, 
                height
            )
            
            return result

        except Exception as e:
            print(f"Error in detection: {e}")
            return None

    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate input image meets requirements."""
        if image is None:
            print("Input image is None")
            return False
        if not isinstance(image, np.ndarray):
            print("Input image must be a numpy array")
            return False
        if len(image.shape) != 3 or image.shape[2] != 3:
            print("Input image must be a 3-channel BGR image")
            return False
        return True

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Apply Sobel edge detection to the image."""
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
            sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitude = np.uint8(sobel_magnitude)
            return cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None

    def _get_quadrants(self, width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Divide the image into 4 quadrants."""
        mid_x, mid_y = width // 2, height // 2
        return {
            "Q1": (mid_x, 0, width, mid_y),
            "Q2": (0, 0, mid_x, mid_y),
            "Q3": (0, mid_y, mid_x, height),
            "Q4": (mid_x, mid_y, width, height)
        }

    def _process_detections(self, detection_results, edges_bgr, quadrants, width, height):
        """Process detection results and return formatted output."""
        # Initialize counts and tracking variables
        healthy_count = 0
        faulty_count = 0
        damaged_quadrants = set()
        missing_quadrants = set()
        excess_quadrants = set()
        quadrant_counts = {q: 0 for q in quadrants}
        quadrant_detections = {q: [] for q in quadrants}

        mid_x, mid_y = width // 2, height // 2

        # Collect all detections
        all_boxes = []
        all_confidences = []
        all_class_ids = []

        for result in detection_results:
            boxes = result.boxes
            for box in boxes:
                try:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.class_names[cls]

                    # Apply confidence thresholds
                    if (class_name == "healthy" and conf < self.config['healthy_threshold']) or \
                       (class_name == "faulty" and conf < self.config['faulty_threshold']):
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    all_boxes.append([x1, y1, x2, y2])
                    all_confidences.append(conf)
                    all_class_ids.append(cls)
                except Exception as e:
                    print(f"Error processing detection box: {e}")
                    continue

        # Apply Non-Maximum Suppression
        nms_indices = []
        if all_boxes:
            try:
                nms_indices = cv2.dnn.NMSBoxes(
                    all_boxes, 
                    all_confidences, 
                    score_threshold=min(self.config['healthy_threshold'], self.config['faulty_threshold']), 
                    nms_threshold=0.4
                )
                if isinstance(nms_indices, np.ndarray):
                    nms_indices = nms_indices.flatten()
            except Exception as e:
                print(f"Error during NMS: {e}")

        # Process detections after NMS
        for i in nms_indices:
            try:
                x1, y1, x2, y2 = all_boxes[i]
                conf = all_confidences[i]
                cls = all_class_ids[i]
                class_name = self.class_names[cls]

                # Determine quadrant
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if center_x >= mid_x and center_y <= mid_y:
                    quadrant = "Q1"
                elif center_x <= mid_x and center_y <= mid_y:
                    quadrant = "Q2"
                elif center_x <= mid_x and center_y >= mid_y:
                    quadrant = "Q3"
                else:
                    quadrant = "Q4"

                quadrant_detections[quadrant].append((x1, y1, x2, y2, conf, cls))
            except Exception as e:
                print(f"Error processing NMS detection: {e}")
                continue

        # Limit detections per quadrant and draw results
        for quadrant in quadrant_detections:
            detections = sorted(
                quadrant_detections[quadrant], 
                key=lambda x: x[4], 
                reverse=True
            )[:self.config['max_per_quadrant']]
            
            quadrant_detections[quadrant] = detections
            quadrant_counts[quadrant] = len(detections)

            for x1, y1, x2, y2, conf, cls in detections:
                class_name = self.class_names[cls]
                if class_name == "healthy":
                    healthy_count += 1
                elif class_name == "faulty":
                    faulty_count += 1
                    damaged_quadrants.add(quadrant)

                # Draw bounding box and label
                label = f"{class_name} {conf:.2f}"
                color = self.config['healthy_color'] if class_name == "healthy" else self.config['faulty_color']
                cv2.rectangle(edges_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    edges_bgr, label, (x1, y1 - 10),
                    self.config['font'], 0.5, color, 2
                )

            if len(quadrant_detections[quadrant]) > self.config['max_per_quadrant']:
                excess_quadrants.add(quadrant)

        # Check for missing propellers
        for quadrant, count in quadrant_counts.items():
            if count < self.config['min_per_quadrant']:
                missing_quadrants.add(quadrant)

        # Draw quadrant lines
        cv2.line(edges_bgr, (mid_x, 0), (mid_x, height), self.config['quadrant_line_color'], 2)
        cv2.line(edges_bgr, (0, mid_y), (width, mid_y), self.config['quadrant_line_color'], 2)

        # Determine health status
        total_count = healthy_count + faulty_count
        status = "Healthy" if (
            faulty_count == 0 and 
            total_count == self.config['expected_total_count'] and 
            not missing_quadrants and 
            not excess_quadrants
        ) else "Faulty"

        # Add status text to image
        status_text = (
            f"Status: {status} "
            f"(Healthy: {healthy_count}, "
            f"Faulty: {faulty_count}, "
            f"Total: {total_count}/{self.config['expected_total_count']})"
        )
        cv2.putText(
            edges_bgr, status_text, (10, 30),
            self.config['font'], 1, self.config['text_color'], 2
        )

        # Add quadrant information
        y_offset = 70
        if damaged_quadrants:
            damage_text = f"Damaged Quadrants: {', '.join(sorted(damaged_quadrants))}"
            cv2.putText(
                edges_bgr, damage_text, (10, y_offset),
                self.config['font'], 1, self.config['text_color'], 2
            )
            y_offset += 40

        if missing_quadrants:
            missing_text = f"Missing Propellers in: {', '.join(sorted(missing_quadrants))}"
            cv2.putText(
                edges_bgr, missing_text, (10, y_offset),
                self.config['font'], 1, self.config['text_color'], 2
            )
            y_offset += 40

        if excess_quadrants:
            excess_text = f"Excess Propellers in: {', '.join(sorted(excess_quadrants))}"
            cv2.putText(
                edges_bgr, excess_text, (10, y_offset),
                self.config['font'], 1, self.config['text_color'], 2
            )

        return DetectionResult(
            healthy_count=healthy_count,
            faulty_count=faulty_count,
            total_count=total_count,
            quadrant_counts=quadrant_counts,
            status=status,
            damaged_quadrants=sorted(damaged_quadrants),
            missing_quadrants=sorted(missing_quadrants),
            excess_quadrants=sorted(excess_quadrants),
            processed_image=edges_bgr
        )