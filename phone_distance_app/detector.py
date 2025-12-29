"""
YOLO-based phone detector module.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    """Represents a single object detection."""

    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]


class PhoneDetector:
    """
    YOLO-based phone detector.

    Uses Ultralytics YOLO to detect cell phones in images.
    COCO dataset class 67 = "cell phone"
    """

    # COCO dataset class ID for "cell phone"
    CELL_PHONE_CLASS_ID = 67

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize the phone detector.

        Args:
            model_name: YOLO model name (e.g., "yolov8n.pt", "yolov8s.pt")
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ("cuda", "cpu", or "auto")
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load the YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_name)
                self._model.to(self.device)
            except ImportError:
                raise ImportError(
                    "ultralytics is required. Install with: pip install ultralytics"
                )
        return self._model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect phones in a frame.

        Args:
            frame: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of Detection objects for detected phones
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                # Filter for cell phones only (COCO class 67)
                if class_id == self.CELL_PHONE_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])

                    detections.append(
                        Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=confidence,
                            class_id=class_id,
                            class_name="cell phone",
                        )
                    )

        return detections

    def get_largest_phone(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Get the largest phone detection by bounding box area.

        Args:
            detections: List of Detection objects

        Returns:
            The largest Detection or None if no detections
        """
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)

    def get_closest_phone(
        self, detections: List[Detection], depth_map: Optional[np.ndarray] = None
    ) -> Optional[Detection]:
        """
        Get the closest phone detection by depth.

        If depth map is not available, returns the largest phone.

        Args:
            detections: List of Detection objects
            depth_map: Optional depth map for determining distance

        Returns:
            The closest Detection or None if no detections
        """
        if not detections:
            return None

        if depth_map is None:
            return self.get_largest_phone(detections)

        # Find detection with smallest depth at center
        closest = None
        min_depth = float("inf")

        for detection in detections:
            cx, cy = detection.center
            h, w = depth_map.shape

            # Check bounds
            if 0 <= cy < h and 0 <= cx < w:
                depth = depth_map[cy, cx]
                if depth > 0 and depth < min_depth:
                    min_depth = depth
                    closest = detection

        return closest if closest is not None else self.get_largest_phone(detections)
