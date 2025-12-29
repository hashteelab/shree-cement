"""
Pipeline processor for combining YOLO detection and depth estimation.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np
import cv2

from .config import AppConfig
from .detector import PhoneDetector, Detection
from .distance_estimator import DistanceEstimator


class ProcessedFrame:
    """Result of processing a frame through the pipeline."""

    def __init__(
        self,
        original_frame: np.ndarray,
        detection: Optional[Detection] = None,
        distance: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None,
        fps: float = 0.0,
        processing_time: float = 0.0,
    ):
        self.original_frame = original_frame
        self.detection = detection
        self.distance = distance
        self.depth_map = depth_map
        self.fps = fps
        self.processing_time = processing_time


class PipelineProcessor:
    """
    Manages the processing pipeline for real-time phone distance estimation.

    YOLO runs on every frame (fast), while DepthAnything runs periodically
    (slow) to maintain real-time performance.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: Optional[PhoneDetector] = None,
        estimator: Optional[DistanceEstimator] = None,
    ):
        """
        Initialize the pipeline processor.

        Args:
            config: Application configuration
            detector: Optional pre-configured PhoneDetector
            estimator: Optional pre-configured DistanceEstimator
        """
        self.config = config

        # Initialize components
        if detector is None:
            self.detector = PhoneDetector(
                model_name=config.yolo_model,
                confidence_threshold=config.yolo_confidence_threshold,
                device=config.yolo_device_actual,
            )
        else:
            self.detector = detector

        if estimator is None:
            self.estimator = DistanceEstimator(
                model_path=config.da3_model_path,
                device=config.da3_device_actual,
                process_res=config.da3_process_res,
            )
        else:
            self.estimator = estimator

        # Set smoothing parameters
        self.estimator.set_smoothing_alpha(config.distance_smoothing_alpha)

        # Timing state
        self._last_depth_time = 0.0
        self._depth_update_interval = config.depth_update_interval

        # Cached depth map (thread-safe access)
        self._depth_map: Optional[np.ndarray] = None
        self._depth_lock = threading.Lock()

        # FPS tracking
        self._frame_times = []
        self._last_fps_update = 0.0
        self._current_fps = 0.0

    def should_update_depth(self) -> bool:
        """Check if enough time has elapsed for a new depth estimation."""
        return time.time() - self._last_depth_time >= self._depth_update_interval

    def update_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run depth estimation and update the cached depth map.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            New depth map, or None if estimation failed
        """
        try:
            depth_map = self.estimator.estimate_depth(frame)
            with self._depth_lock:
                self._depth_map = depth_map
            self._last_depth_time = time.time()
            return depth_map
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None

    def get_cached_depth(self) -> Optional[np.ndarray]:
        """Get the currently cached depth map (thread-safe)."""
        with self._depth_lock:
            return self._depth_map

    def process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            ProcessedFrame with detection and distance information
        """
        start_time = time.time()

        if frame is None or frame.size == 0:
            return ProcessedFrame(original_frame=frame)

        # Step 1: Run YOLO detection (fast, every frame)
        detections = self.detector.detect(frame)

        # Get the largest/closest phone detection
        detection = self.detector.get_largest_phone(detections)

        # Step 2: Update depth map if needed (slow, periodic)
        depth_map = None
        distance = None

        if self.should_update_depth():
            depth_map = self.update_depth(frame)
        else:
            depth_map = self.get_cached_depth()

        # Step 3: Calculate distance if we have both detection and depth
        if detection is not None and depth_map is not None:
            distance = self.estimator.calculate_distance(
                depth_map,
                detection.center,
                radius=self.config.depth_sample_radius,
            )

        # Update FPS tracking
        processing_time = time.time() - start_time
        self._update_fps(processing_time)

        return ProcessedFrame(
            original_frame=frame,
            detection=detection,
            distance=distance,
            depth_map=depth_map,
            fps=self._current_fps,
            processing_time=processing_time,
        )

    def _update_fps(self, processing_time: float):
        """Update FPS calculation."""
        current_time = time.time()
        self._frame_times.append(current_time)

        # Remove frame times older than 1 second
        cutoff = current_time - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]

        # Update FPS every 0.5 seconds
        if current_time - self._last_fps_update > 0.5:
            if len(self._frame_times) > 1:
                self._current_fps = len(self._frame_times)
            self._last_fps_update = current_time

    def annotate_frame(self, result: ProcessedFrame) -> np.ndarray:
        """
        Annotate a frame with detection and distance information.

        Args:
            result: ProcessedFrame from process_frame()

        Returns:
            Annotated frame with bounding boxes and text overlays
        """
        frame = result.original_frame.copy()

        if result.detection is not None:
            # Draw bounding box
            x1, y1, x2, y2 = result.detection.bbox
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.config.bbox_color,
                self.config.bbox_thickness,
            )

            # Prepare info text
            info_lines = []

            # Phone label with confidence
            conf_text = f"{result.detection.class_name}: {result.detection.confidence:.2f}"
            info_lines.append(conf_text)

            # Distance if available
            if result.distance is not None:
                dist_text = f"Distance: {result.distance:.2f} m"
                info_lines.append(dist_text)

            # Draw info text with background
            y_offset = y1 - 10
            for line in reversed(info_lines):
                if y_offset < 20:
                    y_offset = y2 + 20  # Move below box if too close to top

                (text_w, text_h), _ = cv2.getTextSize(
                    line,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    self.config.font_thickness,
                )

                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (x1, y_offset - text_h - 5),
                    (x1 + text_w + 10, y_offset + 5),
                    self.config.text_bg_color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    frame,
                    line,
                    (x1 + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    self.config.text_color,
                    self.config.font_thickness,
                )

                y_offset -= text_h + 10

        else:
            # No phone detected - show message
            text = "No phone detected"
            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_thickness,
            )

            # Draw in center of frame
            h, w = frame.shape[:2]
            x = (w - text_w) // 2
            y = h // 2

            cv2.rectangle(
                frame,
                (x - 10, y - text_h - 10),
                (x + text_w + 10, y + 10),
                self.config.text_bg_color,
                -1,
            )

            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.text_color,
                self.config.font_thickness,
            )

        # Draw FPS and processing info
        info_text = f"FPS: {result.fps:.1f} | Processing: {result.processing_time*1000:.1f}ms"
        if result.depth_map is not None:
            info_text += " | Depth: Active"
        else:
            info_text += " | Depth: Initializing..."

        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return frame

    def reset(self):
        """Reset the processor state."""
        self._last_depth_time = 0.0
        with self._depth_lock:
            self._depth_map = None
        self.estimator.reset_smoothing()
        self._frame_times = []
        self._current_fps = 0.0
