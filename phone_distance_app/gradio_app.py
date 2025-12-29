"""
Gradio web interface for real-time phone detection with depth estimation.
"""

import gradio as gr
import numpy as np
import cv2
from typing import Optional
import time
import os
import threading
from datetime import datetime

from .config import AppConfig
from .detector import PhoneDetector
from .distance_estimator import DistanceEstimator


class PhoneDistanceApp:
    """Gradio application for real-time phone detection with depth estimation."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the application."""
        print("Initializing PhoneDistanceApp...")

        self.config = config or AppConfig()

        # Initialize YOLO detector
        print("Loading YOLO model...")
        self.detector = PhoneDetector(
            model_name=self.config.yolo_model,
            confidence_threshold=self.config.yolo_confidence_threshold,
            device=self.config.yolo_device_actual,
        )
        print("YOLO model loaded!")

        # Initialize Depth Anything 3 estimator (optional)
        self.estimator: Optional[DistanceEstimator] = None
        self._depth_map: Optional[np.ndarray] = None
        self._depth_lock = threading.Lock()
        self._last_depth_time = 0.0
        self._current_distance: Optional[float] = None

        if self.config.enable_depth:
            print("Loading Depth Anything 3 model...")
            try:
                self.estimator = DistanceEstimator(
                    model_path=self.config.da3_model_path,
                    device=self.config.da3_device_actual,
                    process_res=self.config.da3_process_res,
                )
                self.estimator.set_smoothing_alpha(self.config.distance_smoothing_alpha)
                print("Depth Anything 3 model loaded!")
            except Exception as e:
                print(f"Warning: Could not load Depth Anything 3: {e}")
                print("Continuing without depth estimation...")
                self.estimator = None

        # State for FPS tracking
        self._frame_times = []

        # State for image capture
        self._last_capture_time = 0.0

        # Create capture folder if enabled
        if self.config.enable_capture:
            os.makedirs(self.config.capture_folder, exist_ok=True)
            print(f"Capture folder: {self.config.capture_folder}")

        print("PhoneDistanceApp initialized!")

    def _should_update_depth(self) -> bool:
        """Check if enough time has elapsed for a new depth estimation."""
        return time.time() - self._last_depth_time >= self.config.depth_update_interval

    def _update_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run depth estimation and update the cached depth map."""
        if self.estimator is None:
            return None

        try:
            depth_map = self.estimator.estimate_depth(frame)
            with self._depth_lock:
                self._depth_map = depth_map
            self._last_depth_time = time.time()
            return depth_map
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None

    def _get_cached_depth(self) -> Optional[np.ndarray]:
        """Get the currently cached depth map (thread-safe)."""
        with self._depth_lock:
            return self._depth_map

    def _calculate_distance(self, detection) -> Optional[float]:
        """Calculate distance at phone bounding box centroid."""
        if self.estimator is None or detection is None:
            return None

        depth_map = self._get_cached_depth()
        if depth_map is None:
            return None

        return self.estimator.calculate_distance(
            depth_map,
            detection.center,
            radius=self.config.depth_sample_radius,
        )

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> np.ndarray:
        """
        Process a single webcam frame.

        Args:
            frame: Input frame (RGB format)
            conf_threshold: YOLO confidence threshold

        Returns:
            Annotated frame (RGB format)
        """
        current_time = time.time()

        if frame is None or frame.size == 0:
            return None

        try:
            # Convert RGB to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Update confidence threshold
            self.detector.confidence_threshold = conf_threshold

            # Run YOLO detection
            detections = self.detector.detect(frame_bgr)
            detection = self.detector.get_largest_phone(detections)

            # Update depth map if needed (periodic, not every frame for performance)
            if self.estimator is not None and self._should_update_depth():
                self._update_depth(frame_bgr)

            # Calculate distance if we have both detection and depth
            distance = None
            if detection is not None:
                distance = self._calculate_distance(detection)
                if distance is not None:
                    self._current_distance = distance

            # Annotate frame with detection and distance
            annotated = self._annotate_frame(frame_bgr, detection, self._current_distance)

            # Save image when phone is detected (with interval check)
            if self.config.enable_capture and detection is not None:
                if current_time - self._last_capture_time >= self.config.capture_interval:
                    self._save_image(annotated, distance)
                    self._last_capture_time = current_time

            # Update FPS
            self._frame_times.append(current_time)
            self._frame_times = [t for t in self._frame_times if current_time - t < 1.0]
            fps = len(self._frame_times)

            # Add FPS overlay
            cv2.putText(
                annotated,
                f"FPS: {fps:.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Add depth status overlay
            depth_status = "Active" if self._get_cached_depth() is not None else "Initializing..."
            cv2.putText(
                annotated,
                f"Depth: {depth_status}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 200, 255) if self._get_cached_depth() is not None else (100, 100, 100),
                1,
            )

            # Convert back to RGB
            result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return result

        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detection,
        distance: Optional[float] = None
    ) -> np.ndarray:
        """Annotate frame with bounding box and info."""
        annotated = frame.copy()

        if detection is not None:
            x1, y1, x2, y2 = detection.bbox

            # Green bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw centroid point
            cx, cy = detection.center
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            # Prepare info lines
            info_lines = []

            # Phone label with confidence
            info_lines.append(f"Phone: {detection.confidence:.2f}")

            # Distance if available
            if distance is not None:
                info_lines.append(f"Distance: {distance:.2f} m")

            # Draw info text with background
            y_offset = y1 - 10
            for line in reversed(info_lines):
                if y_offset < 25:
                    y_offset = y2 + 25  # Move below box if too close to top

                (text_w, text_h), _ = cv2.getTextSize(
                    line,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )

                # Black background
                cv2.rectangle(
                    annotated,
                    (x1 - 5, y_offset - text_h - 5),
                    (x1 + text_w + 10, y_offset + 5),
                    (0, 0, 0),
                    -1,
                )

                # Green text
                cv2.putText(
                    annotated,
                    line,
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                y_offset -= text_h + 10
        else:
            # No phone detected
            cv2.putText(
                annotated,
                "No phone detected",
                (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        return annotated

    def _save_image(self, frame: np.ndarray, distance: Optional[float] = None) -> None:
        """
        Save annotated frame to disk with bounding box and distance info.

        Args:
            frame: Annotated frame (BGR format) to save
            distance: Optional distance in meters
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            if distance is not None:
                filename = f"phone_{timestamp}_d{distance:.2f}m.jpg"
            else:
                filename = f"phone_{timestamp}.jpg"
            filepath = os.path.join(self.config.capture_folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def launch(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        share: bool = False,
        **kwargs,
    ):
        """Launch the Gradio application."""
        print("=" * 60)
        print("Phone Detection App with Depth Estimation")
        print("=" * 60)
        print(f"URL: http://{host}:{port}")
        print("Press Ctrl+C to stop")
        print("=" * 60)

        # Custom CSS to fix layout issues
        custom_css = """
        footer {
            display: none !important;
        }
        .gradio-container {
            max-width: 1400px !important;
        }
        """

        with gr.Blocks(css=custom_css, title="Phone Detection App") as demo:
            gr.Markdown("# 📱 Real-Time Phone Detection with Depth Estimation")
            gr.Markdown("Adjust the confidence threshold and start your webcam to detect phones and estimate distance in real-time.")

            with gr.Row():
                with gr.Column(scale=1):
                    # Webcam input with streaming
                    webcam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Webcam Feed",
                        mirror_webcam=False,
                    )

                with gr.Column(scale=1):
                    # Output display
                    output = gr.Image(
                        label="Detection Output",
                    )

            with gr.Row():
                # Confidence threshold slider
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    label="Confidence Threshold",
                    info="Adjust detection sensitivity"
                )

            with gr.Row():
                gr.Markdown("""
                ### Instructions:
                1. Click the camera icon on the webcam feed to start your camera
                2. Adjust the confidence threshold slider to change detection sensitivity
                3. Green boxes indicate detected phones with confidence scores
                4. Red dot shows the centroid where depth is measured
                5. Distance is shown in meters when phone is detected
                6. FPS counter shows real-time performance
                """)

            # Stream processing
            webcam.stream(
                fn=self.process_frame,
                inputs=[webcam, conf_slider],
                outputs=output,
                show_progress=False,
            )

        # Launch the app
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            show_api=False,
            **kwargs,
        )


def create_app(config: Optional[AppConfig] = None) -> PhoneDistanceApp:
    """Convenience function to create a PhoneDistanceApp."""
    return PhoneDistanceApp(config)


if __name__ == "__main__":
    app = create_app()
    app.launch()
