"""
Gradio web interface for real-time phone detection.
"""

import gradio as gr
import numpy as np
import cv2
from typing import Optional
import time
import os
from datetime import datetime

from .config import AppConfig
from .detector import PhoneDetector


class PhoneDistanceApp:
    """Gradio application for real-time phone detection."""

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

        # State for FPS tracking
        self._frame_times = []

        # State for image capture
        self._last_capture_time = 0.0

        # Create capture folder if enabled
        if self.config.enable_capture:
            os.makedirs(self.config.capture_folder, exist_ok=True)
            print(f"Capture folder: {self.config.capture_folder}")

        print("PhoneDistanceApp initialized!")

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

            # Annotate frame with detection
            annotated = self._annotate_frame(frame_bgr, detection)

            # Save image when phone is detected (with interval check)
            if self.config.enable_capture and detection is not None:
                if current_time - self._last_capture_time >= self.config.capture_interval:
                    self._save_image(annotated)
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
        detection
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

            # Phone label with confidence
            info_text = f"Phone: {detection.confidence:.2f}"

            # Draw info text with background
            y_offset = y1 - 10
            (text_w, text_h), _ = cv2.getTextSize(
                info_text,
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
                info_text,
                (x1, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
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

    def _save_image(self, frame: np.ndarray) -> None:
        """
        Save annotated frame to disk.

        Args:
            frame: Annotated frame (BGR format) to save
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
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
        print("Phone Detection App")
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
            gr.Markdown("# 📱 Real-Time Phone Detection")
            gr.Markdown("Adjust the confidence threshold and start your webcam to detect phones in real-time.")

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
                    value=0.1,
                    label="Confidence Threshold",
                    info="Adjust detection sensitivity"
                )

            with gr.Row():
                gr.Markdown("""
                ### Instructions:
                1. Click the camera icon on the webcam feed to start your camera
                2. Adjust the confidence threshold slider to change detection sensitivity
                3. Green boxes indicate detected phones with confidence scores
                4. Red dot shows the centroid of the detected phone
                5. FPS counter shows real-time performance
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
