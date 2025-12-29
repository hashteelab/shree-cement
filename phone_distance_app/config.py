"""
Configuration module for Phone Detection App.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AppConfig:
    """Configuration settings for the phone detection application."""

    # Camera settings
    camera_id: int = 0
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480

    # YOLO settings
    yolo_model: str = "yolov8n.pt"  # nano model for speed
    yolo_confidence_threshold: float = 0.5
    yolo_nms_threshold: float = 0.45
    yolo_device: str = ""  # Empty = auto-detect

    # Display settings
    display_width: int = 640
    display_height: int = 480
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR format)
    bbox_thickness: int = 2
    font_scale: float = 1.0
    font_thickness: int = 2
    text_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black

    # Image capture settings
    enable_capture: bool = True
    capture_folder: str = "images"
    capture_interval: float = 2.0  # seconds between captures

    @property
    def yolo_device_actual(self) -> str:
        """Get actual YOLO device (auto-detect if empty)."""
        if self.yolo_device:
            return self.yolo_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
