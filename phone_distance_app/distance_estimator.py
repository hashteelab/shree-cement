"""
Distance estimation module using DepthAnything3.
"""

import sys
import os
from typing import Optional, Tuple
import numpy as np
import torch

# Add depth-anything-3 to Python path
_da3_path = os.path.join(os.path.dirname(__file__), "..", "depth-anything-3")
if os.path.exists(_da3_path) and _da3_path not in sys.path:
    sys.path.insert(0, _da3_path)


class DistanceEstimator:
    """
    Estimates distance using DepthAnything3 depth maps.

    Uses the da3metric-large model which outputs metric depth in meters.
    """

    def __init__(
        self,
        model_path: str = "depth-anything/DA3METRIC-LARGE",
        device: str = "cuda",
        process_res: int = 504,
    ):
        """
        Initialize the distance estimator.

        Args:
            model_path: Hugging Face model path or local path
            device: Device to run inference on ("cuda" or "cpu")
            process_res: Processing resolution for DA3
        """
        self.model_path = model_path
        self.device = device
        self.process_res = process_res
        self._model = None
        self._smoothed_distance: Optional[float] = None
        self._alpha = 0.3  # Exponential smoothing factor

    @property
    def model(self):
        """Lazy load the DepthAnything3 model."""
        if self._model is None:
            try:
                from depth_anything_3.api import DepthAnything3

                self._model = DepthAnything3.from_pretrained(self.model_path)
                self._model.to(self.device)
                self._model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load DepthAnything3 model: {e}")
        return self._model

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Run DepthAnything inference to get depth map.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            depth_map: Depth values (H, W) in meters for metric models
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (typical from OpenCV) by looking at channel statistics
            # RGB image from PIL, BGR from cv2
            image = image[:, :, ::-1]  # BGR to RGB conversion

        # Run inference
        prediction = self.model.inference(
            [image],
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
        )

        # Get depth map
        depth_map = prediction.depth[0]  # First image in batch

        return depth_map

    def calculate_distance(
        self,
        depth_map: np.ndarray,
        bbox_center: Tuple[int, int],
        radius: int = 2,
    ) -> Optional[float]:
        """
        Extract distance at phone location from depth map.

        Samples a small region around the center point and uses the median
        for robustness against noise.

        Args:
            depth_map: Depth map (H, W) with values in meters
            bbox_center: (cx, cy) center of phone bounding box
            radius: Radius of region to sample around center

        Returns:
            Distance in meters, or None if out of bounds or invalid
        """
        if depth_map is None:
            return None

        cx, cy = bbox_center
        h, w = depth_map.shape

        # Check bounds
        if not (0 <= cy < h and 0 <= cx < w):
            return None

        # Sample region around center
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)

        region = depth_map[y_min:y_max, x_min:x_max]

        # Filter out invalid values (zeros, negatives, inf)
        valid_mask = (region > 0) & np.isfinite(region)
        if not np.any(valid_mask):
            return None

        valid_depths = region[valid_mask]
        distance = float(np.median(valid_depths))

        # Apply exponential smoothing
        if self._smoothed_distance is None:
            self._smoothed_distance = distance
        else:
            self._smoothed_distance = (
                self._alpha * distance + (1 - self._alpha) * self._smoothed_distance
            )

        return self._smoothed_distance

    def reset_smoothing(self):
        """Reset the exponential smoothing state."""
        self._smoothed_distance = None

    def set_smoothing_alpha(self, alpha: float):
        """
        Set the exponential smoothing factor.

        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing.
        """
        self._alpha = max(0.0, min(1.0, alpha))

    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device if hasattr(self, "_device") else "cuda"

    @device.setter
    def device(self, value: str):
        """Set the device and reload model if necessary."""
        if hasattr(self, "_device") and self._device != value:
            self._model = None  # Force reload
        self._device = value
