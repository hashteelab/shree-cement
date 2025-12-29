#!/usr/bin/env python3
"""
Download required models for Phone Distance Estimation.

Run this script once to download all models before running the main app.
"""

import os
import sys

# Add depth-anything-3 to path
sys.path.insert(0, "/Users/hashteelab/Documents/GitHub/shree-cement/depth-anything-3")


def download_yolo_model():
    """Download YOLOv8 model."""
    print("=" * 60)
    print("Downloading YOLOv8n model...")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        # This will download yolov8n.pt if not cached
        model = YOLO("yolov8n.pt")
        print("YOLOv8n model downloaded successfully!")
        return True
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"ERROR downloading YOLO model: {e}")
        return False


def download_depthanything_model():
    """Download DepthAnything3 model."""
    print("\n" + "=" * 60)
    print("Downloading DepthAnything3 (da3metric-large) model...")
    print("=" * 60)

    try:
        from depth_anything_3.api import DepthAnything3

        # This will download from Hugging Face if not cached
        model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        print("DepthAnything3 model downloaded successfully!")
        return True
    except ImportError as e:
        print(f"ERROR: DepthAnything3 not available: {e}")
        return False
    except Exception as e:
        print(f"ERROR downloading DepthAnything3 model: {e}")
        return False


def main():
    """Download all models."""
    print("\n" + "#" * 60)
    print("# Phone Distance Estimation - Model Downloader")
    print("#" * 60 + "\n")

    results = []

    # Download YOLO
    results.append(("YOLOv8n", download_yolo_model()))

    # Download DepthAnything3
    results.append(("DepthAnything3", download_depthanything_model()))

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"{name}: {status}")

    all_success = all(r[1] for r in results)

    if all_success:
        print("\nAll models downloaded successfully!")
        print("You can now run: python main.py")
        return 0
    else:
        print("\nSome models failed to download. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
