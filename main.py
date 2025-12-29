#!/usr/bin/env python3
"""
Phone Detection - Main Entry Point

Real-time phone detection using YOLO.

Usage:
    python main.py                    # Launch with default settings
    python main.py --port 7861        # Custom port
    python main.py --host 0.0.0.0     # Allow external access
    python main.py --share            # Create public link
"""

import argparse
import sys

from phone_distance_app import PhoneDistanceApp, AppConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone Detection using YOLO"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold (default: 0.5)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration from command line args
    config = AppConfig(
        camera_id=args.camera_id,
        yolo_model=args.yolo_model,
        yolo_confidence_threshold=args.confidence,
    )

    # Create and launch app
    app = PhoneDistanceApp(config)

    print("=" * 50)
    print("Phone Detection")
    print("=" * 50)
    print(f"Camera ID: {config.camera_id}")
    print(f"YOLO Model: {config.yolo_model}")
    print(f"Confidence Threshold: {config.yolo_confidence_threshold}")
    print("=" * 50)

    try:
        app.launch(
            host=args.host,
            port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
