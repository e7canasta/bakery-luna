"""
Video I/O utilities for Bakery CLI.
"""
import sys
import cv2
from pathlib import Path
from typing import Tuple

from rich.console import Console

console = Console()

def open_video_source(video_path: str) -> Tuple[cv2.VideoCapture, int, int, int, int]:
    """
    Open video file or camera.

    Args:
        video_path: Path to video file or camera index (0, 1, etc.)

    Returns:
        Tuple of (VideoCapture, width, height, fps, total_frames)
    """
    # Try to parse as camera index
    try:
        camera_idx = int(video_path)
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            console.print(f"\n📹 Opened camera {camera_idx}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30  # Default for webcam
            total_frames = -1 # Infinite
            return cap, width, height, fps, total_frames
    except ValueError:
        pass

    # Try as file path
    path = Path(video_path)
    if not path.exists():
        console.print(f"[red]❌ Video file not found: {video_path}[/red]")
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        console.print(f"[red]❌ Failed to open video: {video_path}[/red]")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    console.print(f"\n📹 Video opened: {path.name}")
    console.print(f"   Resolution: {width}x{height}")
    console.print(f"   FPS: {fps}")
    console.print(f"   Total frames: {total_frames}")

    return cap, width, height, fps, total_frames


def create_video_writer(output_path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """
    Create video writer for output.

    Args:
        output_path: Output video path
        width: Frame width
        height: Frame height
        fps: Frames per second

    Returns:
        cv2.VideoWriter object
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        console.print(f"[red]❌ Failed to create video writer: {output_path}[/red]")
        sys.exit(1)

    console.print(f"\n💾 Output: {output_path}")

    return writer
