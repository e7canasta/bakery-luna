"""
Performance metrics tracking utilities.

Simple utilities for measuring FPS, latency, and other performance metrics
in video processing pipelines.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerformanceMetrics:
    """
    Track pipeline performance metrics.

    Tracks frame count, timing, and provides convenience methods for
    calculating FPS and other performance statistics.

    Example:
        >>> metrics = PerformanceMetrics()
        >>> metrics.start()
        >>> # ... process frames ...
        >>> metrics.update_frame_count(100)
        >>> metrics.stop()
        >>> print(f"FPS: {metrics.fps:.2f}")
    """

    total_frames: int = 0
    seg_runs: int = 0
    pose_runs: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    _frame_times: list = field(default_factory=list)

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self):
        """Stop timing."""
        self.end_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    @property
    def fps(self) -> float:
        """Calculate frames per second."""
        elapsed = self.elapsed_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0

    @property
    def seg_skip_rate(self) -> float:
        """
        Calculate segmentation skip rate.

        Skip rate = 1 - (seg_runs / total_frames)

        Returns:
            Skip rate in range [0, 1]
        """
        if self.total_frames == 0:
            return 0.0
        return 1.0 - (self.seg_runs / self.total_frames)

    @property
    def avg_latency_ms(self) -> float:
        """
        Calculate average latency per frame in milliseconds.

        Returns:
            Average latency in ms
        """
        if len(self._frame_times) == 0:
            return 0.0
        return (sum(self._frame_times) / len(self._frame_times)) * 1000

    def update_frame_count(self, count: int = 1):
        """
        Increment frame counter.

        Args:
            count: Number of frames to add (default: 1)
        """
        self.total_frames += count

    def record_seg_run(self):
        """Record a segmentation inference run."""
        self.seg_runs += 1

    def record_pose_run(self):
        """Record a pose inference run."""
        self.pose_runs += 1

    def record_frame_time(self, duration: float):
        """
        Record frame processing time for latency tracking.

        Args:
            duration: Frame processing duration in seconds
        """
        self._frame_times.append(duration)

    def get_summary(self) -> dict:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with all metrics
        """
        return {
            "total_frames": self.total_frames,
            "elapsed_time_s": self.elapsed_time,
            "fps": self.fps,
            "seg_runs": self.seg_runs,
            "pose_runs": self.pose_runs,
            "seg_skip_rate": self.seg_skip_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }

    def print_summary(self):
        """Print formatted summary of metrics."""
        summary = self.get_summary()
        print("\n📊 Performance Metrics")
        print(f"⏱  Total time: {summary['elapsed_time_s']:.2f}s")
        print(f"🎞  FPS: {summary['fps']:.2f}")
        print(f"🖼️  Total frames: {summary['total_frames']}")
        print(f"🔧 Segmentation runs: {summary['seg_runs']} ({summary['seg_skip_rate']*100:.1f}% skipped)")
        print(f"🦴 Pose runs: {summary['pose_runs']}")
        if summary['avg_latency_ms'] > 0:
            print(f"⚡ Avg latency: {summary['avg_latency_ms']:.2f}ms")


class FPSCounter:
    """
    Simple FPS counter for real-time monitoring.

    Uses a rolling window to calculate FPS over recent frames.

    Example:
        >>> fps_counter = FPSCounter(window_size=30)
        >>> for frame in video:
        ...     fps_counter.tick()
        ...     print(f"Current FPS: {fps_counter.fps:.1f}")
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()

    def tick(self):
        """Record a frame tick."""
        current_time = time.time()
        delta = current_time - self.last_time
        self.frame_times.append(delta)

        # Keep only last N frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        self.last_time = current_time

    @property
    def fps(self) -> float:
        """Calculate current FPS from rolling window."""
        if len(self.frame_times) == 0:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def reset(self):
        """Reset the counter."""
        self.frame_times.clear()
        self.last_time = time.time()
