"""
Bakery Runtime - Benchmark API
===============================

Programmatic API for measuring inference performance.
Extracted from scripts/int8_vnni/benchmark_cpu.py.

Usage:
    from bakery_runtime.benchmark import BenchmarkRunner, BenchmarkResult

    runner = BenchmarkRunner()
    result = runner.run(model_instance, video_path, max_frames=300)
    print(result.fps, result.avg_cpu_percent)
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import psutil


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    model_name: str
    device: str
    total_frames: int
    elapsed_seconds: float
    fps: float

    # CPU metrics
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    cpu_seconds: float = 0.0  # elapsed × (cpu% / 100)

    # Memory metrics
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0

    # Temperature (optional)
    avg_temp_celsius: float = 0.0

    # Latency (per-frame)
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0


class CPUMonitor:
    """
    Background thread that samples CPU%, memory, and temperature.

    Usage:
        monitor = CPUMonitor(interval=0.1)
        monitor.start()
        # ... do work ...
        stats = monitor.stop()  # returns dict with avg/max values
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.temp_samples: List[float] = []
        self.process = psutil.Process()

    def _monitor_loop(self):
        while self.running:
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)

            mem_info = self.process.memory_info()
            self.memory_samples.append(mem_info.rss / 1024 / 1024)

            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps and temps['coretemp']:
                    avg_temp = sum(t.current for t in temps['coretemp']) / len(temps['coretemp'])
                    self.temp_samples.append(avg_temp)
            except (AttributeError, KeyError):
                pass

            time.sleep(self.interval)

    def start(self):
        """Start background monitoring."""
        self.running = True
        self.cpu_samples = []
        self.memory_samples = []
        self.temp_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated stats."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        return {
            'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            'max_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'max_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
            'avg_temp_celsius': sum(self.temp_samples) / len(self.temp_samples) if self.temp_samples else 0,
        }


class BenchmarkRunner:
    """
    Run inference benchmarks using ModelInstance.

    Measures FPS, latency percentiles, CPU load, and memory usage.
    """

    def run(
        self,
        model_instance,  # bakery_runtime.ModelInstance
        video_path: str,
        max_frames: Optional[int] = None,
        device: str = "GPU",
    ) -> BenchmarkResult:
        """
        Run a benchmark on a video file.

        Args:
            model_instance: A ModelInstance to benchmark.
            video_path: Path to video file.
            max_frames: Max frames to process (None = all).
            device: Device label for reporting.

        Returns:
            BenchmarkResult with all metrics.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = min(max_frames, total_available) if max_frames else total_available

        monitor = CPUMonitor(interval=0.1)
        monitor.start()

        latencies: List[float] = []
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_count >= max_frames:
                break

            t0 = time.perf_counter()
            tensor, meta = model_instance.preprocess(frame)
            _ = model_instance.infer_tensor(tensor)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms
            frame_count += 1

        elapsed = time.time() - start_time
        cpu_stats = monitor.stop()
        cap.release()

        fps = frame_count / elapsed if elapsed > 0 else 0

        # Latency percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        result = BenchmarkResult(
            model_name=model_instance.model_name if hasattr(model_instance, 'model_name') else "unknown",
            device=device,
            total_frames=frame_count,
            elapsed_seconds=elapsed,
            fps=fps,
            avg_cpu_percent=cpu_stats['avg_cpu_percent'],
            max_cpu_percent=cpu_stats['max_cpu_percent'],
            cpu_seconds=elapsed * (cpu_stats['avg_cpu_percent'] / 100.0),
            avg_memory_mb=cpu_stats['avg_memory_mb'],
            max_memory_mb=cpu_stats['max_memory_mb'],
            avg_temp_celsius=cpu_stats['avg_temp_celsius'],
            latency_avg_ms=sum(latencies) / n if n else 0,
            latency_p50_ms=latencies_sorted[int(n * 0.50)] if n else 0,
            latency_p95_ms=latencies_sorted[int(n * 0.95)] if n else 0,
            latency_p99_ms=latencies_sorted[min(int(n * 0.99), n - 1)] if n else 0,
        )

        return result
