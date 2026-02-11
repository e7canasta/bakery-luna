"""
Tests for Adaptive Shift Lens Strategy.
"""

import pytest
import numpy as np
import supervision as sv

from bakery_lens import FocusLensConfig
from bakery_lens.strategies._adaptive_shift import AdaptiveShiftLensStrategy


class TestAdaptiveShiftLensStrategy:
    """Feature: Adaptive shift strategy behavior."""

    def test_initial_position_centered(self):
        """
        Scenario: Initial crop position is centered
            Given 1920x1080 frame, focus_size=640
            When computing first crop
            Then should be at center x=640
            And size should be 640x640
        """
        config = FocusLensConfig(focus_size=640, adaptive=True)
        strategy = AdaptiveShiftLensStrategy(config)
        
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        assert x == 640
        assert y == 220
        assert w == 640
        assert h == 640

    def test_initial_position_explicit(self):
        """
        Scenario: Initial crop position is explicit
            Given focus_x=100
            When computing first crop
            Then should start at x=100
        """
        config = FocusLensConfig(focus_size=640, focus_x=100, adaptive=True)
        strategy = AdaptiveShiftLensStrategy(config)
        
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        assert x == 100

    def test_shift_right_on_right_edge_detection(self):
        """
        Scenario: Shift right when person is near right edge
            Given person near right edge (x2 > focus_size - threshold)
            When updating strategy
            Then target center should increase (shift right)
        """
        config = FocusLensConfig(
            focus_size=640, 
            adaptive=True, 
            edge_threshold=40, 
            shift_step=80,
            smoothing=0.0  # Instant update for testing
        )
        strategy = AdaptiveShiftLensStrategy(config)
        
        # Initialize
        strategy.compute_crop_params(1920, 1080)
        initial_center = strategy.current_center_x
        
        # Detection near right edge (630 > 640-40)
        detections = sv.Detections(
            xyxy=np.array([[500, 100, 630, 200]]), # x1, y1, x2, y2
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        
        strategy.update(detections)
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        # Should have shifted right by shift_step
        expected_center = initial_center + 80
        # Re-calculate expected x
        expected_x = int(expected_center - 320)
        
        assert x == expected_x
        assert x > 640
        assert w == 640 # Size unchanged

    def test_shift_left_on_left_edge_detection(self):
        """
        Scenario: Shift left when person is near left edge
            Given person near left edge (x1 < threshold)
            When updating strategy
            Then target center should decrease (shift left)
        """
        config = FocusLensConfig(
            focus_size=640, 
            adaptive=True, 
            edge_threshold=40, 
            shift_step=80,
            smoothing=0.0
        )
        strategy = AdaptiveShiftLensStrategy(config)
        
        # Initialize
        strategy.compute_crop_params(1920, 1080)
        initial_center = strategy.current_center_x
        
        # Detection near left edge (10 < 40)
        detections = sv.Detections(
            xyxy=np.array([[10, 100, 50, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        
        strategy.update(detections)
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        # Should have shifted left
        expected_center = initial_center - 80
        expected_x = int(expected_center - 320)
        
        assert x == expected_x
        assert x < 640

    def test_expand_on_both_edges(self):
        """
        Scenario: Expand size when people are on both edges
            Given detections on both left and right edges
            And allow_expand=True
            When updating strategy
            Then size should increase
        """
        config = FocusLensConfig(
            focus_size=640, 
            adaptive=True, 
            allow_expand=True,
            edge_threshold=40, 
            shift_step=80,
            smoothing=0.0
        )
        strategy = AdaptiveShiftLensStrategy(config)
        
        # Initialize
        strategy.compute_crop_params(1920, 1080)
        
        # Detections: One left, one right
        # Left: 10..50, Right: 600..630 (relative to 640 crop)
        detections = sv.Detections(
            xyxy=np.array([
                [10, 100, 50, 200],
                [600, 100, 630, 200]
            ]),
            confidence=np.array([0.9, 0.9]),
            class_id=np.array([0, 0])
        )
        
        strategy.update(detections)
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        # Size should increase by shift_step
        assert w == 640 + 80
        assert h == 640 + 80
        
        # Center should remain mostly same (if symmetric expansion)
        # 640 + 80 = 720. Half = 360.
        # Initial center 960. New center = 960.
        # x = 960 - 360 = 600.
        # old x = 960 - 320 = 640.
        # So x moved left by 40 to accommodate expansion.
        assert x == 600

    def test_decay_size_when_clear(self):
        """
        Scenario: Decay size when no pressure
            Given expanded state
            And detections verify no pressure on edges
            When updating
            Then size should decrease towards base focus_size
        """
        config = FocusLensConfig(
            focus_size=640, 
            adaptive=True, 
            allow_expand=True,
            shift_step=80,
            smoothing=0.0
        )
        strategy = AdaptiveShiftLensStrategy(config)
        
        # Initialize
        strategy.compute_crop_params(1920, 1080)
        
        # Force expanded state manually for test setup
        strategy.current_size = 800.0
        strategy.target_size = 800.0
        
        # Detection safely in middle (no edge pressure)
        detections = sv.Detections(
            xyxy=np.array([[300, 100, 400, 200]]), # 300..400 in 800-wide crop
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        
        strategy.update(detections)
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        # Should decay
        # decay rule: target - (step/2) -> 800 - 40 = 760
        assert w == 760
        assert w < 800
        assert w >= 640

    def test_clamp_to_boundaries(self):
        """
        Scenario: Shift respects frame boundaries
        """
        config = FocusLensConfig(
            focus_size=640, 
            adaptive=True, 
            shift_step=1000, # Large shift
            smoothing=0.0
        )
        strategy = AdaptiveShiftLensStrategy(config)
        
        # Initialize
        strategy.compute_crop_params(1920, 1080)
        
        # Push right
        detections = sv.Detections(
            xyxy=np.array([[600, 100, 630, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        strategy.update(detections)
        
        x, y, w, h = strategy.compute_crop_params(1920, 1080)
        
        # Max valid x = 1920 - 640 = 1280
        assert x == 1280

    def test_smoothing(self):
        """
        Scenario: Smoothing dampens movement
        """
        config = FocusLensConfig(
            focus_size=640, 
            adaptive=True, 
            shift_step=80,
            smoothing=0.5
        )
        strategy = AdaptiveShiftLensStrategy(config)
        
        # Initialize (center = 960)
        strategy.compute_crop_params(1920, 1080) 
        
        # Update target to 960 + 80 = 1040
        detections = sv.Detections(
            xyxy=np.array([[600, 100, 630, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        strategy.update(detections)
        
        # Compute new pos
        # current = current * 0.5 + target * 0.5
        # 960 * 0.5 + 1040 * 0.5 = 480 + 520 = 1000
        strategy.compute_crop_params(1920, 1080)
        
        assert strategy.current_center_x == 1000.0
