"""
Tests for Focus Lens functionality.

BDD-style tests for FocusLensConfig entity and focus lens utilities.
"""

import pytest
import numpy as np
import supervision as sv

from bakery.core.entities import Frame, CropInfo, FocusLensConfig
from bakery.utils.focus_lens import (
    apply_focus_lens,
    map_detections_to_full_frame,
    map_keypoints_to_full_frame,
    crop_info_to_tuple,
)


class TestFocusLensConfig:
    """Feature: FocusLensConfig entity validation."""

    def test_valid_config_default_strategy(self):
        """
        Scenario: Create config with valid focus_size
            Given focus_size=640 (multiple of 80)
            When creating FocusLensConfig
            Then config should be created with default zoom strategy
        """
        config = FocusLensConfig(focus_size=640)

        assert config.focus_size == 640
        assert config.focus_x is None
        assert config.focus_y is None
        assert config.strategy == "zoom"
        assert config.is_centered is True

    def test_valid_config_with_position(self):
        """
        Scenario: Create config with explicit position
            Given focus_size=480 and position (100, 200)
            When creating FocusLensConfig
            Then config should have specified position
        """
        config = FocusLensConfig(focus_size=480, focus_x=100, focus_y=200)

        assert config.focus_size == 480
        assert config.focus_x == 100
        assert config.focus_y == 200
        assert config.is_centered is False

    def test_valid_config_pad_strategy(self):
        """
        Scenario: Create config with pad strategy
            Given focus_size=320 and strategy="pad"
            When creating FocusLensConfig
            Then config should use pad strategy
        """
        config = FocusLensConfig(focus_size=320, strategy="pad")

        assert config.strategy == "pad"

    def test_invalid_focus_size_not_multiple_of_80(self):
        """
        Scenario: Reject focus_size not multiple of 80
            Given focus_size=650 (not multiple of 80)
            When creating FocusLensConfig
            Then should raise ValueError
        """
        with pytest.raises(ValueError, match="multiple of 80"):
            FocusLensConfig(focus_size=650)

    def test_invalid_focus_size_zero(self):
        """
        Scenario: Reject zero focus_size
            Given focus_size=0
            When creating FocusLensConfig
            Then should raise ValueError
        """
        with pytest.raises(ValueError, match="positive"):
            FocusLensConfig(focus_size=0)

    def test_invalid_focus_size_negative(self):
        """
        Scenario: Reject negative focus_size
            Given focus_size=-80
            When creating FocusLensConfig
            Then should raise ValueError
        """
        with pytest.raises(ValueError, match="positive"):
            FocusLensConfig(focus_size=-80)

    def test_invalid_strategy(self):
        """
        Scenario: Reject invalid strategy
            Given strategy="stretch"
            When creating FocusLensConfig
            Then should raise ValueError
        """
        with pytest.raises(ValueError, match="zoom.*pad"):
            FocusLensConfig(focus_size=640, strategy="stretch")

    def test_invalid_focus_x_negative(self):
        """
        Scenario: Reject negative focus_x
            Given focus_x=-10
            When creating FocusLensConfig
            Then should raise ValueError
        """
        with pytest.raises(ValueError, match="focus_x"):
            FocusLensConfig(focus_size=640, focus_x=-10)

    def test_immutable(self):
        """
        Scenario: Config is immutable (frozen dataclass)
            Given a FocusLensConfig
            When trying to modify it
            Then should raise FrozenInstanceError
        """
        config = FocusLensConfig(focus_size=640)

        with pytest.raises(Exception):  # FrozenInstanceError
            config.focus_size = 320

    def test_all_valid_multiples_of_80(self):
        """
        Scenario: Accept all valid multiples of 80
            Given various multiples of 80
            When creating FocusLensConfig
            Then all should succeed
        """
        valid_sizes = [80, 160, 240, 320, 400, 480, 560, 640, 720, 800]

        for size in valid_sizes:
            config = FocusLensConfig(focus_size=size)
            assert config.focus_size == size


class TestApplyFocusLens:
    """Feature: Apply focus lens to frame."""

    def test_crop_centered(self):
        """
        Scenario: Crop frame with centered focus lens
            Given 1920x1080 frame, focus_size=640
            When applying focus lens (centered)
            Then crop should be at center: x=640, y=220
        """
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        config = FocusLensConfig(focus_size=640)

        cropped, crop_info = apply_focus_lens(frame, config)

        assert cropped.width == 640
        assert cropped.height == 640
        assert crop_info.x == 640  # (1920 - 640) // 2
        assert crop_info.y == 220  # (1080 - 640) // 2
        assert crop_info.scale_factor == 1.0

    def test_crop_at_position(self):
        """
        Scenario: Crop frame at specified position
            Given 1920x1080 frame, focus_size=480, position (100, 50)
            When applying focus lens
            Then crop should be at specified position
        """
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        config = FocusLensConfig(focus_size=480, focus_x=100, focus_y=50)

        cropped, crop_info = apply_focus_lens(frame, config)

        assert cropped.width == 480
        assert cropped.height == 480
        assert crop_info.x == 100
        assert crop_info.y == 50

    def test_crop_position_clamped_to_boundary(self):
        """
        Scenario: Clamp crop position to frame boundaries
            Given 1920x1080 frame, focus_size=640, position (2000, 2000)
            When applying focus lens
            Then position should be clamped to valid range
        """
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        config = FocusLensConfig(focus_size=640, focus_x=2000, focus_y=2000)

        cropped, crop_info = apply_focus_lens(frame, config)

        # Should be clamped to max valid position
        assert crop_info.x == 1920 - 640  # 1280
        assert crop_info.y == 1080 - 640  # 440

    def test_zoom_strategy_small_frame(self):
        """
        Scenario: Zoom strategy on frame smaller than focus_size
            Given 400x400 frame, focus_size=640, strategy="zoom"
            When applying focus lens
            Then frame should be scaled up and scale_factor > 1
        """
        frame_data = np.zeros((400, 400, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        config = FocusLensConfig(focus_size=640, strategy="zoom")

        cropped, crop_info = apply_focus_lens(frame, config)

        assert cropped.width == 640
        assert cropped.height == 640
        assert crop_info.scale_factor == 1.6  # 640 / 400

    def test_pad_strategy_small_frame(self):
        """
        Scenario: Pad strategy on frame smaller than focus_size
            Given 400x400 frame, focus_size=640, strategy="pad"
            When applying focus lens
            Then frame should be padded and crop extracted
        """
        frame_data = np.ones((400, 400, 3), dtype=np.uint8) * 128
        frame = Frame.from_array(frame_data, frame_id=0)
        config = FocusLensConfig(focus_size=640, strategy="pad")

        cropped, crop_info = apply_focus_lens(frame, config)

        assert cropped.width == 640
        assert cropped.height == 640
        assert crop_info.scale_factor == 1.0  # No scaling with pad

    def test_cropped_frame_has_correct_data(self):
        """
        Scenario: Cropped frame contains correct pixel data
            Given frame with colored region
            When applying focus lens over that region
            Then cropped frame should contain the colored pixels
        """
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw white square at center
        frame_data[190:290, 270:370] = 255
        frame = Frame.from_array(frame_data, frame_id=0)
        config = FocusLensConfig(focus_size=320)

        cropped, crop_info = apply_focus_lens(frame, config)

        # Center of cropped should have white pixels
        center = cropped.data[160, 160]
        assert np.all(center == 255)


class TestMapDetectionsToFullFrame:
    """Feature: Map detections from crop to full frame."""

    def test_offset_bounding_boxes(self):
        """
        Scenario: Offset bounding boxes by crop position
            Given detection at (10, 10, 50, 50) in crop
            And crop at (100, 100)
            When mapping to full frame
            Then bbox should be at (110, 110, 150, 150)
        """
        crop_info = CropInfo(x=100, y=100, width=640, height=640)
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )

        mapped = map_detections_to_full_frame(detections, crop_info, 1920, 1080)

        np.testing.assert_array_almost_equal(
            mapped.xyxy[0],
            [110, 110, 150, 150]
        )

    def test_expand_masks_to_full_frame(self):
        """
        Scenario: Expand masks to full frame size
            Given 640x640 mask in crop
            And crop at (100, 100)
            And full frame is 1920x1080
            When mapping to full frame
            Then mask should be 1920x1080 with content at crop position
        """
        crop_info = CropInfo(x=100, y=100, width=640, height=640)

        # Create mask with square in center
        mask_data = np.zeros((640, 640), dtype=bool)
        mask_data[300:340, 300:340] = True

        detections = sv.Detections(
            xyxy=np.array([[300, 300, 340, 340]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            mask=np.array([mask_data])
        )

        mapped = map_detections_to_full_frame(detections, crop_info, 1920, 1080)

        # Mask should be at offset position in full frame
        assert mapped.mask.shape == (1, 1080, 1920)
        assert mapped.mask[0, 400, 400] == True  # 100 + 300
        assert mapped.mask[0, 0, 0] == False  # Outside crop region

    def test_empty_detections(self):
        """
        Scenario: Handle empty detections
            Given empty sv.Detections
            When mapping to full frame
            Then should return empty detections
        """
        crop_info = CropInfo(x=100, y=100, width=640, height=640)
        detections = sv.Detections.empty()

        mapped = map_detections_to_full_frame(detections, crop_info, 1920, 1080)

        assert len(mapped) == 0

    def test_apply_inverse_scale_factor(self):
        """
        Scenario: Apply inverse scale when zoom was used
            Given detection at (100, 100, 200, 200) in crop
            And crop with scale_factor=2.0
            When mapping to full frame
            Then coordinates should be divided by scale
        """
        crop_info = CropInfo(x=0, y=0, width=640, height=640, scale_factor=2.0)
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )

        mapped = map_detections_to_full_frame(detections, crop_info, 960, 540)

        # Coordinates should be halved
        np.testing.assert_array_almost_equal(
            mapped.xyxy[0],
            [50, 50, 100, 100]
        )


class TestMapKeypointsToFullFrame:
    """Feature: Map keypoints from crop to full frame."""

    def test_offset_visible_keypoints(self):
        """
        Scenario: Offset visible keypoints by crop position
            Given visible keypoint at (50, 50) in crop
            And crop at (100, 100)
            When mapping to full frame
            Then keypoint should be at (150, 150)
        """
        crop_info = CropInfo(x=100, y=100, width=640, height=640)

        # Single person with 17 keypoints
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        xy[0, 0] = [50, 50]  # Nose
        xy[0, 5] = [100, 150]  # Left shoulder

        keypoints = sv.KeyPoints(
            xy=xy,
            confidence=np.ones((1, 17), dtype=np.float32),
            class_id=np.zeros(1, dtype=int)
        )

        mapped = map_keypoints_to_full_frame(keypoints, crop_info)

        # Keypoints should be offset
        np.testing.assert_array_almost_equal(mapped.xy[0, 0], [150, 150])
        np.testing.assert_array_almost_equal(mapped.xy[0, 5], [200, 250])

    def test_preserve_invisible_keypoints(self):
        """
        Scenario: Preserve invisible keypoints at origin
            Given invisible keypoint at (0, 0)
            When mapping to full frame
            Then keypoint should remain at (0, 0)
        """
        crop_info = CropInfo(x=100, y=100, width=640, height=640)

        xy = np.zeros((1, 17, 2), dtype=np.float32)
        xy[0, 0] = [50, 50]  # Visible
        xy[0, 1] = [0, 0]    # Invisible (should stay at origin)

        keypoints = sv.KeyPoints(
            xy=xy,
            confidence=np.array([[0.9, 0.0] + [0.5] * 15], dtype=np.float32),
            class_id=np.zeros(1, dtype=int)
        )

        mapped = map_keypoints_to_full_frame(keypoints, crop_info)

        # Visible keypoint offset
        np.testing.assert_array_almost_equal(mapped.xy[0, 0], [150, 150])
        # Invisible keypoint stays at origin
        np.testing.assert_array_almost_equal(mapped.xy[0, 1], [0, 0])

    def test_empty_keypoints(self):
        """
        Scenario: Handle empty keypoints
            Given empty sv.KeyPoints
            When mapping to full frame
            Then should return empty keypoints
        """
        crop_info = CropInfo(x=100, y=100, width=640, height=640)
        keypoints = sv.KeyPoints.empty()

        mapped = map_keypoints_to_full_frame(keypoints, crop_info)

        assert len(mapped) == 0

    def test_apply_inverse_scale_factor(self):
        """
        Scenario: Apply inverse scale when zoom was used
            Given keypoint at (100, 100)
            And crop with scale_factor=2.0
            When mapping to full frame
            Then coordinates should be divided by scale
        """
        crop_info = CropInfo(x=0, y=0, width=640, height=640, scale_factor=2.0)

        xy = np.zeros((1, 17, 2), dtype=np.float32)
        xy[0, 0] = [100, 100]

        keypoints = sv.KeyPoints(
            xy=xy,
            confidence=np.ones((1, 17), dtype=np.float32),
            class_id=np.zeros(1, dtype=int)
        )

        mapped = map_keypoints_to_full_frame(keypoints, crop_info)

        # Coordinates should be halved
        np.testing.assert_array_almost_equal(mapped.xy[0, 0], [50, 50])


class TestCropInfoToTuple:
    """Feature: Convert CropInfo to tuple for annotator."""

    def test_convert_crop_info(self):
        """
        Scenario: Convert CropInfo to tuple
            Given CropInfo(x=100, y=200, width=640, height=480)
            When converting to tuple
            Then should return (100, 200, 640, 480)
        """
        crop_info = CropInfo(x=100, y=200, width=640, height=480)

        result = crop_info_to_tuple(crop_info)

        assert result == (100, 200, 640, 480)

    def test_none_returns_none(self):
        """
        Scenario: Handle None input
            Given None
            When converting to tuple
            Then should return None
        """
        result = crop_info_to_tuple(None)

        assert result is None
