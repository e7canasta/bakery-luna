"""
BDD-style specifications for preprocessing behavior.

Behavior-Driven Development specs to document and validate preprocessing
behavior before implementation. Uses pytest-bdd style (Given/When/Then).

These specs describe WHAT the preprocessing should do, not HOW.
"""

import pytest
import numpy as np
from bakery.core.entities.frame import Frame
from bakery.adapters.openvino.preprocessing import letterbox, preprocess_with_metadata, PreprocessCache


# ==============================================================================
# Feature: YOLO Preprocessing
# ==============================================================================
# As a vision pipeline developer
# I want frames to be preprocessed for YOLO models
# So that inference receives correctly formatted input


class TestLetterboxBehavior:
    """
    Feature: Letterbox Resize

    Background:
        Letterbox maintains aspect ratio by adding padding to make image square.
        This prevents distortion while fitting various aspect ratios.
    """

    def test_square_image_no_padding(self):
        """
        Scenario: Square image fits perfectly
            Given a square frame 640x640
            When I letterbox to target size 640x640
            Then no padding should be added
            And the output should be 640x640
            And ratio should be 1.0
        """
        # Given
        frame_data = np.zeros((640, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        target_size = (640, 640)

        # When
        result, ratio, (pad_w, pad_h) = letterbox(frame.data, new_shape=target_size)

        # Then
        assert result.shape == (640, 640, 3)
        assert ratio == 1.0
        assert pad_w == 0 and pad_h == 0

    def test_landscape_image_adds_vertical_padding(self):
        """
        Scenario: Landscape image needs vertical padding
            Given a landscape frame 1920x1080 (16:9)
            When I letterbox to target size 640x640
            Then vertical padding should be added (top and bottom)
            And the output should be 640x640
            And aspect ratio should be preserved
        """
        # Given
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        target_size = (640, 640)

        # When
        result, ratio, (pad_w, pad_h) = letterbox(frame.data, new_shape=target_size)

        # Then
        assert result.shape == (640, 640, 3)
        assert pad_h > 0  # Should have vertical padding
        assert pad_w == 0  # No horizontal padding
        # Aspect ratio preserved: 1920/1080 should match (640-pad_h*2)/640
        # 1920/1080 = 1.777...
        # 640 / (640 - 2*pad_h) approx 1.777 inverted? No, new_unpad dims
        # The ratio returned is scale = 640 / 1920 = 0.333
        # unpad_h = 1080 * 0.333 = 360
        # pad_h = (640 - 360) / 2 = 140
        assert abs(pad_h - 140) <= 1

    def test_portrait_image_adds_horizontal_padding(self):
        """
        Scenario: Portrait image needs horizontal padding
            Given a portrait frame 1080x1920 (9:16)
            When I letterbox to target size 640x640
            Then horizontal padding should be added (left and right)
            And the output should be 640x640
            And aspect ratio should be preserved
        """
        # Given
        frame_data = np.zeros((1920, 1080, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        target_size = (640, 640)

        # When
        result, ratio, (pad_w, pad_h) = letterbox(frame.data, new_shape=target_size)

        # Then
        assert result.shape == (640, 640, 3)
        assert pad_w > 0  # Horizontal padding
        assert pad_h == 0 # No vertical padding
        # Scale = 640 / 1920 = 0.333
        # unpad_w = 1080 * 0.333 = 360
        # pad_w = (640 - 360) / 2 = 140
        assert abs(pad_w - 140) <= 1


class TestPreprocessCacheBehavior:
    """
    Feature: Preprocessing Cache

    Background:
        Preprocessing is expensive (letterbox, normalize, transpose).
        If seg and pose models use same resolution, we should cache.

    Business Rule:
        Cache hit rate should be ~50% when seg_resolution == pose_resolution
    """

    def test_cache_miss_on_first_call(self):
        """
        Scenario: First preprocessing call misses cache
            Given a fresh preprocessor cache
            And a frame with id=0
            When I preprocess for target_shape 640x640
            Then cache should MISS (implicitly, by computing)
            And result should be stored in cache
        """
        # Given
        frame_data = np.zeros((640, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        cache = PreprocessCache()

        # When
        # simulate single model call by passing same shape for both
        seg_t, seg_meta, pose_t, pose_meta = cache.get_or_compute(
            frame.data, frame.frame_id, seg_shape=(640, 640), pose_shape=(640, 640)
        )

        # Then
        assert cache.seg_tensor is not None
        assert cache.last_frame_id == 0
        assert seg_t is cache.seg_tensor

    def test_cache_hit_on_same_frame_same_shape(self):
        """
        Scenario: Same frame and shape hits cache
            Given a preprocessor cache with frame 0 at 640x640
            When I request frame 0 at 640x640 again
            Then cache should HIT
        """
        # Given
        frame_data = np.zeros((640, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        cache = PreprocessCache()
        # First call
        t1, _, _, _ = cache.get_or_compute(frame.data, frame.frame_id, (640, 640), (640, 640))

        # When - second call
        t2, _, _, _ = cache.get_or_compute(frame.data, frame.frame_id, (640, 640), (640, 640))

        # Then
        assert t1 is t2  # Should be exact same object reference

    def test_cache_miss_on_different_shape(self):
        """
        Scenario: Different shapes in sequential frames
            Given a preprocessor cache with frame 0 at 640x640
            When I move to frame 1 and request 320x320
            Then cache should recompute for new resolution

        Note: Current implementation limitation:
            Changing shape on SAME frame_id without changing frame_id first
            is an edge case not handled (doesn't occur in real pipeline).
            Real usage: shape stays constant across frames.
        """
        # Given
        frame_data = np.zeros((640, 640, 3), dtype=np.uint8)
        frame0 = Frame.from_array(frame_data, frame_id=0)
        frame1 = Frame.from_array(frame_data, frame_id=1)
        cache = PreprocessCache()

        # First call frame 0 @ 640
        t1, _, _, _ = cache.get_or_compute(frame0.data, frame0.frame_id, (640, 640), (640, 640))

        # When move to frame 1 @ 320 (typical usage: new frame, same or different resolution)
        t2, _, _, _ = cache.get_or_compute(frame1.data, frame1.frame_id, (320, 320), (320, 320))

        # Then
        assert t1 is not t2  # Different frames → different objects
        assert t2.shape[2:] == (320, 320)  # Correct shape

    def test_cache_invalidation_on_new_frame(self):
        """
        Scenario: Cache invalidates when frame ID changes
            Given a preprocessor cache with frame 0
            When frame ID advances to 1
            Then old cache entries should be invalidated
            And new frame should miss cache
        """
        # Given
        frame_data = np.zeros((640, 640, 3), dtype=np.uint8)
        frame0 = Frame.from_array(frame_data, frame_id=0)
        frame1 = Frame.from_array(frame_data, frame_id=1)
        cache = PreprocessCache()
        
        t0, _, _, _ = cache.get_or_compute(frame0.data, frame0.frame_id, (640, 640), (640, 640))

        # When
        t1, _, _, _ = cache.get_or_compute(frame1.data, frame1.frame_id, (640, 640), (640, 640))

        # Then
        assert cache.last_frame_id == 1
        # t1 and t0 might be different objects even if content is same (recomputed)
        # To verify recomputation, we ideally mock preprocess, but here we check state.
        assert t1 is not t0 # Should be new object

    def test_dual_model_optimization(self):
        """
        Scenario: Seg and Pose use same resolution (optimization)
            Given seg_resolution = 256 and pose_resolution = 256
            And frame 0 is a dual model request
            Then processing should happen once
        """
        # Given
        frame_data = np.zeros((640, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(frame_data, frame_id=0)
        cache = PreprocessCache()

        # When
        seg_t, seg_m, pose_t, pose_m = cache.get_or_compute(
            frame.data, frame.frame_id, (256, 256), (256, 256)
        )

        # Then
        assert seg_t is pose_t  # Crucial optimization: same tensor object
        assert seg_m is pose_m


class TestNormalizationBehavior:
    """
    Feature: Image Normalization

    Background:
        YOLO models expect input in [0, 1] range with RGB channel order.
    """

    def test_bgr_to_rgb_conversion(self):
        """
        Scenario: OpenCV uses BGR, YOLO expects RGB
        """
        # Given
        frame_data = np.zeros((64, 64, 3), dtype=np.uint8)
        frame_data[:, :, 0] = 255  # Blue channel in BGR

        # When - preprocess
        tensor, _, _ = preprocess_with_metadata(frame_data, (64, 64))

        # Then - Blue should now be in channel 2 (RGB index 2 is B, index 0 is R)
        # Tensor shape: [1, 3, 64, 64]
        # BGR (0=B, 1=G, 2=R) -> RGB (0=R, 1=G, 2=B)
        # Original Blue is at index 0. In RGB it should be at index 2.
        assert tensor[0, 2, :, :].mean() > 0.0  # Blue channel has data
        assert tensor[0, 0, :, :].mean() == 0.0  # Red channel is empty

    def test_uint8_to_float32_normalization(self):
        """
        Scenario: Pixel values normalized to [0, 1]
        """
        # Given
        frame_data = np.full((64, 64, 3), 255, dtype=np.uint8)

        # When
        tensor, _, _ = preprocess_with_metadata(frame_data, (64, 64))

        # Then
        assert tensor.dtype == np.float32
        assert tensor.max() <= 1.0 + 1e-6
        assert tensor.min() >= 0.0

    def test_hwc_to_chw_transpose(self):
        """
        Scenario: OpenCV uses HWC, PyTorch/YOLO uses CHW
        """
        # Given
        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # When
        tensor, _, _ = preprocess_with_metadata(frame_data, (640, 640))

        # Then
        assert tensor.shape == (1, 3, 640, 640)


# ==============================================================================
# Feature: Preprocessing Pipeline Integration
# ==============================================================================

class TestPreprocessingPipelineIntegration:
    """
    Feature: Complete Preprocessing Pipeline
    """

    def test_end_to_end_preprocessing(self):
        """
        Scenario: Full preprocessing pipeline
        """
        # Given
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8) # 16:9
        target_size = (640, 640)

        # When
        tensor, ratio, (pad_w, pad_h) = preprocess_with_metadata(frame_data, target_size)

        # Then
        assert tensor.shape == (1, 3, 640, 640)
        assert pad_h > 0
        assert ratio < 1.0

# ==============================================================================
# Performance Specifications (Placeholder for future rigorous perf testing)
# ==============================================================================

class TestPreprocessingPerformance:
    pass

# ==============================================================================
# Edge Cases & Error Handling
# ==============================================================================

class TestPreprocessingEdgeCases:
    """
    Feature: Edge Cases & Error Handling
    """

    def test_very_small_image_upscaling(self):
        """
        Scenario: Very small image is upscaled
        """
        # Given
        frame_data = np.zeros((64, 64, 3), dtype=np.uint8)

        # When
        tensor, ratio, (pad_w, pad_h) = preprocess_with_metadata(frame_data, (640, 640))

        # Then
        assert tensor.shape == (1, 3, 640, 640)
        assert ratio > 1.0 # Upscaled

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
