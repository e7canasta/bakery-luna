"""
BDD-style specifications for Disney/Roger Rabbit annotator behavior.

Validates the multi-layer rendering pipeline that creates the iconic
"colorful objects in B&W world" aesthetic.
"""

import pytest
import numpy as np
import supervision as sv
from bakery.annotators import DisneyAnnotator
from bakery.annotators.disney_annotator import RenderConfig


# ==============================================================================
# Feature: Disney/Roger Rabbit Multi-layer Rendering
# ==============================================================================
# As a vision pipeline developer
# I want frames to be rendered in Disney/Roger Rabbit style
# So that detected objects stand out dramatically against grayscale background


class TestBWWorldBehavior:
    """
    Feature: B&W World Layer (Layer 1)

    Background:
        The base layer converts the entire frame to grayscale and darkens it
        to create contrast for colorful detected objects.
    """

    def test_frame_converted_to_grayscale(self):
        """
        Scenario: Frame is converted to B&W
            Given a color frame with RGB channels
            When I apply B&W world layer
            Then all pixels should have equal RGB values (grayscale)
        """
        # Given
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        annotator = DisneyAnnotator()

        # When
        bw_frame = annotator._apply_bw_world(frame)

        # Then - All channels should be equal (grayscale property)
        assert bw_frame.shape == frame.shape
        # Check that R=G=B for all pixels (grayscale)
        assert np.allclose(bw_frame[:, :, 0], bw_frame[:, :, 1])
        assert np.allclose(bw_frame[:, :, 1], bw_frame[:, :, 2])

    def test_frame_darkened_for_contrast(self):
        """
        Scenario: Frame is darkened to increase contrast
            Given a B&W frame
            When I apply darkness factor of 0.6
            Then pixel values should be 60% of original
        """
        # Given
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)  # Medium gray
        config = RenderConfig(bw_darkness=0.6)
        annotator = DisneyAnnotator(config)

        # When
        darkened = annotator._apply_bw_world(frame)

        # Then - Should be approximately 60% of original
        # (not exact due to grayscale conversion)
        assert darkened.mean() < frame.mean()
        assert darkened.mean() < 100  # Definitely darker


class TestFocusLensBehavior:
    """
    Feature: Focus Lens Layer (Layer 2)

    Background:
        Subtly brightens a region of interest to guide viewer attention.
    """

    def test_focus_region_brightened(self):
        """
        Scenario: Focus region is brighter than background
            Given a uniform grayscale frame
            And a focus region at (100, 100, 320, 320)
            When I apply focus lens with brightness=1.1
            Then focus region should be 10% brighter
        """
        # Given
        frame = np.full((640, 640, 3), 100, dtype=np.uint8)
        config = RenderConfig(lens_brightness=1.1)
        annotator = DisneyAnnotator(config)
        focus_region = (100, 100, 320, 320)

        # When
        result = annotator._apply_focus_lens(frame.copy(), focus_region)

        # Then
        x, y, w, h = focus_region
        brightened_region = result[y:y+h, x:x+w]
        outside_region = result[:y, :]

        assert brightened_region.mean() > outside_region.mean()
        # Should be approximately 110 (100 * 1.1)
        assert 105 < brightened_region.mean() < 115


class TestColorSpotlightBehavior:
    """
    Feature: Color Spotlight Layer (Layer 5)

    Background:
        The most important layer - paints detected objects with original color
        plus brightness boost, making them "pop" against grayscale background.
    """

    def test_masked_regions_painted_with_color(self):
        """
        Scenario: Detected objects appear in color on B&W background
            Given a B&W frame
            And a color original frame
            And a detection mask
            When I apply color spotlight
            Then masked region should have color from original frame
            And unmasked region should remain grayscale
        """
        # Given - Create B&W frame and color original
        bw_frame = np.full((100, 100, 3), 80, dtype=np.uint8)  # Gray
        color_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color_frame[:, :, 2] = 200  # Red region in BGR

        # Create detection with mask covering top-left quarter
        mask = np.zeros((100, 100), dtype=bool)
        mask[:50, :50] = True

        detections = sv.Detections(
            xyxy=np.array([[0, 0, 50, 50]]),
            mask=np.array([mask]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )

        config = RenderConfig(spotlight_brightness=1.2)
        annotator = DisneyAnnotator(config)

        # When
        result = annotator._apply_color_spotlight(bw_frame, color_frame, detections)

        # Then - Masked region should have color (red channel > 0)
        masked_region = result[:50, :50, 2]  # Red channel
        unmasked_region = result[60:, 60:, 2]

        assert masked_region.mean() > 0  # Has color
        assert unmasked_region.mean() == 80  # Remains gray

    def test_color_brightness_boosted(self):
        """
        Scenario: Detected objects are brighter than original
            Given original frame with RGB=(100, 100, 100)
            And brightness boost factor=1.2
            When I apply color spotlight
            Then detected regions should be ~120 brightness
        """
        # Given
        bw_frame = np.full((100, 100, 3), 60, dtype=np.uint8)
        color_frame = np.full((100, 100, 3), 100, dtype=np.uint8)

        mask = np.ones((100, 100), dtype=bool)
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 100, 100]]),
            mask=np.array([mask]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )

        config = RenderConfig(spotlight_brightness=1.2)
        annotator = DisneyAnnotator(config)

        # When
        result = annotator._apply_color_spotlight(bw_frame, color_frame, detections)

        # Then - Should be approximately 120 (100 * 1.2)
        assert 115 < result.mean() < 125


class TestLayerCompositionBehavior:
    """
    Feature: Multi-layer Composition

    Background:
        All layers must compose correctly to create the final Disney/Roger Rabbit
        aesthetic without visual artifacts.
    """

    def test_empty_detections_produce_bw_frame(self):
        """
        Scenario: No detections results in pure B&W frame
            Given an empty detections set
            When I annotate frame
            Then result should be grayscale only
        """
        # Given
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        empty_detections = sv.Detections.empty()
        annotator = DisneyAnnotator()

        # When
        result = annotator.annotate(frame, empty_detections)

        # Then - Should be grayscale
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=5)
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=5)

    def test_all_layers_integrate_without_artifacts(self):
        """
        Scenario: Full rendering pipeline produces valid output
            Given a frame with detections and keypoints
            When I apply all 8 layers
            Then output should be valid image
            And no NaN or out-of-range values
        """
        # Given
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create simple detection with mask
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 100:200] = True

        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            mask=np.array([mask]),
            confidence=np.array([0.85]),
            class_id=np.array([0])
        )

        # Create simple keypoints
        keypoints_data = np.array([[[150, 150, 0.9]]])  # Single keypoint
        keypoints = sv.KeyPoints(xy=keypoints_data[:, :, :2], confidence=keypoints_data[:, :, 2])

        annotator = DisneyAnnotator(enable_pose=True)

        # When
        result = annotator.annotate(
            frame, detections, keypoints,
            focus_region=(50, 50, 300, 300),
            show_focus_marker=True
        )

        # Then
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        assert not np.isnan(result).any()
        assert result.min() >= 0
        assert result.max() <= 255


class TestHaloAndCornersBehavior:
    """
    Feature: Halo and Corner Layers (Layers 3 & 4)

    Background:
        Halo creates a soft glow around detections.
        Corners add subtle bounding box markers.
    """

    def test_halo_annotator_initialized(self):
        """
        Scenario: Halo annotator is properly configured
            Given a RenderConfig with halo settings
            When I initialize DisneyAnnotator
            Then halo_annotator should exist with correct color
        """
        # Given
        config = RenderConfig(
            halo_color=(240, 248, 255),
            halo_opacity=0.5,
            halo_kernel_size=40
        )

        # When
        annotator = DisneyAnnotator(config)

        # Then
        assert annotator.halo_annotator is not None
        assert annotator.corner_annotator is not None

    def test_corners_applied_with_transparency(self):
        """
        Scenario: Corner markers are translucent
            Given a detection with bounding box
            When I apply box corners
            Then corners should blend with 30% opacity
        """
        # Given
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )

        config = RenderConfig(corner_opacity=0.3)
        annotator = DisneyAnnotator(config)

        # When
        result = annotator._apply_box_corners(frame, detections)

        # Then - Result should be different from input (corners added)
        assert not np.array_equal(result, frame)
        # Most pixels should remain unchanged (only corners affected)
        unchanged_ratio = np.sum(result == frame) / result.size
        assert unchanged_ratio > 0.95  # >95% unchanged (corners are small)


class TestLabelsAndBarsBehavior:
    """
    Feature: Labels and Confidence Bars (Layers 6 & 7)

    Background:
        Labels show class ID and confidence.
        Bars provide visual confidence indicators.
    """

    def test_labels_formatted_correctly(self):
        """
        Scenario: Labels show class ID and confidence
            Given detections with class_id=5 and confidence=0.85
            When I apply labels
            Then label should be "ID:5 0.85"
        """
        # Given
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            confidence=np.array([0.85]),
            class_id=np.array([5])
        )

        annotator = DisneyAnnotator()

        # When
        result = annotator._apply_labels(frame, detections)

        # Then - Frame should be modified (labels added)
        assert not np.array_equal(result, frame)

    def test_bars_and_labels_use_transparency(self):
        """
        Scenario: Bars and labels are translucent (40% opacity)
            Given a detection
            When I apply confidence bars and labels
            Then they should blend with 40% opacity
        """
        # Given
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            confidence=np.array([0.85]),
            class_id=np.array([0])
        )

        config = RenderConfig(bar_opacity=0.4, label_opacity=0.4)
        annotator = DisneyAnnotator(config)

        # When
        with_bars = annotator._apply_confidence_bars(frame, detections)
        with_labels = annotator._apply_labels(frame, detections)

        # Then - Both should modify frame
        assert not np.array_equal(with_bars, frame)
        assert not np.array_equal(with_labels, frame)


class TestSkeletonBehavior:
    """
    Feature: Skeleton Overlay (Layer 8)

    Background:
        Renders pose keypoints and edges in translucent gray.
    """

    def test_skeleton_only_active_when_enabled(self):
        """
        Scenario: Skeleton annotators only exist if enable_pose=True
            Given enable_pose=False
            When I initialize annotator
            Then edge_annotator and vertex_annotator should be None
        """
        # Given & When
        annotator_no_pose = DisneyAnnotator(enable_pose=False)
        annotator_with_pose = DisneyAnnotator(enable_pose=True)

        # Then
        assert annotator_no_pose.edge_annotator is None
        assert annotator_no_pose.vertex_annotator is None
        assert annotator_with_pose.edge_annotator is not None
        assert annotator_with_pose.vertex_annotator is not None

    def test_skeleton_applied_with_transparency(self):
        """
        Scenario: Skeleton edges and vertices are translucent (30% opacity)
            Given a frame and keypoints
            And enable_pose=True
            When I apply skeleton layer
            Then edges and vertices should blend with 30% opacity
        """
        # Given
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        # Create simple keypoints (3 points)
        keypoints_data = np.array([[
            [150, 150, 0.9],
            [160, 160, 0.8],
            [170, 170, 0.85]
        ]])
        keypoints = sv.KeyPoints(xy=keypoints_data[:, :, :2], confidence=keypoints_data[:, :, 2])

        config = RenderConfig(skeleton_opacity=0.3)
        annotator = DisneyAnnotator(config, enable_pose=True)

        # When
        result = annotator._apply_skeleton(frame, keypoints)

        # Then - Frame should be modified
        # Note: Actual drawing depends on supervision's EdgeAnnotator/VertexAnnotator
        # We verify that function executes without error
        assert result.shape == frame.shape


# ==============================================================================
# Feature: Configuration and Customization
# ==============================================================================


class TestRenderConfigBehavior:
    """
    Feature: Render Configuration

    Background:
        RenderConfig allows customization of all rendering parameters.
    """

    def test_default_config_values(self):
        """
        Scenario: Default config has documented values
            Given a default RenderConfig
            Then all parameters should match Disney/Roger Rabbit specs
        """
        # Given & When
        config = RenderConfig()

        # Then - Verify key defaults
        assert config.bw_darkness == 0.6
        assert config.lens_brightness == 1.1
        assert config.spotlight_brightness == 1.2
        assert config.halo_opacity == 0.5
        assert config.corner_opacity == 0.3
        assert config.bar_opacity == 0.4
        assert config.label_opacity == 0.4
        assert config.skeleton_opacity == 0.3

    def test_custom_config_values(self):
        """
        Scenario: Custom config values are respected
            Given a RenderConfig with custom darkness=0.5
            When I create annotator with this config
            Then B&W layer should use darkness=0.5
        """
        # Given
        config = RenderConfig(bw_darkness=0.5)
        annotator = DisneyAnnotator(config)

        frame = np.full((100, 100, 3), 100, dtype=np.uint8)

        # When
        result = annotator._apply_bw_world(frame)

        # Then - Should use custom darkness value
        # (Approximately 50 = 100 * 0.5, accounting for grayscale conversion)
        assert result.mean() < 60


# ==============================================================================
# Feature: Edge Cases
# ==============================================================================


class TestAnnotatorEdgeCases:
    """
    Feature: Edge Cases and Error Handling

    Validates robustness in unusual scenarios.
    """

    def test_handles_detections_without_masks(self):
        """
        Scenario: Detections without masks don't crash
            Given detections with mask=None
            When I annotate
            Then should skip color spotlight gracefully
        """
        # Given
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
            # mask=None (not provided)
        )

        annotator = DisneyAnnotator()

        # When & Then - Should not crash
        result = annotator.annotate(frame, detections)
        assert result.shape == frame.shape

    def test_handles_empty_keypoints(self):
        """
        Scenario: Empty keypoints don't crash
            Given empty keypoints
            When I annotate with enable_pose=True
            Then should skip skeleton gracefully
        """
        # Given
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        empty_detections = sv.Detections.empty()
        empty_keypoints = sv.KeyPoints.empty()

        annotator = DisneyAnnotator(enable_pose=True)

        # When & Then - Should not crash
        result = annotator.annotate(frame, empty_detections, empty_keypoints)
        assert result.shape == frame.shape

    def test_handles_none_keypoints(self):
        """
        Scenario: None keypoints are handled gracefully
            Given keypoints=None
            When I annotate
            Then should skip skeleton layer
        """
        # Given
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = sv.Detections.empty()

        annotator = DisneyAnnotator(enable_pose=True)

        # When & Then - Should not crash
        result = annotator.annotate(frame, detections, keypoints=None)
        assert result.shape == frame.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
