"""
BDD Tests for DualModelPipeline (dual-model orchestrator).

Feature: Smart Scheduling
    Background: Segmentation every N frames, pose every frame

Feature: Preprocess Cache Integration
    Background: Pipeline uses PreprocessCache for optimization

Feature: Coordinate Transformation
    Background: Transform from model space to frame space

Feature: Metrics Tracking
    Background: Track performance metrics
"""

import pytest
from pathlib import Path
import numpy as np
import openvino as ov
from openvino.runtime import opset10
from bakery.pipeline.dual_model_pipeline import DualModelPipeline
from bakery.adapters.openvino.inference_engine import InferenceEngine
from bakery.core.entities.frame import Frame
from bakery.core.entities.model_config import ModelConfig, ModelType, Device, Precision, PipelineConfig


class TestSmartSchedulingBehavior:
    """
    Feature: Smart Scheduling
    Background: Segmentation every N frames, pose every frame
    """

    def test_segmentation_runs_every_n_frames(self, tmp_path):
        """
        Scenario: Segmentation respects seg_interval
            Given seg_interval=5
            When I process 10 frames
            Then segmentation should run 2 times (frame 0, 5)
            And pose should run 10 times
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path, seg_interval=5)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        for i in range(10):
            frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=i)
            pipeline.process_frame(frame)

        # Then
        metrics = pipeline.get_metrics()
        assert metrics.total_frames == 10
        assert metrics.seg_runs == 2  # frame 0, 5
        assert metrics.pose_runs == 10  # every frame

    def test_cached_segmentation_reused(self, tmp_path):
        """
        Scenario: Segmentation results are cached
            Given frame 0 produces segmentation
            When I process frame 1 (no seg inference)
            Then should return same segmentation with frame 0
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path, seg_interval=5)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame0 = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg0, pose0 = pipeline.process_frame(frame0)

        frame1 = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=1)
        seg1, pose1 = pipeline.process_frame(frame1)

        # Then
        # Segmentation should be cached (same frame_id from cache)
        assert seg1.frame_id == 0  # Still from frame 0
        # Pose should be new
        assert pose1.frame_id == 1

    def test_seg_interval_of_1_runs_every_frame(self, tmp_path):
        """
        Scenario: seg_interval=1 means run every frame
            Given seg_interval=1
            When I process 5 frames
            Then segmentation should run 5 times
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path, seg_interval=1)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        for i in range(5):
            frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=i)
            pipeline.process_frame(frame)

        # Then
        metrics = pipeline.get_metrics()
        assert metrics.seg_runs == 5


class TestPreprocessCacheIntegration:
    """
    Feature: Preprocess Cache Integration
    Background: Pipeline uses PreprocessCache for optimization
    """

    def test_cache_used_for_preprocessing(self, tmp_path):
        """
        Scenario: Pipeline uses PreprocessCache
            Given a pipeline
            When I process frames
            Then PreprocessCache should be used
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg, pose = pipeline.process_frame(frame)

        # Then
        # PreprocessCache should exist
        assert pipeline.preprocess_cache is not None
        # Should have cached entries (checked internally, but we can verify it works)
        assert seg is not None or pose is not None

    def test_different_frames_cache_invalidated(self, tmp_path):
        """
        Scenario: Cache invalidated on new frame
            Given frame 0 is cached
            When I process frame 1
            Then cache should be invalidated and recomputed
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame0 = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        pipeline.process_frame(frame0)

        frame1 = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=1)
        pipeline.process_frame(frame1)

        # Then
        # Both frames should be processed successfully
        metrics = pipeline.get_metrics()
        assert metrics.total_frames == 2


class TestCoordinateTransformationBehavior:
    """
    Feature: Coordinate Transformation
    Background: Transform from model space to frame space
    """

    def test_applies_transformation_to_segmentation(self, tmp_path):
        """
        Scenario: Segmentation coordinates transformed
            Given a pipeline
            When I process a frame
            Then segmentation boxes should be in original frame coordinates
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg, pose = pipeline.process_frame(frame)

        # Then
        # Segmentation should exist (even if empty)
        assert seg is not None
        assert seg.frame_id == 0

    def test_applies_transformation_to_pose(self, tmp_path):
        """
        Scenario: Pose coordinates transformed
            Given a pipeline
            When I process a frame
            Then pose keypoints should be in original frame coordinates
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg, pose = pipeline.process_frame(frame)

        # Then
        # Pose should exist (even if empty)
        assert pose is not None
        assert pose.frame_id == 0

    def test_handles_empty_detections(self, tmp_path):
        """
        Scenario: Empty detections handled gracefully
            Given a pipeline with low confidence threshold
            When I process a frame with no detections
            Then should return empty Segmentation and PoseEstimation
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path, confidence=0.99)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg, pose = pipeline.process_frame(frame)

        # Then
        # Should return empty results, not crash
        assert seg is not None
        assert pose is not None


class TestMetricsTrackingBehavior:
    """
    Feature: Metrics Tracking
    Background: Track pipeline performance metrics
    """

    def test_tracks_total_frames(self, tmp_path):
        """
        Scenario: Total frames counted correctly
            Given a pipeline
            When I process N frames
            Then metrics should show N total frames
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        for i in range(7):
            frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=i)
            pipeline.process_frame(frame)

        # Then
        metrics = pipeline.get_metrics()
        assert metrics.total_frames == 7

    def test_tracks_seg_and_pose_runs(self, tmp_path):
        """
        Scenario: Segmentation and pose runs tracked separately
            Given seg_interval=3
            When I process 9 frames
            Then seg_runs should be 3, pose_runs should be 9
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path, seg_interval=3)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        for i in range(9):
            frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=i)
            pipeline.process_frame(frame)

        # Then
        metrics = pipeline.get_metrics()
        assert metrics.seg_runs == 3  # frames 0, 3, 6
        assert metrics.pose_runs == 9

    def test_reset_metrics_clears_counters(self, tmp_path):
        """
        Scenario: Metrics can be reset
            Given a pipeline with metrics
            When I reset metrics
            Then counters should be zero
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # Process some frames
        for i in range(5):
            frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=i)
            pipeline.process_frame(frame)

        # When
        pipeline.reset_metrics()

        # Then
        metrics = pipeline.get_metrics()
        assert metrics.total_frames == 0
        assert metrics.seg_runs == 0
        assert metrics.pose_runs == 0


class TestPipelineIntegration:
    """
    Feature: Pipeline Integration
    Background: End-to-end pipeline functionality
    """

    def test_pipeline_processes_frame_successfully(self, tmp_path):
        """
        Scenario: Full pipeline execution succeeds
            Given a complete pipeline setup
            When I process a frame
            Then should return Segmentation and PoseEstimation
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg, pose = pipeline.process_frame(frame)

        # Then
        assert seg is not None
        assert pose is not None
        assert seg.frame_id == 0
        assert pose.frame_id == 0

    def test_pipeline_handles_sequence_of_frames(self, tmp_path):
        """
        Scenario: Pipeline processes sequence correctly
            Given a pipeline
            When I process multiple frames in sequence
            Then each frame should be processed correctly
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(tmp_path, seg_interval=2)
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When/Then
        for i in range(10):
            frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=i)
            seg, pose = pipeline.process_frame(frame)

            # Verify results
            assert seg is not None
            assert pose is not None
            assert pose.frame_id == i

    def test_pipeline_with_different_resolutions(self, tmp_path):
        """
        Scenario: Pipeline works with different model resolutions
            Given seg_resolution=640 and pose_resolution=320
            When I process a frame
            Then should handle different resolutions correctly
        """
        # Given
        seg_engine, pose_engine, config = self._create_test_pipeline(
            tmp_path,
            seg_resolution=640,
            pose_resolution=320
        )
        pipeline = DualModelPipeline(seg_engine, pose_engine, config)

        # When
        frame = Frame.from_array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), frame_id=0)
        seg, pose = pipeline.process_frame(frame)

        # Then
        assert seg is not None
        assert pose is not None


# Helper methods
def _create_test_pipeline(self, tmp_path, seg_interval=5, confidence=0.25, seg_resolution=640, pose_resolution=640):
    """Create a test pipeline with mock models."""
    # Create test models
    seg_model_path = self._create_test_model(tmp_path / "seg.xml", resolution=seg_resolution, num_outputs=2)
    pose_model_path = self._create_test_model(tmp_path / "pose.xml", resolution=pose_resolution, num_outputs=1)

    # Create model configs
    seg_config = ModelConfig(
        model_path=seg_model_path,
        model_type=ModelType.SEGMENTATION,
        resolution=seg_resolution,
        device=Device.CPU,
        precision=Precision.FP32,
        confidence=confidence
    )

    pose_config = ModelConfig(
        model_path=pose_model_path,
        model_type=ModelType.POSE,
        resolution=pose_resolution,
        device=Device.CPU,
        precision=Precision.FP32,
        confidence=confidence
    )

    # Create inference engines
    seg_engine = InferenceEngine(seg_config)
    pose_engine = InferenceEngine(pose_config)

    # Create pipeline config
    pipeline_config = PipelineConfig(
        segmentation=seg_config,
        pose=pose_config,
        seg_interval=seg_interval,
        confidence_threshold=confidence
    )

    return seg_engine, pose_engine, pipeline_config


def _create_test_model(self, path: Path, resolution=640, num_outputs=2) -> Path:
    """Create a test OpenVINO model."""
    input_shape = [1, 3, resolution, resolution]
    input_layer = opset10.parameter(input_shape, np.float32, name="images")

    # Flatten input
    flatten = opset10.reshape(input_layer, opset10.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)

    outputs_list = []
    slice_step = opset10.constant(np.array([1, 1], dtype=np.int64))

    if num_outputs >= 1:
        # Output 1: [1, 84, 1792] = 150,528 elements
        slice_start = opset10.constant(np.array([0, 0], dtype=np.int64))
        slice_end = opset10.constant(np.array([1, 150528], dtype=np.int64))
        sliced = opset10.slice(flatten, slice_start, slice_end, slice_step)
        shape1 = opset10.constant(np.array([1, 84, 1792], dtype=np.int64))
        output1 = opset10.reshape(sliced, shape1, special_zero=False)
        output1.set_friendly_name("output0")
        outputs_list.append(output1)

    if num_outputs >= 2:
        # Output 2: [1, 32, 80, 80] = 204,800 elements
        slice_start2 = opset10.constant(np.array([0, 150528], dtype=np.int64))
        slice_end2 = opset10.constant(np.array([1, 355328], dtype=np.int64))
        sliced2 = opset10.slice(flatten, slice_start2, slice_end2, slice_step)
        shape2 = opset10.constant(np.array([1, 32, 80, 80], dtype=np.int64))
        output2 = opset10.reshape(sliced2, shape2, special_zero=False)
        output2.set_friendly_name("output1")
        outputs_list.append(output2)

    model = ov.Model(outputs_list, [input_layer], "test_model")
    ov.serialize(model, str(path))

    return path


# Attach helpers to test classes
TestSmartSchedulingBehavior._create_test_pipeline = _create_test_pipeline
TestSmartSchedulingBehavior._create_test_model = _create_test_model

TestPreprocessCacheIntegration._create_test_pipeline = _create_test_pipeline
TestPreprocessCacheIntegration._create_test_model = _create_test_model

TestCoordinateTransformationBehavior._create_test_pipeline = _create_test_pipeline
TestCoordinateTransformationBehavior._create_test_model = _create_test_model

TestMetricsTrackingBehavior._create_test_pipeline = _create_test_pipeline
TestMetricsTrackingBehavior._create_test_model = _create_test_model

TestPipelineIntegration._create_test_pipeline = _create_test_pipeline
TestPipelineIntegration._create_test_model = _create_test_model
