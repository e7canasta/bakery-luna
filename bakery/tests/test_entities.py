"""
Unit tests for core entities.
"""

import numpy as np
import pytest
from pathlib import Path
from bakery.core.entities import (
    Frame,
    CropInfo,
    BoundingBox,
    Mask,
    Segmentation,
    KeyPoint,
    Skeleton,
    PoseEstimation,
    ModelConfig,
    PipelineConfig,
    ModelSize,
    ModelType,
    Device,
    Precision,
)


class TestCropInfo:
    """Tests for CropInfo entity."""

    def test_create_valid_crop(self):
        """Test creating valid crop info."""
        crop = CropInfo(x=100, y=100, width=320, height=320)
        assert crop.x == 100
        assert crop.y == 100
        assert crop.width == 320
        assert crop.height == 320
        assert crop.scale_factor == 1.0

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError, match="Crop dimensions must be positive"):
            CropInfo(x=0, y=0, width=0, height=320)

    def test_invalid_scale_factor(self):
        """Test that invalid scale factor raises error."""
        with pytest.raises(ValueError, match="Scale factor must be positive"):
            CropInfo(x=0, y=0, width=320, height=320, scale_factor=-1.0)

    def test_immutable(self):
        """Test that CropInfo is immutable (frozen)."""
        crop = CropInfo(x=100, y=100, width=320, height=320)
        with pytest.raises(Exception):  # FrozenInstanceError
            crop.x = 200


class TestFrame:
    """Tests for Frame entity."""

    def test_create_from_array(self):
        """Test creating frame from numpy array."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=0)

        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame_id == 0
        assert frame.shape == (480, 640, 3)
        assert not frame.is_cropped

    def test_apply_crop(self):
        """Test applying crop to frame."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=0)

        crop = CropInfo(x=100, y=100, width=320, height=320)
        cropped = frame.apply_crop(crop)

        assert cropped.width == 320
        assert cropped.height == 320
        assert cropped.is_cropped
        assert cropped.crop_info == crop

    def test_invalid_dimensions(self):
        """Test that invalid data raises error."""
        # 2D array (missing channel dimension)
        data_2d = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame data must be 3D array"):
            Frame.from_array(data_2d, frame_id=0)

        # Wrong number of channels
        data_4ch = np.zeros((480, 640, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame data must have 3 channels"):
            Frame.from_array(data_4ch, frame_id=0)

    def test_copy(self):
        """Test that copy creates independent frame."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1 = Frame.from_array(data, frame_id=0)
        frame2 = frame1.copy()

        # Modify frame2 data
        frame2.data[0, 0, 0] = 255

        # frame1 should be unchanged
        assert frame1.data[0, 0, 0] == 0


class TestBoundingBox:
    """Tests for BoundingBox entity."""

    def test_create_valid_bbox(self):
        """Test creating valid bounding box."""
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_id=0)
        assert bbox.width == 40
        assert bbox.height == 40
        assert bbox.area == 1600
        assert bbox.center == (30, 30)

    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise error."""
        with pytest.raises(ValueError, match="x2 must be >= x1"):
            BoundingBox(x1=50, y1=10, x2=10, y2=50, confidence=0.9, class_id=0)

    def test_contains_point(self):
        """Test point containment check."""
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_id=0)

        assert bbox.contains_point(30, 30)  # Inside
        assert bbox.contains_point(10, 10)  # On edge
        assert not bbox.contains_point(5, 5)  # Outside

    def test_from_xyxy(self):
        """Test creating bbox from array."""
        xyxy = np.array([10, 10, 50, 50])
        bbox = BoundingBox.from_xyxy(xyxy, confidence=0.9, class_id=0)

        assert bbox.x1 == 10
        assert bbox.y1 == 10
        assert bbox.x2 == 50
        assert bbox.y2 == 50


class TestSegmentation:
    """Tests for Segmentation entity."""

    def test_create_empty(self):
        """Test creating empty segmentation."""
        seg = Segmentation.empty(frame_id=0)
        assert len(seg) == 0

    def test_filter_by_class(self):
        """Test filtering by class ID."""
        bbox1 = BoundingBox(10, 10, 50, 50, 0.9, class_id=0)  # person
        bbox2 = BoundingBox(60, 60, 100, 100, 0.8, class_id=2)  # car
        mask1 = Mask(np.ones((40, 40), dtype=bool), bbox1)
        mask2 = Mask(np.ones((40, 40), dtype=bool), bbox2)

        seg = Segmentation(bboxes=[bbox1, bbox2], masks=[mask1, mask2], frame_id=0)

        # Filter to only persons
        person_seg = seg.filter_by_class([0])
        assert len(person_seg) == 1
        assert person_seg.bboxes[0].class_id == 0

    def test_filter_by_confidence(self):
        """Test filtering by confidence."""
        bbox1 = BoundingBox(10, 10, 50, 50, 0.9, class_id=0)
        bbox2 = BoundingBox(60, 60, 100, 100, 0.3, class_id=0)  # Low confidence
        mask1 = Mask(np.ones((40, 40), dtype=bool), bbox1)
        mask2 = Mask(np.ones((40, 40), dtype=bool), bbox2)

        seg = Segmentation(bboxes=[bbox1, bbox2], masks=[mask1, mask2], frame_id=0)

        # Filter to confidence >= 0.5
        filtered = seg.filter_by_confidence(0.5)
        assert len(filtered) == 1
        assert filtered.bboxes[0].confidence == 0.9


class TestKeyPoint:
    """Tests for KeyPoint entity."""

    def test_create_valid_keypoint(self):
        """Test creating valid keypoint."""
        kp = KeyPoint(x=100, y=200, confidence=0.9)
        assert kp.xy == (100, 200)
        assert kp.is_valid

    def test_create_invalid_keypoint(self):
        """Test creating invalid keypoint."""
        kp = KeyPoint.invalid()
        assert kp.xy == (0, 0)
        assert kp.confidence == 0
        assert not kp.visible
        assert not kp.is_valid


class TestSkeleton:
    """Tests for Skeleton entity."""

    def test_create_valid_skeleton(self):
        """Test creating valid skeleton with 17 keypoints."""
        keypoints = [KeyPoint(i * 10, i * 10, 0.9) for i in range(17)]
        bbox = BoundingBox(0, 0, 100, 100, 0.9, class_id=0)
        skeleton = Skeleton(keypoints=keypoints, bbox=bbox)

        assert len(skeleton.keypoints) == 17
        assert skeleton.num_valid_keypoints == 17

    def test_invalid_keypoint_count(self):
        """Test that wrong number of keypoints raises error."""
        keypoints = [KeyPoint(0, 0, 0.9) for _ in range(10)]  # Only 10 keypoints
        bbox = BoundingBox(0, 0, 100, 100, 0.9, class_id=0)

        with pytest.raises(ValueError, match="COCO skeleton must have 17 keypoints"):
            Skeleton(keypoints=keypoints, bbox=bbox)

    def test_centroid_calculation(self):
        """Test centroid calculation from valid keypoints."""
        # Create skeleton with keypoints at known positions
        keypoints = [KeyPoint(i * 10, i * 10, 0.9 if i < 5 else 0.0) for i in range(17)]
        bbox = BoundingBox(0, 0, 100, 100, 0.9, class_id=0)
        skeleton = Skeleton(keypoints=keypoints, bbox=bbox)

        # Only first 5 keypoints are valid
        centroid = skeleton.centroid
        # Centroid should be average of (0,0), (10,10), (20,20), (30,30), (40,40)
        assert centroid == pytest.approx((20, 20))

    def test_from_array(self):
        """Test creating skeleton from numpy array."""
        # Create array with proper confidence values [0, 1]
        array = np.random.rand(17, 3)  # Random [x, y, conf] in [0, 1]
        array[:, :2] *= 100  # Scale x, y to [0, 100]
        bbox = BoundingBox(0, 0, 100, 100, 0.9, class_id=0)
        skeleton = Skeleton.from_array(array, bbox)

        assert len(skeleton.keypoints) == 17
        assert skeleton.keypoints[0].x == pytest.approx(array[0, 0])


class TestModelConfig:
    """Tests for ModelConfig entity."""

    def test_create_valid_config(self, tmp_path):
        """Test creating valid model config."""
        model_path = tmp_path / "model.xml"
        model_path.touch()  # Create dummy file

        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            model_size=ModelSize.LARGE,
            resolution=256,
            device=Device.GPU,
            precision=Precision.FP16,
            confidence=0.25,
        )

        assert config.model_size == ModelSize.LARGE
        assert config.resolution == 256
        assert config.input_shape == (1, 3, 256, 256)

    def test_invalid_resolution(self, tmp_path):
        """Test that invalid resolution raises error."""
        model_path = tmp_path / "model.xml"
        model_path.touch()

        # Not multiple of 32
        with pytest.raises(ValueError, match="Resolution must be multiple of 32"):
            ModelConfig(
                model_path=model_path,
                model_type=ModelType.SEGMENTATION,
                model_size=ModelSize.SMALL,
                resolution=100,
                device=Device.GPU,
                precision=Precision.FP16,
                confidence=0.25,
            )

    def test_from_args(self, tmp_path):
        """Test creating config from string arguments."""
        model_path = tmp_path / "model.xml"
        model_path.touch()

        config = ModelConfig.from_args(
            model_path=model_path,
            model_type="segmentation",
            model_size="l",
            resolution=256,
            device="GPU",
            precision="fp16",
            confidence=0.25
        )

        assert config.model_size == ModelSize.LARGE
        assert config.device == Device.GPU
        assert config.precision == Precision.FP16
        assert config.model_type == ModelType.SEGMENTATION


class TestPipelineConfig:
    """Tests for PipelineConfig entity."""

    def test_create_valid_config(self, tmp_path):
        """Test creating valid pipeline config."""
        seg_path = tmp_path / "seg.xml"
        pose_path = tmp_path / "pose.xml"
        seg_path.touch()
        pose_path.touch()

        seg_config = ModelConfig.from_args(seg_path, "segmentation", 192, "CPU", "int8", 0.25, "s")
        pose_config = ModelConfig.from_args(pose_path, "pose", 256, "GPU", "fp16", 0.25, "l")

        pipeline_config = PipelineConfig(
            segmentation=seg_config,
            pose=pose_config,
            seg_interval=5,
            filter_keypoints=True,
        )

        assert pipeline_config.seg_interval == 5
        assert pipeline_config.filter_keypoints
        assert not pipeline_config.is_same_resolution  # 192 != 256

    def test_same_resolution_detection(self, tmp_path):
        """Test detection of same resolution for both models."""
        seg_path = tmp_path / "seg.xml"
        pose_path = tmp_path / "pose.xml"
        seg_path.touch()
        pose_path.touch()

        seg_config = ModelConfig.from_args(seg_path, "segmentation", 256, "CPU", "int8", 0.25, "s")
        pose_config = ModelConfig.from_args(pose_path, "pose", 256, "GPU", "fp16", 0.25, "l")

        pipeline_config = PipelineConfig(segmentation=seg_config, pose=pose_config)
        assert pipeline_config.is_same_resolution

    def test_invalid_seg_interval(self, tmp_path):
        """Test that invalid seg_interval raises error."""
        seg_path = tmp_path / "seg.xml"
        pose_path = tmp_path / "pose.xml"
        seg_path.touch()
        pose_path.touch()

        seg_config = ModelConfig.from_args(seg_path, "segmentation", 192, "CPU", "int8", 0.25, "s")
        pose_config = ModelConfig.from_args(pose_path, "pose", 256, "GPU", "fp16", 0.25, "l")

        with pytest.raises(ValueError, match="seg_interval must be >= 1"):
            PipelineConfig(segmentation=seg_config, pose=pose_config, seg_interval=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
