"""
BDD Tests for Model Repository (OpenVINO model discovery and validation).

Feature: Model Discovery
    Background: Repository scans directory for .xml models

Feature: Model Validation
    Background: Validate YOLO segmentation/pose format

Feature: Metadata Extraction
    Background: Extract shapes and precision from models
"""

import pytest
from pathlib import Path
import tempfile
import openvino as ov
from openvino.runtime import opset10
import numpy as np
from bakery.adapters.openvino.model_repository import ModelRepository
from bakery.core.entities.model_config import ModelType, Precision


class TestModelDiscoveryBehavior:
    """
    Feature: Model Discovery
    Background: Repository scans directory for .xml models
    """

    def test_discovers_all_xml_models(self, tmp_path):
        """
        Scenario: All .xml models in directory are found
            Given a directory with 2 valid .xml models
            When I discover models
            Then should return 2 ModelConfig objects
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()

        # Create 2 mock models (segmentation and pose)
        seg_model = self._create_mock_segmentation_model(repo_dir / "seg_model.xml")
        pose_model = self._create_mock_pose_model(repo_dir / "pose_model.xml")

        # When
        repository = ModelRepository(repo_dir)
        models = repository.discover_models()

        # Then
        assert len(models) == 2
        model_types = {model.model_type for model in models}
        assert ModelType.SEGMENTATION in model_types
        assert ModelType.POSE in model_types

    def test_filters_invalid_models(self, tmp_path):
        """
        Scenario: Invalid models are excluded
            Given a directory with 1 valid and 1 invalid model
            When I discover models
            Then should only return 1 valid model
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()

        # Create 1 valid model
        valid_model = self._create_mock_segmentation_model(repo_dir / "valid.xml")

        # Create 1 invalid model (3 outputs - not YOLO format)
        invalid_model = self._create_mock_invalid_model(repo_dir / "invalid.xml")

        # When
        repository = ModelRepository(repo_dir)
        models = repository.discover_models()

        # Then
        assert len(models) == 1
        assert models[0].model_type == ModelType.SEGMENTATION

    def test_handles_empty_directory(self, tmp_path):
        """
        Scenario: Empty directory returns empty list
            Given an empty directory
            When I discover models
            Then should return empty list
        """
        # Given
        repo_dir = tmp_path / "empty"
        repo_dir.mkdir()

        # When
        repository = ModelRepository(repo_dir)
        models = repository.discover_models()

        # Then
        assert len(models) == 0

    def test_discovers_models_recursively(self, tmp_path):
        """
        Scenario: Models in subdirectories are found
            Given models in nested subdirectories
            When I discover models
            Then should find all models recursively
        """
        # Given
        repo_dir = tmp_path / "models"
        subdir1 = repo_dir / "lens1" / "fp16"
        subdir2 = repo_dir / "lens2" / "fp32"
        subdir1.mkdir(parents=True)
        subdir2.mkdir(parents=True)

        # Create models in different subdirectories
        # Note: filenames must match detection patterns (seg_ or -seg for segmentation,
        # pose_ or -pose for pose) otherwise the model type detection fails
        self._create_mock_segmentation_model(subdir1 / "yolo-seg_model.xml")
        self._create_mock_pose_model(subdir2 / "yolo-pose_model.xml")

        # When
        repository = ModelRepository(repo_dir)
        models = repository.discover_models()

        # Then
        assert len(models) == 2


class TestModelValidationBehavior:
    """
    Feature: Model Validation
    Background: Validate YOLO segmentation/pose format
    """

    def test_validates_segmentation_model(self, tmp_path):
        """
        Scenario: Segmentation model has correct outputs
            Given a model with 2 outputs (boxes, masks)
            When I validate
            Then should return True
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        model_path = self._create_mock_segmentation_model(repo_dir / "seg.xml")

        # When
        repository = ModelRepository(repo_dir)
        is_valid = repository.validate_model(model_path)

        # Then
        assert is_valid is True

    def test_validates_pose_model(self, tmp_path):
        """
        Scenario: Pose model has correct output
            Given a model with 1 output (keypoints + boxes)
            When I validate
            Then should return True
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        model_path = self._create_mock_pose_model(repo_dir / "pose.xml")

        # When
        repository = ModelRepository(repo_dir)
        is_valid = repository.validate_model(model_path)

        # Then
        assert is_valid is True

    def test_rejects_invalid_model(self, tmp_path):
        """
        Scenario: Model with wrong number of outputs is rejected
            Given a model with 3 outputs (not YOLO format)
            When I validate
            Then should return False
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        model_path = self._create_mock_invalid_model(repo_dir / "invalid.xml")

        # When
        repository = ModelRepository(repo_dir)
        is_valid = repository.validate_model(model_path)

        # Then
        assert is_valid is False

    def test_rejects_non_xml_file(self, tmp_path):
        """
        Scenario: Non-.xml files are rejected
            Given a .txt file
            When I validate
            Then should return False
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        txt_file = repo_dir / "not_a_model.txt"
        txt_file.write_text("not a model")

        # When
        repository = ModelRepository(repo_dir)
        is_valid = repository.validate_model(txt_file)

        # Then
        assert is_valid is False


class TestMetadataExtractionBehavior:
    """
    Feature: Metadata Extraction
    Background: Extract shapes and precision from models
    """

    def test_extracts_input_shape(self, tmp_path):
        """
        Scenario: Input shape is correctly extracted
            Given a model with input [1, 3, 640, 640]
            When I extract metadata
            Then should return input_shape (1, 3, 640, 640)
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        model_path = self._create_mock_segmentation_model(
            repo_dir / "seg.xml",
            input_shape=[1, 3, 640, 640]
        )

        # When
        repository = ModelRepository(repo_dir)
        metadata = repository.extract_metadata(model_path)

        # Then
        assert metadata["input_shape"] == (1, 3, 640, 640)

    def test_extracts_output_shapes(self, tmp_path):
        """
        Scenario: Output shapes are correctly extracted
            Given a segmentation model
            When I extract metadata
            Then should return 2 output shapes (boxes, masks)
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        model_path = self._create_mock_segmentation_model(repo_dir / "seg.xml")

        # When
        repository = ModelRepository(repo_dir)
        metadata = repository.extract_metadata(model_path)

        # Then
        assert metadata["num_outputs"] == 2
        assert len(metadata["output_shapes"]) == 2

    def test_detects_fp16_precision(self, tmp_path):
        """
        Scenario: FP16 precision detected from path
            Given a model path containing "fp16"
            When I discover models
            Then ModelConfig should have Precision.FP16
        """
        # Given
        repo_dir = tmp_path / "models_fp16"
        repo_dir.mkdir()
        self._create_mock_segmentation_model(repo_dir / "model_fp16.xml")

        # When
        repository = ModelRepository(repo_dir)
        models = repository.discover_models()

        # Then
        assert len(models) == 1
        assert models[0].precision == Precision.FP16


class TestGetModelByTypeBehavior:
    """
    Feature: Get Model By Type
    Background: Find models by their type (segmentation/pose)
    """

    def test_finds_segmentation_model(self, tmp_path):
        """
        Scenario: Finds first segmentation model
            Given a directory with segmentation and pose models
            When I get model by type SEGMENTATION
            Then should return segmentation ModelConfig
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        self._create_mock_segmentation_model(repo_dir / "seg.xml")
        self._create_mock_pose_model(repo_dir / "pose.xml")

        # When
        repository = ModelRepository(repo_dir)
        seg_model = repository.get_model_by_type(ModelType.SEGMENTATION)

        # Then
        assert seg_model is not None
        assert seg_model.model_type == ModelType.SEGMENTATION

    def test_returns_none_when_not_found(self, tmp_path):
        """
        Scenario: Returns None when type not found
            Given a directory with only segmentation models
            When I get model by type POSE
            Then should return None
        """
        # Given
        repo_dir = tmp_path / "models"
        repo_dir.mkdir()
        self._create_mock_segmentation_model(repo_dir / "seg.xml")

        # When
        repository = ModelRepository(repo_dir)
        pose_model = repository.get_model_by_type(ModelType.POSE)

        # Then
        assert pose_model is None


# Helper methods for creating mock OpenVINO models
def _create_mock_segmentation_model(self, path: Path, input_shape=None) -> Path:
    """Create a mock YOLO segmentation model (2 outputs)."""
    if input_shape is None:
        input_shape = [1, 3, 640, 640]

    # Create OpenVINO model with proper structure
    input_layer = opset10.parameter(input_shape, np.float32, name="images")

    # Flatten input: [1, 3, 640, 640] = 1,228,800 elements
    flatten = opset10.reshape(input_layer, opset10.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)

    # Split flatten into two outputs
    # Output 1 (boxes): [1, 84, 1792] = 150,528 elements
    # Output 2 (masks): [1, 32, 80, 80] = 204,800 elements
    # Total = 355,328 (less than 1,228,800, so we just use a slice)

    # For simplicity, just use two different compatible shapes
    # Output 1: Take first 150,528 elements → reshape to [1, 84, 1792]
    slice_start_1 = opset10.constant(np.array([0, 0], dtype=np.int64))
    slice_end_1 = opset10.constant(np.array([1, 150528], dtype=np.int64))
    slice_step = opset10.constant(np.array([1, 1], dtype=np.int64))
    sliced_1 = opset10.slice(flatten, slice_start_1, slice_end_1, slice_step)
    boxes_shape = opset10.constant(np.array([1, 84, 1792], dtype=np.int64))
    output_boxes = opset10.reshape(sliced_1, boxes_shape, special_zero=False)

    # Output 2: Take next 204,800 elements → reshape to [1, 32, 80, 80]
    slice_start_2 = opset10.constant(np.array([0, 150528], dtype=np.int64))
    slice_end_2 = opset10.constant(np.array([1, 355328], dtype=np.int64))
    sliced_2 = opset10.slice(flatten, slice_start_2, slice_end_2, slice_step)
    masks_shape = opset10.constant(np.array([1, 32, 80, 80], dtype=np.int64))
    output_masks = opset10.reshape(sliced_2, masks_shape, special_zero=False)

    output_boxes.set_friendly_name("output0")
    output_masks.set_friendly_name("output1")

    model = ov.Model([output_boxes, output_masks], [input_layer], "segmentation_model")

    # Serialize model
    ov.serialize(model, str(path))

    return path


def _create_mock_pose_model(self, path: Path) -> Path:
    """Create a mock YOLO pose model (1 output)."""
    input_shape = [1, 3, 640, 640]
    input_layer = opset10.parameter(input_shape, np.float32, name="images")

    # Flatten input: [1, 3, 640, 640] = 1,228,800 elements
    flatten = opset10.reshape(input_layer, opset10.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)

    # Reshape to pose output: [1, 56, 21943] = 1,228,808 elements (close, but OpenVINO will complain)
    # Let's use exact: [1, 56, 21942] = 1,228,752 elements (close but not exact)
    # Actually: 1,228,800 / 56 = 21942.857...
    # Let's use [1, 56, 2688] * 8 = 1,204,224 or just take a slice
    # Simpler: [1, 84, 14628] = 1,228,752 or [1, 56, 21943] = 1,228,808 (too many)
    # Use [1, 1228800] reshape directly - no, need 3D
    # Let's use: 1228800 = 56 * 21942 + 48, so use [1, 56, 21942] and slice to that

    slice_start = opset10.constant(np.array([0, 0], dtype=np.int64))
    slice_end = opset10.constant(np.array([1, 1228752], dtype=np.int64))  # 56 * 21942 = 1,228,752
    slice_step = opset10.constant(np.array([1, 1], dtype=np.int64))
    sliced = opset10.slice(flatten, slice_start, slice_end, slice_step)

    pose_shape = opset10.constant(np.array([1, 56, 21942], dtype=np.int64))
    output = opset10.reshape(sliced, pose_shape, special_zero=False)
    output.set_friendly_name("output0")

    model = ov.Model([output], [input_layer], "pose_model")

    # Serialize model
    ov.serialize(model, str(path))

    return path


def _create_mock_invalid_model(self, path: Path) -> Path:
    """Create a mock invalid model (3 outputs - not YOLO format)."""
    input_shape = [1, 3, 640, 640]
    input_layer = opset10.parameter(input_shape, np.float32, name="images")

    # Flatten input: [1, 3, 640, 640] = 1,228,800 elements
    flatten = opset10.reshape(input_layer, opset10.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)

    # Invalid: 3 outputs (YOLO only has 1 or 2)
    # Split elements evenly: 1,228,800 / 3 ≈ 409,600 each
    # Let's use: 100 * 4096 = 409,600 elements per output
    slice_step = opset10.constant(np.array([1, 1], dtype=np.int64))

    # Output 1: [1, 100, 4096] = 409,600
    slice_start_1 = opset10.constant(np.array([0, 0], dtype=np.int64))
    slice_end_1 = opset10.constant(np.array([1, 409600], dtype=np.int64))
    sliced_1 = opset10.slice(flatten, slice_start_1, slice_end_1, slice_step)
    shape1 = opset10.constant(np.array([1, 100, 4096], dtype=np.int64))
    out1 = opset10.reshape(sliced_1, shape1, special_zero=False)

    # Output 2: [1, 200, 2048] = 409,600
    slice_start_2 = opset10.constant(np.array([0, 409600], dtype=np.int64))
    slice_end_2 = opset10.constant(np.array([1, 819200], dtype=np.int64))
    sliced_2 = opset10.slice(flatten, slice_start_2, slice_end_2, slice_step)
    shape2 = opset10.constant(np.array([1, 200, 2048], dtype=np.int64))
    out2 = opset10.reshape(sliced_2, shape2, special_zero=False)

    # Output 3: [1, 50, 8192] = 409,600
    slice_start_3 = opset10.constant(np.array([0, 819200], dtype=np.int64))
    slice_end_3 = opset10.constant(np.array([1, 1228800], dtype=np.int64))
    sliced_3 = opset10.slice(flatten, slice_start_3, slice_end_3, slice_step)
    shape3 = opset10.constant(np.array([1, 50, 8192], dtype=np.int64))
    out3 = opset10.reshape(sliced_3, shape3, special_zero=False)

    model = ov.Model([out1, out2, out3], [input_layer], "invalid_model")

    # Serialize model
    ov.serialize(model, str(path))

    return path


# Add helper methods to test classes
TestModelDiscoveryBehavior._create_mock_segmentation_model = _create_mock_segmentation_model
TestModelDiscoveryBehavior._create_mock_pose_model = _create_mock_pose_model
TestModelDiscoveryBehavior._create_mock_invalid_model = _create_mock_invalid_model

TestModelValidationBehavior._create_mock_segmentation_model = _create_mock_segmentation_model
TestModelValidationBehavior._create_mock_pose_model = _create_mock_pose_model
TestModelValidationBehavior._create_mock_invalid_model = _create_mock_invalid_model

TestMetadataExtractionBehavior._create_mock_segmentation_model = _create_mock_segmentation_model

TestGetModelByTypeBehavior._create_mock_segmentation_model = _create_mock_segmentation_model
TestGetModelByTypeBehavior._create_mock_pose_model = _create_mock_pose_model
