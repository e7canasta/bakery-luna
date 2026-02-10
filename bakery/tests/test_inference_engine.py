"""
BDD Tests for Inference Engine (OpenVINO inference wrapper).

Feature: OpenVINO Inference
    Background: Engine compiles and runs models

Feature: Device Management
    Background: Automatic GPU→CPU fallback

Feature: Model Metadata
    Background: Extract shapes and device info
"""

import pytest
from pathlib import Path
import numpy as np
import openvino as ov
from openvino.runtime import opset10
from bakery.adapters.openvino.inference_engine import InferenceEngine
from bakery.core.entities.model_config import ModelConfig, ModelType, Device, Precision


class TestEngineCompilationBehavior:
    """
    Feature: Engine Compilation
    Background: Engine compiles models for target device
    """

    def test_compiles_model_successfully(self, tmp_path):
        """
        Scenario: Model compilation succeeds
            Given a valid ModelConfig
            When I create InferenceEngine
            Then model should be compiled
            And device should be set
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml")
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,  # Use CPU to ensure it works
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)

        # Then
        assert engine.compiled_model is not None
        assert engine.device is not None
        assert engine.device in ["GPU", "CPU"]

    def test_extracts_input_shape_correctly(self, tmp_path):
        """
        Scenario: Input shape extracted from model
            Given a model with input [1, 3, 640, 640]
            When I get input shape
            Then should return (640, 640)
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml", input_shape=[1, 3, 640, 640])
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)
        h, w = engine.get_input_shape()

        # Then
        assert (h, w) == (640, 640)

    def test_raises_error_on_missing_model(self, tmp_path):
        """
        Scenario: Missing model file raises error
            Given a ModelConfig with non-existent path
            When I try to create InferenceEngine
            Then should raise FileNotFoundError or RuntimeError
        """
        # Given
        config = ModelConfig(
            model_path=tmp_path / "nonexistent.xml",
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )

        # When/Then
        with pytest.raises((FileNotFoundError, RuntimeError)):
            engine = InferenceEngine(config)


class TestInferenceBehavior:
    """
    Feature: Inference Execution
    Background: Engine runs inference on input tensors
    """

    def test_inference_returns_correct_outputs(self, tmp_path):
        """
        Scenario: Inference produces expected outputs
            Given a compiled segmentation model
            When I infer with tensor [1, 3, 640, 640]
            Then should return dict with 2 outputs
        """
        # Given
        model_path = self._create_test_model(tmp_path / "seg_model.xml", num_outputs=2)
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )
        engine = InferenceEngine(config)

        # When
        tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = engine.infer(tensor)

        # Then
        assert isinstance(outputs, dict)
        assert len(outputs) == 2
        assert "output0" in outputs or "Reshape_8" in outputs  # Output names may vary

    def test_inference_output_shapes_match_model(self, tmp_path):
        """
        Scenario: Inference output shapes match model definition
            Given a compiled model
            When I run inference
            Then output shapes should match get_output_shapes()
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml")
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )
        engine = InferenceEngine(config)
        expected_shapes = engine.get_output_shapes()

        # When
        tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = engine.infer(tensor)

        # Then
        for output_name, output_array in outputs.items():
            # Output names might differ slightly, just check count matches
            assert output_array.shape == expected_shapes[output_name]

    def test_handles_single_output_model(self, tmp_path):
        """
        Scenario: Pose model with 1 output works correctly
            Given a pose model with 1 output
            When I run inference
            Then should return dict with 1 output
        """
        # Given
        model_path = self._create_test_model(tmp_path / "pose_model.xml", num_outputs=1)
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.POSE,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )
        engine = InferenceEngine(config)

        # When
        tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = engine.infer(tensor)

        # Then
        assert isinstance(outputs, dict)
        assert len(outputs) == 1


class TestDeviceFallbackBehavior:
    """
    Feature: Device Fallback
    Background: GPU → CPU fallback when GPU unavailable
    """

    def test_compiles_on_cpu_when_cpu_requested(self, tmp_path):
        """
        Scenario: CPU device requested and used
            Given a ModelConfig with device=CPU
            When I create InferenceEngine
            Then device should be CPU
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml")
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)

        # Then
        assert engine.get_device() == "CPU"

    def test_device_fallback_information_available(self, tmp_path):
        """
        Scenario: Device fallback information is accessible
            Given an InferenceEngine
            When I check the device
            Then should return actual device used (CPU or GPU)
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml")
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.GPU,  # Request GPU
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)
        device = engine.get_device()

        # Then
        # Should fallback to CPU if GPU not available
        assert device in ["GPU", "CPU"]


class TestModelMetadataExtraction:
    """
    Feature: Model Metadata
    Background: Extract model information
    """

    def test_get_num_outputs_returns_correct_count(self, tmp_path):
        """
        Scenario: Number of outputs is correct
            Given a segmentation model with 2 outputs
            When I get num outputs
            Then should return 2
        """
        # Given
        model_path = self._create_test_model(tmp_path / "seg_model.xml", num_outputs=2)
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)
        num_outputs = engine.get_num_outputs()

        # Then
        assert num_outputs == 2

    def test_get_output_shapes_returns_all_shapes(self, tmp_path):
        """
        Scenario: All output shapes are returned
            Given a compiled model
            When I get output shapes
            Then should return dict with all output names and shapes
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml", num_outputs=2)
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)
        shapes = engine.get_output_shapes()

        # Then
        assert isinstance(shapes, dict)
        assert len(shapes) == 2
        for shape in shapes.values():
            assert isinstance(shape, tuple)

    def test_get_model_info_returns_comprehensive_info(self, tmp_path):
        """
        Scenario: Model info contains all metadata
            Given a compiled model
            When I get model info
            Then should return dict with path, device, shapes, etc.
        """
        # Given
        model_path = self._create_test_model(tmp_path / "model.xml")
        config = ModelConfig(
            model_path=model_path,
            model_type=ModelType.SEGMENTATION,
            resolution=640,
            device=Device.CPU,
            precision=Precision.FP32,
            confidence=0.25
        )

        # When
        engine = InferenceEngine(config)
        info = engine.get_model_info()

        # Then
        assert "model_path" in info
        assert "device" in info
        assert "input_shape" in info
        assert "output_shapes" in info
        assert "num_outputs" in info
        assert info["device"] in ["CPU", "GPU"]
        assert isinstance(info["input_shape"], tuple)
        assert isinstance(info["output_shapes"], dict)


# Helper method for creating test models
def _create_test_model(self, path: Path, input_shape=None, num_outputs=2) -> Path:
    """Create a test OpenVINO model."""
    if input_shape is None:
        input_shape = [1, 3, 640, 640]

    input_layer = opset10.parameter(input_shape, np.float32, name="images")

    # Flatten input: [1, 3, 640, 640] = 1,228,800 elements
    flatten = opset10.reshape(input_layer, opset10.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)

    outputs_list = []

    if num_outputs >= 1:
        # Output 1: [1, 84, 1792] = 150,528 elements
        slice_start = opset10.constant(np.array([0, 0], dtype=np.int64))
        slice_end = opset10.constant(np.array([1, 150528], dtype=np.int64))
        slice_step = opset10.constant(np.array([1, 1], dtype=np.int64))
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


# Attach helper to test classes
TestEngineCompilationBehavior._create_test_model = _create_test_model
TestInferenceBehavior._create_test_model = _create_test_model
TestDeviceFallbackBehavior._create_test_model = _create_test_model
TestModelMetadataExtraction._create_test_model = _create_test_model
