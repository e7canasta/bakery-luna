"""
Unit tests for geometry utilities.
"""

import numpy as np
import pytest
from bakery.utils.geometry import xywh2xyxy, nms, bbox_iou


class TestXywh2xyxy:
    """Tests for xywh2xyxy function."""

    def test_single_box(self):
        """Test converting single box from center format to corner format."""
        boxes = np.array([[100, 100, 50, 50]])  # center (100, 100), size 50x50
        expected = np.array([[75, 75, 125, 125]])  # top-left (75, 75), bottom-right (125, 125)
        result = xywh2xyxy(boxes)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_boxes(self):
        """Test converting multiple boxes."""
        boxes = np.array([
            [100, 100, 50, 50],
            [200, 200, 100, 100],
            [50, 50, 20, 20]
        ])
        expected = np.array([
            [75, 75, 125, 125],
            [150, 150, 250, 250],
            [40, 40, 60, 60]
        ])
        result = xywh2xyxy(boxes)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_size_box(self):
        """Test box with zero width/height."""
        boxes = np.array([[100, 100, 0, 0]])
        expected = np.array([[100, 100, 100, 100]])
        result = xywh2xyxy(boxes)
        np.testing.assert_array_almost_equal(result, expected)

    def test_preserves_original_array(self):
        """Test that original array is not modified (copy is returned)."""
        boxes = np.array([[100, 100, 50, 50]])
        original_copy = boxes.copy()
        _ = xywh2xyxy(boxes)
        np.testing.assert_array_equal(boxes, original_copy)


class TestNMS:
    """Tests for NMS (Non-Maximum Suppression) function."""

    def test_no_overlap(self):
        """Test boxes with no overlap - all should be kept."""
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250]
        ])
        scores = np.array([0.9, 0.8, 0.95])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 3  # All boxes should be kept

    def test_full_overlap(self):
        """Test identical boxes - only highest score should be kept."""
        boxes = np.array([
            [10, 10, 50, 50],
            [10, 10, 50, 50],  # Identical to first
            [10, 10, 50, 50]   # Identical to first
        ])
        scores = np.array([0.7, 0.9, 0.8])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 1  # Index 1 has highest score (0.9)

    def test_partial_overlap(self):
        """Test partially overlapping boxes."""
        boxes = np.array([
            [10, 10, 50, 50],
            [30, 30, 70, 70],  # Partial overlap with first (IoU ~0.14)
            [100, 100, 150, 150]  # No overlap
        ])
        scores = np.array([0.9, 0.8, 0.95])
        keep = nms(boxes, scores, iou_threshold=0.1)  # Lower threshold to suppress overlaps
        # Box 2 (score 0.95) and Box 0 (score 0.9) should be kept
        # Box 1 (score 0.8) suppressed by Box 0 (IoU 0.14 > 0.1)
        assert len(keep) == 2
        assert 2 in keep  # Highest score
        assert 0 in keep

    def test_empty_input(self):
        """Test empty input."""
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 0

    def test_single_box(self):
        """Test single box input."""
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_different_thresholds(self):
        """Test that different IoU thresholds produce different results."""
        boxes = np.array([
            [10, 10, 50, 50],
            [30, 30, 70, 70]  # Partial overlap
        ])
        scores = np.array([0.9, 0.8])

        # Low threshold - strict NMS, suppresses overlapping boxes
        keep_strict = nms(boxes, scores, iou_threshold=0.1)
        assert len(keep_strict) == 1

        # High threshold - lenient NMS, keeps both boxes
        keep_lenient = nms(boxes, scores, iou_threshold=0.9)
        assert len(keep_lenient) == 2


class TestBboxIoU:
    """Tests for bbox_iou function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([10, 10, 50, 50])
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes should be 0.0."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([100, 100, 150, 150])
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        bbox1 = np.array([10, 10, 50, 50])  # 40x40 = 1600
        bbox2 = np.array([30, 30, 70, 70])  # 40x40 = 1600
        # Overlap: [30, 30, 50, 50] = 20x20 = 400
        # Union: 1600 + 1600 - 400 = 2800
        # IoU: 400 / 2800 = 0.142857...
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(400 / 2800, rel=1e-5)

    def test_one_inside_other(self):
        """Test IoU when one box is completely inside another."""
        bbox1 = np.array([10, 10, 100, 100])  # 90x90 = 8100
        bbox2 = np.array([30, 30, 70, 70])    # 40x40 = 1600
        # Overlap: bbox2 entirely inside bbox1 = 1600
        # Union: 8100 (bbox1 covers bbox2)
        # IoU: 1600 / 8100 ≈ 0.1975
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(1600 / 8100, rel=1e-5)

    def test_touching_boxes(self):
        """Test IoU of boxes that touch but don't overlap."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([50, 10, 90, 50])  # Shares edge with bbox1
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.0)

    def test_commutative_property(self):
        """Test that IoU(A, B) == IoU(B, A)."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([30, 30, 70, 70])
        iou1 = bbox_iou(bbox1, bbox2)
        iou2 = bbox_iou(bbox2, bbox1)
        assert iou1 == pytest.approx(iou2)

    def test_zero_area_box(self):
        """Test IoU with zero-area box."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([30, 30, 30, 30])  # Zero area
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
