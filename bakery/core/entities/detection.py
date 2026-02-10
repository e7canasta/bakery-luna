"""
Detection entities - represents object detection results.

Domain entities for bounding boxes, masks, and segmentation aggregates.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BoundingBox:
    """
    Represents a bounding box detection.

    Attributes:
        x1: Top-left X coordinate
        y1: Top-left Y coordinate
        x2: Bottom-right X coordinate
        y2: Bottom-right Y coordinate
        confidence: Detection confidence score [0, 1]
        class_id: Class identifier (e.g., 0 for person in COCO)
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    def __post_init__(self):
        """Validate bounding box."""
        if self.x2 < self.x1:
            raise ValueError(f"x2 must be >= x1, got x1={self.x1}, x2={self.x2}")
        if self.y2 < self.y1:
            raise ValueError(f"y2 must be >= y1, got y1={self.y1}, y2={self.y2}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"Class ID must be non-negative, got {self.class_id}")

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def xyxy(self) -> np.ndarray:
        """Get bounding box as numpy array [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @classmethod
    def from_xyxy(cls, xyxy: np.ndarray, confidence: float, class_id: int) -> "BoundingBox":
        """
        Create BoundingBox from [x1, y1, x2, y2] array.

        Args:
            xyxy: Array [x1, y1, x2, y2]
            confidence: Detection confidence
            class_id: Class identifier

        Returns:
            BoundingBox instance
        """
        return cls(
            x1=float(xyxy[0]),
            y1=float(xyxy[1]),
            x2=float(xyxy[2]),
            y2=float(xyxy[3]),
            confidence=confidence,
            class_id=class_id,
        )

    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if point (x, y) is inside bounding box.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside box
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Mask:
    """
    Binary segmentation mask.

    Attributes:
        data: Binary mask array [H, W] with bool values
        bbox: Bounding box associated with this mask
    """

    data: np.ndarray
    bbox: BoundingBox

    def __post_init__(self):
        """Validate mask."""
        if self.data.ndim != 2:
            raise ValueError(f"Mask data must be 2D array [H, W], got shape {self.data.shape}")
        if self.data.dtype != np.bool_:
            raise ValueError(f"Mask data must be bool dtype, got {self.data.dtype}")

    @property
    def shape(self) -> Tuple[int, int]:
        """Get mask shape (height, width)."""
        return self.data.shape

    @property
    def area(self) -> int:
        """Get mask area (number of True pixels)."""
        return int(np.sum(self.data))

    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if point (x, y) is inside mask.

        Args:
            x: X coordinate (integer)
            y: Y coordinate (integer)

        Returns:
            True if point is inside mask (mask[y, x] == True)
        """
        h, w = self.shape
        if 0 <= x < w and 0 <= y < h:
            return bool(self.data[y, x])
        return False


@dataclass
class Segmentation:
    """
    Aggregate of segmentation detection results.

    Represents all detections in a single frame including bounding boxes
    and corresponding masks.

    Attributes:
        bboxes: List of bounding boxes
        masks: List of masks (one per bbox)
        frame_id: Frame identifier this segmentation belongs to
    """

    bboxes: List[BoundingBox]
    masks: List[Mask]
    frame_id: int

    def __post_init__(self):
        """Validate segmentation."""
        if len(self.bboxes) != len(self.masks):
            raise ValueError(
                f"Number of bboxes ({len(self.bboxes)}) must match number of masks ({len(self.masks)})"
            )

    def __len__(self) -> int:
        """Get number of detections."""
        return len(self.bboxes)

    def filter_by_class(self, class_ids: List[int]) -> "Segmentation":
        """
        Filter segmentation by class IDs.

        Args:
            class_ids: List of class IDs to keep

        Returns:
            New Segmentation with only specified classes

        Example:
            >>> # Filter to keep only persons (class 0)
            >>> person_seg = segmentation.filter_by_class([0])
        """
        class_id_set = set(class_ids)
        filtered_bboxes = []
        filtered_masks = []

        for bbox, mask in zip(self.bboxes, self.masks):
            if bbox.class_id in class_id_set:
                filtered_bboxes.append(bbox)
                filtered_masks.append(mask)

        return Segmentation(
            bboxes=filtered_bboxes, masks=filtered_masks, frame_id=self.frame_id
        )

    def filter_by_confidence(self, min_confidence: float) -> "Segmentation":
        """
        Filter segmentation by minimum confidence.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            New Segmentation with only detections above threshold
        """
        filtered_bboxes = []
        filtered_masks = []

        for bbox, mask in zip(self.bboxes, self.masks):
            if bbox.confidence >= min_confidence:
                filtered_bboxes.append(bbox)
                filtered_masks.append(mask)

        return Segmentation(
            bboxes=filtered_bboxes, masks=filtered_masks, frame_id=self.frame_id
        )

    @classmethod
    def empty(cls, frame_id: int) -> "Segmentation":
        """
        Create empty segmentation (no detections).

        Args:
            frame_id: Frame identifier

        Returns:
            Empty Segmentation
        """
        return cls(bboxes=[], masks=[], frame_id=frame_id)

    def to_supervision(self):
        """
        Convert Segmentation to supervision Detections format.

        Returns:
            supervision.Detections object

        Example:
            >>> import supervision as sv
            >>> detections = segmentation.to_supervision()
        """
        import supervision as sv

        if len(self) == 0:
            return sv.Detections.empty()

        # Convert bboxes to xyxy array [N, 4]
        xyxy = np.array([bbox.xyxy for bbox in self.bboxes])

        # Extract confidence scores [N]
        confidence = np.array([bbox.confidence for bbox in self.bboxes])

        # Extract class IDs [N]
        class_id = np.array([bbox.class_id for bbox in self.bboxes])

        # Convert masks to boolean array [N, H, W]
        mask = np.array([mask.data for mask in self.masks])

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=mask
        )
