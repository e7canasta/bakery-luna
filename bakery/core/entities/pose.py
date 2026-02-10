"""
Pose entities - represents pose estimation results.

Domain entities for keypoints, skeletons (17 COCO keypoints), and pose
estimation aggregates.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .detection import BoundingBox


@dataclass
class KeyPoint:
    """
    Single keypoint with position and confidence.

    Attributes:
        x: X coordinate
        y: Y coordinate
        confidence: Keypoint confidence score [0, 1]
        visible: Whether keypoint is visible/valid
    """

    x: float
    y: float
    confidence: float
    visible: bool = True

    def __post_init__(self):
        """Validate keypoint."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @classmethod
    def invalid(cls) -> "KeyPoint":
        """
        Create an invalid/invisible keypoint at origin [0, 0].

        Used for keypoints that were not detected or filtered out.

        Returns:
            Invalid KeyPoint with confidence=0, visible=False
        """
        return cls(x=0.0, y=0.0, confidence=0.0, visible=False)

    @property
    def xy(self) -> Tuple[float, float]:
        """Get keypoint as (x, y) tuple."""
        return (self.x, self.y)

    @property
    def is_valid(self) -> bool:
        """Check if keypoint is valid (visible with confidence > 0)."""
        return self.visible and self.confidence > 0


# COCO keypoint indices (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]


@dataclass
class Skeleton:
    """
    17 COCO keypoints for a person.

    Represents a complete human pose with 17 keypoints following COCO format.

    Attributes:
        keypoints: List of 17 KeyPoint objects
        bbox: Bounding box associated with this person
        person_id: Optional person tracking ID
    """

    keypoints: List[KeyPoint]
    bbox: BoundingBox
    person_id: Optional[int] = None

    def __post_init__(self):
        """Validate skeleton."""
        if len(self.keypoints) != 17:
            raise ValueError(f"COCO skeleton must have 17 keypoints, got {len(self.keypoints)}")

    @property
    def valid_keypoints(self) -> List[KeyPoint]:
        """
        Get only visible/valid keypoints.

        Returns:
            List of valid KeyPoint objects
        """
        return [kp for kp in self.keypoints if kp.is_valid]

    @property
    def num_valid_keypoints(self) -> int:
        """Get count of valid keypoints."""
        return len(self.valid_keypoints)

    @property
    def centroid(self) -> Tuple[float, float]:
        """
        Compute centroid from valid keypoints.

        Falls back to bbox center if no valid keypoints.

        Returns:
            Centroid (x, y) coordinates
        """
        valid = self.valid_keypoints
        if not valid:
            return self.bbox.center

        x_mean = sum(kp.x for kp in valid) / len(valid)
        y_mean = sum(kp.y for kp in valid) / len(valid)
        return (x_mean, y_mean)

    def get_keypoint(self, index: int) -> KeyPoint:
        """
        Get keypoint by index.

        Args:
            index: Keypoint index (0-16)

        Returns:
            KeyPoint at given index

        Example:
            >>> nose = skeleton.get_keypoint(0)
            >>> left_shoulder = skeleton.get_keypoint(5)
        """
        if not 0 <= index < 17:
            raise IndexError(f"Keypoint index must be in [0, 16], got {index}")
        return self.keypoints[index]

    def get_keypoint_by_name(self, name: str) -> KeyPoint:
        """
        Get keypoint by COCO name.

        Args:
            name: Keypoint name (e.g., "nose", "left_shoulder")

        Returns:
            KeyPoint with given name

        Example:
            >>> nose = skeleton.get_keypoint_by_name("nose")
            >>> left_shoulder = skeleton.get_keypoint_by_name("left_shoulder")
        """
        if name not in COCO_KEYPOINT_NAMES:
            raise ValueError(
                f"Invalid keypoint name: {name}. Valid names: {COCO_KEYPOINT_NAMES}"
            )
        index = COCO_KEYPOINT_NAMES.index(name)
        return self.keypoints[index]

    @classmethod
    def from_array(
        cls,
        keypoints_array: np.ndarray,
        bbox: BoundingBox,
        person_id: Optional[int] = None,
    ) -> "Skeleton":
        """
        Create Skeleton from numpy array.

        Args:
            keypoints_array: Array of shape [17, 3] with [x, y, confidence] per keypoint
            bbox: Bounding box for this person
            person_id: Optional person tracking ID

        Returns:
            Skeleton instance
        """
        if keypoints_array.shape != (17, 3):
            raise ValueError(
                f"Keypoints array must have shape [17, 3], got {keypoints_array.shape}"
            )

        keypoints = []
        for i in range(17):
            x, y, conf = keypoints_array[i]
            visible = conf > 0  # Consider keypoint visible if confidence > 0
            keypoints.append(KeyPoint(x=float(x), y=float(y), confidence=float(conf), visible=visible))

        return cls(keypoints=keypoints, bbox=bbox, person_id=person_id)

    def to_array(self) -> np.ndarray:
        """
        Convert skeleton to numpy array.

        Returns:
            Array of shape [17, 3] with [x, y, confidence] per keypoint
        """
        array = np.zeros((17, 3), dtype=np.float32)
        for i, kp in enumerate(self.keypoints):
            array[i] = [kp.x, kp.y, kp.confidence]
        return array


@dataclass
class PoseEstimation:
    """
    Aggregate of pose estimation results.

    Represents all detected skeletons in a single frame.

    Attributes:
        skeletons: List of Skeleton objects
        frame_id: Frame identifier this pose estimation belongs to
    """

    skeletons: List[Skeleton]
    frame_id: int

    def __len__(self) -> int:
        """Get number of detected people."""
        return len(self.skeletons)

    def filter_by_confidence(self, min_valid_keypoints: int = 3) -> "PoseEstimation":
        """
        Filter skeletons by minimum number of valid keypoints.

        Args:
            min_valid_keypoints: Minimum number of valid keypoints required

        Returns:
            New PoseEstimation with filtered skeletons

        Example:
            >>> # Keep only skeletons with at least 5 valid keypoints
            >>> filtered = pose.filter_by_confidence(min_valid_keypoints=5)
        """
        filtered_skeletons = [
            skeleton
            for skeleton in self.skeletons
            if skeleton.num_valid_keypoints >= min_valid_keypoints
        ]
        return PoseEstimation(skeletons=filtered_skeletons, frame_id=self.frame_id)

    @classmethod
    def empty(cls, frame_id: int) -> "PoseEstimation":
        """
        Create empty pose estimation (no skeletons detected).

        Args:
            frame_id: Frame identifier

        Returns:
            Empty PoseEstimation
        """
        return cls(skeletons=[], frame_id=frame_id)

    def to_supervision(self):
        """
        Convert PoseEstimation to supervision KeyPoints format.

        Returns:
            supervision.KeyPoints object (empty if no skeletons)

        Example:
            >>> import supervision as sv
            >>> keypoints = pose_estimation.to_supervision()
        """
        import supervision as sv

        if len(self) == 0:
            return sv.KeyPoints.empty()  # Return empty KeyPoints, not None

        # Convert skeletons to array [N, 17, 2] for xy coordinates
        xy = np.array([
            [[kp.x, kp.y] for kp in skeleton.keypoints]
            for skeleton in self.skeletons
        ])

        # Convert confidence to array [N, 17]
        confidence = np.array([
            [kp.confidence for kp in skeleton.keypoints]
            for skeleton in self.skeletons
        ])

        return sv.KeyPoints(
            xy=xy.astype(np.float32),
            confidence=confidence.astype(np.float32),
            class_id=np.zeros(len(self.skeletons), dtype=int)
        )
