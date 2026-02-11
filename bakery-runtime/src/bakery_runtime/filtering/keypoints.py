"""
COCO Keypoint constants and helpers.

Standard 17-keypoint COCO layout used by YOLO pose models.
Provides human-readable names ↔ indices for CLI ergonomics.
"""

from __future__ import annotations


# COCO 17-keypoint index map
COCO_KEYPOINTS: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Reverse map: index → name
COCO_KEYPOINT_NAMES: dict[int, str] = {v: k for k, v in COCO_KEYPOINTS.items()}

NUM_COCO_KEYPOINTS = 17


def resolve_keypoint_id(name_or_id: str | int) -> int:
    """
    Accept either a keypoint name ('left_wrist') or numeric index (9).

    Args:
        name_or_id: Keypoint name string or integer index.

    Returns:
        Integer keypoint index.

    Raises:
        ValueError: If name is unknown or index is out of range.

    Example:
        >>> resolve_keypoint_id("left_wrist")
        9
        >>> resolve_keypoint_id(9)
        9
    """
    if isinstance(name_or_id, int):
        if not 0 <= name_or_id < NUM_COCO_KEYPOINTS:
            raise ValueError(
                f"Keypoint index must be in [0, {NUM_COCO_KEYPOINTS - 1}], got {name_or_id}"
            )
        return name_or_id

    name = str(name_or_id).lower().strip()
    if name in COCO_KEYPOINTS:
        return COCO_KEYPOINTS[name]

    # Try numeric string
    try:
        idx = int(name)
        return resolve_keypoint_id(idx)
    except ValueError:
        pass

    raise ValueError(
        f"Unknown keypoint: '{name_or_id}'. "
        f"Valid names: {list(COCO_KEYPOINTS.keys())}"
    )
