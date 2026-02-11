"""
FilterPolicy — declarative filtering configuration.

A value object that captures all filtering rules for post-inference results.
Pure data, no numpy dependency, easily serializable to JSON/YAML.

Resolution order (most specific wins):
  Detection confidence: per_class_confidence[cls] > seg/pose_confidence > global_confidence
  Keypoint confidence:  per_keypoint_confidence[kp] > keypoint_confidence > global_confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FilterPolicy:
    """
    Declarative filtering rules for post-inference results.

    Attributes:
        global_confidence: Baseline confidence threshold for everything.
        seg_confidence: Override confidence for segmentation model (None = use global).
        pose_confidence: Override confidence for pose model (None = use global).
        classes: Whitelist of class IDs to keep (None = all classes).
        per_class_confidence: Per-class confidence overrides {class_id: threshold}.
        keypoint_confidence: Blanket threshold for all keypoints (None = use global).
        per_keypoint_confidence: Per-keypoint confidence overrides {kp_index: threshold}.
        keypoint_min_visible: Drop skeleton if fewer than N keypoints are visible.

    Example:
        >>> policy = FilterPolicy(
        ...     global_confidence=0.25,
        ...     seg_confidence=0.3,
        ...     classes=[0, 2],
        ...     per_class_confidence={0: 0.6, 2: 0.3},
        ...     per_keypoint_confidence={9: 0.3, 15: 0.4},
        ...     keypoint_min_visible=5,
        ... )
    """

    # ── Detection Filtering ──────────────────────────────────────
    global_confidence: float = 0.25
    seg_confidence: Optional[float] = None
    pose_confidence: Optional[float] = None

    classes: Optional[tuple[int, ...]] = None
    per_class_confidence: Optional[dict[int, float]] = field(default=None)

    # ── Keypoint Filtering ───────────────────────────────────────
    keypoint_confidence: Optional[float] = None
    per_keypoint_confidence: Optional[dict[int, float]] = field(default=None)
    keypoint_min_visible: Optional[int] = None

    # ── Resolvers ────────────────────────────────────────────────

    def resolve_seg_confidence(self) -> float:
        """Effective confidence for segmentation model."""
        return self.seg_confidence if self.seg_confidence is not None else self.global_confidence

    def resolve_pose_confidence(self) -> float:
        """Effective confidence for pose model."""
        return self.pose_confidence if self.pose_confidence is not None else self.global_confidence

    def resolve_class_confidence(self, class_id: int) -> float:
        """Effective confidence for a specific class ID."""
        if self.per_class_confidence and class_id in self.per_class_confidence:
            return self.per_class_confidence[class_id]
        return self.resolve_seg_confidence()

    def resolve_keypoint_confidence(self, kp_id: int) -> float:
        """Effective confidence for a specific keypoint index."""
        if self.per_keypoint_confidence and kp_id in self.per_keypoint_confidence:
            return self.per_keypoint_confidence[kp_id]
        if self.keypoint_confidence is not None:
            return self.keypoint_confidence
        return self.global_confidence

    # ── Predicates ───────────────────────────────────────────────

    @property
    def has_class_filter(self) -> bool:
        return self.classes is not None

    @property
    def has_per_class_confidence(self) -> bool:
        return self.per_class_confidence is not None and len(self.per_class_confidence) > 0

    @property
    def has_keypoint_filter(self) -> bool:
        return (
            self.keypoint_confidence is not None
            or self.per_keypoint_confidence is not None
            or self.keypoint_min_visible is not None
        )

    # ── Factory ──────────────────────────────────────────────────

    @classmethod
    def from_cli(
        cls,
        confidence: float = 0.25,
        seg_confidence: Optional[float] = None,
        pose_confidence: Optional[float] = None,
        classes: Optional[list[int]] = None,
        class_confidence: Optional[list[float]] = None,
        keypoints: Optional[list[int]] = None,
        keypoint_confidence: Optional[list[float]] = None,
        keypoint_confidence_all: Optional[float] = None,
        keypoint_min_visible: Optional[int] = None,
    ) -> "FilterPolicy":
        """
        Build a FilterPolicy from CLI arguments.

        Paired lists are zipped together:
          --classes 0 2 --class-confidence 0.6 0.3
          → per_class_confidence = {0: 0.6, 2: 0.3}

        Args:
            confidence: Global confidence threshold.
            seg_confidence: Segmentation model confidence override.
            pose_confidence: Pose model confidence override.
            classes: List of class IDs to whitelist.
            class_confidence: Paired confidence values for each class in `classes`.
            keypoints: List of keypoint indices to apply specific thresholds.
            keypoint_confidence: Paired confidence values for each keypoint in `keypoints`.
            keypoint_confidence_all: Blanket keypoint confidence threshold.
            keypoint_min_visible: Minimum visible keypoints to keep a skeleton.

        Returns:
            Configured FilterPolicy.

        Raises:
            ValueError: If paired lists have mismatched lengths.
        """
        # Build per-class confidence map
        per_class_conf = None
        if classes and class_confidence:
            if len(classes) != len(class_confidence):
                raise ValueError(
                    f"--classes ({len(classes)}) and --class-confidence ({len(class_confidence)}) "
                    f"must have the same length"
                )
            per_class_conf = dict(zip(classes, class_confidence))

        # Build per-keypoint confidence map
        per_kp_conf = None
        if keypoints and keypoint_confidence:
            if len(keypoints) != len(keypoint_confidence):
                raise ValueError(
                    f"--keypoints ({len(keypoints)}) and --keypoint-confidence ({len(keypoint_confidence)}) "
                    f"must have the same length"
                )
            per_kp_conf = dict(zip(keypoints, keypoint_confidence))

        return cls(
            global_confidence=confidence,
            seg_confidence=seg_confidence,
            pose_confidence=pose_confidence,
            classes=tuple(classes) if classes else None,
            per_class_confidence=per_class_conf,
            keypoint_confidence=keypoint_confidence_all,
            per_keypoint_confidence=per_kp_conf,
            keypoint_min_visible=keypoint_min_visible,
        )
