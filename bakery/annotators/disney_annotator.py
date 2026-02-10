"""
Disney/Roger Rabbit Style Annotator - Multi-layer rendering for vision pipeline.

Implements the iconic "colorful objects in B&W world" aesthetic with 8 rendering layers:
  1. B&W World: Grayscale background (darkened 0.6x for contrast)
  2. Focus Lens: Subtle brightening (1.1x) of region of interest
  3. Halo Effect: Soft glow around detections (defines boundaries)
  4. Box Corners: Subtle corner markers (30% opacity)
  5. Color Spotlight: Original color + brightness boost (1.2x) for detected objects
  6. Confidence Bars: Percentage indicators (40% opacity)
  7. Labels: Class ID + confidence text (40% opacity)
  8. Skeleton Overlay: Pose edges + vertices (30% opacity)

Philosophy: "Complejidad por diseño, no por accidente"
"""

import cv2
import numpy as np
import supervision as sv
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RenderConfig:
    """Configuration for Disney/Roger Rabbit rendering style."""

    # Layer 1: B&W World
    bw_darkness: float = 0.6  # Darken factor (0.6 = 60% of original)

    # Layer 2: Focus Lens
    lens_brightness: float = 1.1  # Brighten factor (1.1 = 10% brighter)

    # Layer 3: Halo Effect
    halo_color: Tuple[int, int, int] = (240, 248, 255)  # Alice blue (light glow)
    halo_opacity: float = 0.5
    halo_kernel_size: int = 40

    # Layer 4: Box Corners
    corner_color: Tuple[int, int, int] = (200, 200, 200)  # Light gray
    corner_thickness: int = 1
    corner_length: int = 15
    corner_opacity: float = 0.3  # 30% blend

    # Layer 5: Color Spotlight
    spotlight_brightness: float = 1.2  # 20% brighter than original

    # Layer 6: Confidence Bars
    bar_height: int = 16
    bar_width: int = 55
    bar_color: Tuple[int, int, int] = (90, 90, 90)  # Medium gray
    bar_border_color: Tuple[int, int, int] = (70, 70, 70)  # Dark gray
    bar_border_thickness: int = 1
    bar_opacity: float = 0.4  # 40% blend

    # Layer 7: Labels
    label_text_scale: float = 0.4
    label_text_thickness: int = 1
    label_text_padding: int = 8
    label_text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    label_background_color: Tuple[int, int, int] = (40, 40, 40)  # Very dark gray
    label_border_radius: int = 3
    label_opacity: float = 0.4  # 40% blend

    # Layer 8: Skeleton
    skeleton_edge_color: Tuple[int, int, int] = (200, 200, 200)  # Light gray
    skeleton_edge_thickness: int = 1
    skeleton_vertex_color: Tuple[int, int, int] = (160, 160, 160)  # Medium gray
    skeleton_vertex_radius: int = 3
    skeleton_opacity: float = 0.3  # 30% blend

    # Focus lens marker (optional)
    lens_marker_color: Tuple[int, int, int] = (150, 150, 150)  # Medium gray
    lens_marker_thickness: int = 1
    lens_marker_opacity: float = 0.3


class DisneyAnnotator:
    """
    Multi-layer annotator implementing Disney/Roger Rabbit aesthetic.

    Creates the iconic "colorful objects emerging from grayscale world" effect
    with translucent ghost-like annotations.

    Example:
        >>> config = RenderConfig()
        >>> annotator = DisneyAnnotator(config, enable_pose=True)
        >>> annotated = annotator.annotate(
        ...     frame=frame,
        ...     detections=detections,
        ...     keypoints=keypoints,
        ...     focus_region=(100, 100, 640, 640)
        ... )
    """

    def __init__(self, config: Optional[RenderConfig] = None, enable_pose: bool = False):
        """
        Initialize Disney annotator with supervision components.

        Args:
            config: Rendering configuration (uses defaults if None)
            enable_pose: If True, initialize pose skeleton annotators
        """
        self.config = config or RenderConfig()
        self.enable_pose = enable_pose

        # Initialize supervision annotators
        self._init_annotators()

    def _init_annotators(self):
        """Initialize all supervision annotator components."""
        # Layer 3: Halo
        self.halo_annotator = sv.HaloAnnotator(
            color=sv.Color.from_rgb_tuple(self.config.halo_color),
            opacity=self.config.halo_opacity,
            kernel_size=self.config.halo_kernel_size
        )

        # Layer 4: Box Corners
        self.corner_annotator = sv.BoxCornerAnnotator(
            color=sv.Color.from_rgb_tuple(self.config.corner_color),
            thickness=self.config.corner_thickness,
            corner_length=self.config.corner_length
        )

        # Layer 6: Percentage Bar
        self.bar_annotator = sv.PercentageBarAnnotator(
            height=self.config.bar_height,
            width=self.config.bar_width,
            color=sv.Color.from_rgb_tuple(self.config.bar_color),
            border_color=sv.Color.from_rgb_tuple(self.config.bar_border_color),
            border_thickness=self.config.bar_border_thickness,
            position=sv.Position.BOTTOM_RIGHT
        )

        # Layer 7: Labels
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.BOTTOM_LEFT,
            text_scale=self.config.label_text_scale,
            text_thickness=self.config.label_text_thickness,
            text_padding=self.config.label_text_padding,
            text_color=sv.Color.from_rgb_tuple(self.config.label_text_color),
            color=sv.Color.from_rgb_tuple(self.config.label_background_color),
            border_radius=self.config.label_border_radius
        )

        # Layer 8: Pose (optional)
        if self.enable_pose:
            self.edge_annotator = sv.EdgeAnnotator(
                color=sv.Color.from_rgb_tuple(self.config.skeleton_edge_color),
                thickness=self.config.skeleton_edge_thickness
            )

            self.vertex_annotator = sv.VertexAnnotator(
                color=sv.Color.from_rgb_tuple(self.config.skeleton_vertex_color),
                radius=self.config.skeleton_vertex_radius
            )
        else:
            self.edge_annotator = None
            self.vertex_annotator = None

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        keypoints: Optional[sv.KeyPoints] = None,
        focus_region: Optional[Tuple[int, int, int, int]] = None,
        show_focus_marker: bool = True
    ) -> np.ndarray:
        """
        Apply Disney/Roger Rabbit style rendering to frame.

        Args:
            frame: Input frame (BGR, OpenCV format)
            detections: Segmentation detections with masks
            keypoints: Pose keypoints (optional)
            focus_region: (x, y, width, height) of focus lens region (optional)
            show_focus_marker: If True, draw subtle focus lens rectangle

        Returns:
            Annotated frame with Disney/Roger Rabbit rendering
        """
        # Layer 1: B&W World (darken for contrast)
        annotated = self._apply_bw_world(frame)

        # Layer 2: Focus Lens (brighten region of interest)
        if focus_region is not None:
            annotated = self._apply_focus_lens(annotated, focus_region)

        # Layer 3: Halo Effect (glow around detections)
        if len(detections) > 0:
            annotated = self._apply_halo(annotated, detections)

        # Layer 4: Box Corners (subtle markers)
        if len(detections) > 0:
            annotated = self._apply_box_corners(annotated, detections)

        # Layer 5: Color Spotlight (paint objects with brightness boost)
        if len(detections) > 0 and detections.mask is not None:
            annotated = self._apply_color_spotlight(annotated, frame, detections)

        # Layer 6: Confidence Bars
        if len(detections) > 0:
            annotated = self._apply_confidence_bars(annotated, detections)

        # Layer 7: Labels (class ID + confidence)
        if len(detections) > 0:
            annotated = self._apply_labels(annotated, detections)

        # Layer 7.5: Focus Lens Marker (optional)
        if focus_region is not None and show_focus_marker:
            annotated = self._apply_focus_marker(annotated, focus_region)

        # Layer 8: Skeleton Overlay (edges + vertices)
        if keypoints is not None and len(keypoints) > 0:
            annotated = self._apply_skeleton(annotated, keypoints)

        return annotated

    def _apply_bw_world(self, frame: np.ndarray) -> np.ndarray:
        """Layer 1: Convert frame to grayscale and darken."""
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.cvtColor(frame_bw, cv2.COLOR_GRAY2BGR)
        # Darken to medium gray for more contrast
        frame_bw = np.clip(frame_bw.astype(np.float32) * self.config.bw_darkness, 0, 255).astype(np.uint8)
        return frame_bw

    def _apply_focus_lens(self, annotated: np.ndarray, focus_region: Tuple[int, int, int, int]) -> np.ndarray:
        """Layer 2: Brighten focus lens region subtly."""
        x, y, width, height = focus_region
        lens_region = annotated[y:y+height, x:x+width].copy()
        lens_region = np.clip(
            lens_region.astype(np.float32) * self.config.lens_brightness,
            0, 255
        ).astype(np.uint8)
        annotated[y:y+height, x:x+width] = lens_region
        return annotated

    def _apply_halo(self, annotated: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Layer 3: Apply halo glow around detections."""
        return self.halo_annotator.annotate(scene=annotated, detections=detections)

    def _apply_box_corners(self, annotated: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Layer 4: Apply subtle corner markers with transparency."""
        overlay = annotated.copy()
        overlay = self.corner_annotator.annotate(scene=overlay, detections=detections)
        # Blend: 70% frame + 30% corners
        alpha = self.config.corner_opacity
        return cv2.addWeighted(annotated, 1.0 - alpha, overlay, alpha, 0)

    def _apply_color_spotlight(
        self,
        annotated: np.ndarray,
        original_frame: np.ndarray,
        detections: sv.Detections
    ) -> np.ndarray:
        """Layer 5: Paint detected objects with original color + brightness boost."""
        for mask in detections.mask:
            # Expand mask to 3 channels
            mask_3ch = np.stack([mask] * 3, axis=-1)

            # Brighten original frame
            frame_bright = np.clip(
                original_frame.astype(np.float32) * self.config.spotlight_brightness,
                0, 255
            ).astype(np.uint8)

            # Where mask=True, use bright color; where mask=False, keep B&W
            annotated = np.where(mask_3ch, frame_bright, annotated)

        return annotated

    def _apply_confidence_bars(self, annotated: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Layer 6: Apply confidence percentage bars with transparency."""
        overlay = annotated.copy()
        overlay = self.bar_annotator.annotate(scene=overlay, detections=detections)
        # Blend: 60% frame + 40% bars
        alpha = self.config.bar_opacity
        return cv2.addWeighted(annotated, 1.0 - alpha, overlay, alpha, 0)

    def _apply_labels(self, annotated: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Layer 7: Apply class ID and confidence labels with transparency."""
        labels = [
            f"ID:{class_id} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
        overlay = annotated.copy()
        overlay = self.label_annotator.annotate(
            scene=overlay,
            detections=detections,
            labels=labels
        )
        # Blend: 60% frame + 40% labels
        alpha = self.config.label_opacity
        return cv2.addWeighted(annotated, 1.0 - alpha, overlay, alpha, 0)

    def _apply_focus_marker(self, annotated: np.ndarray, focus_region: Tuple[int, int, int, int]) -> np.ndarray:
        """Layer 7.5: Draw subtle focus lens boundary marker."""
        x, y, width, height = focus_region
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            color=self.config.lens_marker_color,
            thickness=self.config.lens_marker_thickness
        )
        # Blend with subtle opacity
        alpha = self.config.lens_marker_opacity
        return cv2.addWeighted(annotated, 1.0 - alpha, overlay, alpha, 0)

    def _apply_skeleton(self, annotated: np.ndarray, keypoints: sv.KeyPoints) -> np.ndarray:
        """Layer 8: Apply pose skeleton (edges + vertices) with transparency."""
        if self.edge_annotator is None or self.vertex_annotator is None:
            return annotated

        # Edges first (lines connecting keypoints)
        overlay_edges = annotated.copy()
        overlay_edges = self.edge_annotator.annotate(
            scene=overlay_edges,
            key_points=keypoints
        )
        # Blend: 70% frame + 30% edges
        alpha = self.config.skeleton_opacity
        annotated = cv2.addWeighted(annotated, 1.0 - alpha, overlay_edges, alpha, 0)

        # Vertices on top (circles at keypoints)
        overlay_vertices = annotated.copy()
        overlay_vertices = self.vertex_annotator.annotate(
            scene=overlay_vertices,
            key_points=keypoints
        )
        # Blend: 70% frame + 30% vertices
        annotated = cv2.addWeighted(annotated, 1.0 - alpha, overlay_vertices, alpha, 0)

        return annotated
