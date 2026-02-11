"""
Focus Lens configuration entity.

Configuration for the focus lens (crop-based inference) feature.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FocusLensConfig:
    """
    Immutable Focus Lens configuration.

    Focus Lens applies a square crop to the frame before inference,
    concentrating pixels on a region of interest for improved accuracy.

    Attributes:
        focus_size: Size of the focus region in pixels (must be multiple of 32)
        focus_x: X coordinate of crop origin (None = centered)
        focus_y: Y coordinate of crop origin (None = centered)
        strategy: How to handle frames smaller than focus_size:
                  - "zoom": Scale up frame to focus_size
                  - "pad": Pad frame with black borders

    Example:
        >>> config = FocusLensConfig(focus_size=640)  # Centered 640x640 crop
        >>> config = FocusLensConfig(focus_size=480, focus_x=100, focus_y=100)
        >>> config = FocusLensConfig(focus_size=320, strategy="pad")
    """

    focus_size: int
    focus_x: Optional[int] = None
    focus_y: Optional[int] = None
    strategy: str = "zoom"

    def __post_init__(self):
        """Validate Focus Lens configuration."""
        # Validate focus_size is positive
        if self.focus_size <= 0:
            raise ValueError(f"focus_size must be positive, got {self.focus_size}")

        # Validate focus_size is multiple of 32
        if self.focus_size % 32 != 0:
            raise ValueError(
                f"focus_size must be multiple of 32 (32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, ...), "
                f"got {self.focus_size}"
            )

        # Validate strategy
        if self.strategy not in ("zoom", "pad"):
            raise ValueError(
                f"strategy must be 'zoom' or 'pad', got '{self.strategy}'"
            )

        # Validate focus_x/focus_y if provided
        if self.focus_x is not None and self.focus_x < 0:
            raise ValueError(f"focus_x must be non-negative, got {self.focus_x}")
        if self.focus_y is not None and self.focus_y < 0:
            raise ValueError(f"focus_y must be non-negative, got {self.focus_y}")

    @property
    def is_centered(self) -> bool:
        """Check if focus lens is centered (no explicit position)."""
        return self.focus_x is None and self.focus_y is None
