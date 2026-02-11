"""
Focus Lens configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FocusLensConfig:
    """
    Immutable Focus Lens configuration.

    Attributes:
        focus_size: Size of the focus region in pixels (must be multiple of 32)
        focus_x: X coordinate of crop origin (None = centered)
        focus_y: Y coordinate of crop origin (None = centered)
        strategy: How to handle frames smaller than focus_size:
                  - "zoom": Scale up frame to focus_size
                  - "pad": Pad frame with black borders
        adaptive: Enable adaptive strategies (shift/expand)
        edge_threshold: Pixels from edge to trigger adaptive shift (default: 40)
        shift_step: Max pixels to shift per update (default: 80)
        smoothing: EMA smoothing factor 0-1 (default: 0.3)
        allow_expand: Allow dynamic expansion of focus_size (Phase 2)
    """

    focus_size: int
    focus_x: Optional[int] = None
    focus_y: Optional[int] = None
    strategy: str = "zoom"
    
    # Adaptive configuration
    adaptive: bool = False
    edge_threshold: int = 40
    shift_step: int = 80
    smoothing: float = 0.3
    allow_expand: bool = False

    def __post_init__(self):
        """Validate Focus Lens configuration."""
        # Validate focus_size is positive
        if self.focus_size <= 0:
            raise ValueError(f"focus_size must be positive, got {self.focus_size}")

        # Validate focus_size is multiple of 32
        if self.focus_size % 32 != 0:
            raise ValueError(
                f"focus_size must be multiple of 32 (32, 64, 96, 128, 160...), "
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
            
        # Validate adaptive params
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold must be non-negative")
        if self.shift_step <= 0:
            raise ValueError("shift_step must be positive")
        if not (0 <= self.smoothing <= 1):
            raise ValueError("smoothing must be between 0 and 1")

    @property
    def is_centered(self) -> bool:
        """Check if focus lens is centered (no explicit position)."""
        return self.focus_x is None and self.focus_y is None
