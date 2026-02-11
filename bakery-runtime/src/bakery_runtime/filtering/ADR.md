# Architectural Decision Records (ADR) - bakery-runtime.filtering

## ADR-001: Sub-Package Extraction from Inline Logic
**Context**: Class filtering was initially inlined as a `classes` parameter in `postprocess_segmentation`. As more filtering dimensions emerged (per-model confidence, per-class confidence, keypoint thresholds), the postprocess signature would grow unbounded.
**Decision**: Extract filtering into `bakery_runtime.filtering/` as a dedicated sub-package, separate from `processing/` (raw numerical decoding).
**Consequences**:
- (+) `postprocess_segmentation` / `postprocess_pose` stay focused on numerical NMS + decoding.
- (+) Filtering logic is testable in isolation (no model or engine dependency).
- (+) New filter dimensions don't touch existing postprocess functions (Open/Closed).

## ADR-002: FilterPolicy as Frozen Value Object
**Context**: Filtering rules could be mutable state on the pipeline, or a strategy object, or a config dataclass.
**Decision**: `FilterPolicy` is a `@dataclass(frozen=True)` — immutable, serializable, no numpy dependency.
**Reasoning**:
- Value objects are thread-safe by construction.
- Frozen dataclasses hash correctly (can be used as dict keys / cache keys).
- Easy to serialize to JSON/YAML for reproducibility and logging.
**Consequences**:
- (+) No side effects, trivially testable.
- (-) Changing a filter at runtime requires creating a new `FilterPolicy` instance.

## ADR-003: Stateless Pure Functions over Strategy Pattern
**Context**: Filtering could be implemented as a `FilterStrategy` class hierarchy or as stateless functions.
**Decision**: Use stateless pure functions (`filter_detections`, `filter_keypoints`) that take numpy arrays and a `FilterPolicy`.
**Reasoning**:
- Filters have no state between frames — they're pure `(input, policy) → output` transformations.
- A class hierarchy would add indirection without benefit. Functions compose naturally.
- Matches the pattern used by `supervision` library (`sv.Detections` filtering is functional).
**Consequences**:
- (+) Zero allocation overhead beyond the filtered arrays.
- (+) Easy to compose: chain multiple filter calls if needed.
- (-) If we later need stateful filters (e.g., temporal smoothing), we'd add a separate abstraction.

## ADR-004: Filtering Stage — Post-NMS, Pre-Entity
**Context**: Filtering can happen at three points: (1) during postprocess (pre-NMS), (2) after postprocess but before domain entity construction, (3) after entity construction.
**Decision**: Filter at stage (2) — after NMS, before `_create_segmentation` / `_create_pose_estimation`.
**Reasoning**:
- Stage (1) is too early: NMS needs all candidates for correct suppression.
- Stage (3) is too late: we'd allocate domain objects only to discard them.
- Stage (2) is the earliest useful point where filtering is safe and avoids wasted work.
**Consequences**:
- (+) No wasted domain object allocation.
- (+) NMS correctness preserved.
- (+) Filters operate on raw numpy — fast, no entity dependency.

## ADR-005: Paired CLI Lists for Per-Class/Per-Keypoint Config
**Context**: We needed a CLI mechanism for mapping `class_id → confidence` and `keypoint_id → confidence`.
**Decision**: Use paired positional lists: `--classes 0 2 --class-confidence 0.6 0.3`. The lists are zipped in `FilterPolicy.from_cli()`. Mismatched lengths raise `ValueError`.
**Reasoning**:
- Familiar pattern from ffmpeg (`-map 0:1 -c:a aac`), gstreamer, and similar tools.
- No need for custom parsers or YAML config files for simple use cases.
- Alternative (dict syntax `--class-confidence 0=0.6 2=0.3`) requires custom type parsing in Typer.
**Consequences**:
- (+) Minimal cognitive load for CLI users.
- (+) Validation is trivial (length check).
- (-) Order matters — user must keep lists aligned.

## ADR-006: COCO Keypoint Constants in filtering/
**Context**: Keypoint indices are magic numbers (e.g., 9 = left_wrist). Users need ergonomic access.
**Decision**: Place COCO keypoint constants and `resolve_keypoint_id()` in `filtering/keypoints.py`.
**Reasoning**:
- The primary consumer of keypoint names is the filtering layer (per-keypoint thresholds).
- If other packages need it, they can import from `bakery_runtime.filtering.keypoints`.
- In the future, if we support non-COCO layouts, the resolver can be extended.
**Consequences**:
- (+) CLI can accept `--keypoints left_wrist left_ankle` in addition to `--keypoints 9 15`.
- (+) Agent-friendly: names are self-documenting.
