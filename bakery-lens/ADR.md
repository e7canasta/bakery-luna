# Architectural Decision Records (ADR) - bakery-lens

## ADR-001: Package Extraction
**Context**: `bakery-luna` contained mixed concerns (pipeline orchestration vs geometric transformations).
**Decision**: Extract `bakery-lens` as a standalone UV workspace member.
**Consequences**:
- (+) Decoupled development of cropping logic.
- (+) Easier testing (no dependency on heavy inference engines).
- (-) Explicit dependency management in root `pyproject.toml`.

## ADR-002: Lens Protocol Definition
**Context**: We needed a unified way to handle different cropping behaviors (Static, Shift, Expand).
**Decision**: Define a `Lens` protocol that delegates to a `LensStrategy`.
- `Lens` handles the "Physical" ops (cropping bytes, mapping coordinates).
- `Strategy` handles the "Logical" decisions (where to crop, how big).
**Consequences**:
- (+) `base_lens.py` is closed for modification, open for extension via strategies.
- (+) Strategies are pure logic (math), easy to test.

## ADR-003: Dynamic Crop Parameters (x, y, w, h)
**Context**: Initially, strategies only computed `(x, y)`. `FocusLensConfig` only had a fixed `focus_size`.
**Decision**: Update `LensStrategy` protocol to return `Tuple[int, int, int, int]` representing `(x, y, w, h)`.
**Reasoning**: `AdaptiveExpand` requires changing the window size dynamically.
**Consequences**:
- (+) Allows breathing/expanding crops.
- (-) Requires `BaseLens` to construct dynamic configurations on the fly to pass to legacy `ops` functions.

## ADR-004: Merged Adaptive Strategy
**Context**: We considered `AdaptiveShift` and `AdaptiveExpand` as separate strategies.
**Decision**: Merge into a single class `AdaptiveShiftLensStrategy` (conceptually `AdaptiveLens`) with feature flags (`allow_expand`).
**Reasoning**:
- Access to shared state (current center, current edges) is tighter.
- Preventing conflict (e.g., Shift wanting to move left while Expand wants to grow) is easier in a single cohesive logic block.
**Consequences**:
- (+) Simpler instantiation factory.
- (-) Class complexity increases; implies need for rigorous testing (mitigated by `test_adaptive_shift.py`).

## ADR-005: Feedback Loop Integration
**Context**: The lens needs to know about detections to adapt.
**Decision**: Explicit `lens.update(detections)` method called by the pipeline *after* inference but *before* mapping.
**Reasoning**: Using the crop-space detections is more direct for the strategy than using global-space detections.
