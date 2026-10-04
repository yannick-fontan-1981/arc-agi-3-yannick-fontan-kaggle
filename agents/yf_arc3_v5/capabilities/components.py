"""Pure component extraction and ambiguity-preserving correspondence."""

from __future__ import annotations

from functools import lru_cache

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    ComponentDescription,
    ComponentExtractionInput,
    ComponentExtractionResult,
    ComponentMatchCandidate,
    ComponentMatchInput,
    ComponentMatchResult,
    Position,
)
from agents.yf_arc3_v5.logos.types import stable_digest


@lru_cache(maxsize=32)
def detect_components(value: ComponentExtractionInput) -> ComponentExtractionResult:
    """Extract components once per immutable frame/connectivity pair.

    Several pure capabilities consume the same normalized frame during one
    reflection.  Their registry cache keys differ, so this small bounded cache
    prevents identical mechanical decompositions without sharing semantics.
    """
    frame = value.frame
    width = frame.width
    height = frame.height
    visited = bytearray(width * height)
    components: list[ComponentDescription] = []
    shape_instances: dict[tuple[Position, ...], int] = {}
    component_kind = "block" if value.connectivity == 4 else "zone"
    directions: tuple[tuple[int, int], ...] = (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    )
    if value.connectivity == 8:
        directions = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    for row in range(height):
        for col in range(width):
            origin_index = row * width + col
            if visited[origin_index]:
                continue
            pixel_value = frame.rows[row][col]
            pending = [origin_index]
            visited[origin_index] = 1
            pixels: list[Position] = []
            while pending:
                current_row, current_col = divmod(pending.pop(), width)
                pixels.append((current_row, current_col))
                for delta_row, delta_col in directions:
                    neighbor_row = current_row + delta_row
                    neighbor_col = current_col + delta_col
                    if (
                        neighbor_row < 0
                        or neighbor_col < 0
                        or neighbor_row >= height
                        or neighbor_col >= width
                    ):
                        continue
                    neighbor_index = neighbor_row * width + neighbor_col
                    if (
                        visited[neighbor_index]
                        or frame.rows[neighbor_row][neighbor_col] != pixel_value
                    ):
                        continue
                    visited[neighbor_index] = 1
                    pending.append(neighbor_index)
            ordered = tuple(sorted(pixels))
            bbox = BoundingBox(
                top=min(item[0] for item in ordered),
                left=min(item[1] for item in ordered),
                bottom=max(item[0] for item in ordered),
                right=max(item[1] for item in ordered),
            )
            relative_pixels = tuple(
                (pixel_row - bbox.top, pixel_col - bbox.left)
                for pixel_row, pixel_col in ordered
            )
            shape_instances[relative_pixels] = shape_instances.get(relative_pixels, 0) + 1
            shape_ref = stable_digest(relative_pixels)[:8]
            components.append(
                ComponentDescription(
                    component_id=(
                        f"{component_kind}.shape_{bbox.height}x{bbox.width}"
                        f".area_{len(ordered)}.pattern_{shape_ref}"
                        f".instance_{shape_instances[relative_pixels]:03d}"
                    ),
                    value=pixel_value,
                    pixels=ordered,
                    relative_pixels=relative_pixels,
                    bbox=bbox,
                    area=len(ordered),
                    touches_frame_boundary=(
                        bbox.top == 0
                        or bbox.left == 0
                        or bbox.bottom == frame.height - 1
                        or bbox.right == frame.width - 1
                    ),
                )
            )
    return ComponentExtractionResult(components=tuple(components))


@lru_cache(maxsize=16)
def enumerate_component_matches(value: ComponentMatchInput) -> ComponentMatchResult:
    """Reuse exact correspondence enumeration across pure transition consumers."""
    after_by_signature: dict[tuple[int, tuple[Position, ...]], list[ComponentDescription]] = {}
    for after in value.after.components:
        after_by_signature.setdefault(
            (after.value, after.relative_pixels), []
        ).append(after)

    candidate_rows: list[ComponentMatchCandidate] = []
    before_counts: dict[str, int] = {}
    after_counts = {component.component_id: 0 for component in value.after.components}
    for before in value.before.components:
        matches = after_by_signature.get((before.value, before.relative_pixels), ())
        before_counts[before.component_id] = len(matches)
        for after in matches:
            candidate_rows.append(
                ComponentMatchCandidate(
                    before_component_id=before.component_id,
                    after_component_id=after.component_id,
                    delta_row=after.bbox.top - before.bbox.top,
                    delta_col=after.bbox.left - before.bbox.left,
                )
            )
            after_counts[after.component_id] += 1
    candidates = tuple(candidate_rows)
    return ComponentMatchResult(
        candidates=candidates,
        ambiguous_before_component_ids=tuple(
            component_id for component_id, count in sorted(before_counts.items()) if count > 1
        ),
        ambiguous_after_component_ids=tuple(
            component_id for component_id, count in sorted(after_counts.items()) if count > 1
        ),
        unmatched_before_component_ids=tuple(
            component_id for component_id, count in sorted(before_counts.items()) if count == 0
        ),
        unmatched_after_component_ids=tuple(
            component_id for component_id, count in sorted(after_counts.items()) if count == 0
        ),
        action_ref=value.action_ref,
    )
