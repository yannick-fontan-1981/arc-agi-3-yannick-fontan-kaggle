"""Bounded cell-lattice measurements for composed orthogonal reflections.

This module measures masks, axes, finite transform closures, overlaps and exact
translation residuals.  It deliberately assigns no visual role and chooses no
goal; DRM/SRC own every semantic interpretation and action commitment.
"""

from __future__ import annotations

from math import gcd

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    ComponentDescription,
    ComponentExtractionInput,
    FrameGrid,
    InteractionProbeInput,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


_MAXIMUM_CELL_QUANTUM = 16
_MAXIMUM_SUPPORT_ANCHORS = 64
_MAXIMUM_REFLECTION_COMPLETIONS = 3
_MAXIMUM_SUPPORT_AXIS_PAIR_DESCRIPTIONS = 4096
_MAXIMUM_TRANSLATION_OFFSETS = 512
_MAXIMUM_TRANSLATION_WITNESSES = 3


Cell = tuple[int, int]


def _axis_multiplicity_cell_count(
    cells: frozenset[Cell], *, row_axis_twice: int, col_axis_twice: int
) -> int:
    """Count coincident reflection layers: 1 off axes, 2 on one, 4 on both."""
    return sum(
        (2 if 2 * row == row_axis_twice else 1)
        * (2 if 2 * col == col_axis_twice else 1)
        for row, col in cells
    )


def _orthogonally_connected_cell_sets(
    cells: frozenset[Cell],
) -> tuple[frozenset[Cell], ...]:
    """Decompose persistent logical occupancy, independent of rendered layers."""

    remaining = set(cells)
    components: list[frozenset[Cell]] = []
    while remaining:
        frontier = [min(remaining)]
        connected: set[Cell] = set()
        while frontier:
            cell = frontier.pop()
            if cell not in remaining:
                continue
            remaining.remove(cell)
            connected.add(cell)
            row, col = cell
            for peer in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if peer in remaining:
                    frontier.append(peer)
        components.append(frozenset(connected))
    return tuple(sorted(components, key=lambda component: min(component)))


def _bounded_morphology_decomposition(
    cells: frozenset[Cell],
    morphology_collections: tuple[tuple[Cell, ...], ...],
) -> tuple[frozenset[Cell], ...]:
    """Recover one exact layered decomposition of a merged same-value body."""

    morphologies = tuple(
        frozenset((int(row), int(col)) for row, col in morphology)
        for morphology in morphology_collections
        if morphology
    )
    if (
        len(morphologies) < 2
        or len(cells) > sum(len(morphology) for morphology in morphologies)
        or len(cells) <= max(len(morphology) for morphology in morphologies)
    ):
        return ()
    placements_by_morphology: list[tuple[frozenset[Cell], ...]] = []
    for morphology in morphologies:
        placements: set[frozenset[Cell]] = set()
        for source_row, source_col in morphology:
            for target_row, target_col in cells:
                delta_row = target_row - source_row
                delta_col = target_col - source_col
                translated = frozenset(
                    (row + delta_row, col + delta_col)
                    for row, col in morphology
                )
                if translated.issubset(cells):
                    placements.add(translated)
                if len(placements) > _MAXIMUM_TRANSLATION_OFFSETS:
                    return ()
        if not placements:
            return ()
        placements_by_morphology.append(
            tuple(sorted(placements, key=lambda item: tuple(sorted(item))))
        )
    decompositions: set[tuple[tuple[Cell, ...], ...]] = set()
    expansion_count = 0

    def extend(index: int, placed: tuple[frozenset[Cell], ...]) -> None:
        nonlocal expansion_count
        if expansion_count >= _MAXIMUM_TRANSLATION_OFFSETS:
            return
        if index == len(placements_by_morphology):
            if frozenset(cell for item in placed for cell in item) == cells:
                decompositions.add(
                    tuple(tuple(sorted(item)) for item in placed)
                )
            return
        for placement in placements_by_morphology[index]:
            expansion_count += 1
            if expansion_count > _MAXIMUM_TRANSLATION_OFFSETS:
                return
            extend(index + 1, (*placed, placement))

    extend(0, ())
    if len(decompositions) != 1:
        return ()
    return tuple(frozenset(item) for item in next(iter(decompositions)))


def _action_conditioned_bbox_supports_candidate_focus(
    *,
    candidate_bbox: tuple[int, int, int, int] | None,
    change_bbox: tuple[int, int, int, int],
    quantum: int,
) -> bool:
    """Measure either a near-complete or contained focus-change witness.

    A selector animation need not repaint the whole controlled component: it
    can expose only a thin highlight strictly inside that component.  The
    containment direction matters.  A scene-wide or axis-wide change which
    merely contains the candidate is not evidence that the candidate itself
    has focus.
    """

    if candidate_bbox is None:
        return False
    near_complete = max(
        abs(change_bbox[0] - candidate_bbox[0]),
        abs(change_bbox[1] - candidate_bbox[1]),
        abs(change_bbox[2] - candidate_bbox[2]),
        abs(change_bbox[3] - candidate_bbox[3]),
    ) <= quantum
    contained_change = (
        candidate_bbox[0] <= change_bbox[0] <= change_bbox[2] <= candidate_bbox[2]
        and candidate_bbox[1]
        <= change_bbox[1]
        <= change_bbox[3]
        <= candidate_bbox[3]
    )
    return near_complete or contained_change


def _current_components(value: InteractionProbeInput):
    frame = value.current_observed_frame or value.frame
    return detect_components(ComponentExtractionInput(frame=frame)).components


def _observed_cell_quantum(value: InteractionProbeInput) -> int:
    carried_quanta = tuple(
        sorted(
            {
                abs(int(quantum))
                for quantum in value.prior_context_effect_carried_reflection_axis_quanta
                if 1 <= abs(int(quantum)) <= _MAXIMUM_CELL_QUANTUM
            }
        )
    )
    if len(carried_quanta) == 1:
        return carried_quanta[0]
    directional_axes = {
        action_ref: (delta_row != 0, delta_col != 0)
        for action_ref, delta_row, delta_col in (
            value.interface_action_translation_deltas
        )
    }
    directional_action_refs = frozenset(directional_axes)
    row_magnitudes: set[int] = set()
    col_magnitudes: set[int] = set()
    magnitudes = tuple(
        abs(int(item))
        # Interface deltas are abstract directional intents.  Only measured
        # action-conditioned raster translations establish a cell quantum.
        for _action_ref, delta_row, delta_col in value.known_action_translation_deltas
        if _action_ref in directional_action_refs
        for item in (delta_row, delta_col)
        if int(item) != 0
    )
    for action_ref, delta_row, delta_col in value.known_action_translation_deltas:
        declared_axis = directional_axes.get(action_ref)
        if declared_axis is None:
            continue
        row_declared, col_declared = declared_axis
        if row_declared and not col_declared and delta_row and not delta_col:
            row_magnitudes.add(abs(int(delta_row)))
        if col_declared and not row_declared and delta_col and not delta_row:
            col_magnitudes.add(abs(int(delta_col)))
    # A square cellular structure has one shared raster quantum across its two
    # orthogonal interface axes.  An exact magnitude independently observed on
    # both axes is therefore stronger scale evidence than one isolated alias
    # produced by repeated-marker tracking.  Preserve every observed delta for
    # later reconciliation, but do not let a lone contradictory magnitude
    # collapse this local structural measurement.  Multiple shared magnitudes
    # remain ambiguous and fall through to the historical exact GCD rule.
    shared_axis_magnitudes = tuple(sorted(row_magnitudes & col_magnitudes))
    if len(shared_axis_magnitudes) == 1:
        unique_shared = shared_axis_magnitudes[0]
        if 1 <= unique_shared <= _MAXIMUM_CELL_QUANTUM:
            return unique_shared
    quantum = 0
    for magnitude in magnitudes:
        quantum = gcd(quantum, magnitude)
    return quantum if 1 <= quantum <= _MAXIMUM_CELL_QUANTUM else 0


def _occupied_cells(
    *,
    rows: tuple[tuple[int, ...], ...],
    palette_value: int,
    quantum: int,
    row_phase: int,
    col_phase: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> frozenset[Cell]:
    occupied: set[Cell] = set()
    first_row = top + ((row_phase - top) % quantum)
    first_col = left + ((col_phase - left) % quantum)
    for pixel_row in range(first_row, bottom + 1, quantum):
        for pixel_col in range(first_col, right + 1, quantum):
            if any(
                rows[row][col] == palette_value
                for row in range(pixel_row, min(pixel_row + quantum, len(rows)))
                for col in range(
                    pixel_col,
                    min(pixel_col + quantum, len(rows[0])),
                )
            ):
                occupied.add(
                    (
                        (pixel_row - row_phase) // quantum,
                        (pixel_col - col_phase) // quantum,
                    )
                )
    return frozenset(occupied)


def _fully_palette_cells(
    *,
    rows: tuple[tuple[int, ...], ...],
    palette_value: int,
    quantum: int,
    row_phase: int,
    col_phase: int,
    candidate_cells: frozenset[Cell],
) -> frozenset[Cell]:
    """Return logical cells whose complete rendered area has one value."""

    fully_palette: set[Cell] = set()
    for cell_row, cell_col in sorted(candidate_cells):
        pixel_row = row_phase + cell_row * quantum
        pixel_col = col_phase + cell_col * quantum
        pixel_rows = range(pixel_row, min(pixel_row + quantum, len(rows)))
        pixel_cols = range(pixel_col, min(pixel_col + quantum, len(rows[0])))
        if all(
            rows[row][col] == palette_value
            for row in pixel_rows
            for col in pixel_cols
        ):
            fully_palette.add((cell_row, cell_col))
    return frozenset(fully_palette)


def measure_orthogonal_reflection_lattice_object_audit(
    *,
    frame: FrameGrid | tuple[tuple[int, ...], ...],
    quantum: int,
    row_phase: int | None,
    col_phase: int | None,
    support_component_ref: str | None,
    support_palette_value: int,
    completed_cells: tuple[Cell, ...],
    carrier_palette_value: int,
    carrier_row_axis_coordinate_twice: int | None,
    carrier_col_axis_coordinate_twice: int | None,
    direct_body_palette_values: tuple[int, ...] = (),
    current_focus_bbox_tuple: tuple[int, int, int, int] | None = None,
    current_focus_horizontal_axis_like: bool = False,
    current_focus_vertical_axis_like: bool = False,
) -> FrozenMap:
    """Build a bounded cell/layer audit without entering the runtime hot path."""

    if quantum <= 0:
        return FrozenMap()
    frame_grid = frame if isinstance(frame, FrameGrid) else FrameGrid(rows=frame)
    rows = frame_grid.rows
    components = detect_components(
        ComponentExtractionInput(frame=frame_grid)
    ).components
    if row_phase is None or col_phase is None:
        audit_palette_values = frozenset(
            {
                int(support_palette_value),
                int(carrier_palette_value),
                *(int(value) for value in direct_body_palette_values),
            }
        )
        phase_components = tuple(
            component
            for component in components
            if component.value in audit_palette_values
            and not component.touches_frame_boundary
        )
        row_phase_support: dict[int, int] = {}
        col_phase_support: dict[int, int] = {}
        for component in phase_components:
            row_candidate = component.bbox.top % quantum
            col_candidate = component.bbox.left % quantum
            row_phase_support[row_candidate] = (
                row_phase_support.get(row_candidate, 0) + component.area
            )
            col_phase_support[col_candidate] = (
                col_phase_support.get(col_candidate, 0) + component.area
            )
        maximum_row_support = max(row_phase_support.values(), default=0)
        maximum_col_support = max(col_phase_support.values(), default=0)
        row_phases = tuple(
            phase
            for phase, support in sorted(row_phase_support.items())
            if support == maximum_row_support
        )
        col_phases = tuple(
            phase
            for phase, support in sorted(col_phase_support.items())
            if support == maximum_col_support
        )
        if row_phase is None:
            if len(row_phases) != 1:
                return FrozenMap()
            row_phase = next(iter(row_phases))
        if col_phase is None:
            if len(col_phases) != 1:
                return FrozenMap()
            col_phase = next(iter(col_phases))
    measured_visible = _occupied_cells(
        rows=rows,
        palette_value=int(support_palette_value),
        quantum=quantum,
        row_phase=row_phase,
        col_phase=col_phase,
        top=0,
        left=0,
        bottom=len(rows) - 1,
        right=len(rows[0]) - 1,
    )
    completed = frozenset(
        (int(row), int(col)) for row, col in completed_cells
    )
    if completed:
        support_object_rows: list[dict[str, object]] = []
        for persistent_component in _orthogonally_connected_cell_sets(completed):
            component_visible = measured_visible.intersection(persistent_component)
            support_object_rows.append(
                {
                "object_ref": (
                    "measurement.persistent_support_component."
                    + stable_digest(
                        (
                            support_component_ref,
                            support_palette_value,
                            tuple(sorted(persistent_component)),
                        )
                    )[:16]
                ),
                "palette_value": int(support_palette_value),
                "persistent_support_cells": tuple(sorted(persistent_component)),
                "currently_visible_palette_cells": tuple(
                    sorted(component_visible)
                ),
                "currently_nonvisible_support_cells": tuple(
                    sorted(persistent_component - component_visible)
                ),
                "persistent_cell_count": len(persistent_component),
                "currently_visible_cell_count": len(component_visible),
                }
            )
        support_objects = tuple(support_object_rows)
        visible = measured_visible.intersection(completed)
    else:
        support_object_rows: list[dict[str, object]] = []
        for component in components:
            if (
                component.value != support_palette_value
                or component.bbox.height < 2 * quantum
                or component.bbox.width < 2 * quantum
                or component.bbox.height % quantum != 0
                or component.bbox.width % quantum != 0
            ):
                continue
            cells = _occupied_cells(
                rows=rows,
                palette_value=int(component.value),
                quantum=quantum,
                row_phase=row_phase,
                col_phase=col_phase,
                top=component.bbox.top,
                left=component.bbox.left,
                bottom=component.bbox.bottom,
                right=component.bbox.right,
            )
            if not cells:
                continue
            support_object_rows.append(
                {
                    "object_ref": str(component.component_id),
                    "palette_value": int(support_palette_value),
                    "persistent_support_cells": tuple(sorted(cells)),
                    "currently_visible_palette_cells": tuple(sorted(cells)),
                    "currently_nonvisible_support_cells": (),
                    "persistent_cell_count": len(cells),
                    "currently_visible_cell_count": len(cells),
                }
            )
        support_objects = tuple(support_object_rows)
        completed = frozenset(
            cell
            for support_object in support_objects
            for cell in support_object["persistent_support_cells"]
        )
        visible = completed
    if not completed or not support_objects:
        return FrozenMap()
    carrier_cells = _occupied_cells(
        rows=rows,
        palette_value=int(carrier_palette_value),
        quantum=quantum,
        row_phase=row_phase,
        col_phase=col_phase,
        top=0,
        left=0,
        bottom=len(rows) - 1,
        right=len(rows[0]) - 1,
    )
    if (
        carrier_row_axis_coordinate_twice is None
        or carrier_col_axis_coordinate_twice is None
    ):
        completed_row_span = (
            max(row for row, _col in completed)
            - min(row for row, _col in completed)
            + 1
        )
        completed_col_span = (
            max(col for _row, col in completed)
            - min(col for _row, col in completed)
            + 1
        )
        inferred_row, inferred_col, _row_count, _col_count = (
            _unique_crossing_axis_lines(
                carrier_cells,
                minimum_row_span=completed_col_span,
                minimum_col_span=completed_row_span,
            )
        )
        if inferred_row is None or inferred_col is None:
            return FrozenMap()
        carrier_row = inferred_row
        carrier_col = inferred_col
    else:
        carrier_row = carrier_row_axis_coordinate_twice // 2
        carrier_col = carrier_col_axis_coordinate_twice // 2
    horizontal_cells = frozenset(
        cell for cell in carrier_cells if cell[0] == carrier_row
    )
    vertical_cells = frozenset(
        cell for cell in carrier_cells if cell[1] == carrier_col
    )
    shared_axis_cells = horizontal_cells.intersection(vertical_cells)
    axis_objects = tuple(
        {
            "object_ref": (
                "measurement.lattice_line."
                + stable_digest(
                    (carrier_palette_value, orientation, tuple(sorted(cells)))
                )[:16]
            ),
            "orientation": orientation,
            "palette_value": int(carrier_palette_value),
            "support_cells": tuple(sorted(cells)),
            "cell_count": len(cells),
            "shared_with_orthogonal_line_cells": tuple(sorted(shared_axis_cells)),
            "shared_with_support_cells": tuple(
                sorted(cells.intersection(completed))
            ),
            "matches_action_conditioned_current_control": bool(
                (orientation == "horizontal" and current_focus_horizontal_axis_like)
                or (orientation == "vertical" and current_focus_vertical_axis_like)
            ),
        }
        for orientation, cells in (
            ("horizontal", horizontal_cells),
            ("vertical", vertical_cells),
        )
    )
    direct_values = frozenset(int(value) for value in direct_body_palette_values)
    body_objects: list[dict[str, object]] = []
    for component in components:
        if (
            component.area <= 1
            or component.touches_frame_boundary
            or component.value in {support_palette_value, carrier_palette_value}
            or (direct_values and component.value not in direct_values)
            or component.bbox.height < 2 * quantum
            or component.bbox.width < 2 * quantum
            or component.bbox.height % quantum != 0
            or component.bbox.width % quantum != 0
        ):
            continue
        cells = _occupied_cells(
            rows=rows,
            palette_value=int(component.value),
            quantum=quantum,
            row_phase=row_phase,
            col_phase=col_phase,
            top=component.bbox.top,
            left=component.bbox.left,
            bottom=component.bbox.bottom,
            right=component.bbox.right,
        )
        if not 2 <= len(cells) <= len(completed):
            continue
        min_row = min(row for row, _col in cells)
        min_col = min(col for _row, col in cells)
        body_objects.append(
            {
                "object_ref": str(component.component_id),
                "palette_value": int(component.value),
                "support_cells": tuple(sorted(cells)),
                "cell_count": len(cells),
                "bbox_cells": (
                    min_row,
                    min_col,
                    max(row for row, _col in cells),
                    max(col for _row, col in cells),
                ),
                "normalized_cell_offsets": tuple(
                    sorted((row - min_row, col - min_col) for row, col in cells)
                ),
                "matches_action_conditioned_current_control": bool(
                    current_focus_bbox_tuple
                    == (
                        component.bbox.top,
                        component.bbox.left,
                        component.bbox.bottom,
                        component.bbox.right,
                    )
                ),
            }
        )
    transformed_layers: list[dict[str, object]] = []
    for body_object in body_objects:
        body_cells = frozenset(body_object["support_cells"])
        row_image = _reflect_rows(body_cells, 2 * carrier_row)
        col_image = _reflect_columns(body_cells, 2 * carrier_col)
        composed_image = _reflect_rows(col_image, 2 * carrier_row)
        for transform_ref, layer_cells in (
            ("reflection.horizontal_line", row_image),
            ("reflection.vertical_line", col_image),
            ("reflection.horizontal_then_vertical", composed_image),
        ):
            transformed_layers.append(
                {
                    "object_ref": (
                        "measurement.transformed_layer."
                        + stable_digest(
                            (
                                body_object["object_ref"],
                                transform_ref,
                                tuple(sorted(layer_cells)),
                            )
                        )[:16]
                    ),
                    "source_object_ref": body_object["object_ref"],
                    "transform_ref": transform_ref,
                    "support_cells": tuple(sorted(layer_cells)),
                    "cell_count": len(layer_cells),
                    "shared_with_support_cells": tuple(
                        sorted(layer_cells.intersection(completed))
                    ),
                }
            )
    object_cells = tuple(
        (
            str(item["object_ref"]),
            frozenset(
                item.get("support_cells")
                or item.get("persistent_support_cells")
                or ()
            ),
        )
        for item in (
            *axis_objects,
            *support_objects,
            *body_objects,
            *transformed_layers,
        )
    )
    all_cells = frozenset(
        cell for _object_ref, cells in object_cells for cell in cells
    )
    return FrozenMap(
        {
            "orthogonal_reflection_lattice_axis_object_measurements": axis_objects,
            "orthogonal_reflection_lattice_support_object_measurements": support_objects,
            "orthogonal_reflection_lattice_body_object_measurements": tuple(
                body_objects
            ),
            "orthogonal_reflection_lattice_transformed_layer_measurements": tuple(
                transformed_layers
            ),
            "orthogonal_reflection_lattice_overlap_memberships": tuple(
                (
                    cell,
                    tuple(
                        object_ref
                        for object_ref, cells in object_cells
                        if cell in cells
                    ),
                )
                for cell in sorted(all_cells)
                if sum(cell in cells for _object_ref, cells in object_cells) > 1
            ),
            "orthogonal_reflection_lattice_direct_body_value_binding_present": bool(
                direct_values
            ),
        }
    )


def _reflect_rows(cells: frozenset[Cell], coordinate_twice: int) -> frozenset[Cell]:
    return frozenset((coordinate_twice - row, col) for row, col in cells)


def _reflect_columns(
    cells: frozenset[Cell], coordinate_twice: int
) -> frozenset[Cell]:
    return frozenset((row, coordinate_twice - col) for row, col in cells)


def _quadrant_signs(
    cells: frozenset[Cell],
    *,
    row_axis_twice: int,
    col_axis_twice: int,
) -> tuple[int, int] | None:
    """Measure the strict bbox quadrant occupied by one cell mask."""

    row_center_twice = min(row for row, _col in cells) + max(
        row for row, _col in cells
    )
    col_center_twice = min(col for _row, col in cells) + max(
        col for _row, col in cells
    )
    if row_center_twice == row_axis_twice or col_center_twice == col_axis_twice:
        return None
    return (
        -1 if row_center_twice < row_axis_twice else 1,
        -1 if col_center_twice < col_axis_twice else 1,
    )


def _directional_corner_front(
    cells: frozenset[Cell],
    *,
    row_sign: int,
    col_sign: int,
) -> tuple[Cell, ...]:
    """Return every non-dominated cell at one diagonal cardinal corner."""

    front = tuple(
        cell
        for cell in cells
        if not any(
            row_sign * other[0] >= row_sign * cell[0]
            and col_sign * other[1] >= col_sign * cell[1]
            and (
                row_sign * other[0] > row_sign * cell[0]
                or col_sign * other[1] > col_sign * cell[1]
            )
            for other in cells
        )
    )
    return tuple(
        sorted(
            front,
            key=lambda cell: (-row_sign * cell[0], -col_sign * cell[1]),
        )
    )


def _same_quadrant_cardinal_offsets(
    body: frozenset[Cell],
    target_components: tuple[frozenset[Cell], ...],
    *,
    row_axis_twice: int,
    col_axis_twice: int,
) -> tuple[tuple[int, int], ...]:
    """Enumerate corner correspondences without leaving the shared quadrant."""

    body_quadrant = _quadrant_signs(
        body,
        row_axis_twice=row_axis_twice,
        col_axis_twice=col_axis_twice,
    )
    if body_quadrant is None:
        return ()
    row_sign, col_sign = body_quadrant
    body_front = _directional_corner_front(
        body,
        row_sign=row_sign,
        col_sign=col_sign,
    )
    complete_containment_offsets: list[tuple[int, int]] = []
    guided_offsets: list[tuple[int, int]] = []
    for target in target_components:
        if _quadrant_signs(
            target,
            row_axis_twice=row_axis_twice,
            col_axis_twice=col_axis_twice,
        ) != body_quadrant:
            continue
        target_front = _directional_corner_front(
            target,
            row_sign=row_sign,
            col_sign=col_sign,
        )
        mechanically_possible_offsets = tuple(
            sorted(
                {
                    (target_row - body_row, target_col - body_col)
                    for target_row, target_col in target
                    for body_row, body_col in body
                }
            )
        )
        for offset in mechanically_possible_offsets:
            translated = frozenset(
                (row + offset[0], col + offset[1]) for row, col in body
            )
            if target.issubset(translated):
                complete_containment_offsets.append(offset)
        for body_cell in body_front:
            for target_cell in target_front:
                offset = (
                    target_cell[0] - body_cell[0],
                    target_cell[1] - body_cell[1],
                )
                if offset not in guided_offsets:
                    guided_offsets.append(offset)
    ordered_complete_offsets = tuple(
        offset
        for offset in guided_offsets
        if offset in frozenset(complete_containment_offsets)
    ) + tuple(
        offset
        for offset in complete_containment_offsets
        if offset not in frozenset(guided_offsets)
    )
    return tuple(dict.fromkeys((*ordered_complete_offsets, *guided_offsets)))


def _exact_cover_orbits(
    measurements: tuple[dict[str, object], ...],
    *,
    row_axis_twice: int,
    col_axis_twice: int,
) -> tuple[tuple[dict[str, object], ...], ...]:
    """Quotient exact-cover offsets by the already measured reflection group."""

    orbits: list[tuple[dict[str, object], ...]] = []
    remaining = list(measurements)
    while remaining:
        first = remaining.pop(0)
        first_cells = first["translated_cells"]
        assert isinstance(first_cells, frozenset)
        orbit_cells = frozenset(
            {
                first_cells,
                _reflect_rows(first_cells, row_axis_twice),
                _reflect_columns(first_cells, col_axis_twice),
                _reflect_rows(
                    _reflect_columns(first_cells, col_axis_twice),
                    row_axis_twice,
                ),
            }
        )
        orbit_members = [first]
        retained: list[dict[str, object]] = []
        for item in remaining:
            item_cells = item["translated_cells"]
            assert isinstance(item_cells, frozenset)
            if (
                item["body_component_ref"] == first["body_component_ref"]
                and item_cells in orbit_cells
            ):
                orbit_members.append(item)
            else:
                retained.append(item)
        remaining = retained
        orbits.append(tuple(orbit_members))
    return tuple(orbits)


def _orthogonal_completions(
    visible: frozenset[Cell],
) -> tuple[tuple[int, int, frozenset[Cell]], ...]:
    if len(visible) < 4:
        return ()
    row_min = min(row for row, _col in visible)
    row_max = max(row for row, _col in visible)
    col_min = min(col for _row, col in visible)
    col_max = max(col for _row, col in visible)
    if row_min == row_max or col_min == col_max:
        return ()
    measured: list[tuple[int, int, frozenset[Cell]]] = []
    for row_coordinate_twice in range(2 * row_min, 2 * row_max + 1):
        row_completion = visible | _reflect_rows(
            visible, row_coordinate_twice
        )
        if any(
            row < row_min or row > row_max or col < col_min or col > col_max
            for row, col in row_completion
        ):
            continue
        for col_coordinate_twice in range(2 * col_min, 2 * col_max + 1):
            column_completion = visible | _reflect_columns(
                visible, col_coordinate_twice
            )
            if row_completion != column_completion:
                continue
            if _reflect_rows(row_completion, row_coordinate_twice) != row_completion:
                continue
            if (
                _reflect_columns(row_completion, col_coordinate_twice)
                != row_completion
            ):
                continue
            measured.append(
                (
                    row_coordinate_twice,
                    col_coordinate_twice,
                    frozenset(row_completion),
                )
            )
            if len(measured) > _MAXIMUM_REFLECTION_COMPLETIONS:
                return tuple(measured)
    return tuple(measured)


def _unique_crossing_axis_lines(
    cells: frozenset[Cell],
    *,
    minimum_row_span: int,
    minimum_col_span: int,
) -> tuple[int | None, int | None, int, int]:
    row_positions: dict[int, set[int]] = {}
    col_positions: dict[int, set[int]] = {}
    for row, col in cells:
        row_positions.setdefault(row, set()).add(col)
        col_positions.setdefault(col, set()).add(row)
    row_spans = {
        row: max(cols) - min(cols) + 1 for row, cols in sorted(row_positions.items())
    }
    col_spans = {
        col: max(rows) - min(rows) + 1 for col, rows in sorted(col_positions.items())
    }
    pairs = tuple(
        (row, col, row_span + col_span)
        for row, row_span in sorted(row_spans.items())
        if row_span >= minimum_row_span
        for col, col_span in sorted(col_spans.items())
        if col_span >= minimum_col_span
        and min(row_positions[row]) <= col <= max(row_positions[row])
        and min(col_positions[col]) <= row <= max(col_positions[col])
    )
    maximum_extent = max((extent for _row, _col, extent in pairs), default=0)
    longest_pairs = tuple(
        (row, col) for row, col, extent in pairs if extent == maximum_extent
    )
    if len(longest_pairs) != 1:
        return None, None, 0, 0
    unique_row, unique_col = longest_pairs[0]
    return (
        unique_row,
        unique_col,
        len(row_positions[unique_row]),
        len(col_positions[unique_col]),
    )


def _carrier_induced_support_completions(
    value: InteractionProbeInput,
    *,
    rows: tuple[tuple[int, ...], ...],
    quantum: int,
) -> tuple[dict[str, object], ...]:
    """Close currently visible support cells under one measured carrier pair.

    This is the layer-preserving counterpart of ``_orthogonal_completions``.
    A target layer may already be partly hidden by direct or reflected bodies,
    so its two one-generator completions need not be equal in the rendered
    frame.  Once an action-conditioned carrier palette exposes one unique
    crossing row/column pair, Python may mechanically compute the finite
    four-member closure of every still-provisional support-value candidate.
    DRM remains responsible for assigning the target role to an actionable
    description.
    """

    if not value.prior_context_effect_carried_reflection_axis_observed:
        return ()
    carrier_values = tuple(
        sorted(
            {
                int(palette_value)
                for palette_value in (
                    value.prior_context_effect_carried_reflection_axis_carrier_values
                )
            }
        )
    )
    if len(carrier_values) != 1:
        return ()
    support_values = tuple(
        sorted(
            {
                int(palette_value)
                for palette_value in (
                    value.prior_context_orthogonal_reflection_support_values
                    or value.prior_game_episode_orthogonal_reflection_support_values
                )
                if int(palette_value) != carrier_values[0]
            }
        )
    )
    if not support_values:
        return ()

    components = _current_components(value)
    measurements: list[dict[str, object]] = []
    for support_value in support_values:
        eligible_support_components = tuple(
            component
            for component in components
            if int(component.value) == support_value
            and component.area > 1
            and component.bbox.height >= 2 * quantum
            and component.bbox.width >= 2 * quantum
            and component.bbox.height % quantum == 0
            and component.bbox.width % quantum == 0
        )
        phase_pairs = tuple(
            sorted(
                {
                    (
                        int(component.bbox.top % quantum),
                        int(component.bbox.left % quantum),
                    )
                    for component in eligible_support_components
                }
            )
        )
        for row_phase, col_phase in phase_pairs:
            phased_support_components = tuple(
                component
                for component in eligible_support_components
                if component.bbox.top % quantum == row_phase
                and component.bbox.left % quantum == col_phase
            )
            visible = frozenset(
                cell
                for component in phased_support_components
                for cell in _occupied_cells(
                    rows=rows,
                    palette_value=support_value,
                    quantum=quantum,
                    row_phase=row_phase,
                    col_phase=col_phase,
                    top=component.bbox.top,
                    left=component.bbox.left,
                    bottom=component.bbox.bottom,
                    right=component.bbox.right,
                )
            )
            if len(visible) < 4:
                continue
            support_component_cell_counts = frozenset(
                len(
                    _occupied_cells(
                        rows=rows,
                        palette_value=support_value,
                        quantum=quantum,
                        row_phase=row_phase,
                        col_phase=col_phase,
                        top=component.bbox.top,
                        left=component.bbox.left,
                        bottom=component.bbox.bottom,
                        right=component.bbox.right,
                    )
                )
                for component in phased_support_components
            )
            provisional_body_values = frozenset(
                int(palette_value)
                for palette_value in (
                    value.prior_context_orthogonal_reflection_direct_body_values
                    or value.prior_game_episode_orthogonal_reflection_direct_body_values
                )
            )
            provisional_body_cell_counts = tuple(
                len(
                    _occupied_cells(
                        rows=rows,
                        palette_value=int(component.value),
                        quantum=quantum,
                        row_phase=row_phase,
                        col_phase=col_phase,
                        top=component.bbox.top,
                        left=component.bbox.left,
                        bottom=component.bbox.bottom,
                        right=component.bbox.right,
                    )
                )
                for component in components
                if int(component.value) in provisional_body_values
                and component.area > 1
                and not component.touches_frame_boundary
            )
            # This completion is admitted only for the recurrent finite
            # partition relation: several movable bodies together have the
            # logical-cell cardinality of one target quarter.  The target axes
            # are first inferred from the target union itself.  A previously
            # proven aligned carrier pair is only a fallback when occlusion
            # leaves exactly two equal target components.
            partition_cell_count = sum(provisional_body_cell_counts)
            if (
                len(phased_support_components) < 2
                or len(provisional_body_cell_counts) < 2
                or partition_cell_count <= 0
                or not (
                    2 * partition_cell_count
                    <= len(visible)
                    <= 4 * partition_cell_count
                )
            ):
                continue
            grid_height = (len(rows) - row_phase + quantum - 1) // quantum
            grid_width = (len(rows[0]) - col_phase + quantum - 1) // quantum
            row_axis_candidates = tuple(
                range(
                    2 * min(row for row, _col in visible),
                    2 * max(row for row, _col in visible) + 1,
                )
            )
            col_axis_candidates = tuple(
                range(
                    2 * min(col for _row, col in visible),
                    2 * max(col for _row, col in visible) + 1,
                )
            )
            intrinsic_closure_descriptions: list[
                tuple[int, int, frozenset[Cell]]
            ] = []
            if (
                len(row_axis_candidates) * len(col_axis_candidates)
                <= _MAXIMUM_SUPPORT_AXIS_PAIR_DESCRIPTIONS
            ):
                for candidate_row_axis_twice in row_axis_candidates:
                    for candidate_col_axis_twice in col_axis_candidates:
                        candidate_completed = (
                            visible
                            | _reflect_rows(
                                visible, candidate_row_axis_twice
                            )
                            | _reflect_columns(
                                visible, candidate_col_axis_twice
                            )
                            | _reflect_rows(
                                _reflect_columns(
                                    visible, candidate_col_axis_twice
                                ),
                                candidate_row_axis_twice,
                            )
                        )
                        if len(candidate_completed) != 4 * partition_cell_count:
                            continue
                        if any(
                            row < 0
                            or row >= grid_height
                            or col < 0
                            or col >= grid_width
                            for row, col in candidate_completed
                        ):
                            continue
                        if (
                            _reflect_rows(
                                candidate_completed,
                                candidate_row_axis_twice,
                            )
                            != candidate_completed
                            or _reflect_columns(
                                candidate_completed,
                                candidate_col_axis_twice,
                            )
                            != candidate_completed
                        ):
                            continue
                        intrinsic_closure_descriptions.append(
                            (
                                candidate_row_axis_twice,
                                candidate_col_axis_twice,
                                frozenset(candidate_completed),
                            )
                        )
                        if (
                            len(intrinsic_closure_descriptions)
                            > _MAXIMUM_REFLECTION_COMPLETIONS
                        ):
                            break
                    if (
                        len(intrinsic_closure_descriptions)
                        > _MAXIMUM_REFLECTION_COMPLETIONS
                    ):
                        break
            intrinsic_closed_completion = (
                intrinsic_closure_descriptions[0]
                if len(intrinsic_closure_descriptions) == 1
                else None
            )
            if intrinsic_closed_completion is not None:
                row_axis_twice, col_axis_twice, completed = (
                    intrinsic_closed_completion
                )
            else:
                if not (
                    value.prior_context_orthogonal_reflection_axis_zero_count == 2
                    and len(phased_support_components) == 2
                    and len(support_component_cell_counts) == 1
                    and next(iter(support_component_cell_counts))
                    == partition_cell_count
                ):
                    continue
                carrier_cells = _occupied_cells(
                    rows=rows,
                    palette_value=carrier_values[0],
                    quantum=quantum,
                    row_phase=row_phase,
                    col_phase=col_phase,
                    top=0,
                    left=0,
                    bottom=len(rows) - 1,
                    right=len(rows[0]) - 1,
                )
                carrier_row, carrier_col, _row_count, _col_count = (
                    _unique_crossing_axis_lines(
                        carrier_cells,
                        minimum_row_span=2,
                        minimum_col_span=2,
                    )
                )
                if carrier_row is None or carrier_col is None:
                    continue
                row_axis_twice = 2 * carrier_row
                col_axis_twice = 2 * carrier_col
                completed = (
                    visible
                    | _reflect_rows(visible, row_axis_twice)
                    | _reflect_columns(visible, col_axis_twice)
                    | _reflect_rows(
                        _reflect_columns(visible, col_axis_twice),
                        row_axis_twice,
                    )
                )
            if (
                intrinsic_closed_completion is None
                and (
                    completed == visible
                    or len(completed) != 4 * partition_cell_count
                )
            ) or any(
                row < 0 or row >= grid_height or col < 0 or col >= grid_width
                for row, col in completed
            ):
                continue
            if (
                _reflect_rows(completed, row_axis_twice) != completed
                or _reflect_columns(completed, col_axis_twice) != completed
            ):
                continue
            measurements.append(
                {
                    "support_component_ref": (
                        "measurement.carrier_closed_support."
                        + stable_digest(
                            (
                                support_value,
                                row_phase,
                                col_phase,
                                row_axis_twice,
                                col_axis_twice,
                                tuple(sorted(visible)),
                            )
                        )[:12]
                    ),
                    "support_palette_value": support_value,
                    "row_phase": row_phase,
                    "col_phase": col_phase,
                    "support_bbox": (),
                    "visible": visible,
                    "completed": frozenset(completed),
                    "row_axis_twice": row_axis_twice,
                    "col_axis_twice": col_axis_twice,
                    "missing": frozenset(completed - visible),
                }
            )
            if len(measurements) > _MAXIMUM_REFLECTION_COMPLETIONS:
                return tuple(measurements)
    return tuple(measurements)


def _measure_orthogonal_reflection_structure(
    value: InteractionProbeInput,
) -> FrozenMap:
    """Measure a unique two-generator completion and compatible cell offsets.

    Candidate support anchors are bounded connected components.  Palette values
    remain literal observations.  A support completion is admitted only when
    reflecting the visible occupancy separately across the row and column axes
    produces the same finite mask and that mask is invariant under both
    generators.  Translation witnesses are finite offsets induced by aligning
    observed cells; no path family is enumerated.
    """

    quantum = _observed_cell_quantum(value)
    if not quantum:
        return FrozenMap(
            {
                "orthogonal_reflection_cell_quantum": 0,
                "orthogonal_reflection_completion_candidate_count": 0,
                "orthogonal_reflection_completion_enumeration_truncated": False,
                "unique_orthogonal_reflection_completion": False,
                "orthogonal_reflection_carrier_measurement_pair_present": False,
                "orthogonal_reflection_translation_candidate_count": 0,
                "orthogonal_reflection_translation_enumeration_truncated": False,
            }
        )
    frame = value.current_observed_frame or value.frame
    rows = frame.rows
    support_measurements: list[dict[str, object]] = []
    transported_completed = frozenset(
        tuple(position)
        for position in value.prior_context_orthogonal_reflection_completed_cells
    )
    transported_support_values = tuple(
        value.prior_context_orthogonal_reflection_support_values
    )
    if (
        transported_completed
        and len(transported_support_values) == 1
        and value.prior_context_orthogonal_reflection_row_phase is not None
        and value.prior_context_orthogonal_reflection_col_phase is not None
        and value.prior_context_orthogonal_reflection_row_axis_coordinate_twice
        is not None
        and value.prior_context_orthogonal_reflection_col_axis_coordinate_twice
        is not None
    ):
        transported_row_phase = int(
            value.prior_context_orthogonal_reflection_row_phase
        )
        transported_col_phase = int(
            value.prior_context_orthogonal_reflection_col_phase
        )
        transported_visible = _occupied_cells(
            rows=rows,
            palette_value=int(transported_support_values[0]),
            quantum=quantum,
            row_phase=transported_row_phase,
            col_phase=transported_col_phase,
            top=0,
            left=0,
            bottom=len(rows) - 1,
            right=len(rows[0]) - 1,
        ).intersection(transported_completed)
        support_measurements.append(
            {
                "support_component_ref": (
                    "measurement.transported_orthogonal_reflection_support"
                ),
                "support_palette_value": int(transported_support_values[0]),
                "row_phase": transported_row_phase,
                "col_phase": transported_col_phase,
                "support_bbox": (),
                "visible": frozenset(transported_visible),
                "completed": transported_completed,
                "row_axis_twice": int(
                    value.prior_context_orthogonal_reflection_row_axis_coordinate_twice
                ),
                "col_axis_twice": int(
                    value.prior_context_orthogonal_reflection_col_axis_coordinate_twice
                ),
                "missing": transported_completed - transported_visible,
            }
        )
    if not support_measurements:
        support_measurements.extend(
            _carrier_induced_support_completions(
                value,
                rows=rows,
                quantum=quantum,
            )
        )
    anchors = tuple(
        component
        for component in _current_components(value)
        if component.area > 1
        and component.value
        not in frozenset(
            value.prior_context_effect_carried_reflection_axis_carrier_values
        )
        and not component.touches_frame_boundary
        and component.bbox.height >= 2 * quantum
        and component.bbox.width >= 2 * quantum
        and component.bbox.height % quantum == 0
        and component.bbox.width % quantum == 0
    )[:_MAXIMUM_SUPPORT_ANCHORS]
    for component in (() if support_measurements else anchors):
        if (
            value.prior_context_orthogonal_reflection_support_values
            and component.value
            not in frozenset(
                value.prior_context_orthogonal_reflection_support_values
            )
        ):
            continue
        row_phase = component.bbox.top % quantum
        col_phase = component.bbox.left % quantum
        visible = _occupied_cells(
            rows=rows,
            palette_value=component.value,
            quantum=quantum,
            row_phase=row_phase,
            col_phase=col_phase,
            top=component.bbox.top,
            left=component.bbox.left,
            bottom=component.bbox.bottom,
            right=component.bbox.right,
        )
        completions = _orthogonal_completions(visible)
        for row_axis_twice, col_axis_twice, completed in completions:
            missing = completed - visible
            # A completion must add at least one latent locus.  Already closed
            # symmetric decorations remain valid measured shapes, but they do
            # not compete as explanations of an incomplete target support.
            if not missing:
                continue
            support_measurements.append(
                {
                    "support_component_ref": component.component_id,
                    "support_palette_value": int(component.value),
                    "row_phase": row_phase,
                    "col_phase": col_phase,
                    "support_bbox": (
                        component.bbox.top,
                        component.bbox.left,
                        component.bbox.bottom,
                        component.bbox.right,
                    ),
                    "visible": visible,
                    "completed": completed,
                    "row_axis_twice": row_axis_twice,
                    "col_axis_twice": col_axis_twice,
                    "missing": missing,
                }
            )
            if len(support_measurements) > _MAXIMUM_REFLECTION_COMPLETIONS:
                return FrozenMap(
                    {
                        "orthogonal_reflection_cell_quantum": quantum,
                        "orthogonal_reflection_completion_candidate_count": len(
                            support_measurements
                        ),
                        "orthogonal_reflection_completion_enumeration_truncated": True,
                        "unique_orthogonal_reflection_completion": False,
                        "orthogonal_reflection_carrier_measurement_pair_present": False,
                        "orthogonal_reflection_translation_candidate_count": 0,
                        "orthogonal_reflection_translation_enumeration_truncated": False,
                    }
                )
    if len(support_measurements) != 1:
        return FrozenMap(
            {
                "orthogonal_reflection_cell_quantum": quantum,
                "orthogonal_reflection_completion_candidate_count": len(
                    support_measurements
                ),
                "orthogonal_reflection_completion_enumeration_truncated": False,
                "unique_orthogonal_reflection_completion": False,
                "orthogonal_reflection_carrier_measurement_pair_present": False,
                "orthogonal_reflection_translation_candidate_count": 0,
                "orthogonal_reflection_translation_enumeration_truncated": False,
            }
        )

    unique_support = support_measurements[0]
    visible = unique_support["visible"]
    completed = unique_support["completed"]
    missing = unique_support["missing"]
    assert isinstance(visible, frozenset)
    assert isinstance(completed, frozenset)
    assert isinstance(missing, frozenset)
    row_phase = int(unique_support["row_phase"])
    col_phase = int(unique_support["col_phase"])
    row_axis_twice = int(unique_support["row_axis_twice"])
    col_axis_twice = int(unique_support["col_axis_twice"])
    fully_visible_target_cells = _fully_palette_cells(
        rows=rows,
        palette_value=int(unique_support["support_palette_value"]),
        quantum=quantum,
        row_phase=row_phase,
        col_phase=col_phase,
        candidate_cells=completed,
    )

    missing_value_sets = tuple(
        tuple(
            sorted(
                {
                    rows[pixel_row][pixel_col]
                    for pixel_row in range(
                        row_phase + row * quantum,
                        min(row_phase + (row + 1) * quantum, len(rows)),
                    )
                    for pixel_col in range(
                        col_phase + col * quantum,
                        min(col_phase + (col + 1) * quantum, len(rows[0])),
                    )
                }
            )
        )
        for row, col in sorted(missing)
    )

    completed_row_span = (
        max(row for row, _col in completed)
        - min(row for row, _col in completed)
        + 1
    )
    completed_col_span = (
        max(col for _row, col in completed)
        - min(col for _row, col in completed)
        + 1
    )
    carrier_measurements: list[tuple[int, int, int, int, int]] = []
    measured_carrier_values = tuple(
        sorted(
            frozenset(
                value.prior_context_effect_carried_reflection_axis_carrier_values
            )
        )
    )
    preserved_terminal_carrier_pair = bool(
        value.prior_context_orthogonal_reflection_axis_zero_count == 2
        and len(measured_carrier_values) == 1
        and value.prior_context_orthogonal_reflection_carrier_row_axis_coordinate_twice
        is not None
        and value.prior_context_orthogonal_reflection_carrier_col_axis_coordinate_twice
        is not None
        and int(
            value.prior_context_orthogonal_reflection_carrier_row_axis_coordinate_twice
        )
        % 2
        == 0
        and int(
            value.prior_context_orthogonal_reflection_carrier_col_axis_coordinate_twice
        )
        % 2
        == 0
    )
    for palette_value in measured_carrier_values:
        occupied = _occupied_cells(
            rows=rows,
            palette_value=palette_value,
            quantum=quantum,
            row_phase=row_phase,
            col_phase=col_phase,
            top=0,
            left=0,
            bottom=len(rows) - 1,
            right=len(rows[0]) - 1,
        )
        if preserved_terminal_carrier_pair:
            unique_row = int(
                value.prior_context_orthogonal_reflection_carrier_row_axis_coordinate_twice
            ) // 2
            unique_col = int(
                value.prior_context_orthogonal_reflection_carrier_col_axis_coordinate_twice
            ) // 2
            row_count = sum(1 for row, _col in occupied if row == unique_row)
            col_count = sum(1 for _row, col in occupied if col == unique_col)
        else:
            unique_row, unique_col, row_count, col_count = (
                _unique_crossing_axis_lines(
                    occupied,
                    minimum_row_span=completed_col_span,
                    minimum_col_span=completed_row_span,
                )
            )
        if unique_row is not None and unique_col is not None:
            carrier_measurements.append(
                (palette_value, unique_row, unique_col, row_count, col_count)
            )

    translation_measurements: list[dict[str, object]] = []
    exact_cover_measurements: list[dict[str, object]] = []
    partial_partition_measurements_by_body: dict[
        str, list[dict[str, object]]
    ] = {}
    body_cell_counts_by_ref: dict[str, int] = {}
    all_body_cell_counts_by_value: dict[int, dict[str, int]] = {}
    body_palette_values_by_ref: dict[str, int] = {}
    body_normalized_cell_offsets_by_ref: dict[str, tuple[Cell, ...]] = {}
    current_covered_target_cells: set[Cell] = set()
    current_covered_target_cells_by_body_ref: dict[str, frozenset[Cell]] = {}
    all_translation_measurements: list[dict[str, object]] = []
    joint_minimum_anchor_measurements: list[dict[str, object]] = []
    positive_coverage_measurements: list[dict[str, object]] = []
    cardinal_positive_coverage_measurements: list[dict[str, object]] = []
    first_complete_same_quadrant_cardinal_measurement: dict[str, object] | None = None
    same_quadrant_cardinal_simulation_attempt_count = 0
    best_positive_coverage_key: tuple[int, int] | None = None
    maximum_coverage_enumeration_truncated = False
    cardinal_maximum_coverage_enumeration_truncated = False
    exact_cover_enumeration_truncated = False
    partial_partition_enumeration_truncated = False
    joint_minimum_enumeration_truncated = False
    support_component_ref = str(unique_support["support_component_ref"])
    support_palette_value = int(unique_support["support_palette_value"])
    support_component_cell_sets = tuple(
        _occupied_cells(
            rows=rows,
            palette_value=support_palette_value,
            quantum=quantum,
            row_phase=row_phase,
            col_phase=col_phase,
            top=component.bbox.top,
            left=component.bbox.left,
            bottom=component.bbox.bottom,
            right=component.bbox.right,
        )
        for component in value.components.components
        if component.value == support_palette_value
        and component.bbox.height >= 2 * quantum
        and component.bbox.width >= 2 * quantum
        and component.bbox.height % quantum == 0
        and component.bbox.width % quantum == 0
    )
    non_singleton_support_component_cell_sets = tuple(
        cells for cells in support_component_cell_sets if len(cells) > 1
    )
    persistent_support_component_cell_sets = tuple(
        cells
        for cells in _orthogonally_connected_cell_sets(completed)
        if len(cells) > 1
    )
    placement_support_component_cell_sets = (
        persistent_support_component_cell_sets
        if persistent_support_component_cell_sets
        else non_singleton_support_component_cell_sets
    )
    # Only completely target-valued logical cells remain terminally uncovered.
    # Mixed cells still belong to the persistent target layer, but a rendered
    # cyan/direct/derived layer already covers them for the current objective.
    coverage_target = fully_visible_target_cells
    carrier_values = frozenset(
        value.prior_context_effect_carried_reflection_axis_carrier_values
    )
    committed_direct_body_values = frozenset(
        value.prior_context_orthogonal_reflection_direct_body_values
    )
    committed_direct_body_normalized_offsets = frozenset(
        value.prior_context_orthogonal_reflection_direct_body_normalized_cell_offsets
    )
    remaining_body_focus_transfer_active = bool(
        value.committed_plan_kind_ref
        == "orthogonal_reflection_focus_transfer_series"
        and committed_direct_body_normalized_offsets
    )
    placement_target = completed
    prior_committed_body_morphology_filtered_count = 0
    candidate_direct_body_values = (
        committed_direct_body_values
        or frozenset(value.prior_game_episode_orthogonal_reflection_direct_body_values)
    )
    anchor_body_cell_overrides: dict[str, frozenset[Cell]] = {}
    expanded_anchors: list[ComponentDescription] = []
    carried_body_morphologies = tuple(
        tuple((int(row), int(col)) for row, col in morphology)
        for morphology in (
            value.prior_context_orthogonal_reflection_direct_body_morphology_collections
        )
        if morphology
    )
    for component in anchors:
        raw_body_cells = _occupied_cells(
            rows=rows,
            palette_value=int(component.value),
            quantum=quantum,
            row_phase=row_phase,
            col_phase=col_phase,
            top=component.bbox.top,
            left=component.bbox.left,
            bottom=component.bbox.bottom,
            right=component.bbox.right,
        )
        decomposition = (
            _bounded_morphology_decomposition(
                raw_body_cells,
                carried_body_morphologies,
            )
            if carried_body_morphologies
            and int(component.value) in candidate_direct_body_values
            else ()
        )
        if not decomposition:
            expanded_anchors.append(component)
            continue
        for index, virtual_cells in enumerate(decomposition):
            virtual_pixels = tuple(
                sorted(
                    (
                        row_phase + row * quantum + pixel_row,
                        col_phase + col * quantum + pixel_col,
                    )
                    for row, col in virtual_cells
                    for pixel_row in range(quantum)
                    for pixel_col in range(quantum)
                )
            )
            virtual_bbox = BoundingBox(
                top=min(row for row, _col in virtual_pixels),
                left=min(col for _row, col in virtual_pixels),
                bottom=max(row for row, _col in virtual_pixels),
                right=max(col for _row, col in virtual_pixels),
            )
            virtual_ref = (
                "measurement.layered_body."
                + stable_digest(
                    (
                        str(component.component_id),
                        index,
                        tuple(sorted(virtual_cells)),
                    )
                )[:16]
            )
            expanded_anchors.append(
                ComponentDescription(
                    component_id=virtual_ref,
                    value=int(component.value),
                    pixels=virtual_pixels,
                    relative_pixels=tuple(
                        (row - virtual_bbox.top, col - virtual_bbox.left)
                        for row, col in virtual_pixels
                    ),
                    bbox=virtual_bbox,
                    area=len(virtual_pixels),
                    touches_frame_boundary=(
                        virtual_bbox.top == 0
                        or virtual_bbox.left == 0
                        or virtual_bbox.bottom == len(rows) - 1
                        or virtual_bbox.right == len(rows[0]) - 1
                    ),
                )
            )
            anchor_body_cell_overrides[virtual_ref] = virtual_cells
    anchors = tuple(expanded_anchors[:_MAXIMUM_SUPPORT_ANCHORS])
    action_conditioned_anchor_refs = frozenset(
        component.component_id
        for component in anchors
        if any(
            max(
                abs(change_bbox.top - component.bbox.top),
                abs(change_bbox.left - component.bbox.left),
                abs(change_bbox.bottom - component.bbox.bottom),
                abs(change_bbox.right - component.bbox.right),
            )
            <= quantum
            for change_bbox in value.transient_change_bboxes
        )
    )
    action_conditioned_body_cell_counts: list[int] = []
    for component in anchors:
        component_bbox_tuple = (
            component.bbox.top,
            component.bbox.left,
            component.bbox.bottom,
            component.bbox.right,
        )
        action_conditioned_identity_candidate = (
            component.component_id in action_conditioned_anchor_refs
        )
        if (
            component.component_id == support_component_ref
            or component.value == support_palette_value
            or component.value in carrier_values
            or (
                candidate_direct_body_values
                and component.value not in candidate_direct_body_values
                and not action_conditioned_identity_candidate
            )
        ):
            continue
        body = anchor_body_cell_overrides.get(str(component.component_id))
        if body is None:
            body = _occupied_cells(
                rows=rows,
                palette_value=component.value,
                quantum=quantum,
                row_phase=row_phase,
                col_phase=col_phase,
                top=component.bbox.top,
                left=component.bbox.left,
                bottom=component.bbox.bottom,
                right=component.bbox.right,
            )
        body_min_row = min((row for row, _col in body), default=0)
        body_min_col = min((col for _row, col in body), default=0)
        body_normalized_cell_offsets = tuple(
            sorted(
                (row - body_min_row, col - body_min_col)
                for row, col in body
            )
        )
        if not 2 <= len(body) <= len(completed):
            continue
        body_component_ref = str(component.component_id)
        all_body_cell_counts_by_value.setdefault(int(component.value), {})[
            body_component_ref
        ] = len(body)
        current_body_image_union = body
        if len(carrier_measurements) == 1:
            current_carrier_row_axis_twice = 2 * carrier_measurements[0][1]
            current_carrier_col_axis_twice = 2 * carrier_measurements[0][2]
            current_body_image_union = (
                body
                | _reflect_rows(body, current_carrier_row_axis_twice)
                | _reflect_columns(body, current_carrier_col_axis_twice)
                | _reflect_rows(
                    _reflect_columns(body, current_carrier_col_axis_twice),
                    current_carrier_row_axis_twice,
                )
            )
        current_body_covered_target_cells = completed.intersection(
            current_body_image_union
        )
        current_covered_target_cells_by_body_ref[body_component_ref] = (
            current_body_covered_target_cells
        )
        current_covered_target_cells.update(current_body_covered_target_cells)
        if (
            remaining_body_focus_transfer_active
            and frozenset(body_normalized_cell_offsets)
            == committed_direct_body_normalized_offsets
            and value.committed_focus_transfer_body_context_exhausted_declared
        ):
            prior_committed_body_morphology_filtered_count += 1
            continue
        if action_conditioned_identity_candidate:
            action_conditioned_body_cell_counts.append(len(body))
        body_cell_counts_by_ref[body_component_ref] = len(body)
        body_palette_values_by_ref[body_component_ref] = int(component.value)
        body_normalized_cell_offsets_by_ref[body_component_ref] = (
            body_normalized_cell_offsets
        )
        body_anchor_row, body_anchor_col = min(body)
        cardinal_corner_key_functions = (
            lambda cell: (cell[0], cell[1]),
            lambda cell: (cell[0], -cell[1]),
            lambda cell: (-cell[0], cell[1]),
            lambda cell: (-cell[0], -cell[1]),
        )
        cardinal_corner_offsets = tuple(
            (
                target_corner[0] - body_corner[0],
                target_corner[1] - body_corner[1],
            )
            for corner_key in cardinal_corner_key_functions
            for body_corner, target_corner in (
                (min(body, key=corner_key), min(coverage_target, key=corner_key)),
            )
        ) if coverage_target else ()
        same_quadrant_cardinal_offsets = _same_quadrant_cardinal_offsets(
            body,
            placement_support_component_cell_sets,
            row_axis_twice=row_axis_twice,
            col_axis_twice=col_axis_twice,
        )
        exact_cover_offsets = tuple(
            sorted(
                {
                    (
                        target_row - body_anchor_row,
                        target_col - body_anchor_col,
                    )
                    for target_row, target_col in completed
                }
            )
        )
        if len(exact_cover_offsets) > _MAXIMUM_TRANSLATION_OFFSETS:
            continue
        component_cover_offsets = tuple(
            sorted(
                {
                    (target_row - body_row, target_col - body_col)
                    for target_component in non_singleton_support_component_cell_sets
                    for target_row, target_col in target_component
                    for body_row, body_col in body
                }
            )
        )
        if len(component_cover_offsets) > _MAXIMUM_TRANSLATION_OFFSETS:
            component_cover_offsets = ()
        broad_offsets = tuple(
            sorted(
                {
                    (target_row - body_row, target_col - body_col)
                    for target_row, target_col in placement_target
                    for body_row, body_col in body
                }
            )
        )
        primary_offsets = tuple(
            dict.fromkeys(
                (
                    *same_quadrant_cardinal_offsets,
                    *cardinal_corner_offsets,
                    *exact_cover_offsets,
                    *component_cover_offsets,
                )
            )
        )
        offsets = primary_offsets + tuple(
            offset
            for offset in (
                broad_offsets
                if len(broad_offsets) <= _MAXIMUM_TRANSLATION_OFFSETS
                else ()
            )
            if offset not in frozenset(primary_offsets)
        )
        if len(offsets) > _MAXIMUM_TRANSLATION_OFFSETS:
            offsets = offsets[:_MAXIMUM_TRANSLATION_OFFSETS]
            exact_cover_enumeration_truncated = True
            partial_partition_enumeration_truncated = True
            maximum_coverage_enumeration_truncated = True
            cardinal_maximum_coverage_enumeration_truncated = True
        for delta_row, delta_col in offsets:
            translated = frozenset(
                (row + delta_row, col + delta_col) for row, col in body
            )
            row_image = _reflect_rows(translated, row_axis_twice)
            col_image = _reflect_columns(translated, col_axis_twice)
            composed_image = _reflect_rows(col_image, row_axis_twice)
            image_union = translated | row_image | col_image | composed_image
            translated_body_covers_support_component = any(
                target_component.issubset(translated)
                for target_component in non_singleton_support_component_cell_sets
            )
            target_covered_by_image_union = completed.issubset(image_union)
            translated_body_is_current_target_subset = translated.issubset(
                placement_target
            )
            image_union_is_current_target_subset = image_union.issubset(
                placement_target
            )
            covered_target_cell_count = len(
                placement_target.intersection(image_union)
            )
            extra_image_cell_count = len(image_union - placement_target)
            translated_touches_joint_minimum_extremum = bool(
                min(row for row, _col in translated)
                == min(row for row, _col in completed)
                and min(col for _row, col in translated)
                == min(col for _row, col in completed)
            )
            measurement: dict[str, object] = {
                    "body_component_ref": component.component_id,
                    "body_bbox_tuple": (
                        component.bbox.top,
                        component.bbox.left,
                        component.bbox.bottom,
                        component.bbox.right,
                    ),
                    "body_palette_value": int(component.value),
                    "body_cell_count": len(body),
                    "body_normalized_cell_offsets": body_normalized_cell_offsets,
                    "delta_row": delta_row,
                    "delta_col": delta_col,
                    "image_union_count": len(image_union),
                    "image_union_cells": image_union,
                    "image_union_extra_count": extra_image_cell_count,
                    "covered_target_cell_count": covered_target_cell_count,
                    "cardinal_corner_alignment": (
                        (delta_row, delta_col)
                        in frozenset(
                            (*same_quadrant_cardinal_offsets, *cardinal_corner_offsets)
                        )
                    ),
                    "image_overlap_count": (
                        4 * len(body) - len(image_union)
                    ),
                    "translated_body_is_target_subset": translated.issubset(
                        completed
                    ),
                    "translated_body_is_current_target_subset": (
                        translated_body_is_current_target_subset
                    ),
                    "image_union_is_current_target_subset": (
                        image_union_is_current_target_subset
                    ),
                    "current_target_remaining_cell_count_after_image_union": len(
                        placement_target - image_union
                    ),
                    "translated_body_covers_support_component": (
                        translated_body_covers_support_component
                    ),
                    "target_covered_by_image_union": target_covered_by_image_union,
                    "translated_body_touches_joint_minimum_extremum": (
                        translated_touches_joint_minimum_extremum
                    ),
                    "translated_cells": translated,
                }
            all_translation_measurements.append(measurement)
            if (
                translated_body_is_current_target_subset
                and image_union_is_current_target_subset
            ):
                partial_partition_measurements_by_body.setdefault(
                    body_component_ref, []
                ).append(measurement)
            same_quadrant_cardinal_candidate = (
                (delta_row, delta_col)
                in frozenset(same_quadrant_cardinal_offsets)
            )
            if (
                same_quadrant_cardinal_candidate
                and (
                    not action_conditioned_anchor_refs
                    or action_conditioned_identity_candidate
                )
            ):
                same_quadrant_cardinal_simulation_attempt_count += 1
                if (
                    translated_body_covers_support_component
                    and first_complete_same_quadrant_cardinal_measurement is None
                ):
                    first_complete_same_quadrant_cardinal_measurement = measurement
            if (
                covered_target_cell_count > 0
                and (
                    not action_conditioned_anchor_refs
                    or action_conditioned_identity_candidate
                )
            ):
                coverage_key = (
                    covered_target_cell_count,
                    -extra_image_cell_count,
                )
                if (
                    best_positive_coverage_key is None
                    or coverage_key > best_positive_coverage_key
                ):
                    best_positive_coverage_key = coverage_key
                    positive_coverage_measurements = [measurement]
                    cardinal_positive_coverage_measurements = (
                        [measurement]
                        if bool(measurement["cardinal_corner_alignment"])
                        else []
                    )
                    maximum_coverage_enumeration_truncated = False
                    cardinal_maximum_coverage_enumeration_truncated = False
                elif (
                    coverage_key == best_positive_coverage_key
                    and len(positive_coverage_measurements)
                    < _MAXIMUM_TRANSLATION_WITNESSES
                ):
                    positive_coverage_measurements.append(measurement)
                elif coverage_key == best_positive_coverage_key:
                    maximum_coverage_enumeration_truncated = True
                if (
                    coverage_key == best_positive_coverage_key
                    and bool(measurement["cardinal_corner_alignment"])
                    and measurement
                    not in cardinal_positive_coverage_measurements
                ):
                    if (
                        len(cardinal_positive_coverage_measurements)
                        < _MAXIMUM_TRANSLATION_WITNESSES
                    ):
                        cardinal_positive_coverage_measurements.append(
                            measurement
                        )
                    else:
                        cardinal_maximum_coverage_enumeration_truncated = True
            if (
                not target_covered_by_image_union
                and not translated_body_covers_support_component
            ):
                continue
            if len(translation_measurements) < _MAXIMUM_TRANSLATION_WITNESSES:
                translation_measurements.append(measurement)
            if bool(measurement["translated_body_is_target_subset"]) or bool(
                measurement["translated_body_covers_support_component"]
            ):
                if len(exact_cover_measurements) < _MAXIMUM_TRANSLATION_WITNESSES:
                    exact_cover_measurements.append(measurement)
                else:
                    exact_cover_enumeration_truncated = True
            if bool(measurement["translated_body_touches_joint_minimum_extremum"]):
                if (
                    len(joint_minimum_anchor_measurements)
                    < _MAXIMUM_TRANSLATION_WITNESSES
                ):
                    joint_minimum_anchor_measurements.append(measurement)
                else:
                    joint_minimum_enumeration_truncated = True
            if first_complete_same_quadrant_cardinal_measurement is measurement:
                break

    for measurement in all_translation_measurements:
        candidate_body_ref = str(measurement["body_component_ref"])
        # Two commuting reflection generators produce four image occurrences
        # per source cell. Coincident axis layers keep their multiplicity.
        measurement["direct_family_cell_occurrence_count"] = 4 * sum(
            all_body_cell_counts_by_value[int(measurement["body_palette_value"])].values()
        )
        measurement["target_cell_occurrence_count"] = _axis_multiplicity_cell_count(
            placement_target, row_axis_twice=row_axis_twice,
            col_axis_twice=col_axis_twice,
        )
        measurement["direct_family_target_cardinality_delta"] = (
            measurement["direct_family_cell_occurrence_count"]
            - measurement["target_cell_occurrence_count"]
        )
        measurement["strict_placement_count_for_body"] = len(
            partial_partition_measurements_by_body.get(candidate_body_ref, ())
        )
        other_current_coverage = frozenset(
            cell
            for body_ref, covered_cells in (
                sorted(current_covered_target_cells_by_body_ref.items())
            )
            if body_ref != candidate_body_ref
            for cell in covered_cells
        )
        candidate_image_union = measurement["image_union_cells"]
        assert isinstance(candidate_image_union, frozenset)
        measurement[
            "current_target_remaining_cell_count_after_image_union"
        ] = len(completed - (other_current_coverage | candidate_image_union))

    cumulative_terminal_measurements = tuple(
        measurement
        for measurement in all_translation_measurements
        if int(
            measurement["current_target_remaining_cell_count_after_image_union"]
        )
        == 0
    )
    cumulative_terminal_orbits = _exact_cover_orbits(
        cumulative_terminal_measurements,
        row_axis_twice=row_axis_twice,
        col_axis_twice=col_axis_twice,
    )
    cumulative_terminal_representatives = tuple(
        min(
            orbit,
            key=lambda item: (
                abs(int(item["delta_row"])) + abs(int(item["delta_col"])),
                str(item["body_component_ref"]),
                int(item["delta_row"]),
                int(item["delta_col"]),
            ),
        )
        for orbit in cumulative_terminal_orbits
    )
    # Exact distance reduction over already bounded terminal witnesses. Keep
    # ties unresolved; cardinality-dependent admissibility belongs to DRM.
    minimum_terminal_distance = min(
        (abs(int(item["delta_row"])) + abs(int(item["delta_col"]))
         for item in cumulative_terminal_representatives),
        default=None,
    )
    cumulative_terminal_representatives = tuple(
        item for item in cumulative_terminal_representatives
        if abs(int(item["delta_row"])) + abs(int(item["delta_col"]))
        == minimum_terminal_distance
    )

    if first_complete_same_quadrant_cardinal_measurement is not None:
        cardinal_positive_coverage_measurements = [
            first_complete_same_quadrant_cardinal_measurement
        ]
        cardinal_maximum_coverage_enumeration_truncated = False

    bounded_exact_cover_measurements = tuple(
        exact_cover_measurements[:_MAXIMUM_TRANSLATION_WITNESSES]
    )
    bounded_joint_minimum_measurements = tuple(
        joint_minimum_anchor_measurements[:_MAXIMUM_TRANSLATION_WITNESSES]
    )
    maximum_coverage_measurements = tuple(positive_coverage_measurements)
    maximum_coverage_orbits = _exact_cover_orbits(
        maximum_coverage_measurements,
        row_axis_twice=row_axis_twice,
        col_axis_twice=col_axis_twice,
    )
    maximum_coverage_representatives = tuple(
        min(
            orbit,
            key=lambda item: (
                str(item["body_component_ref"]),
                int(item["delta_row"]),
                int(item["delta_col"]),
            ),
        )
        for orbit in maximum_coverage_orbits
    )
    cardinal_maximum_coverage_measurements = tuple(
        cardinal_positive_coverage_measurements
    )
    cardinal_maximum_coverage_orbits = _exact_cover_orbits(
        cardinal_maximum_coverage_measurements,
        row_axis_twice=row_axis_twice,
        col_axis_twice=col_axis_twice,
    )
    cardinal_maximum_coverage_representatives = tuple(
        min(
            orbit,
            key=lambda item: (
                str(item["body_component_ref"]),
                int(item["delta_row"]),
                int(item["delta_col"]),
            ),
        )
        for orbit in cardinal_maximum_coverage_orbits
    )
    exact_cover_orbits = _exact_cover_orbits(
        bounded_exact_cover_measurements,
        row_axis_twice=row_axis_twice,
        col_axis_twice=col_axis_twice,
    )
    exact_cover_orbit_representatives = tuple(
        min(
            orbit,
            key=lambda item: (
                str(item["body_component_ref"]),
                int(item["delta_row"]),
                int(item["delta_col"]),
            ),
        )
        for orbit in exact_cover_orbits
    )
    target_component_cell_counts = frozenset(
        len(cells) for cells in placement_support_component_cell_sets
    )
    body_refs_by_palette: dict[int, list[str]] = {}
    for body_component_ref, palette_value in sorted(
        body_palette_values_by_ref.items()
    ):
        body_refs_by_palette.setdefault(palette_value, []).append(
            body_component_ref
        )
    partial_partition_body_families = tuple(
        tuple(sorted(body_refs))
        for _palette_value, body_refs in sorted(body_refs_by_palette.items())
        if len(target_component_cell_counts) == 1
        and len(body_refs) >= 2
        and sum(body_cell_counts_by_ref[ref] for ref in body_refs)
        == next(iter(target_component_cell_counts))
        and all(partial_partition_measurements_by_body.get(ref) for ref in body_refs)
    )
    if len(partial_partition_body_families) > _MAXIMUM_TRANSLATION_WITNESSES:
        partial_partition_enumeration_truncated = True
    unique_partial_partition_family = (
        partial_partition_body_families[0]
        if len(partial_partition_body_families) == 1
        and not partial_partition_enumeration_truncated
        else ()
    )
    unique_placement_partial_partition_measurements = tuple(
        partial_partition_measurements_by_body[body_component_ref][0]
        for body_component_ref in unique_partial_partition_family
        if len(partial_partition_measurements_by_body[body_component_ref]) == 1
    )
    unique_partial_partition_translation = (
        unique_placement_partial_partition_measurements[0]
        if len(unique_placement_partial_partition_measurements) == 1
        else None
    )
    partial_partition_candidate_count = sum(
        len(partial_partition_measurements_by_body[body_component_ref])
        for body_component_ref in unique_partial_partition_family
    )
    measured_selectable_body_count = len(
        {
            body_component_ref
            for palette_value in sorted(body_refs_by_palette)
            for body_component_ref in sorted(
                body_refs_by_palette[palette_value]
            )
        }
    )
    current_focus_bbox_tuples = tuple(
        (bbox.top, bbox.left, bbox.bottom, bbox.right)
        for bbox in value.input_aligned_entity_bboxes
    )
    common: dict[str, object] = {
        "orthogonal_reflection_cell_quantum": quantum,
        "orthogonal_reflection_completion_candidate_count": 1,
        "orthogonal_reflection_completion_enumeration_truncated": False,
        "unique_orthogonal_reflection_completion": True,
        "orthogonal_reflection_carrier_measurement_pair_present": False,
        "orthogonal_reflection_terminal_carrier_pair_preserved_through_occlusion": (
            preserved_terminal_carrier_pair
        ),
        "orthogonal_reflection_current_focus_bbox_measurement_present": (
            len(current_focus_bbox_tuples) == 1
        ),
        "orthogonal_reflection_current_focus_bbox_tuple": (
            current_focus_bbox_tuples[0]
            if len(current_focus_bbox_tuples) == 1
            else None
        ),
        "orthogonal_reflection_current_focus_horizontal_axis_like": False,
        "orthogonal_reflection_current_focus_vertical_axis_like": False,
        "orthogonal_reflection_current_focus_matches_open_axis": False,
        "orthogonal_reflection_support_component_ref": support_component_ref,
        "orthogonal_reflection_support_palette_value": support_palette_value,
        "orthogonal_reflection_completed_cell_positions": tuple(
            sorted(completed)
        ),
        "orthogonal_reflection_row_phase": row_phase,
        "orthogonal_reflection_col_phase": col_phase,
        "orthogonal_reflection_visible_cell_count": len(visible),
        # Exact current terminal predicate: count the persistent target cells
        # whose target attribute remains fully visible now.  Simulated body
        # images guide a future placement, but must never be mistaken for a
        # layer that the environment has actually rendered.
        "orthogonal_reflection_uncovered_target_cell_count": len(
            fully_visible_target_cells
        ),
        "orthogonal_reflection_current_covered_target_cell_count": (
            len(completed) - len(fully_visible_target_cells)
        ),
        "orthogonal_reflection_coverage_target_cell_count": len(
            coverage_target
        ),
        "orthogonal_reflection_completed_cell_count": len(completed),
        "orthogonal_reflection_missing_cell_count": len(missing),
        "orthogonal_reflection_action_conditioned_body_cell_counts": tuple(
            action_conditioned_body_cell_counts[:3]
        ),
        "orthogonal_reflection_measured_selectable_body_count": (
            measured_selectable_body_count
        ),
        "orthogonal_reflection_direct_body_morphology_collections": tuple(
            sorted(
                set(body_normalized_cell_offsets_by_ref.values()),
                key=lambda morphology: (len(morphology), morphology),
            )
        ),
        "orthogonal_reflection_prior_committed_body_morphology_filtered_count": (
            prior_committed_body_morphology_filtered_count
        ),
        "orthogonal_reflection_missing_cell_value_sets": missing_value_sets,
        "orthogonal_reflection_row_axis_coordinate_twice": row_axis_twice,
        "orthogonal_reflection_col_axis_coordinate_twice": col_axis_twice,
        "orthogonal_reflection_generator_count": 2,
        "orthogonal_reflection_closure_member_count": 4,
        "orthogonal_reflection_generators_involutive": True,
        "orthogonal_reflection_generators_commute": True,
        "orthogonal_reflection_separate_completions_equal": True,
        "orthogonal_reflection_translation_candidate_count": len(
            bounded_exact_cover_measurements
        ),
        "orthogonal_reflection_translation_candidate_tuples": tuple(
            (
                str(item["body_component_ref"]),
                int(item["body_palette_value"]),
                int(item["body_cell_count"]),
                int(item["delta_row"]),
                int(item["delta_col"]),
                int(item["image_union_count"]),
                int(item["image_union_extra_count"]),
                int(item["image_overlap_count"]),
                bool(item["translated_body_is_target_subset"]),
                bool(item["translated_body_touches_joint_minimum_extremum"]),
            )
            for item in bounded_exact_cover_measurements
        ),
        "orthogonal_reflection_joint_minimum_anchor_candidate_count": len(
            bounded_joint_minimum_measurements
        ),
        "orthogonal_reflection_maximum_coverage_candidate_count": len(
            maximum_coverage_measurements
        ),
        "orthogonal_reflection_maximum_coverage_enumeration_truncated": (
            maximum_coverage_enumeration_truncated
        ),
        "orthogonal_reflection_maximum_coverage_orbit_count": len(
            maximum_coverage_orbits
        ),
        "orthogonal_reflection_cumulative_terminal_candidate_count": len(
            cumulative_terminal_measurements
        ),
        "orthogonal_reflection_cumulative_terminal_orbit_count": len(
            cumulative_terminal_orbits
        ),
        "orthogonal_reflection_unique_cumulative_terminal_measurement_present": (
            len(cumulative_terminal_representatives) == 1
        ),
        "orthogonal_reflection_cardinal_maximum_coverage_candidate_count": len(
            cardinal_maximum_coverage_measurements
        ),
        "orthogonal_reflection_same_quadrant_cardinal_simulation_attempt_count": (
            same_quadrant_cardinal_simulation_attempt_count
        ),
        "orthogonal_reflection_same_quadrant_cardinal_first_complete_cover_present": (
            first_complete_same_quadrant_cardinal_measurement is not None
        ),
        "orthogonal_reflection_cardinal_maximum_coverage_enumeration_truncated": (
            cardinal_maximum_coverage_enumeration_truncated
        ),
        "orthogonal_reflection_cardinal_maximum_coverage_orbit_count": len(
            cardinal_maximum_coverage_orbits
        ),
        "orthogonal_reflection_unique_cardinal_maximum_coverage_measurement_present": (
            len(cardinal_maximum_coverage_representatives) == 1
        ),
        "orthogonal_reflection_cardinal_maximum_coverage_translation_component_matches_action_conditioned_focus_bbox": False,
        "orthogonal_reflection_cardinal_maximum_coverage_translation_identity_binding_satisfied": False,
        "orthogonal_reflection_cardinal_maximum_coverage_translated_body_is_current_target_subset": False,
        "orthogonal_reflection_cardinal_maximum_coverage_image_union_is_current_target_subset": False,
        "orthogonal_reflection_cardinal_maximum_coverage_strict_placement_count_for_body": 0,
        "orthogonal_reflection_unique_maximum_coverage_measurement_present": (
            len(maximum_coverage_representatives) == 1
        ),
        "orthogonal_reflection_maximum_coverage_translated_body_is_current_target_subset": False,
        "orthogonal_reflection_maximum_coverage_image_union_is_current_target_subset": False,
        "orthogonal_reflection_maximum_coverage_strict_placement_count_for_body": 0,
        "orthogonal_reflection_exact_cover_candidate_count": len(
            bounded_exact_cover_measurements
        ),
        "orthogonal_reflection_exact_cover_orbit_count": len(
            exact_cover_orbits
        ),
        "orthogonal_reflection_exact_cover_orbit_member_counts": tuple(
            len(orbit) for orbit in exact_cover_orbits
        ),
        "orthogonal_reflection_exact_cover_candidates_symmetry_equivalent": (
            bool(bounded_exact_cover_measurements)
            and len(exact_cover_orbits) == 1
            and not exact_cover_enumeration_truncated
        ),
        "orthogonal_reflection_exact_cover_or_unique_joint_measurement_present": (
            (
                bool(bounded_exact_cover_measurements)
                and len(exact_cover_orbits) == 1
                and not exact_cover_enumeration_truncated
            )
            or (
                len(bounded_joint_minimum_measurements) == 1
                and not joint_minimum_enumeration_truncated
            )
        ),
        "orthogonal_reflection_partial_partition_body_family_count": len(
            partial_partition_body_families
        ),
        "orthogonal_reflection_partial_partition_body_count": len(
            unique_partial_partition_family
        ),
        "orthogonal_reflection_partial_partition_candidate_count": (
            partial_partition_candidate_count
        ),
        "orthogonal_reflection_cumulative_terminal_translation_candidate_count": len(
            cumulative_terminal_representatives
        ),
        "orthogonal_reflection_partial_partition_unique_placement_body_count": len(
            unique_placement_partial_partition_measurements
        ),
        "orthogonal_reflection_partial_partition_ambiguous_body_count": sum(
            1
            for body_component_ref in unique_partial_partition_family
            if len(partial_partition_measurements_by_body[body_component_ref]) > 1
        ),
        "orthogonal_reflection_partial_partition_enumeration_truncated": (
            partial_partition_enumeration_truncated
        ),
        "orthogonal_reflection_unique_partial_partition_measurement_present": (
            unique_partial_partition_translation is not None
        ),
        "orthogonal_reflection_translation_enumeration_truncated": (
            exact_cover_enumeration_truncated
            or joint_minimum_enumeration_truncated
        ),
        "orthogonal_reflection_measurement_digest": stable_digest(
            (
                quantum,
                support_component_ref,
                row_axis_twice,
                col_axis_twice,
                tuple(sorted(visible)),
                tuple(sorted(completed)),
            )
        )[:16],
    }
    if len(carrier_measurements) == 1:
        (
            carrier_value,
            carrier_row,
            carrier_col,
            carrier_row_count,
            carrier_col_count,
        ) = carrier_measurements[0]
        row_residual_twice = row_axis_twice - 2 * carrier_row
        col_residual_twice = col_axis_twice - 2 * carrier_col
        if row_residual_twice % 2 == 0 and col_residual_twice % 2 == 0:
            row_residual = row_residual_twice // 2
            col_residual = col_residual_twice // 2
            current_focus_bbox = (
                value.input_aligned_entity_bboxes[0]
                if len(value.input_aligned_entity_bboxes) == 1
                else None
            )
            focus_horizontal_axis_like = bool(
                current_focus_bbox is not None
                and current_focus_bbox.height <= quantum
                and current_focus_bbox.width >= completed_col_span * quantum
            )
            focus_vertical_axis_like = bool(
                current_focus_bbox is not None
                and current_focus_bbox.width <= quantum
                and current_focus_bbox.height >= completed_row_span * quantum
            )
            common.update(
                {
                    "orthogonal_reflection_carrier_measurement_count": 2,
                    "orthogonal_reflection_carrier_measurement_pair_present": True,
                    "orthogonal_reflection_carrier_palette_value": carrier_value,
                    "orthogonal_reflection_carrier_row_axis_coordinate_twice": 2
                    * carrier_row,
                    "orthogonal_reflection_carrier_col_axis_coordinate_twice": 2
                    * carrier_col,
                    "orthogonal_reflection_carrier_row_support_count": carrier_row_count,
                    "orthogonal_reflection_carrier_col_support_count": carrier_col_count,
                    "orthogonal_reflection_row_residual_signed": row_residual,
                    "orthogonal_reflection_col_residual_signed": col_residual,
                    "orthogonal_reflection_current_focus_horizontal_axis_like": (
                        focus_horizontal_axis_like
                    ),
                    "orthogonal_reflection_current_focus_vertical_axis_like": (
                        focus_vertical_axis_like
                    ),
                    "orthogonal_reflection_current_focus_matches_open_axis": bool(
                        (row_residual != 0 and focus_horizontal_axis_like)
                        or (col_residual != 0 and focus_vertical_axis_like)
                    ),
                }
            )
    else:
        common["orthogonal_reflection_carrier_measurement_count"] = 0

    unique_translation = (
        unique_partial_partition_translation
        if unique_partial_partition_translation is not None
        else cumulative_terminal_representatives[0]
        if len(cumulative_terminal_representatives) == 1
        else exact_cover_orbit_representatives[0]
        if len(exact_cover_orbit_representatives) == 1
        and not exact_cover_enumeration_truncated
        else maximum_coverage_representatives[0]
        if len(maximum_coverage_representatives) == 1
        else bounded_joint_minimum_measurements[0]
        if len(bounded_joint_minimum_measurements) == 1
        and not joint_minimum_enumeration_truncated
        else first_complete_same_quadrant_cardinal_measurement
        if first_complete_same_quadrant_cardinal_measurement is not None
        else None
    )
    if unique_translation is not None:
        translation_component = next(
            (
                component
                for component in anchors
                if component.component_id
                == unique_translation["body_component_ref"]
            ),
            None,
        )
        translation_component_bbox = (
            (
                translation_component.bbox.top,
                translation_component.bbox.left,
                translation_component.bbox.bottom,
                translation_component.bbox.right,
            )
            if translation_component is not None
            else unique_translation.get("body_bbox_tuple")
        )
        current_focus_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.input_aligned_entity_bboxes
        )
        transient_change_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.transient_change_bboxes
        )
        appeared_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.action_conditioned_appeared_bboxes
        )
        action_conditioned_focus_match = any(
            _action_conditioned_bbox_supports_candidate_focus(
                candidate_bbox=translation_component_bbox,
                change_bbox=change_bbox,
                quantum=quantum,
            )
            for change_bbox in transient_change_bbox_tuples
        )
        same_palette_component_count = sum(
            1
            for component in anchors
            if int(component.value)
            == int(unique_translation["body_palette_value"])
        )
        common.update(
            {
                "orthogonal_reflection_translation_component_ref": unique_translation[
                    "body_component_ref"
                ],
                "orthogonal_reflection_translation_palette_value": unique_translation[
                    "body_palette_value"
                ],
                "orthogonal_reflection_body_cell_count": unique_translation[
                    "body_cell_count"
                ],
                "orthogonal_reflection_body_normalized_cell_offsets": (
                    unique_translation["body_normalized_cell_offsets"]
                ),
                "orthogonal_reflection_translation_residual_row_signed": unique_translation[
                    "delta_row"
                ],
                "orthogonal_reflection_translation_residual_col_signed": unique_translation[
                    "delta_col"
                ],
                "orthogonal_reflection_image_union_cell_count": unique_translation[
                    "image_union_count"
                ],
                "orthogonal_reflection_direct_family_cell_occurrence_count": unique_translation["direct_family_cell_occurrence_count"],
                "orthogonal_reflection_target_cell_occurrence_count": unique_translation["target_cell_occurrence_count"],
                "orthogonal_reflection_direct_family_target_cardinality_delta": unique_translation["direct_family_target_cardinality_delta"],
                "orthogonal_reflection_image_union_target_cardinality_delta": (
                    _axis_multiplicity_cell_count(
                        unique_translation["image_union_cells"],
                        row_axis_twice=row_axis_twice,
                        col_axis_twice=col_axis_twice,
                    )
                    - _axis_multiplicity_cell_count(
                        placement_target,
                        row_axis_twice=row_axis_twice,
                        col_axis_twice=col_axis_twice,
                    )
                ),
                "orthogonal_reflection_image_union_extra_cell_count": unique_translation[
                    "image_union_extra_count"
                ],
                "orthogonal_reflection_covered_target_cell_count": unique_translation[
                    "covered_target_cell_count"
                ],
                "orthogonal_reflection_selected_destination_is_cardinal_corner_alignment": unique_translation[
                    "cardinal_corner_alignment"
                ],
                "orthogonal_reflection_image_overlap_cell_count": unique_translation[
                    "image_overlap_count"
                ],
                "orthogonal_reflection_translated_body_is_target_subset": unique_translation[
                    "translated_body_is_target_subset"
                ],
                "orthogonal_reflection_translated_body_is_current_target_subset": unique_translation[
                    "translated_body_is_current_target_subset"
                ],
                "orthogonal_reflection_image_union_is_current_target_subset": unique_translation[
                    "image_union_is_current_target_subset"
                ],
                "orthogonal_reflection_current_target_remaining_cell_count_after_image_union": unique_translation[
                    "current_target_remaining_cell_count_after_image_union"
                ],
                "orthogonal_reflection_translated_body_covers_support_component": unique_translation[
                    "translated_body_covers_support_component"
                ],
                "orthogonal_reflection_target_covered_by_image_union": unique_translation[
                    "target_covered_by_image_union"
                ],
                "orthogonal_reflection_complete_target_or_support_component_covered": (
                    bool(unique_translation["target_covered_by_image_union"])
                    or int(unique_translation[
                        "current_target_remaining_cell_count_after_image_union"
                    ]) == 0
                    or bool(
                        unique_translation[
                            "translated_body_covers_support_component"
                        ]
                    )
                ),
                "orthogonal_reflection_translation_component_bbox_tuple": translation_component_bbox,
                "orthogonal_reflection_translation_same_palette_component_count": same_palette_component_count,
                "orthogonal_reflection_current_focus_bbox_measurement_present": len(current_focus_bbox_tuples) == 1,
                "orthogonal_reflection_current_focus_bbox_tuple": (
                    current_focus_bbox_tuples[0]
                    if len(current_focus_bbox_tuples) == 1
                    else None
                ),
                "orthogonal_reflection_transient_change_bbox_tuples": transient_change_bbox_tuples,
                "orthogonal_reflection_action_conditioned_appeared_bbox_tuples": appeared_bbox_tuples,
                "orthogonal_reflection_translation_component_matches_current_focus_bbox": (
                    translation_component_bbox is not None
                    and len(current_focus_bbox_tuples) == 1
                    and current_focus_bbox_tuples[0] == translation_component_bbox
                ),
                "orthogonal_reflection_translation_component_matches_action_conditioned_focus_bbox": action_conditioned_focus_match,
                "orthogonal_reflection_translation_identity_binding_satisfied": (
                    action_conditioned_focus_match
                    or (
                        translation_component_bbox is not None
                        and len(current_focus_bbox_tuples) == 1
                        and current_focus_bbox_tuples[0] == translation_component_bbox
                    )
                    or (
                        same_palette_component_count == 1
                        and not current_focus_bbox_tuples
                    )
                ),
            }
        )
    unique_maximum_coverage_translation = (
        maximum_coverage_representatives[0]
        if len(maximum_coverage_representatives) == 1
        else None
    )
    if unique_maximum_coverage_translation is not None:
        maximum_bbox = unique_maximum_coverage_translation.get("body_bbox_tuple")
        current_focus_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.input_aligned_entity_bboxes
        )
        transient_change_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.transient_change_bboxes
        )
        maximum_action_conditioned_focus_match = any(
            _action_conditioned_bbox_supports_candidate_focus(
                candidate_bbox=maximum_bbox,
                change_bbox=change_bbox,
                quantum=quantum,
            )
            for change_bbox in transient_change_bbox_tuples
        )
        maximum_same_palette_component_count = sum(
            1
            for component in anchors
            if int(component.value)
            == int(unique_maximum_coverage_translation["body_palette_value"])
        )
        common.update(
            {
                "orthogonal_reflection_maximum_coverage_translation_residual_row_signed": unique_maximum_coverage_translation[
                    "delta_row"
                ],
                "orthogonal_reflection_maximum_coverage_translation_residual_col_signed": unique_maximum_coverage_translation[
                    "delta_col"
                ],
                "orthogonal_reflection_maximum_coverage_body_cell_count": unique_maximum_coverage_translation[
                    "body_cell_count"
                ],
                "orthogonal_reflection_maximum_coverage_current_target_remaining_cell_count_after_image_union": unique_maximum_coverage_translation[
                    "current_target_remaining_cell_count_after_image_union"
                ],
                "orthogonal_reflection_maximum_coverage_translation_component_bbox_tuple": maximum_bbox,
                "orthogonal_reflection_maximum_coverage_destination_is_cardinal_corner_alignment": unique_maximum_coverage_translation[
                    "cardinal_corner_alignment"
                ],
                "orthogonal_reflection_maximum_coverage_translated_body_is_current_target_subset": unique_maximum_coverage_translation[
                    "translated_body_is_current_target_subset"
                ],
                "orthogonal_reflection_maximum_coverage_image_union_is_current_target_subset": unique_maximum_coverage_translation[
                    "image_union_is_current_target_subset"
                ],
                "orthogonal_reflection_maximum_coverage_strict_placement_count_for_body": unique_maximum_coverage_translation[
                    "strict_placement_count_for_body"
                ],
                "orthogonal_reflection_maximum_coverage_direct_family_target_cardinality_delta": unique_maximum_coverage_translation["direct_family_target_cardinality_delta"],
                "orthogonal_reflection_maximum_coverage_translated_body_covers_support_component": unique_maximum_coverage_translation["translated_body_covers_support_component"],
                "orthogonal_reflection_maximum_coverage_translation_component_matches_action_conditioned_focus_bbox": maximum_action_conditioned_focus_match,
                "orthogonal_reflection_maximum_coverage_translation_identity_binding_satisfied": (
                    maximum_action_conditioned_focus_match
                    or (
                        maximum_bbox is not None
                        and len(current_focus_bbox_tuples) == 1
                        and current_focus_bbox_tuples[0] == maximum_bbox
                    )
                    or (
                        maximum_same_palette_component_count == 1
                        and not current_focus_bbox_tuples
                    )
                ),
            }
        )
    unique_cardinal_maximum_coverage_translation = (
        cardinal_maximum_coverage_representatives[0]
        if len(cardinal_maximum_coverage_representatives) == 1
        else None
    )
    if unique_cardinal_maximum_coverage_translation is not None:
        cardinal_bbox = unique_cardinal_maximum_coverage_translation.get(
            "body_bbox_tuple"
        )
        current_focus_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.input_aligned_entity_bboxes
        )
        transient_change_bbox_tuples = tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.transient_change_bboxes
        )
        cardinal_action_conditioned_focus_match = any(
            _action_conditioned_bbox_supports_candidate_focus(
                candidate_bbox=cardinal_bbox,
                change_bbox=change_bbox,
                quantum=quantum,
            )
            for change_bbox in transient_change_bbox_tuples
        )
        cardinal_same_palette_component_count = sum(
            1
            for component in anchors
            if int(component.value)
            == int(
                unique_cardinal_maximum_coverage_translation[
                    "body_palette_value"
                ]
            )
        )
        common.update(
            {
                "orthogonal_reflection_cardinal_maximum_coverage_translation_residual_row_signed": unique_cardinal_maximum_coverage_translation[
                    "delta_row"
                ],
                "orthogonal_reflection_cardinal_maximum_coverage_translation_residual_col_signed": unique_cardinal_maximum_coverage_translation[
                    "delta_col"
                ],
                "orthogonal_reflection_cardinal_maximum_coverage_body_cell_count": unique_cardinal_maximum_coverage_translation[
                    "body_cell_count"
                ],
                "orthogonal_reflection_cardinal_maximum_coverage_current_target_remaining_cell_count_after_image_union": max(
                    0,
                    len(coverage_target)
                    - int(
                        unique_cardinal_maximum_coverage_translation[
                            "covered_target_cell_count"
                        ]
                    ),
                ),
                "orthogonal_reflection_cardinal_maximum_coverage_translation_component_bbox_tuple": cardinal_bbox,
                "orthogonal_reflection_cardinal_maximum_coverage_translated_body_is_current_target_subset": unique_cardinal_maximum_coverage_translation[
                    "translated_body_is_current_target_subset"
                ],
                "orthogonal_reflection_cardinal_maximum_coverage_image_union_is_current_target_subset": unique_cardinal_maximum_coverage_translation[
                    "image_union_is_current_target_subset"
                ],
                "orthogonal_reflection_cardinal_maximum_coverage_strict_placement_count_for_body": unique_cardinal_maximum_coverage_translation[
                    "strict_placement_count_for_body"
                ],
                "orthogonal_reflection_cardinal_maximum_coverage_direct_family_target_cardinality_delta": unique_cardinal_maximum_coverage_translation["direct_family_target_cardinality_delta"],
                "orthogonal_reflection_cardinal_maximum_coverage_translated_body_covers_support_component": unique_cardinal_maximum_coverage_translation["translated_body_covers_support_component"],
                "orthogonal_reflection_cardinal_maximum_coverage_translation_component_matches_action_conditioned_focus_bbox": cardinal_action_conditioned_focus_match,
                "orthogonal_reflection_cardinal_maximum_coverage_translation_identity_binding_satisfied": (
                    cardinal_action_conditioned_focus_match
                    or (
                        cardinal_bbox is not None
                        and len(current_focus_bbox_tuples) == 1
                        and current_focus_bbox_tuples[0] == cardinal_bbox
                    )
                    or (
                        cardinal_same_palette_component_count == 1
                        and not current_focus_bbox_tuples
                    )
                ),
            }
        )
    row_residual = common.get("orthogonal_reflection_row_residual_signed")
    col_residual = common.get("orthogonal_reflection_col_residual_signed")
    if row_residual is not None and col_residual is not None:
        common["orthogonal_reflection_axis_zero_count"] = int(
            int(row_residual) == 0
        ) + int(int(col_residual) == 0)
        common["orthogonal_reflection_axis_open_count"] = int(
            int(row_residual) != 0
        ) + int(int(col_residual) != 0)
    else:
        common["orthogonal_reflection_axis_zero_count"] = 0
        common["orthogonal_reflection_axis_open_count"] = 0
    body_row = common.get("orthogonal_reflection_translation_residual_row_signed")
    body_col = common.get("orthogonal_reflection_translation_residual_col_signed")
    common["orthogonal_reflection_body_open_residual_count"] = (
        int(int(body_row) != 0) + int(int(body_col) != 0)
        if body_row is not None and body_col is not None
        else 0
    )
    return FrozenMap(common)


def _disconnected_support_measurements(
    value: InteractionProbeInput,
    *,
    quantum: int,
    established_axis_pair: tuple[int, int] | None = None,
) -> tuple[dict[str, object], ...]:
    """Enumerate bounded palette-wide supports containing disconnected cells."""

    if not quantum:
        return ()
    frame = value.current_observed_frame or value.frame
    rows = frame.rows
    carrier_values = frozenset(
        value.prior_context_effect_carried_reflection_axis_carrier_values
    )
    grouped: dict[tuple[int, int, int], list[object]] = {}
    for component in _current_components(value):
        if (
            component.value in carrier_values
            or component.touches_frame_boundary
            or component.bbox.height % quantum != 0
            or component.bbox.width % quantum != 0
        ):
            continue
        grouped.setdefault(
            (
                int(component.value),
                component.bbox.top % quantum,
                component.bbox.left % quantum,
            ),
            [],
        ).append(component)
    measurements: list[dict[str, object]] = []
    for (palette_value, row_phase, col_phase), raw_components in sorted(
        grouped.items()
    ):
        if len(raw_components) < 3:
            continue
        component_cells = tuple(
            _occupied_cells(
                rows=rows,
                palette_value=palette_value,
                quantum=quantum,
                row_phase=row_phase,
                col_phase=col_phase,
                top=component.bbox.top,
                left=component.bbox.left,
                bottom=component.bbox.bottom,
                right=component.bbox.right,
            )
            for component in raw_components
        )
        non_singleton_component_cells = tuple(
            cells for cells in component_cells if len(cells) > 1
        )
        repeated_non_singleton_support = bool(
            len(non_singleton_component_cells) == len(component_cells)
            and len(component_cells) >= 2
            and len({len(cells) for cells in component_cells}) == 1
            and carrier_values
            and value.prior_context_effect_carried_reflection_axis_observed
        )
        if not non_singleton_component_cells or (
            not any(len(cells) == 1 for cells in component_cells)
            and not repeated_non_singleton_support
        ):
            continue
        visible = frozenset(
            cell for cells in component_cells for cell in cells
        )
        for row_axis_twice, col_axis_twice, completed in _orthogonal_completions(
            visible
        ):
            if (
                completed == visible
                and established_axis_pair != (row_axis_twice, col_axis_twice)
                and not repeated_non_singleton_support
            ):
                continue
            measurements.append(
                {
                    "support_component_ref": (
                        "measurement.disconnected_support."
                        f"{stable_digest((palette_value, row_phase, col_phase, tuple(sorted(visible))))[:12]}"
                    ),
                    "support_palette_value": palette_value,
                    "row_phase": row_phase,
                    "col_phase": col_phase,
                    "visible": visible,
                    "completed": completed,
                    "row_axis_twice": row_axis_twice,
                    "col_axis_twice": col_axis_twice,
                    "missing": completed - visible,
                    "disconnected_component_count": len(raw_components),
                }
            )
            if len(measurements) > _MAXIMUM_REFLECTION_COMPLETIONS:
                return tuple(measurements)
    return tuple(measurements)


def _unique_focus_bound_reflection_palette_measurement(
    value: InteractionProbeInput,
    *,
    quantum: int,
) -> FrozenMap | None:
    """Simulate bounded palette-role alternatives after an observed focus transfer.

    Palette values remain literal counterfactual bindings.  A binding survives
    only when the focus bbox identifies one body value, the proposed carrier
    exposes a crossing axis pair, and the existing strict-cover measurement
    admits the focused body.  Returning a measurement requires one exact
    surviving binding; DRM still decides whether its route is eligible.
    """

    if quantum <= 0:
        return None
    directional_action_refs = frozenset(
        str(item[0]) for item in value.interface_action_translation_deltas
    )
    nondirectional_unilateral_transfer = bool(
        value.last_action_ref
        and value.last_action_ref not in directional_action_refs
        and len(value.unilateral_enclosed_change_candidate_bboxes) == 1
    )
    if not (
        value.action_conditioned_control_focus_transfer_supported
        or nondirectional_unilateral_transfer
    ):
        return None
    if nondirectional_unilateral_transfer:
        focus_bboxes = value.unilateral_enclosed_change_candidate_bboxes
    elif len(value.reciprocal_enclosed_change_candidate_bboxes) == 1:
        focus_bboxes = value.reciprocal_enclosed_change_candidate_bboxes
    else:
        focus_bboxes = value.input_aligned_entity_bboxes
    if len(focus_bboxes) != 1:
        return None
    focus_bbox = focus_bboxes[0]
    components = _current_components(value)
    body_values = tuple(
        sorted(
            {
                int(component.value)
                for component in components
                if component.area > 1 and component.bbox == focus_bbox
            }
        )
    )
    palette_values = tuple(sorted({int(component.value) for component in components}))
    if len(body_values) != 1 or not 3 <= len(palette_values) <= 8:
        return None

    body_value = body_values[0]
    measurements: list[FrozenMap] = []
    attempted_binding_count = 0
    for carrier_value in palette_values:
        if carrier_value == body_value:
            continue
        for support_value in palette_values:
            if support_value in {body_value, carrier_value}:
                continue
            attempted_binding_count += 1
            if attempted_binding_count > 64:
                return None
            counterfactual = value.model_copy(
                update={
                    "input_aligned_entity_bboxes": (focus_bbox,),
                    "prior_context_effect_carried_reflection_axis_observed": True,
                    "prior_context_effect_carried_reflection_axis_carrier_values": (
                        carrier_value,
                    ),
                    "prior_context_orthogonal_reflection_axis_zero_count": 2,
                    "prior_game_episode_orthogonal_reflection_support_values": (
                        support_value,
                    ),
                    "prior_game_episode_orthogonal_reflection_direct_body_values": (
                        body_value,
                    ),
                }
            )
            measured = _measure_orthogonal_reflection_structure(counterfactual)
            if not (
                measured.get(
                    "orthogonal_reflection_unique_partial_partition_measurement_present"
                )
                is True
                and measured.get(
                    "orthogonal_reflection_translated_body_is_current_target_subset"
                )
                is True
                and measured.get(
                    "orthogonal_reflection_image_union_is_current_target_subset"
                )
                is True
                and int(
                    measured.get(
                        "orthogonal_reflection_maximum_coverage_strict_placement_count_for_body",
                        0,
                    )
                    or 0
                )
                == 1
            ):
                continue
            measurements.append(measured)
            if len(measurements) > 1:
                return None
    if len(measurements) != 1:
        return None
    result = dict(measurements[0])
    result["orthogonal_reflection_palette_binding_attempt_count"] = (
        attempted_binding_count
    )
    result["orthogonal_reflection_unique_focus_bound_palette_binding"] = True
    return FrozenMap(result)


def measure_orthogonal_reflection_structure(
    value: InteractionProbeInput,
) -> FrozenMap:
    """Measure all bounded support descriptions and retain one actionable class."""

    quantum = _observed_cell_quantum(value)
    base = _measure_orthogonal_reflection_structure(value)
    if not base.get(
        "orthogonal_reflection_unique_partial_partition_measurement_present",
        False,
    ):
        focus_bound = _unique_focus_bound_reflection_palette_measurement(
            value,
            quantum=quantum,
        )
        if focus_bound is not None:
            base = focus_bound
    action_conditioned_anchor_summaries = tuple(
        (
            component.bbox.top,
            component.bbox.left,
            component.bbox.bottom,
            component.bbox.right,
            int(component.value),
            int(component.area),
        )
        for component in _current_components(value)
        if any(
            max(
                abs(component.bbox.top - change_bbox.top),
                abs(component.bbox.left - change_bbox.left),
                abs(component.bbox.bottom - change_bbox.bottom),
                abs(component.bbox.right - change_bbox.right),
            )
            <= quantum
            for change_bbox in value.transient_change_bboxes
        )
    )[:3]
    if action_conditioned_anchor_summaries:
        base_data = dict(base)
        base_data[
            "orthogonal_reflection_action_conditioned_anchor_summaries"
        ] = action_conditioned_anchor_summaries
        base = FrozenMap(base_data)
    base_axis_pair = None
    if (
        base.get("orthogonal_reflection_row_residual_signed") == 0
        and base.get("orthogonal_reflection_col_residual_signed") == 0
        and base.get("orthogonal_reflection_row_axis_coordinate_twice") is not None
        and base.get("orthogonal_reflection_col_axis_coordinate_twice") is not None
    ):
        base_axis_pair = (
            int(base["orthogonal_reflection_row_axis_coordinate_twice"]),
            int(base["orthogonal_reflection_col_axis_coordinate_twice"]),
        )
    disconnected = _disconnected_support_measurements(
        value,
        quantum=quantum,
        established_axis_pair=base_axis_pair,
    )
    measured = [base]
    for support in disconnected[:_MAXIMUM_REFLECTION_COMPLETIONS]:
        if (
            base.get("orthogonal_reflection_support_palette_value")
            == support["support_palette_value"]
            and frozenset(
                base.get("orthogonal_reflection_completed_cell_positions", ())
            )
            == support["completed"]
            and base.get(
                "orthogonal_reflection_row_axis_coordinate_twice"
            )
            == support["row_axis_twice"]
            and base.get(
                "orthogonal_reflection_col_axis_coordinate_twice"
            )
            == support["col_axis_twice"]
        ):
            continue
        transported = value.model_copy(
            update={
                "prior_context_orthogonal_reflection_completed_cells": tuple(
                    sorted(support["completed"])
                ),
                "prior_context_orthogonal_reflection_support_values": (
                    int(support["support_palette_value"]),
                ),
                "prior_context_orthogonal_reflection_row_phase": int(
                    support["row_phase"]
                ),
                "prior_context_orthogonal_reflection_col_phase": int(
                    support["col_phase"]
                ),
                "prior_context_orthogonal_reflection_row_axis_coordinate_twice": int(
                    support["row_axis_twice"]
                ),
                "prior_context_orthogonal_reflection_col_axis_coordinate_twice": int(
                    support["col_axis_twice"]
                ),
            }
        )
        candidate = _measure_orthogonal_reflection_structure(transported)
        candidate_data = dict(candidate)
        candidate_data["orthogonal_reflection_disconnected_support_component_count"] = int(
            support["disconnected_component_count"]
        )
        measured.append(FrozenMap(candidate_data))
    actionable = tuple(
        item
        for item in measured
        if item.get(
            "orthogonal_reflection_exact_cover_or_unique_joint_measurement_present",
            False,
        )
        or item.get(
            "orthogonal_reflection_unique_partial_partition_measurement_present",
            False,
        )
        or item.get(
            "orthogonal_reflection_unique_maximum_coverage_measurement_present",
            False,
        )
    )
    disconnected_actionable = tuple(
        item
        for item in measured[1:]
        if item.get(
            "orthogonal_reflection_exact_cover_or_unique_joint_measurement_present",
            False,
        )
        or item.get(
            "orthogonal_reflection_unique_partial_partition_measurement_present",
            False,
        )
        or item.get(
            "orthogonal_reflection_unique_maximum_coverage_measurement_present",
            False,
        )
    )

    def with_parallel_disconnected_measurement(result: FrozenMap) -> FrozenMap:
        result_data = dict(result)
        result_data.setdefault(
            "orthogonal_reflection_unique_partial_partition_measurement_present",
            False,
        )
        result_data.setdefault(
            "disconnected_reflection_unique_partial_partition_measurement_present",
            False,
        )
        result_data["disconnected_reflection_support_description_count"] = len(
            disconnected
        )
        if len(measured) == 2:
            raw_disconnected = measured[1]
            result_data["disconnected_reflection_measured_exact_cover_candidate_count"] = raw_disconnected.get(
                "orthogonal_reflection_exact_cover_candidate_count"
            )
            result_data["disconnected_reflection_measured_exact_cover_orbit_count"] = raw_disconnected.get(
                "orthogonal_reflection_exact_cover_orbit_count"
            )
            result_data["disconnected_reflection_measured_joint_minimum_candidate_count"] = raw_disconnected.get(
                "orthogonal_reflection_joint_minimum_anchor_candidate_count"
            )
        if len(disconnected_actionable) == 1:
            disconnected_measurement = disconnected_actionable[0]
            field_map = {
                "orthogonal_reflection_disconnected_support_component_count": "orthogonal_reflection_disconnected_support_component_count",
                "orthogonal_reflection_translation_residual_row_signed": "disconnected_reflection_translation_residual_row_signed",
                "orthogonal_reflection_translation_residual_col_signed": "disconnected_reflection_translation_residual_col_signed",
                "orthogonal_reflection_body_open_residual_count": "disconnected_reflection_body_open_residual_count",
                "orthogonal_reflection_exact_cover_candidate_count": "disconnected_reflection_cover_candidate_count",
                "orthogonal_reflection_exact_cover_orbit_count": "disconnected_reflection_cover_orbit_count",
                "orthogonal_reflection_exact_cover_candidates_symmetry_equivalent": "disconnected_reflection_cover_candidates_symmetry_equivalent",
                "orthogonal_reflection_partial_partition_body_family_count": "disconnected_reflection_partial_partition_body_family_count",
                "orthogonal_reflection_partial_partition_body_count": "disconnected_reflection_partial_partition_body_count",
                "orthogonal_reflection_partial_partition_candidate_count": "disconnected_reflection_partial_partition_candidate_count",
                "orthogonal_reflection_partial_partition_unique_placement_body_count": "disconnected_reflection_partial_partition_unique_placement_body_count",
                "orthogonal_reflection_partial_partition_ambiguous_body_count": "disconnected_reflection_partial_partition_ambiguous_body_count",
                "orthogonal_reflection_partial_partition_enumeration_truncated": "disconnected_reflection_partial_partition_enumeration_truncated",
                "orthogonal_reflection_unique_partial_partition_measurement_present": "disconnected_reflection_unique_partial_partition_measurement_present",
                "orthogonal_reflection_unique_cumulative_terminal_measurement_present": "disconnected_reflection_unique_cumulative_terminal_measurement_present",
                "orthogonal_reflection_complete_target_or_support_component_covered": "disconnected_reflection_complete_target_or_support_component_covered",
                "orthogonal_reflection_translated_body_is_current_target_subset": "disconnected_reflection_translated_body_is_current_target_subset",
                "orthogonal_reflection_image_union_is_current_target_subset": "disconnected_reflection_image_union_is_current_target_subset",
                "orthogonal_reflection_current_target_remaining_cell_count_after_image_union": "disconnected_reflection_current_target_remaining_cell_count_after_image_union",
                "orthogonal_reflection_image_overlap_cell_count": "disconnected_reflection_image_overlap_cell_count",
                "orthogonal_reflection_translation_same_palette_component_count": "disconnected_reflection_translation_same_palette_component_count",
                "orthogonal_reflection_translation_component_bbox_tuple": "disconnected_reflection_translation_component_bbox_tuple",
                "orthogonal_reflection_current_focus_bbox_measurement_present": "disconnected_reflection_current_focus_bbox_measurement_present",
                "orthogonal_reflection_current_focus_bbox_tuple": "disconnected_reflection_current_focus_bbox_tuple",
                "orthogonal_reflection_translation_component_matches_current_focus_bbox": "disconnected_reflection_translation_component_matches_current_focus_bbox",
                "orthogonal_reflection_translation_component_matches_action_conditioned_focus_bbox": "disconnected_reflection_translation_component_matches_action_conditioned_focus_bbox",
                "orthogonal_reflection_translation_identity_binding_satisfied": "disconnected_reflection_translation_identity_binding_satisfied",
                "orthogonal_reflection_translation_enumeration_truncated": "disconnected_reflection_translation_enumeration_truncated",
            }
            for source_field, target_field in sorted(field_map.items()):
                result_data[target_field] = disconnected_measurement.get(
                    source_field
                )
            result_data["disconnected_reflection_actionable_support_description_count"] = 1
        else:
            result_data["disconnected_reflection_actionable_support_description_count"] = len(
                disconnected_actionable
            )
        return FrozenMap(result_data)

    if len(actionable) == 1:
        result = dict(actionable[0])
        result["orthogonal_reflection_actionable_support_description_count"] = 1
        result["orthogonal_reflection_support_description_enumeration_truncated"] = (
            len(disconnected) > _MAXIMUM_REFLECTION_COMPLETIONS
        )
        return with_parallel_disconnected_measurement(FrozenMap(result))
    if len(actionable) > 1:
        result = dict(base)
        result.update(
            {
                "orthogonal_reflection_actionable_support_description_count": len(
                    actionable
                ),
                "orthogonal_reflection_support_description_enumeration_truncated": (
                    len(disconnected) > _MAXIMUM_REFLECTION_COMPLETIONS
                ),
            }
        )
        return with_parallel_disconnected_measurement(FrozenMap(result))
    result = dict(base)
    result["orthogonal_reflection_actionable_support_description_count"] = 0
    result["orthogonal_reflection_support_description_enumeration_truncated"] = (
        len(disconnected) > _MAXIMUM_REFLECTION_COMPLETIONS
    )
    return with_parallel_disconnected_measurement(FrozenMap(result))
