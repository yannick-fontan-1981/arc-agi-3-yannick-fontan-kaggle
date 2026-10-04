"""Pure, role-neutral measurements over an action-conditioned frame transition."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from math import gcd

from agents.yf_arc3_v5.capabilities.bearer_support import (
    measure_adjacent_translation_groups,
    measure_translated_enclosed_member_refs,
)
from agents.yf_arc3_v5.capabilities.transformation_pairs import measure_entity_changes
from agents.yf_arc3_v5.capabilities.recipe_pattern_observation import measure_recipe_transition, measure_recipe_edit_context

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundaryDecreaseEvidence,
    BlockGridSceneDescription,
    BoundingBox,
    ComponentDescription,
    ComponentExtractionInput,
    ControlledTransitionAnalysis,
    ControlledTransitionAnalysisInput,
    ExtentTransition,
    FrameDifferenceInput,
    FrameGrid,
    GoalCompletionAnalysis,
    GoalCompletionAnalysisInput,
    IntermediateSceneChangeInput,
    IntermediateSceneChangeMeasurements,
    InteractionProbeInput,
    OrthogonalPerimeterAssemblyInput,
    PeerDistanceChangeCandidate,
    PixelTransitionDelta,
    QuantizedExchangeCandidate,
    RepeatedBoundaryDecreaseAnalysis,
    RepeatedBoundaryDecreaseAnalysisInput,
    SceneTransitionMeasurementInput,
    SceneTransitionMeasurements,
    TrackedComponent,
    TranslationTransition,
)
from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.frame import difference_frames, materialize_retained_periodic_cell_grid
from agents.yf_arc3_v5.capabilities.geometry import (
    measure_orthogonal_perimeter_assemblies,
)
from agents.yf_arc3_v5.capabilities.ordered_focus import (
    measure_left_to_right_outlined_successor,
)
from agents.yf_arc3_v5.logos.types import (
    EffectSignatureNovelty,
    FrozenMap,
    ObservableChangeScope,
    PeriodicCycleWorkflowStatus,
    stable_digest,
)

SCENE_EQUIVALENT_CANDIDATE = "scene_equivalence.equivalent"
SCENE_DIFFERENT_CANDIDATE = "scene_equivalence.different"
REPEATED_BOUNDARY_DECREASE_CANDIDATE = "boundary_decrease.repeated"
SINGLE_BOUNDARY_DECREASE_CANDIDATE = "boundary_decrease.single"
BOUNDARY_FAMILY_PROGRESS_CANDIDATE = "boundary_decrease.progress_group"
UNRESOLVED_BOUNDARY_CHANGE_CANDIDATE = "boundary_decrease.unresolved"
INTERMEDIATE_MEANINGFUL_CHANGE_CANDIDATE = "continue_guard.meaningful_change"
INTERMEDIATE_IGNORED_ONLY_CANDIDATE = "continue_guard.ignored_only"
INTERMEDIATE_NO_CHANGE_CANDIDATE = "continue_guard.no_change"
INTERMEDIATE_CYCLE_INCOMPLETE_CANDIDATE = "continue_guard.cycle_incomplete"


def _exact_quarter_turn_matches(
    before_pixels: tuple[tuple[int, int], ...],
    after_pixels: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Return exact normalized clockwise quarter-turn correspondences."""

    current = frozenset(before_pixels)
    target = frozenset(after_pixels)
    matches: list[int] = []
    for quarter_turns in range(1, 4):
        current = frozenset((column, -row) for row, column in current)
        minimum_row = min(row for row, _column in current)
        minimum_column = min(column for _row, column in current)
        current = frozenset(
            (row - minimum_row, column - minimum_column)
            for row, column in current
        )
        if current == target:
            matches.append(quarter_turns)
    return tuple(matches)


def _unique_nearest_component_proposals(
    prior_components: dict[str, ComponentDescription],
    current_components: tuple[ComponentDescription, ...],
) -> dict[str, ComponentDescription]:
    """Exact nearest correspondence; zero-distance matches need no peer scan."""
    groups: dict[tuple[object, ...], list[tuple[ComponentDescription, tuple[int, int]]]] = {}
    at_center: dict[tuple[object, ...], list[ComponentDescription]] = {}
    for item in current_components:
        signature = (item.value, item.area, item.relative_pixels)
        center = (item.bbox.top + item.bbox.bottom, item.bbox.left + item.bbox.right)
        groups.setdefault(signature, []).append((item, center))
        at_center.setdefault((signature, center), []).append(item)
    proposals: dict[str, ComponentDescription] = {}
    for ref, prior in sorted(prior_components.items()):
        signature = (prior.value, prior.area, prior.relative_pixels)
        center = (prior.bbox.top + prior.bbox.bottom, prior.bbox.left + prior.bbox.right)
        coincident = at_center.get((signature, center), ())
        if coincident:
            if len(coincident) == 1:
                proposals[ref] = coincident[0]
            continue
        minimum: int | None = None
        unique_nearest: ComponentDescription | None = None
        for item, position in groups.get(signature, ()):
            distance = (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
            if minimum is None or distance < minimum:
                minimum, unique_nearest = distance, item
            elif distance == minimum:
                unique_nearest = None
        if unique_nearest is not None:
            proposals[ref] = unique_nearest
    return proposals


def _ordered_point_component_trajectory_facts(
    value: SceneTransitionMeasurementInput,
) -> dict[str, object]:
    """Measure exact persistent trajectories toward a point-action coordinate.

    Mutual unique nearest matching is purely mechanical and bounded by the
    observed component sets.  The result does not name a mechanism; DRM keeps
    local-field, direct-selection, and unrelated-animation meanings separate.
    """

    point_x = value.action_data.get("x")
    point_y = value.action_data.get("y")
    empty = {
        "ordered_point_coordinate_present": False,
        "transient_click_centered_ring_observed": False,
        "transient_click_centered_ring_contracts": False,
        "transient_click_centered_ring_lifetime_frame_count": 0,
        "transient_click_centered_ring_max_chebyshev_radius": 0,
        "transient_click_centered_ring_radius_sequence": (),
        "complete_persistent_component_trajectory_count": 0,
        "moving_persistent_component_trajectory_count": 0,
        "stationary_persistent_component_trajectory_count": 0,
        "unique_persistent_component_monotone_approach_to_point": False,
        "unique_persistent_component_lands_on_point": False,
        "all_other_complete_persistent_components_stationary": False,
    }
    if (
        value.action_ref != "ACTION6"
        or not isinstance(point_x, int)
        or isinstance(point_x, bool)
        or not isinstance(point_y, int)
        or isinstance(point_y, bool)
    ):
        return {}

    frames = (value.before, *value.intermediate_frames, value.after)

    # A tutorial can expose a click's finite influence footprint as a transient
    # concentric ring.  This is a bounded raster measurement only: DRM decides
    # whether the ring is a field, while SRC decides whether that field advances
    # an open objective.  Requiring two strictly contracting shells prevents a
    # lone moving body or an incidental flash from becoming a radius model.
    transient_ring_radii: list[int] = []
    for frame in value.intermediate_frames:
        changed_by_value: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in range(frame.height):
            for col in range(frame.width):
                if frame.rows[row][col] != value.after.rows[row][col]:
                    changed_by_value[frame.rows[row][col]].append((row, col))
        frame_candidates: list[tuple[int, int]] = []
        for pixel_value in sorted(changed_by_value):
            pending_positions = set(changed_by_value[pixel_value])
            connected_groups: list[list[tuple[int, int]]] = []
            while pending_positions:
                seed = min(pending_positions)
                pending_positions.remove(seed)
                group = [seed]
                frontier = [seed]
                while frontier:
                    row, col = frontier.pop()
                    neighbors = tuple(
                        sorted(
                            (row + delta_row, col + delta_col)
                            for delta_row in (-1, 0, 1)
                            for delta_col in (-1, 0, 1)
                            if delta_row != 0 or delta_col != 0
                        )
                    )
                    for neighbor in neighbors:
                        if neighbor not in pending_positions:
                            continue
                        pending_positions.remove(neighbor)
                        group.append(neighbor)
                        frontier.append(neighbor)
                connected_groups.append(group)
            for positions in connected_groups:
                if len(positions) < 8:
                    continue
                top = min(row for row, _col in positions)
                bottom = max(row for row, _col in positions)
                left = min(col for _row, col in positions)
                right = max(col for _row, col in positions)
                extents = (
                    point_y - top,
                    bottom - point_y,
                    point_x - left,
                    right - point_x,
                )
                if min(extents) <= 0 or len(set(extents)) != 1:
                    continue
                radius = extents[0]
                if len(positions) < 4 * radius:
                    continue
                if len(positions) * 2 >= (2 * radius + 1) ** 2:
                    continue
                frame_candidates.append((radius, len(positions)))
        if frame_candidates:
            transient_ring_radii.append(max(frame_candidates)[0])
    distinct_ring_radii = tuple(dict.fromkeys(transient_ring_radii))
    contracting_ring = bool(
        len(distinct_ring_radii) >= 2
        and all(
            after < before
            for before, after in zip(
                distinct_ring_radii, distinct_ring_radii[1:]
            )
        )
    )
    component_sets = tuple(
        detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
        for frame in frames
    )

    def signature(component: ComponentDescription) -> tuple[object, ...]:
        return (
            component.value,
            component.area,
            component.relative_pixels,
        )

    def center_twice(component: ComponentDescription) -> tuple[int, int]:
        return (
            component.bbox.top + component.bbox.bottom,
            component.bbox.left + component.bbox.right,
        )

    trajectories: dict[str, list[ComponentDescription]] = {
        component.component_id: [component] for component in component_sets[0]
    }
    active = dict(trajectories)
    for current_components in component_sets[1:]:
        prior_components = {
            ref: items[-1] for ref, items in sorted(active.items())
        }
        proposals = _unique_nearest_component_proposals(prior_components, current_components)
        reciprocal_counts = Counter(item.component_id for item in (proposals[__yf_order_key] for __yf_order_key in sorted(proposals)))
        next_active: dict[str, list[ComponentDescription]] = {}
        for ref, current in sorted(proposals.items()):
            if reciprocal_counts[current.component_id] != 1:
                continue
            trajectories[ref].append(current)
            next_active[ref] = trajectories[ref]
        active = next_active

    complete = tuple(
        items for items in (trajectories[__yf_order_key] for __yf_order_key in sorted(trajectories)) if len(items) == len(frames)
    )
    point_twice = (2 * point_y, 2 * point_x)
    moving = tuple(
        items
        for items in complete
        if center_twice(items[0]) != center_twice(items[-1])
    )
    stationary = tuple(
        items
        for items in complete
        if all(center_twice(item) == center_twice(items[0]) for item in items[1:])
    )
    monotone_to_point: list[list[ComponentDescription]] = []
    for items in moving:
        distances = tuple(
            abs(center_twice(item)[0] - point_twice[0])
            + abs(center_twice(item)[1] - point_twice[1])
            for item in items
        )
        if (
            all(after <= before for before, after in zip(distances, distances[1:]))
            and any(after < before for before, after in zip(distances, distances[1:]))
        ):
            monotone_to_point.append(items)

    palette_mass_pair: tuple[ComponentDescription, ComponentDescription] | None = None
    if not monotone_to_point:
        landing_components = tuple(
            component
            for component in component_sets[-1]
            if center_twice(component) == point_twice
        )
        pair_candidates: list[
            tuple[int, ComponentDescription, ComponentDescription]
        ] = []
        for landing in landing_components:
            compatible_before = tuple(
                component
                for component in component_sets[0]
                if signature(component) == signature(landing)
                and center_twice(component) != point_twice
            )
            distances = tuple(
                (
                    abs(center_twice(component)[0] - point_twice[0])
                    + abs(center_twice(component)[1] - point_twice[1]),
                    component,
                )
                for component in compatible_before
            )
            if not distances:
                continue
            minimum = min(item[0] for item in distances)
            nearest = tuple(item[1] for item in distances if item[0] == minimum)
            if len(nearest) == 1:
                pair_candidates.append((minimum, nearest[0], landing))
        valid_palette_mass_pairs: list[
            tuple[ComponentDescription, ComponentDescription]
        ] = []
        for _distance, before_candidate, landing_candidate in pair_candidates:
            top = min(before_candidate.bbox.top, landing_candidate.bbox.top)
            left = min(before_candidate.bbox.left, landing_candidate.bbox.left)
            bottom = max(before_candidate.bbox.bottom, landing_candidate.bbox.bottom)
            right = max(before_candidate.bbox.right, landing_candidate.bbox.right)
            palette_distances: list[Fraction] = []
            for frame in frames:
                positions = tuple(
                    (row, col)
                    for row in range(top, bottom + 1)
                    for col in range(left, right + 1)
                    if frame.rows[row][col] == before_candidate.value
                )
                if not positions:
                    palette_distances = []
                    break
                count = len(positions)
                palette_distances.append(
                    Fraction(
                        abs(sum(row for row, _col in positions) - point_y * count)
                        + abs(sum(col for _row, col in positions) - point_x * count),
                        count,
                    )
                )
            peak_index = (
                palette_distances.index(max(palette_distances))
                if palette_distances
                else 0
            )
            approach_suffix = palette_distances[peak_index:]
            if (
                approach_suffix
                and all(
                    after <= before
                    for before, after in zip(
                        approach_suffix, approach_suffix[1:]
                    )
                )
                and any(
                    after < before
                    for before, after in zip(
                        approach_suffix, approach_suffix[1:]
                    )
                )
                and approach_suffix[-1] == 0
            ):
                valid_palette_mass_pairs.append(
                    (before_candidate, landing_candidate)
                )
        if len(valid_palette_mass_pairs) == 1:
            palette_mass_pair = valid_palette_mass_pairs[0]

    result = {
        **empty,
        "ordered_point_coordinate_present": True,
        "transient_click_centered_ring_observed": contracting_ring,
        "transient_click_centered_ring_contracts": contracting_ring,
        "transient_click_centered_ring_lifetime_frame_count": (
            len(distinct_ring_radii) if contracting_ring else 0
        ),
        "transient_click_centered_ring_max_chebyshev_radius": (
            max(distinct_ring_radii) if contracting_ring else 0
        ),
        "transient_click_centered_ring_radius_sequence": (
            distinct_ring_radii if contracting_ring else ()
        ),
        "ordered_point_transition_frame_count": len(frames) - 1,
        "complete_persistent_component_trajectory_count": len(complete),
        "moving_persistent_component_trajectory_count": len(moving),
        "stationary_persistent_component_trajectory_count": len(stationary),
        "unique_persistent_component_monotone_approach_to_point": bool(
            len(monotone_to_point) == 1 or palette_mass_pair is not None
        ),
        "ordered_palette_mass_eventual_monotone_approach_to_point": (
            palette_mass_pair is not None
        ),
        "all_other_complete_persistent_components_stationary": bool(
            len(monotone_to_point) == 1 and len(stationary) == len(complete) - 1
        ),
    }
    if len(monotone_to_point) == 1:
        unique = monotone_to_point[0]
        before_component = unique[0]
        after_component = unique[-1]
    elif palette_mass_pair is not None:
        before_component, after_component = palette_mass_pair
    else:
        return result
    before_center = center_twice(before_component)
    after_center = center_twice(after_component)
    row_gap = max(
        before_component.bbox.top - point_y,
        0,
        point_y - before_component.bbox.bottom,
    )
    col_gap = max(
        before_component.bbox.left - point_x,
        0,
        point_x - before_component.bbox.right,
    )
    result.update(
        {
            "unique_persistent_component_lands_on_point": (
                after_center == point_twice
            ),
            "unique_point_approach_component_ref": before_component.component_id,
            "unique_point_approach_after_component_ref": after_component.component_id,
            "unique_point_approach_component_area": before_component.area,
            "unique_point_approach_before_center_row_twice": before_center[0],
            "unique_point_approach_before_center_column_twice": before_center[1],
            "unique_point_approach_after_center_row_twice": after_center[0],
            "unique_point_approach_after_center_column_twice": after_center[1],
            "unique_point_approach_delta_row_twice": after_center[0] - before_center[0],
            "unique_point_approach_delta_column_twice": after_center[1] - before_center[1],
            "unique_point_approach_integer_delta_present": bool(
                (after_center[0] - before_center[0]) % 2 == 0
                and (after_center[1] - before_center[1]) % 2 == 0
            ),
            "unique_point_approach_delta_row": (
                (after_center[0] - before_center[0]) // 2
            ),
            "unique_point_approach_delta_column": (
                (after_center[1] - before_center[1]) // 2
            ),
            "point_to_before_footprint_row_gap": row_gap,
            "point_to_before_footprint_column_gap": col_gap,
            "point_to_before_footprint_chebyshev_distance": max(row_gap, col_gap),
            "point_to_before_footprint_manhattan_distance": row_gap + col_gap,
        }
    )
    return result


def _ordered_animated_reconfiguration_facts(
    value: SceneTransitionMeasurementInput,
) -> dict[str, object]:
    """Recover a bounded rigid reconfiguration from an animated point packet.

    This is deliberately retrospective: it compares the settled frame before
    the point action with the settled frame after all intermediate frames have
    completed.  It does not infer a colour role or a timing rule.  A persistent
    component whose normalized occupancy is an exact quarter-turn of its prior
    occupancy, together with at least one unchanged component, is simply
    reported as mechanical evidence that a previously observed configuration
    was reconfigured.  DRM decides whether that evidence closes a mechanism
    hypothesis and which route should be resumed.
    """

    empty = {
        "ordered_animation_reconfiguration_frame_count": 0,
        "animated_reconfiguration_persistent_anchor_present": False,
        "animated_reconfiguration_unique_quarter_turn_present": False,
        "animated_reconfiguration_quarter_turns": (),
        "animated_reconfiguration_surface_exchange_candidate": False,
        "animated_reconfiguration_intermediate_effect_observed": False,
        "animated_reconfiguration_reverted_after_unsafe_contact": False,
    }
    if value.action_ref != "ACTION6" or not value.intermediate_frames:
        return {}
    frames = (value.before, *value.intermediate_frames, value.after)
    if len(frames) > 192:
        return {}
    component_sets = tuple(
        detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
        for frame in frames
    )
    before_components = component_sets[0]
    after_components = component_sets[-1]
    def center(component: ComponentDescription) -> tuple[int, int]:
        return (
            component.bbox.top + component.bbox.bottom,
            component.bbox.left + component.bbox.right,
        )
    anchors = tuple(
        before
        for before in before_components
        if not before.touches_frame_boundary
        and any(
            after.value == before.value
            and after.area == before.area
            and after.relative_pixels == before.relative_pixels
            and center(after) == center(before)
            for after in after_components
        )
    )
    candidates: list[tuple[int, ComponentDescription, ComponentDescription]] = []
    for before in before_components:
        if before.touches_frame_boundary or before.area <= 1:
            continue
        for after in after_components:
            if (
                after.value != before.value
                or after.area != before.area
                or after.relative_pixels == before.relative_pixels
                or after.touches_frame_boundary
                or max(
                    abs(center(after)[0] - center(before)[0]),
                    abs(center(after)[1] - center(before)[1]),
                ) > 2
            ):
                continue
            turns = _exact_quarter_turn_matches(before.relative_pixels, after.relative_pixels)
            for turn in turns:
                candidates.append((turn, before, after))
    intermediate_candidates: list[
        tuple[int, ComponentDescription, ComponentDescription]
    ] = []
    for current_components in component_sets[1:-1]:
        for before in before_components:
            if before.touches_frame_boundary or before.area <= 1:
                continue
            for current in current_components:
                if (
                    current.value != before.value
                    or current.area != before.area
                    or current.relative_pixels == before.relative_pixels
                    or current.touches_frame_boundary
                ):
                    continue
                for turn in _exact_quarter_turn_matches(
                    before.relative_pixels, current.relative_pixels
                ):
                    intermediate_candidates.append((turn, before, current))
    pair_keys = {
        (before.value, center(before), center(after))
        for _turn, before, after in candidates
    }
    unique_pair = len(pair_keys) == 1 and bool(candidates)
    turns = tuple(sorted({item[0] for item in candidates}))
    intermediate_pair_keys = {
        (before.value, center(before), center(current))
        for _turn, before, current in intermediate_candidates
    }
    unique_intermediate = len(intermediate_pair_keys) == 1 and bool(
        intermediate_candidates
    )
    intermediate_turns = tuple(sorted({item[0] for item in intermediate_candidates}))
    observed_turns = tuple(sorted(set(turns).union(intermediate_turns)))
    reverted = bool(
        unique_intermediate
        and value.before.rows == value.after.rows
        and not unique_pair
    )
    return {
        "ordered_animation_reconfiguration_frame_count": len(frames) - 1,
        "animated_reconfiguration_persistent_anchor_present": bool(anchors),
        "animated_reconfiguration_unique_quarter_turn_present": (
            unique_pair or unique_intermediate
        ),
        "animated_reconfiguration_quarter_turns": observed_turns,
        "animated_reconfiguration_surface_exchange_candidate": bool(
            (unique_pair or unique_intermediate)
            and anchors
            and value.action_ref == "ACTION6"
        ),
        "animated_reconfiguration_intermediate_effect_observed": unique_intermediate,
        "animated_reconfiguration_reverted_after_unsafe_contact": reverted,
    }


def _consistent_noncolinear_turn(
    centers: tuple[tuple[int, int], ...],
) -> bool:
    """Report one consistent bend of at least one cell off the chord.

    Centers are twice-scaled bbox sums, so one cell is two units. A zero
    cross is a straight step and does not flip the turn. Opposing signs are
    not one curve. The chord test rejects a kink smaller than one cell.
    """

    if len(centers) < 3 or len(set(centers)) < 3:
        return False
    signs: list[int] = []
    for index in range(len(centers) - 2):
        start_row, start_column = centers[index]
        middle_row, middle_column = centers[index + 1]
        end_row, end_column = centers[index + 2]
        first_row = middle_row - start_row
        first_column = middle_column - start_column
        second_row = end_row - middle_row
        second_column = end_column - middle_column
        cross = first_row * second_column - first_column * second_row
        if cross:
            signs.append(1 if cross > 0 else -1)
    if not signs or any(sign != signs[0] for sign in signs):
        return False
    origin_row, origin_column = centers[0]
    chord_row = centers[-1][0] - origin_row
    chord_column = centers[-1][1] - origin_column
    chord_sq = chord_row * chord_row + chord_column * chord_column
    if chord_sq == 0:
        return False
    return any(
        (row - origin_row) * chord_column - (column - origin_column) * chord_row
        for row, column in centers[1:-1]
    ) and any(
        (
            (row - origin_row) * chord_column
            - (column - origin_column) * chord_row
        )
        ** 2
        >= 4 * chord_sq
        for row, column in centers[1:-1]
    )


def _maximal_prefix_suffix_bars(
    values: list[int],
    thin: list[bool],
) -> list[tuple[int, int, int, int, int, int]]:
    """Return inclusion-maximal thickness-1 prefix/suffix spans.

    Each tuple is start, end, prefix count, body span, prefix value, suffix
    value. At most two rare cells are ignored so a single marker can sit
    inside the prefix without splitting the bar.
    """

    if sum(thin) < 8:
        return []
    count_limit = len(values)
    windows: list[tuple[int, int, int, int, int, int]] = []
    for start in range(count_limit):
        counts: dict[int, int] = {}
        best: tuple[int, int, int, int, int, int] | None = None
        for end in range(start, count_limit):
            value = values[end]
            counts[value] = counts.get(value, 0) + 1
            if len(counts) > 3:
                break
            occurrence = counts[value]
            if occurrence == 3 and any(
                not thin[index] and values[index] == value
                for index in range(start, end + 1)
            ):
                break
            if occurrence > 3 and not thin[end]:
                break
            marker_values = {
                item
                for item, item_count in sorted(counts.items())
                if item_count <= 2
            }
            marker_cells = sum(
                item_count
                for item, item_count in sorted(counts.items())
                if item in marker_values
            )
            body_values = [
                item for item in sorted(counts) if item not in marker_values
            ]
            if len(body_values) > 2:
                break
            length = end - start + 1
            if (
                length < 8
                or marker_cells > 2
                or len(body_values) != 2
            ):
                continue
            prefix_value = None
            suffix_value = None
            prefix_count = 0
            suffix_count = 0
            switched = False
            pure = True
            for index in range(start, end + 1):
                cell = values[index]
                if cell in marker_values:
                    continue
                if prefix_value is None:
                    prefix_value = cell
                    prefix_count = 1
                    continue
                if cell == prefix_value and not switched:
                    prefix_count += 1
                    continue
                if suffix_value is None:
                    suffix_value = cell
                    suffix_count = 1
                    switched = True
                    continue
                if cell == suffix_value:
                    suffix_count += 1
                    continue
                pure = False
                break
            if (
                not pure
                or prefix_value is None
                or suffix_value is None
                or prefix_value == suffix_value
            ):
                continue
            best = (
                start,
                end + 1,
                prefix_count,
                prefix_count + suffix_count,
                prefix_value,
                suffix_value,
            )
        if best is not None:
            windows.append(best)
    maximal: list[tuple[int, int, int, int, int, int]] = []
    for window in windows:
        if any(
            other[0] <= window[0]
            and window[1] <= other[1]
            and (other[0], other[1]) != (window[0], window[1])
            for other in windows
        ):
            continue
        if any(
            (other[0], other[1]) == (window[0], window[1]) for other in maximal
        ):
            continue
        maximal.append(window)
    parent = list(range(len(maximal)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(maximal):
        for right_index in range(left_index + 1, len(maximal)):
            right = maximal[right_index]
            if left[0] < right[1] and right[0] < left[1]:
                union(left_index, right_index)
    groups: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for index, window in enumerate(maximal):
        groups.setdefault(find(index), []).append(window)
    return [
        max(group, key=lambda item: (item[1] - item[0], item[3]))
        for _root, group in sorted(groups.items())
    ]


def measure_scalar_fill_display(frame: FrameGrid) -> dict[str, object]:
    """Measure one unique thickness-1 bar with a prefix and a suffix.

    Orthogonal neighbors of the body cells must be a different value, so a
    thick body edge is not a bar. A second qualifying bar, or none, leaves
    the display absent. The reading is a count, not a named control role.
    """

    absent = {
        "scalar_fill_display_present": False,
        "scalar_fill_count": 0,
        "scalar_fill_span": 0,
        "scalar_fill_orientation": "",
        "scalar_fill_index": -1,
        "scalar_fill_start": -1,
        "scalar_fill_end": -1,
        "scalar_fill_prefix_value": -1,
        "scalar_fill_suffix_value": -1,
    }
    height = frame.height
    width = frame.width
    rows = frame.rows
    found: list[dict[str, object]] = []

    def consider(
        orientation: str,
        index: int,
        values: list[int],
        thin: list[bool],
    ) -> None:
        for start, end, count, span, prefix, suffix in _maximal_prefix_suffix_bars(
            values, thin
        ):
            found.append(
                {
                    "scalar_fill_display_present": True,
                    "scalar_fill_count": count,
                    "scalar_fill_span": span,
                    "scalar_fill_orientation": orientation,
                    "scalar_fill_index": index,
                    "scalar_fill_start": start,
                    "scalar_fill_end": end - 1,
                    "scalar_fill_prefix_value": prefix,
                    "scalar_fill_suffix_value": suffix,
                }
            )

    for row in range(height):
        values = [int(rows[row][column]) for column in range(width)]
        thin = [
            row > 0
            and row + 1 < height
            and int(rows[row - 1][column]) != value
            and int(rows[row + 1][column]) != value
            for column, value in enumerate(values)
        ]
        consider("horizontal", row, values, thin)
    for column in range(width):
        values = [int(rows[row][column]) for row in range(height)]
        thin = [
            column > 0
            and column + 1 < width
            and int(rows[row][column - 1]) != value
            and int(rows[row][column + 1]) != value
            for row, value in enumerate(values)
        ]
        consider("vertical", column, values, thin)
    if len(found) != 1:
        return absent
    return found[0]


def _component_bbox_center(component: ComponentDescription) -> tuple[float, float]:
    box = component.bbox
    return (
        (box.top + box.bottom) / 2,
        (box.left + box.right) / 2,
    )


def _linear_slope(values: tuple[float, ...]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    sum_time = (count - 1) * count / 2
    sum_time_square = sum(time * time for time in range(count))
    sum_value = sum(values)
    sum_time_value = sum(time * value for time, value in enumerate(values))
    determinant = count * sum_time_square - sum_time * sum_time
    if determinant == 0:
        return 0.0
    return (count * sum_time_value - sum_time * sum_value) / determinant


def _quadratic_linear_and_square(values: tuple[float, ...]) -> tuple[float, float]:
    """Return the linear and square coefficients of a least-squares quadratic."""

    count = len(values)
    if count < 3:
        return (0.0, 0.0)
    times = tuple(range(count))
    sums = [float(count), 0.0, 0.0, 0.0, 0.0]
    for time in times:
        sums[1] += time
        sums[2] += time * time
        sums[3] += time ** 3
        sums[4] += time ** 4
    moment = [
        sum(values),
        sum(time * value for time, value in zip(times, values)),
        sum(time * time * value for time, value in zip(times, values)),
    ]
    matrix = [
        [sums[0], sums[1], sums[2]],
        [sums[1], sums[2], sums[3]],
        [sums[2], sums[3], sums[4]],
    ]
    for pivot in range(3):
        scale = matrix[pivot][pivot] or 1.0
        for column in range(pivot, 3):
            matrix[pivot][column] /= scale
        moment[pivot] /= scale
        for row in range(3):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            for column in range(pivot, 3):
                matrix[row][column] -= factor * matrix[pivot][column]
            moment[row] -= factor * moment[pivot]
    return (moment[1], moment[2])


def _ballistic_contact_click(
    track: tuple[ComponentDescription, ...],
    source: ComponentDescription,
    peer: ComponentDescription,
    before_components: tuple[ComponentDescription, ...],
    frame: FrameGrid,
    *,
    avoid: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Search a contact click, or one longer flight when the fit cannot meet.

    The fit uses bbox centers of the moving body, the launch body, and the
    one stationary body of comparable shape whose value is not repeated.
    The first cell is the closest arc that meets that body. The second cell
    is only present when no arc meets it: the longest clear flight toward it.
    Python returns cells. It does not name a shooter, a target, or a game.
    """

    if len(track) < 4:
        return (None, None)
    if sum(component.value == peer.value for component in before_components) != 1:
        return (None, None)
    points = tuple(_component_bbox_center(component) for component in track)
    velocity_row, square = _quadratic_linear_and_square(
        tuple(point[0] for point in points)
    )
    gravity = 2 * square
    velocity_column = _linear_slope(tuple(point[1] for point in points))
    speed = math.hypot(velocity_row, velocity_column)
    if speed < 0.5 or abs(gravity) < 0.05:
        return (None, None)
    source_center = _component_bbox_center(source)
    peer_center = _component_bbox_center(peer)
    launch_offset = math.hypot(
        points[0][0] - source_center[0], points[0][1] - source_center[1]
    )
    if launch_offset < 1.0:
        return (None, None)
    frame_height = frame.height
    frame_width = frame.width
    background = max(before_components, key=lambda component: component.area)
    blocked = {
        pixel
        for component in before_components
        if component.component_id
        not in {source.component_id, peer.component_id, background.component_id}
        and component.area > 1
        for pixel in component.pixels
    }
    full_width_tops = tuple(
        component.bbox.top
        for component in before_components
        if component.bbox.width >= frame_width and component.bbox.top > frame_height // 2
    )
    play_bottom = min(full_width_tops) if full_width_tops else frame_height
    fill = measure_scalar_fill_display(frame)
    fill_index = fill.get("scalar_fill_index")
    if (
        fill.get("scalar_fill_display_present") is True
        and fill.get("scalar_fill_orientation") == "horizontal"
        and isinstance(fill_index, int)
        and not isinstance(fill_index, bool)
        and fill_index > 0
    ):
        play_bottom = min(play_bottom, fill_index)
    projectile_extent = max(
        max(component.bbox.height, component.bbox.width) for component in track
    )
    radius = min(peer.bbox.height, peer.bbox.width) / 2 + projectile_extent / 2
    best_gap: tuple[float, int, int] | None = None
    best_flight: tuple[int, int, int] | None = None
    seen: set[tuple[int, int]] = set()
    for step in range(72):
        angle = -math.pi + (2 * math.pi * step / 72)
        click_row = int(round(source_center[0] + math.sin(angle) * 18))
        click_column = int(round(source_center[1] + math.cos(angle) * 18))
        if (click_row, click_column) in seen or (click_row, click_column) == avoid:
            continue
        seen.add((click_row, click_column))
        if not (0 <= click_row < play_bottom and 0 <= click_column < frame_width):
            continue
        delta_row = click_row - source_center[0]
        delta_column = click_column - source_center[1]
        distance = math.hypot(delta_row, delta_column)
        if distance < 1.0:
            continue
        unit_row = delta_row / distance
        unit_column = delta_column / distance
        toward_peer = delta_column * (peer_center[1] - source_center[1]) > 0
        row = source_center[0] + unit_row * launch_offset
        column = source_center[1] + unit_column * launch_offset
        velocity_step_row = unit_row * speed
        velocity_step_column = unit_column * speed
        nearest = float("inf")
        flown = 0
        saw_ascent = velocity_step_row < -0.05
        saw_descent = False
        for _tick in range(160):
            next_row = row + velocity_step_row
            next_column = column + velocity_step_column
            segment = math.hypot(next_row - row, next_column - column)
            samples = max(1, int(math.ceil(segment * 2.0)))
            blocked_segment = False
            for sample in range(1, samples + 1):
                fraction = sample / samples
                sample_row = row + (next_row - row) * fraction
                sample_column = column + (next_column - column) * fraction
                gap = math.hypot(
                    sample_row - peer_center[0], sample_column - peer_center[1]
                )
                if gap < nearest:
                    nearest = gap
                cell = (int(round(sample_row)), int(round(sample_column)))
                if (
                    cell[0] >= play_bottom
                    or cell in blocked
                    or not (0 <= cell[0] < frame_height and 0 <= cell[1] < frame_width)
                ):
                    blocked_segment = True
                    break
            if blocked_segment:
                break
            flown += 1
            row, column = next_row, next_column
            velocity_step_row += gravity
            if saw_ascent and velocity_step_row > 0.05:
                saw_descent = True
        if best_gap is None or nearest < best_gap[0]:
            best_gap = (nearest, click_row, click_column)
        if (
            toward_peer
            and saw_descent
            and abs(unit_column) >= 0.25
            and flown >= 8
            and (best_flight is None or flown > best_flight[0])
        ):
            best_flight = (flown, click_row, click_column)
    contact = None
    if best_gap is not None and best_gap[0] <= radius:
        contact = (best_gap[1], best_gap[2])
    disclosure = None
    if contact is None and best_flight is not None:
        disclosure = (best_flight[1], best_flight[2])
    return (contact, disclosure)


def _ordered_transient_component_path_facts(
    value: SceneTransitionMeasurementInput,
) -> dict[str, object]:
    """Measure one bounded moving component that appears during a point packet.

    The detector reports only ordered raster geometry.  It neither calls the
    body a projectile nor assigns source/target roles; those interpretations
    remain competing DRM suppositions until action-conditioned evidence selects
    them.  Oversized packets fail closed instead of widening the search.
    """

    empty = {
        "transient_component_path_frame_bound_reached": False,
        "transient_component_path_seed_bound_reached": False,
        "moving_transient_component_path_count": 0,
        "unique_moving_transient_component_path_present": False,
        "unique_transient_path_frame_count": 0,
        "unique_transient_path_distinct_position_count": 0,
        "unique_transient_path_horizontal_direction_consistent": False,
        "unique_transient_path_vertical_velocity_changes": False,
        "unique_transient_path_palette_value_preserved": False,
        "unique_transient_path_unique_nearest_continuation_present": False,
        "unique_transient_path_area_span_within_maximum_extent": False,
        "unique_transient_path_orientation_moment_signature_changes": False,
        "unique_transient_path_single_rise_to_fall_reversal": False,
        "unique_transient_path_consistent_noncolinear_turn": False,
        "unique_transient_path_curved": False,
        "unique_transient_path_source_candidate_count": 0,
        "unique_transient_path_source_candidate_present": False,
        "unique_transient_path_source_morphology_peer_count": 0,
        "unique_transient_path_source_morphology_peer_present": False,
        "point_intersects_unique_transient_source_morphology_peer": False,
        "transient_terminal_before_source_peer_along_horizontal_direction": False,
        "transient_terminal_beyond_source_peer_along_horizontal_direction": False,
        "transient_component_absent_from_settled_frame": False,
        "transient_terminal_unique_restored_persistent_component_present": False,
        "transient_terminal_restored_persistent_contact_present": False,
        "transient_terminal_restored_component_differs_from_source_peer": False,
        "unique_transient_path_source_morphology_peer_value_unique": False,
        "ballistic_contact_solution_present": False,
        "ballistic_curve_disclosure_present": False,
    }
    if value.action_ref != "ACTION6" or len(value.intermediate_frames) < 2:
        return {}
    frames = (value.before, *value.intermediate_frames, value.after)
    if len(frames) > 192:
        return {
            **empty,
            "transient_component_path_frame_bound_reached": True,
            "ordered_transient_component_packet_frame_count": len(frames),
        }

    component_sets = tuple(
        detect_components(
            ComponentExtractionInput(frame=frame, connectivity=4)
        ).components
        for frame in frames
    )
    if any(len(components) > 512 for components in component_sets):
        return {
            **empty,
            "transient_component_path_seed_bound_reached": True,
            "ordered_transient_component_packet_frame_count": len(frames),
        }

    def center_twice(component: ComponentDescription) -> tuple[int, int]:
        return (
            component.bbox.top + component.bbox.bottom,
            component.bbox.left + component.bbox.right,
        )

    def overlaps(first: ComponentDescription, second: ComponentDescription) -> bool:
        return bool(
            first.bbox.top <= second.bbox.bottom
            and second.bbox.top <= first.bbox.bottom
            and first.bbox.left <= second.bbox.right
            and second.bbox.left <= first.bbox.right
        )

    def normalized_moment_signature(
        component: ComponentDescription,
    ) -> tuple[int, int]:
        """Return an exact translation-invariant raster orientation measure."""

        count = component.area
        row_sum = sum(row for row, _column in component.relative_pixels)
        column_sum = sum(column for _row, column in component.relative_pixels)
        axis = count * sum(
            (column * column) - (row * row)
            for row, column in component.relative_pixels
        ) - ((column_sum * column_sum) - (row_sum * row_sum))
        cross = 2 * (
            count
            * sum(row * column for row, column in component.relative_pixels)
            - (row_sum * column_sum)
        )
        divisor = gcd(abs(axis), abs(cross))
        if divisor:
            return (axis // divisor, cross // divisor)
        return (0, 0)

    def dihedral_shape_signature(
        component: ComponentDescription,
    ) -> tuple[tuple[int, int], ...]:
        """Canonicalize exact raster occupancy under rotations and reflection."""

        signatures: list[tuple[tuple[int, int], ...]] = []
        source = frozenset(component.relative_pixels)
        for reflected in (False, True):
            pixels = {
                (row, -column) if reflected else (row, column)
                for row, column in source
            }
            for _quarter_turn in range(4):
                minimum_row = min(row for row, _column in pixels)
                minimum_column = min(column for _row, column in pixels)
                signatures.append(
                    tuple(
                        sorted(
                            (row - minimum_row, column - minimum_column)
                            for row, column in pixels
                        )
                    )
                )
                pixels = {(column, -row) for row, column in pixels}
        return min(signatures)

    before_components = component_sets[0]
    first_action_components = component_sets[1]
    seeds = tuple(
        component
        for component in first_action_components
        if not component.touches_frame_boundary
        and 0 < component.area <= 128
        and not any(
            prior.value == component.value and overlaps(prior, component)
            for prior in before_components
        )
    )
    if len(seeds) > 64:
        return {
            **empty,
            "transient_component_path_seed_bound_reached": True,
            "ordered_transient_component_packet_frame_count": len(frames),
            "transient_component_path_seed_count": len(seeds),
        }

    maximum_step_twice = max(2, min(value.before.height, value.before.width))
    tracks: list[tuple[ComponentDescription, ...]] = []
    for seed in seeds:
        track = [seed]
        for current_components in component_sets[2:]:
            prior = track[-1]
            prior_center = center_twice(prior)
            compatible = tuple(
                current
                for current in current_components
                if not current.touches_frame_boundary
                and current.value == prior.value
                and max(1, prior.area // 3) <= current.area <= prior.area * 3
                and max(
                    abs(center_twice(current)[0] - prior_center[0]),
                    abs(center_twice(current)[1] - prior_center[1]),
                )
                <= maximum_step_twice
            )
            if not compatible:
                break
            distances = tuple(
                (
                    (
                        center_twice(current)[0] - prior_center[0]
                    )
                    ** 2
                    + (
                        center_twice(current)[1] - prior_center[1]
                    )
                    ** 2,
                    current,
                )
                for current in compatible
            )
            minimum = min(distance for distance, _current in distances)
            nearest = tuple(
                current for distance, current in distances if distance == minimum
            )
            if len(nearest) != 1:
                break
            track.append(nearest[0])
        centers = tuple(center_twice(component) for component in track)
        # The last packet image is the settled post-action state.  A moving
        # transient may legitimately disappear there after contact, so one
        # missing terminal occurrence is allowed; earlier gaps still fail.
        if len(track) >= len(frames) - 2 and len(set(centers)) >= 4:
            tracks.append(tuple(track))

    result = {
        **empty,
        "ordered_transient_component_packet_frame_count": len(frames),
        "transient_component_path_seed_count": len(seeds),
        "moving_transient_component_path_count": len(tracks),
        "unique_moving_transient_component_path_present": len(tracks) == 1,
    }
    if len(tracks) != 1:
        return result

    track = tracks[0]
    centers = tuple(center_twice(component) for component in track)
    transient_absent_from_settled = len(track) == len(frames) - 2
    before_signatures = frozenset(
        (component.value, component.pixels) for component in before_components
    )
    settled_palette_counts = Counter(
        pixel for row in value.after.rows for pixel in row
    )
    settled_dominant_value = settled_palette_counts.most_common(1)[0][0]
    persistent_settled_components = tuple(
        component
        for component in component_sets[-1]
        if component.value != track[-1].value
        and component.value != settled_dominant_value
        and (component.value, component.pixels) in before_signatures
    )
    terminal_pixels = frozenset(track[-1].pixels)
    persistent_distances = tuple(
        (
            len(terminal_pixels.intersection(component.pixels)),
            min(
                abs(first[0] - second[0]) + abs(first[1] - second[1])
                for first in terminal_pixels
                for second in component.pixels
            ),
            component,
        )
        for component in persistent_settled_components
    )
    best_persistent_key = min(
        ((-overlap_area, gap) for overlap_area, gap, _component in persistent_distances),
        default=None,
    )
    nearest_persistent_components = tuple(
        (overlap_area, gap, component)
        for overlap_area, gap, component in persistent_distances
        if (-overlap_area, gap) == best_persistent_key
    )
    terminal_persistent_component = (
        nearest_persistent_components[0][2]
        if len(nearest_persistent_components) == 1
        else None
    )
    terminal_persistent_overlap_area = (
        nearest_persistent_components[0][0]
        if terminal_persistent_component is not None
        else 0
    )
    terminal_persistent_gap = (
        nearest_persistent_components[0][1]
        if terminal_persistent_component is not None
        else 0
    )
    row_deltas = tuple(
        after[0] - before[0] for before, after in zip(centers, centers[1:])
    )
    column_deltas = tuple(
        after[1] - before[1] for before, after in zip(centers, centers[1:])
    )
    row_second_deltas = tuple(
        after - before for before, after in zip(row_deltas, row_deltas[1:])
    )
    nonzero_row_signs = tuple(
        1 if delta > 0 else -1 for delta in row_deltas if delta
    )
    row_sign_reversal_count = sum(
        before != after
        for before, after in zip(nonzero_row_signs, nonzero_row_signs[1:])
    )
    nonzero_column_signs = {
        1 if delta > 0 else -1 for delta in column_deltas if delta
    }
    areas = tuple(component.area for component in track)
    maximum_extents = tuple(
        max(component.bbox.height, component.bbox.width) for component in track
    )
    normalized_shape_count = len(
        {component.relative_pixels for component in track}
    )
    dihedral_shape_count = len(
        {dihedral_shape_signature(component) for component in track}
    )
    orientation_signature_count = len(
        {normalized_moment_signature(component) for component in track}
    )
    source_distances = tuple(
        (
            (
                center_twice(component)[0] - centers[0][0]
            )
            ** 2
            + (
                center_twice(component)[1] - centers[0][1]
            )
            ** 2,
            component,
        )
        for component in before_components
        if not component.touches_frame_boundary
        and component.area > 1
        and row_deltas
        and column_deltas
        and (
            (center_twice(component)[0] - centers[0][0]) * row_deltas[0]
            + (center_twice(component)[1] - centers[0][1]) * column_deltas[0]
        )
        < 0
    )
    minimum_source_distance = min(
        (distance for distance, _component in source_distances), default=None
    )
    source_candidates = tuple(
        component
        for distance, component in source_distances
        if distance == minimum_source_distance
    )
    result.update(
        {
            "unique_transient_path_frame_count": len(track),
            "unique_transient_path_distinct_position_count": len(set(centers)),
            "unique_transient_path_start_center_row_twice": centers[0][0],
            "unique_transient_path_start_center_column_twice": centers[0][1],
            "unique_transient_path_end_center_row_twice": centers[-1][0],
            "unique_transient_path_end_center_column_twice": centers[-1][1],
            "unique_transient_path_initial_delta_row_twice": (
                row_deltas[0] if row_deltas else 0
            ),
            "unique_transient_path_initial_delta_column_twice": (
                column_deltas[0] if column_deltas else 0
            ),
            "unique_transient_path_final_delta_row_twice": (
                row_deltas[-1] if row_deltas else 0
            ),
            "unique_transient_path_final_delta_column_twice": (
                column_deltas[-1] if column_deltas else 0
            ),
            "unique_transient_path_row_second_delta_min": (
                min(row_second_deltas) if row_second_deltas else 0
            ),
            "unique_transient_path_row_second_delta_max": (
                max(row_second_deltas) if row_second_deltas else 0
            ),
            "unique_transient_path_horizontal_direction_consistent": bool(
                len(nonzero_column_signs) == 1
            ),
            "unique_transient_path_vertical_velocity_changes": bool(
                row_deltas and min(row_deltas) < max(row_deltas)
            ),
            "unique_transient_path_palette_value": track[0].value,
            "unique_transient_path_palette_value_preserved": bool(
                all(component.value == track[0].value for component in track)
            ),
            "unique_transient_path_unique_nearest_continuation_present": bool(
                len(track) > 1
            ),
            "unique_transient_path_area_min": min(areas),
            "unique_transient_path_area_max": max(areas),
            "unique_transient_path_area_span": max(areas) - min(areas),
            "unique_transient_path_maximum_extent_max": max(maximum_extents),
            "unique_transient_path_area_span_within_maximum_extent": bool(
                max(areas) - min(areas) <= max(maximum_extents)
            ),
            "unique_transient_path_normalized_shape_count": normalized_shape_count,
            "unique_transient_path_dihedral_shape_signature_count": (
                dihedral_shape_count
            ),
            "unique_transient_path_orientation_moment_signature_count": (
                orientation_signature_count
            ),
            "unique_transient_path_orientation_moment_signature_changes": bool(
                orientation_signature_count > 1
            ),
            "unique_transient_path_vertical_sign_reversal_count": (
                row_sign_reversal_count
            ),
            "unique_transient_path_single_rise_to_fall_reversal": bool(
                len(nonzero_row_signs) >= 2
                and nonzero_row_signs[0] == -1
                and nonzero_row_signs[-1] == 1
                and row_sign_reversal_count == 1
            ),
            "unique_transient_path_consistent_noncolinear_turn": (
                _consistent_noncolinear_turn(centers)
            ),
            "unique_transient_path_curved": bool(
                (
                    len(nonzero_row_signs) >= 2
                    and nonzero_row_signs[0] == -1
                    and nonzero_row_signs[-1] == 1
                    and row_sign_reversal_count == 1
                )
                or _consistent_noncolinear_turn(centers)
            ),
            "unique_transient_path_source_candidate_count": len(source_candidates),
            "unique_transient_path_source_candidate_present": (
                len(source_candidates) == 1
            ),
            "transient_component_absent_from_settled_frame": (
                transient_absent_from_settled
            ),
            "transient_terminal_unique_restored_persistent_component_present": (
                terminal_persistent_component is not None
            ),
            "transient_terminal_restored_persistent_component_ref": (
                terminal_persistent_component.component_id
                if terminal_persistent_component is not None
                else ""
            ),
            "transient_terminal_restored_persistent_component_overlap_area": (
                terminal_persistent_overlap_area
            ),
            "transient_terminal_restored_persistent_component_manhattan_gap": (
                terminal_persistent_gap
            ),
            "transient_terminal_restored_persistent_contact_present": bool(
                transient_absent_from_settled
                and terminal_persistent_component is not None
                and (
                    terminal_persistent_overlap_area > 0
                    or terminal_persistent_gap <= 1
                )
            ),
        }
    )
    if len(source_candidates) != 1:
        return result

    source = source_candidates[0]
    source_center = center_twice(source)
    source_compact_extent_ratio = bool(
        min(source.bbox.height, source.bbox.width) * 2
        >= max(source.bbox.height, source.bbox.width)
    )
    peers = tuple(
        component
        for component in before_components
        if component.component_id != source.component_id
        and not component.touches_frame_boundary
        and source_compact_extent_ratio
        and min(component.bbox.height, component.bbox.width) * 2
        >= max(component.bbox.height, component.bbox.width)
        and abs(component.area - source.area)
        <= max(source.bbox.height, source.bbox.width)
    )
    result.update(
        {
            "unique_transient_path_source_component_ref": source.component_id,
            "unique_transient_path_source_center_row_twice": source_center[0],
            "unique_transient_path_source_center_column_twice": source_center[1],
            "unique_transient_path_source_bbox_height": source.bbox.height,
            "unique_transient_path_source_bbox_width": source.bbox.width,
            "unique_transient_path_source_compact_extent_ratio_present": (
                source_compact_extent_ratio
            ),
            "unique_transient_path_source_morphology_peer_count": len(peers),
            "unique_transient_path_source_morphology_peer_present": len(peers) == 1,
        }
    )
    if len(peers) != 1:
        return result

    peer = peers[0]
    peer_center = center_twice(peer)
    point_x = value.action_data.get("x")
    point_y = value.action_data.get("y")
    horizontal_direction = (
        1
        if peer_center[1] > source_center[1]
        else -1
        if peer_center[1] < source_center[1]
        else 0
    )
    terminal_signed_residual = (
        (peer_center[1] - centers[-1][1]) * horizontal_direction
        if horizontal_direction
        else 0
    )
    result.update(
        {
            "unique_transient_component_path_ref": (
                "event.transient_component_path."
                + stable_digest(
                    (
                        centers,
                        source.component_id,
                        peer.component_id,
                        value.action_ref,
                    )
                )[:16]
            ),
            "unique_transient_path_source_morphology_peer_ref": peer.component_id,
            "unique_transient_path_source_morphology_peer_bbox_height": (
                peer.bbox.height
            ),
            "unique_transient_path_source_morphology_peer_bbox_width": (
                peer.bbox.width
            ),
            "unique_transient_path_source_morphology_peer_area": peer.area,
            "unique_transient_path_source_morphology_peer_compact_extent_ratio_present": True,
            "unique_transient_path_source_peer_center_row_twice": peer_center[0],
            "unique_transient_path_source_peer_center_column_twice": peer_center[1],
            "unique_transient_path_source_to_peer_horizontal_direction": (
                horizontal_direction
            ),
            "unique_transient_path_horizontal_direction_matches_source_peer": bool(
                horizontal_direction
                and nonzero_column_signs == {horizontal_direction}
            ),
            "point_intersects_unique_transient_source_morphology_peer": bool(
                isinstance(point_x, int)
                and not isinstance(point_x, bool)
                and isinstance(point_y, int)
                and not isinstance(point_y, bool)
                and peer.bbox.top <= point_y <= peer.bbox.bottom
                and peer.bbox.left <= point_x <= peer.bbox.right
            ),
            "transient_terminal_to_source_peer_horizontal_signed_residual_twice": (
                terminal_signed_residual
            ),
            "transient_terminal_before_source_peer_along_horizontal_direction": (
                terminal_signed_residual > 0
            ),
            "transient_terminal_beyond_source_peer_along_horizontal_direction": (
                terminal_signed_residual < 0
            ),
            "transient_terminal_restored_component_differs_from_source_peer": bool(
                terminal_persistent_component is not None
                and terminal_persistent_component.component_id != peer.component_id
            ),
        }
    )
    peer_value_count = sum(
        component.value == peer.value for component in before_components
    )
    third = max(1, len(row_deltas) // 3)
    opening = row_deltas[:third]
    closing = row_deltas[-third:]
    up_then_down = bool(
        opening
        and closing
        and sum(delta < 0 for delta in opening) > sum(delta > 0 for delta in opening)
        and sum(delta > 0 for delta in closing) > sum(delta < 0 for delta in closing)
    )
    solution = None
    disclosure = None
    if peer_value_count == 1 and (
        result.get("unique_transient_path_curved") is True or up_then_down
    ):
        clicked_row = value.action_data.get("y")
        clicked_column = value.action_data.get("x")
        avoid = (
            (clicked_row, clicked_column)
            if isinstance(clicked_row, int)
            and not isinstance(clicked_row, bool)
            and isinstance(clicked_column, int)
            and not isinstance(clicked_column, bool)
            else None
        )
        solution, disclosure = _ballistic_contact_click(
            track,
            source,
            peer,
            before_components,
            value.before,
            avoid=avoid,
        )
    result["unique_transient_path_source_morphology_peer_value_unique"] = (
        peer_value_count == 1
    )
    result["ballistic_contact_solution_present"] = solution is not None
    if solution is not None:
        result["ballistic_contact_solution_row"] = solution[0]
        result["ballistic_contact_solution_column"] = solution[1]
    result["ballistic_curve_disclosure_present"] = disclosure is not None
    if disclosure is not None:
        result["ballistic_curve_disclosure_row"] = disclosure[0]
        result["ballistic_curve_disclosure_column"] = disclosure[1]
    return result


def _marker_axis_internal_state_transitions(
    value: ControlledTransitionAnalysisInput,
) -> tuple[tuple[str, int, int], ...]:
    """Measure an exclusive focus transfer between stationary axial peers.

    The marker-axis capability supplies only bounded geometry.  This transition
    measurement compares its immutable before/after witnesses and abstracts the
    palette-specific centre values into inactive/active state bits.  Meaning and
    control roles remain the responsibility of DRM.
    """

    from agents.yf_arc3_v5.capabilities.interaction import (
        _orthogonal_hollow_marker_axis_measurement,
    )

    available_action_refs = tuple(
        dict.fromkeys((value.action_ref, *value.available_directional_action_refs))
    )

    def measure(scene: object) -> dict[str, object]:
        return _orthogonal_hollow_marker_axis_measurement(
            InteractionProbeInput(
                frame=scene.frame,
                components=scene.blocks,
                available_action_refs=available_action_refs,
            )
        )

    before = measure(value.before_scene)
    after = measure(value.after_scene)
    before_groups = {
        str(item["marker_axis_group_digest"]): item
        for item in before["orthogonal_hollow_marker_axis_groups"]
    }
    after_groups = {
        str(item["marker_axis_group_digest"]): item
        for item in after["orthogonal_hollow_marker_axis_groups"]
    }
    if (
        len(before_groups) < 2
        or before_groups.keys() != after_groups.keys()
        or not before["unique_deviant_axial_body_group_present"]
        or not after["unique_deviant_axial_body_group_present"]
    ):
        return ()
    before_active_ref = str(before["active_marker_axis_group_digest"])
    after_active_ref = str(after["active_marker_axis_group_digest"])
    if not before_active_ref or not after_active_ref or before_active_ref == after_active_ref:
        return ()
    before_active = before_groups[before_active_ref]
    after_active = after_groups[after_active_ref]
    before_position = (
        int(before_active["marker_axis_body_row"]),
        int(before_active["marker_axis_body_col"]),
    )
    after_position = (
        int(after_active["marker_axis_body_row"]),
        int(after_active["marker_axis_body_col"]),
    )
    changes = difference_frames(
        FrameDifferenceInput(
            before=value.before_scene.frame,
            after=value.after_scene.frame,
            action_ref=value.action_ref,
        )
    ).changes
    world_changed_positions = {
        (item.row, item.col)
        for item in changes
        if 0 < item.row < value.before_scene.frame.height - 1
        and 0 < item.col < value.before_scene.frame.width - 1
    }
    if world_changed_positions != {before_position, after_position}:
        return ()
    before_frame = value.before_scene.frame.rows
    after_frame = value.after_scene.frame.rows
    focus_value = before_frame[before_position[0]][before_position[1]]
    if not (
        after_frame[after_position[0]][after_position[1]] == focus_value
        and before_frame[after_position[0]][after_position[1]]
        == int(before_groups[after_active_ref]["marker_axis_group_value"])
        and after_frame[before_position[0]][before_position[1]]
        == int(after_groups[before_active_ref]["marker_axis_group_value"])
    ):
        return ()
    return (
        (before_active_ref, 1, 0),
        (after_active_ref, 0, 1),
    )


def _stationary_inverse_value_axis_focus_measurement(
    transitions: tuple[tuple[TrackedComponent, TrackedComponent], ...],
    *,
    pre_action_input_aligned_bboxes: tuple[BoundingBox, ...] = (),
) -> dict[str, object]:
    """Measure one net-activated collinear state support without naming its role."""

    directed_counts = Counter(
        (before.component.value, after.component.value)
        for before, after in transitions
        if before.component.value != after.component.value
    )
    unordered_pairs = {
        tuple(sorted((before_value, after_value)))
        for before_value, after_value in directed_counts
    }
    if len(unordered_pairs) != 1:
        return {}
    first_value, second_value = next(iter(unordered_pairs))
    first_to_second = int(directed_counts.get((first_value, second_value), 0))
    second_to_first = int(directed_counts.get((second_value, first_value), 0))
    if first_to_second <= 0 or second_to_first <= 0:
        return {}
    directed_positions: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for directed_pair in sorted(directed_counts):
        directed_positions[directed_pair] = tuple(
            sorted(
                (
                    (after.component.bbox.top + after.component.bbox.bottom) // 2,
                    (after.component.bbox.left + after.component.bbox.right) // 2,
                )
                for before, after in transitions
                if (
                    before.component.value,
                    after.component.value,
                )
                == directed_pair
            )
        )

    def bbox_contains_position(
        bbox: BoundingBox,
        position: tuple[int, int],
    ) -> bool:
        row, column = position
        return (
            bbox.top <= row <= bbox.bottom
            and bbox.left <= column <= bbox.right
        )

    input_aligned_departures = tuple(
        (directed_pair, positions)
        for directed_pair, positions in sorted(directed_positions.items())
        if positions
        and any(
            all(bbox_contains_position(bbox, position) for position in positions)
            for bbox in pre_action_input_aligned_bboxes
        )
    )
    input_aligned_inverse_value_measurement: dict[str, object] = {}
    if len(input_aligned_departures) == 1:
        departure_pair, departure_positions = input_aligned_departures[0]
        remote_positions = directed_positions.get(
            (departure_pair[1], departure_pair[0]),
            (),
        )
        remote_positions_are_outside_input_aligned_bboxes = bool(
            remote_positions
            and all(
                not any(
                    bbox_contains_position(bbox, position)
                    for bbox in pre_action_input_aligned_bboxes
                )
                for position in remote_positions
            )
        )
        if remote_positions_are_outside_input_aligned_bboxes:
            remote_rows = {row for row, _column in remote_positions}
            remote_columns = {column for _row, column in remote_positions}
            remote_bbox = BoundingBox(
                top=min(row for row, _column in remote_positions),
                left=min(column for _row, column in remote_positions),
                bottom=max(row for row, _column in remote_positions),
                right=max(column for _row, column in remote_positions),
            )
            input_aligned_inverse_value_measurement = {
                "input_aligned_inverse_value_departure_present": True,
                "input_aligned_inverse_value_departure_position_count": len(
                    departure_positions
                ),
                "inverse_value_remote_support_present": True,
                "inverse_value_remote_support_position_count": len(
                    remote_positions
                ),
                "inverse_value_remote_support_bbox": remote_bbox.model_dump(
                    mode="python"
                ),
                "inverse_value_remote_support_collinear": bool(
                    len(remote_rows) == 1 or len(remote_columns) == 1
                ),
                "input_aligned_inverse_value_transfer_digest": stable_digest(
                    (
                        departure_pair,
                        departure_positions,
                        remote_positions,
                    )
                )[:16],
            }
    net_by_value = {
        first_value: second_to_first - first_to_second,
        second_value: first_to_second - second_to_first,
    }
    net_activated_values = tuple(
        value for value, net in sorted(net_by_value.items()) if net > 0
    )
    if len(net_activated_values) != 1:
        return {}
    activated_value = net_activated_values[0]
    activated_positions = tuple(
        sorted(
            (
                (after.component.bbox.top + after.component.bbox.bottom) // 2,
                (after.component.bbox.left + after.component.bbox.right) // 2,
            )
            for _before, after in transitions
            if after.component.value == activated_value
        )
    )
    if len(activated_positions) < 3:
        return {}
    rows = {row for row, _column in activated_positions}
    columns = {column for _row, column in activated_positions}
    orientation = (
        "horizontal" if len(rows) == 1 else "vertical" if len(columns) == 1 else ""
    )
    bbox = BoundingBox(
        top=min(row for row, _column in activated_positions),
        left=min(column for _row, column in activated_positions),
        bottom=max(row for row, _column in activated_positions),
        right=max(column for _row, column in activated_positions),
    )
    measured = {
        "stationary_inverse_value_transition_pair_present": True,
        "stationary_inverse_value_transition_pair_digest": stable_digest(
            (
                tuple(sorted(directed_counts.items())),
                orientation,
                activated_positions,
            )
        )[:16],
        "stationary_inverse_value_transition_first_direction_count": (
            first_to_second
        ),
        "stationary_inverse_value_transition_second_direction_count": (
            second_to_first
        ),
        "unique_net_activated_support_present": True,
        "unique_net_activated_collinear_support_present": bool(orientation),
        "unique_net_activated_support_position_count": len(activated_positions),
        "unique_net_activated_support_bbox": bbox.model_dump(mode="python"),
        "net_deactivated_position_count": min(first_to_second, second_to_first),
        **input_aligned_inverse_value_measurement,
    }
    if not orientation:
        return measured
    measured.update({
        "unique_net_activated_collinear_support_present": True,
        "net_activated_collinear_support_orientation": orientation,
        "net_activated_collinear_support_position_count": len(
            activated_positions
        ),
        "unique_net_activated_collinear_support_bbox": bbox.model_dump(
            mode="python"
        ),
    })
    return measured


def measure_intermediate_scene_change(
    value: IntermediateSceneChangeInput,
) -> IntermediateSceneChangeMeasurements:
    """Partition raw changes into grounded exclusions and useful scene change.

    This intentionally performs no component extraction, matching or semantic
    inference.  Resource displays and cyclic regions can be excluded only when
    their named entity regions were already grounded by earlier reasoning.
    """

    difference = difference_frames(
        FrameDifferenceInput(
            before=value.before,
            after=value.after,
            action_ref=value.action_ref,
        )
    )
    ignored_positions = {
        position
        for region in value.ignored_regions
        for position in region.pixels
    }
    permitted_positions = {
        position
        for region in value.permitted_regions
        for position in region.pixels
    }
    ignored_changed_count = sum(
        (change.row, change.col) in ignored_positions
        for change in difference.changes
    )
    meaningful_changed_count = difference.changed_count - ignored_changed_count
    permitted_changed_count = sum(
        (change.row, change.col) not in ignored_positions
        and (change.row, change.col) in permitted_positions
        for change in difference.changes
    )
    outside_permitted_changed_count = (
        meaningful_changed_count - permitted_changed_count
    )
    periodic_cycle_workflow_statuses = tuple(
        str(fact.get("workflow_status") or "")
        for fact in value.periodic_cycle_workflow_facts
    )
    all_periodic_cycles_understood = all(
        status == PeriodicCycleWorkflowStatus.CYCLE_UNDERSTOOD.value
        for status in periodic_cycle_workflow_statuses
    )
    alternatives = (
        INTERMEDIATE_CYCLE_INCOMPLETE_CANDIDATE,
        INTERMEDIATE_MEANINGFUL_CHANGE_CANDIDATE,
        INTERMEDIATE_IGNORED_ONLY_CANDIDATE,
        INTERMEDIATE_NO_CHANGE_CANDIDATE,
    )
    shared = {
        "raw_changed_count": difference.changed_count,
        "ignored_changed_count": ignored_changed_count,
        "meaningful_changed_count": meaningful_changed_count,
        "permitted_region_count": len(value.permitted_regions),
        "permitted_changed_count": permitted_changed_count,
        "outside_permitted_changed_count": outside_permitted_changed_count,
        "ignored_region_count": len(value.ignored_regions),
        "ignored_region_refs": tuple(
            region.entity_ref for region in value.ignored_regions
        ),
        "periodic_cycle_workflow_statuses": periodic_cycle_workflow_statuses,
        "all_periodic_cycles_understood": all_periodic_cycles_understood,
    }
    return IntermediateSceneChangeMeasurements(
        alternative_refs=alternatives,
        alternative_facts=FrozenMap(
            {
                alternative: FrozenMap(
                    {
                        **shared,
                        "guard_kind": alternative.rsplit(".", 1)[-1],
                    }
                )
                for alternative in alternatives
            }
        ),
        raw_changed_count=difference.changed_count,
        ignored_changed_count=ignored_changed_count,
        meaningful_changed_count=meaningful_changed_count,
        permitted_region_count=len(value.permitted_regions),
        permitted_changed_count=permitted_changed_count,
        outside_permitted_changed_count=outside_permitted_changed_count,
        ignored_region_refs=tuple(
            region.entity_ref for region in value.ignored_regions
        ),
        action_ref=value.action_ref,
        periodic_cycle_workflow_statuses=periodic_cycle_workflow_statuses,
        all_periodic_cycles_understood=all_periodic_cycles_understood,
    )


def measure_scene_transition(
    value: SceneTransitionMeasurementInput,
) -> SceneTransitionMeasurements:
    """Measure one packet with a single pixel scan and no scene decomposition."""

    pixel_delta = _measure_pixel_packet(value)
    packet_facts = _pixel_delta_facts(pixel_delta)
    if (value.before.height, value.before.width) == (value.after.height, value.after.width):
        packet_facts.update(measure_recipe_edit_context(value.before, value.action_ref, value.action_data, 0))
        packet_facts.update(measure_recipe_transition(
            value.before, value.after, value.action_ref, value.action_data, 0
        ))
    packet_facts.update(
        {
            "boundary_indicator_quantum_delta_measurement": (
                value.boundary_indicator_quantum_delta_measurement
            ),
            "before_score": value.before_score,
        }
    )
    packet_facts.update(
        {
            key: item
            for key, item in sorted(
                {
                    "candidate_ref": value.candidate_ref,
                    "context_epoch": value.context_epoch,
                    "before_configuration_digest": value.before_configuration_digest,
                    "after_configuration_digest": value.after_configuration_digest,
                    "before_available_action_set_digest": (
                        value.before_available_action_set_digest
                    ),
                }.items()
            )
            if item is not None
        }
    )
    compact_packet_facts = {
        key: item
        for key, item in sorted(packet_facts.items())
        if key
        not in {
            "consecutive_changed_counts",
            "intermediate_changed_counts_from_before",
            "intermediate_changed_counts_to_after",
        }
    }

    dimensions_equal = (value.before.height, value.before.width) == (
        value.after.height,
        value.after.width,
    )
    if not dimensions_equal:
        return _measurements(
            action_ref=value.action_ref,
            measurement_input=value,
            pixel_delta=pixel_delta,
            potential_nonzero_translation_count=0,
            shared_facts={
                **compact_packet_facts,
                "frame_dimensions_equal": False,
                "exact_no_pixel_change": False,
                "changed_union_is_collinear": False,
                "changed_union_touches_boundary": False,
                "changed_transition_pair_count": 0,
                "potential_nonzero_translation_count": 0,
                "boundary_collinear_single_transition_without_motion_witness": False,
            },
            ordered_packet_facts=packet_facts,
        )

    boundary_collinear_single_transition_without_motion_witness = bool(
        pixel_delta.final_changed_count > 0
        and pixel_delta.changed_union_is_collinear
        and pixel_delta.changed_union_touches_boundary
        and pixel_delta.changed_transition_pair_count == 1
        and pixel_delta.potential_nonzero_translation_count == 0
    )
    changed_bbox = pixel_delta.changed_bbox
    changed_bbox_edge_gap = (
        min(
            changed_bbox.top,
            changed_bbox.left,
            value.after.height - 1 - changed_bbox.bottom,
            value.after.width - 1 - changed_bbox.right,
        )
        if changed_bbox is not None
        else -1
    )
    boundary_adjacent_collinear_single_transition_without_motion_witness = bool(
        changed_bbox is not None
        and pixel_delta.final_changed_count > 0
        and pixel_delta.changed_union_is_collinear
        and changed_bbox_edge_gap >= 0
        and changed_bbox_edge_gap < value.boundary_search_band_px
        and pixel_delta.changed_transition_pair_count == 1
        and pixel_delta.potential_nonzero_translation_count == 0
    )
    boundary_adjacent_monotonic_strip_change_without_motion_witness = (
        _boundary_adjacent_monotonic_strip_change(value, pixel_delta)
    )
    boundary_adjacent_monotonic_strip_decrease_without_motion_witness = (
        _boundary_adjacent_monotonic_strip_decrease(value, pixel_delta)
    )
    point_x = value.action_data.get("x")
    point_y = value.action_data.get("y")
    point_action_locus_inside_changed_bbox = bool(
        value.action_ref == "ACTION6"
        and isinstance(point_x, int)
        and isinstance(point_y, int)
        and changed_bbox is not None
        and changed_bbox.top <= point_y <= changed_bbox.bottom
        and changed_bbox.left <= point_x <= changed_bbox.right
    )
    ordered_point_trajectory_facts = _ordered_point_component_trajectory_facts(value)
    if value.action_ref == "ACTION6" and value.intermediate_frames:
        from .ordered_traits import measure_focused_word_translations

        ordered_point_trajectory_facts.update(measure_focused_word_translations(
            (value.before, *value.intermediate_frames, value.after)
        ))
    ordered_transient_path_facts = _ordered_transient_component_path_facts(value)
    ordered_animation_reconfiguration_facts = (
        _ordered_animated_reconfiguration_facts(value)
    )
    shared_facts = {
        **compact_packet_facts,
        **ordered_point_trajectory_facts,
        "frame_dimensions_equal": True,
        "exact_no_pixel_change": pixel_delta.final_changed_count == 0,
        "changed_count": pixel_delta.final_changed_count,
        "changed_union_is_collinear": pixel_delta.changed_union_is_collinear,
        "changed_union_touches_boundary": pixel_delta.changed_union_touches_boundary,
        "changed_transition_pair_count": pixel_delta.changed_transition_pair_count,
        "potential_nonzero_translation_count": (
            pixel_delta.potential_nonzero_translation_count
        ),
        "boundary_collinear_single_transition_without_motion_witness": (
            boundary_collinear_single_transition_without_motion_witness
        ),
        "changed_bbox_edge_gap": changed_bbox_edge_gap,
        "boundary_adjacent_collinear_single_transition_without_motion_witness": (
            boundary_adjacent_collinear_single_transition_without_motion_witness
        ),
        "boundary_adjacent_monotonic_strip_change_without_motion_witness": (
            boundary_adjacent_monotonic_strip_change_without_motion_witness
        ),
        **(
            {
                "boundary_adjacent_monotonic_strip_decrease_without_motion_witness": True
            }
            if boundary_adjacent_monotonic_strip_decrease_without_motion_witness
            else {}
        ),
        "point_action_locus_inside_changed_bbox": (
            point_action_locus_inside_changed_bbox
        ),
    }
    return _measurements(
        action_ref=value.action_ref,
        measurement_input=value,
        pixel_delta=pixel_delta,
        potential_nonzero_translation_count=(
            pixel_delta.potential_nonzero_translation_count
        ),
        shared_facts=shared_facts,
        ordered_packet_facts={
            **packet_facts,
            **ordered_transient_path_facts,
            **ordered_animation_reconfiguration_facts,
        },
    )


def analyze_repeated_boundary_decrease(
    value: RepeatedBoundaryDecreaseAnalysisInput,
) -> RepeatedBoundaryDecreaseAnalysis:
    """Measure repeated boundary diminution without assigning a resource role."""

    alternatives = (
        BOUNDARY_FAMILY_PROGRESS_CANDIDATE,
        REPEATED_BOUNDARY_DECREASE_CANDIDATE,
        SINGLE_BOUNDARY_DECREASE_CANDIDATE,
        UNRESOLVED_BOUNDARY_CHANGE_CANDIDATE,
    )
    current = _boundary_decrease_evidence(value)
    related_prior = tuple(
        item
        for item in value.prior_evidence
        if current is not None
        and item.indicator_entity_ref == current.indicator_entity_ref
        and item.evidence_scope_ref == current.evidence_scope_ref
    )
    evidence = (*related_prior, current) if current is not None else ()
    repeated_count = len(evidence)
    observed_decrements = tuple(item.changed_pixel_count for item in evidence)
    decrement_unit_established = (
        repeated_count >= 2 and len(set(observed_decrements)) == 1
    )
    indicator_ref = current.indicator_entity_ref if current is not None else ""
    common = {
        "indicator_entity_ref": indicator_ref,
        "action_ref": value.action_ref,
        "transition_ref": value.transition_ref,
        "evidence_scope_ref": value.evidence_scope_ref,
        "repeated_observation_count": repeated_count,
        "observed_decrements": observed_decrements,
        "decrement_unit_established": decrement_unit_established,
        "decrement_unit": (
            observed_decrements[0] if decrement_unit_established else 0
        ),
        "exact_boundary_decrease_observed": current is not None,
        "changed_pixel_count": (
            current.changed_pixel_count if current is not None else 0
        ),
        "before_pixel_count": (
            current.before_pixel_count if current is not None else 0
        ),
        "after_pixel_count": current.after_pixel_count if current is not None else 0,
        "primitive_count_after": current.primitive_count_after if current is not None else None,
        "render_extent_bbox": (current.render_extent_bbox.model_dump(mode="json")
                               if current is not None and current.render_extent_bbox is not None else None),
        "render_pixel_thickness": current.render_pixel_thickness if current is not None else None,
        "regular_family_member_count": (
            current.regular_family_member_count if current is not None else 0
        ),
        "changed_family_member_count": (
            current.changed_family_member_count if current is not None else 0
        ),
        "remaining_family_member_count": (
            current.remaining_family_member_count if current is not None else 0
        ),
        "homologous_multi_member_state_transfer": (
            current.homologous_multi_member_state_transfer
            if current is not None
            else False
        ),
    }
    facts = FrozenMap(
        {
            BOUNDARY_FAMILY_PROGRESS_CANDIDATE: FrozenMap(
                {**common, "boundary_decrease_candidate_kind": "progress_group"}
            ),
            REPEATED_BOUNDARY_DECREASE_CANDIDATE: FrozenMap(
                {**common, "boundary_decrease_candidate_kind": "repeated"}
            ),
            SINGLE_BOUNDARY_DECREASE_CANDIDATE: FrozenMap(
                {**common, "boundary_decrease_candidate_kind": "single"}
            ),
            UNRESOLVED_BOUNDARY_CHANGE_CANDIDATE: FrozenMap(
                {**common, "boundary_decrease_candidate_kind": "unresolved"}
            ),
        }
    )
    return RepeatedBoundaryDecreaseAnalysis(
        alternative_refs=alternatives,
        alternative_facts=facts,
        evidence=evidence,
        evidence_refs=tuple(
            f"evidence.boundary_decrease:{item.transition_ref}"
            for item in evidence
        ),
    )


def _boundary_decrease_evidence(
    value: RepeatedBoundaryDecreaseAnalysisInput,
) -> BoundaryDecreaseEvidence | None:
    if (value.before.height, value.before.width) != (
        value.after.height,
        value.after.width,
    ):
        return None
    candidates: list[tuple[TrackedComponent, tuple[tuple[int, int], ...]]] = []
    for item in value.before_tracking.entities:
        component = item.component
        if not _long_axis_parallel_to_boundary_band(
            component.bbox, value.before, value.boundary_search_band_px
        ):
            continue
        changed = tuple(
            (row, col)
            for row, col in component.pixels
            if value.before.rows[row][col] == component.value
            and value.after.rows[row][col] != component.value
        )
        if not changed:
            continue
        if component.area != component.bbox.height * component.bbox.width:
            continue
        changed_set = frozenset(changed)
        if component.bbox.width >= component.bbox.height:
            changed_axes = tuple(sorted({col for _row, col in changed_set}))
            terminal_axes = tuple(
                range(component.bbox.left, component.bbox.left + len(changed_axes))
            )
            opposite_terminal_axes = tuple(
                range(component.bbox.right - len(changed_axes) + 1, component.bbox.right + 1)
            )
            expected_terminal_slice = frozenset(
                (row, col)
                for row in range(component.bbox.top, component.bbox.bottom + 1)
                for col in changed_axes
            )
        else:
            changed_axes = tuple(sorted({row for row, _col in changed_set}))
            terminal_axes = tuple(
                range(component.bbox.top, component.bbox.top + len(changed_axes))
            )
            opposite_terminal_axes = tuple(
                range(component.bbox.bottom - len(changed_axes) + 1, component.bbox.bottom + 1)
            )
            expected_terminal_slice = frozenset(
                (row, col)
                for row in changed_axes
                for col in range(component.bbox.left, component.bbox.right + 1)
            )
        if (
            changed_axes not in {terminal_axes, opposite_terminal_axes}
            or changed_set != expected_terminal_slice
        ):
            continue
        transitions = {
            (value.before.rows[row][col], value.after.rows[row][col])
            for row, col in changed
        }
        before_count = sum(
            value.before.rows[row][col] == component.value
            for row, col in component.pixels
        )
        after_count = sum(
            value.after.rows[row][col] == component.value
            for row, col in component.pixels
        )
        if (
            len(transitions) == 1
            and after_count > 0
            and before_count - after_count == len(changed)
        ):
            candidates.append((item, changed))
    regular_family = _regular_boundary_family_decrease_evidence(value)
    if regular_family is not None:
        return regular_family
    if len(candidates) != 1:
        return None
    tracked, changed_positions = candidates[0]
    before_count = sum(
        value.before.rows[row][col] == tracked.component.value
        for row, col in tracked.component.pixels
    )
    after_count = sum(
        value.after.rows[row][col] == tracked.component.value
        for row, col in tracked.component.pixels
    )
    if before_count - after_count != len(changed_positions):
        return None
    return BoundaryDecreaseEvidence(
        indicator_entity_ref=tracked.entity_id,
        transition_ref=value.transition_ref,
        action_ref=value.action_ref,
        evidence_scope_ref=value.evidence_scope_ref,
        changed_pixel_count=len(changed_positions),
        before_pixel_count=before_count,
        after_pixel_count=after_count,
        primitive_count_after=value.primitive_count_after,
        render_extent_bbox=(
            tracked.component.bbox
            if (tracked.component.bbox.width >= tracked.component.bbox.height
                and tracked.component.bbox.left == 0
                and tracked.component.bbox.right == value.before.width - 1)
            or (tracked.component.bbox.height > tracked.component.bbox.width
                and tracked.component.bbox.top == 0
                and tracked.component.bbox.bottom == value.before.height - 1)
            else None
        ),
        render_pixel_thickness=min(tracked.component.bbox.height, tracked.component.bbox.width),
    )


def _regular_boundary_family_decrease_evidence(
    value: RepeatedBoundaryDecreaseAnalysisInput,
) -> BoundaryDecreaseEvidence | None:
    """Measure one-state diminution in a regular boundary-adjacent family.

    Members are grouped by equal extent and collinearity without using palette
    identity. Their internal masks may differ, but every non-state pixel must
    remain exact. A candidate requires at least three equally spaced members,
    one source value losing whole members to one destination value, and every
    other member remaining exact. The stable family reference hashes geometry
    and member support only, so repeated state transfers retain identity.
    """

    components = tuple(
        component
        for component in detect_components(
            ComponentExtractionInput(frame=value.before)
        ).components
        if 0 < component.area <= 64
        and component.area * 2 >= component.bbox.height * component.bbox.width
    )
    bands: dict[tuple[object, ...], list[ComponentDescription]] = {}
    for component in components:
        morphology = (
            component.bbox.height,
            component.bbox.width,
        )
        bands.setdefault(
            ("horizontal", component.bbox.top, component.bbox.bottom, morphology),
            [],
        ).append(component)
        bands.setdefault(
            ("vertical", component.bbox.left, component.bbox.right, morphology),
            [],
        ).append(component)

    evidence: list[BoundaryDecreaseEvidence] = []
    for band_key, raw_members in sorted(bands.items(), key=lambda item: str(item[0])):
        orientation = str(band_key[0])
        members = tuple(
            sorted(
                raw_members,
                key=lambda item: (
                    item.bbox.left if orientation == "horizontal" else item.bbox.top,
                    item.component_id,
                ),
            )
        )
        # This detector feeds the bounded non-revisiting route contract.  A
        # larger regular field is a distinct structural alternative (board,
        # texture, inventory, etc.) and must not be promoted to that horizon.
        if len(members) < 3 or len(members) > 12:
            continue
        coordinates = tuple(
            item.bbox.left if orientation == "horizontal" else item.bbox.top
            for item in members
        )
        gaps = tuple(second - first for first, second in zip(coordinates, coordinates[1:]))
        if not gaps or len(set(gaps)) != 1 or gaps[0] <= 0:
            continue
        family_box = BoundingBox(
            top=min(item.bbox.top for item in members),
            left=min(item.bbox.left for item in members),
            bottom=max(item.bbox.bottom for item in members),
            right=max(item.bbox.right for item in members),
        )
        edge_gap = min(
            family_box.top,
            family_box.left,
            value.before.height - 1 - family_box.bottom,
            value.before.width - 1 - family_box.right,
        )
        member_extent = max(members[0].bbox.height, members[0].bbox.width)
        if edge_gap > max(4, member_extent * 2):
            continue
        family_pixels = tuple(
            sorted(position for item in members for position in item.pixels)
        )
        for source_value in sorted({item.value for item in members}):
            source_members = tuple(
                item for item in members if item.value == source_value
            )
            if len(source_members) < 2:
                continue
            changed_source_members: list[tuple[ComponentDescription, int]] = []
            exact = True
            for member in members:
                before_values = {
                    value.before.rows[row][col] for row, col in member.pixels
                }
                after_values = {
                    value.after.rows[row][col] for row, col in member.pixels
                }
                if len(before_values) != 1 or len(after_values) != 1:
                    exact = False
                    break
                before_value = next(iter(before_values))
                after_value = next(iter(after_values))
                member_pixels = frozenset(member.pixels)
                if any(
                    value.before.rows[row][col] != value.after.rows[row][col]
                    for row in range(member.bbox.top, member.bbox.bottom + 1)
                    for col in range(member.bbox.left, member.bbox.right + 1)
                    if (row, col) not in member_pixels
                ):
                    exact = False
                    break
                if before_value == after_value:
                    continue
                if before_value != source_value:
                    exact = False
                    break
                changed_source_members.append((member, after_value))
            if not exact or not changed_source_members:
                continue
            destination_values = {item[1] for item in changed_source_members}
            changed_pixel_count = sum(item[0].area for item in changed_source_members)
            before_pixel_count = sum(item.area for item in source_members)
            after_pixel_count = before_pixel_count - changed_pixel_count
            if len(destination_values) != 1:
                continue
            family_ref = "entity.boundary-family:" + stable_digest(
                {
                    "orientation": orientation,
                    "member_bboxes": tuple(
                        (
                            item.bbox.top,
                            item.bbox.left,
                            item.bbox.bottom,
                            item.bbox.right,
                        )
                        for item in members
                    ),
                    "family_pixels": family_pixels,
                }
            )[:16]
            evidence.append(
                BoundaryDecreaseEvidence(
                    indicator_entity_ref=family_ref,
                    transition_ref=value.transition_ref,
                    action_ref=value.action_ref,
                    evidence_scope_ref=value.evidence_scope_ref,
                    changed_pixel_count=changed_pixel_count,
                    before_pixel_count=before_pixel_count,
                    after_pixel_count=after_pixel_count,
                    regular_family_member_count=len(members),
                    changed_family_member_count=len(changed_source_members),
                    remaining_family_member_count=(
                        len(source_members) - len(changed_source_members)
                    ),
                    homologous_multi_member_state_transfer=(
                        len(changed_source_members) >= 2
                    ),
                )
            )
    return evidence[0] if len(evidence) == 1 else None


def _long_axis_parallel_to_boundary_band(
    bbox: BoundingBox, frame: FrameGrid, band_px: int
) -> bool:
    """Measure long-axis alignment with an edge inside a bounded inset band.

    The closest face, not the whole thickness, must intersect the search band.
    Padding is independent of thickness and length. No display role is assigned.
    """

    if bbox.width > bbox.height:
        gap = min(bbox.top, frame.height - 1 - bbox.bottom)
    elif bbox.height > bbox.width:
        gap = min(bbox.left, frame.width - 1 - bbox.right)
    else:
        return False
    return 0 <= gap < band_px


def _boundary_adjacent_monotonic_strip_decrease(
    value: SceneTransitionMeasurementInput,
    pixel_delta: PixelTransitionDelta,
) -> bool:
    """Measure an inset narrow component decrement without assigning HUD meaning."""

    if (
        pixel_delta.changed_bbox is None
        or pixel_delta.changed_transition_pair_count != 1
        or pixel_delta.potential_nonzero_translation_count != 0
    ):
        return False
    changed_positions = {
        (row, col)
        for row in range(value.before.height)
        for col in range(value.before.width)
        if value.before.rows[row][col] != value.after.rows[row][col]
    }
    if not changed_positions:
        return False
    before_values = {value.before.rows[row][col] for row, col in changed_positions}
    if len(before_values) != 1:
        return False
    source_value = next(iter(before_values))
    frontier = [next(iter(changed_positions))]
    component: set[tuple[int, int]] = set()
    while frontier:
        row, col = frontier.pop()
        if (row, col) in component or value.before.rows[row][col] != source_value:
            continue
        component.add((row, col))
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < value.before.height
                and 0 <= next_col < value.before.width
                and (next_row, next_col) not in component
            ):
                frontier.append((next_row, next_col))
    if not changed_positions.issubset(component):
        return False
    if any(value.after.rows[row][col] == source_value for row, col in changed_positions):
        return False
    top = min(row for row, _ in component)
    bottom = max(row for row, _ in component)
    left = min(col for _, col in component)
    right = max(col for _, col in component)
    return bool(
        _long_axis_parallel_to_boundary_band(
            BoundingBox(top=top, left=left, bottom=bottom, right=right),
            value.before,
            value.boundary_search_band_px,
        )
        and any(value.after.rows[row][col] == source_value for row, col in component)
    )


def _boundary_adjacent_monotonic_strip_change(
    value: SceneTransitionMeasurementInput,
    pixel_delta: PixelTransitionDelta,
) -> bool:
    """Measure a partial inset narrow-strip change without assigning HUD meaning."""

    if (
        pixel_delta.changed_bbox is None
        or pixel_delta.changed_transition_pair_count != 1
        or pixel_delta.potential_nonzero_translation_count != 0
    ):
        return False
    changed_positions = {
        (row, col)
        for row in range(value.before.height)
        for col in range(value.before.width)
        if value.before.rows[row][col] != value.after.rows[row][col]
    }
    if not changed_positions:
        return False
    before_values = {value.before.rows[row][col] for row, col in changed_positions}
    after_values = {value.after.rows[row][col] for row, col in changed_positions}
    if len(before_values) != 1 or len(after_values) != 1:
        return False
    before_value = next(iter(before_values))
    after_value = next(iter(after_values))
    if before_value == after_value:
        return False

    def qualifies(
        frame: FrameGrid,
        other: FrameGrid,
        component_value: int,
    ) -> bool:
        frontier = [next(iter(changed_positions))]
        component: set[tuple[int, int]] = set()
        while frontier:
            row, col = frontier.pop()
            if (row, col) in component or frame.rows[row][col] != component_value:
                continue
            component.add((row, col))
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= next_row < frame.height
                    and 0 <= next_col < frame.width
                    and (next_row, next_col) not in component
                ):
                    frontier.append((next_row, next_col))
        if not changed_positions.issubset(component):
            return False
        unchanged_support = component.difference(changed_positions)
        if not unchanged_support or not any(
            other.rows[row][col] == component_value
            for row, col in unchanged_support
        ):
            return False
        top = min(row for row, _ in component)
        bottom = max(row for row, _ in component)
        left = min(col for _, col in component)
        right = max(col for _, col in component)
        return _long_axis_parallel_to_boundary_band(
            BoundingBox(top=top, left=left, bottom=bottom, right=right),
            frame,
            value.boundary_search_band_px,
        )

    return qualifies(value.before, value.after, before_value) or qualifies(
        value.after,
        value.before,
        after_value,
    )


def _center_twice(component: ComponentDescription) -> tuple[int, int]:
    return (
        component.bbox.top + component.bbox.bottom,
        component.bbox.left + component.bbox.right,
    )


def _cell_reflection_axis_measurements(scene: object) -> tuple[dict[str, object], ...]:
    """Enumerate bounded palette-independent reflection supports and carriers.

    Hole sprites preserve the exact support of a visually enclosed region even
    when its interior uses several palette values.  Pairing their cells before
    reading values lets perception expose a dark/light or otherwise recolored
    reflection without assigning either palette value a semantic role.
    """

    return _measure_cell_reflection_axis_measurements(
        hole_sprites=tuple(scene.hole_sprites),
        components=tuple(scene.blocks.components),
    )


def _measure_cell_reflection_axis_measurements(
    *,
    hole_sprites: tuple[object, ...],
    components: tuple[ComponentDescription, ...],
    require_carrier: bool = True,
) -> tuple[dict[str, object], ...]:
    """Measure the same reflection relations from one standalone observation."""

    holes = tuple(item for item in hole_sprites if len(item.pixels) > 1)[:64]
    components = tuple(components)[:128]
    measurements: list[dict[str, object]] = []
    structural_count = 0
    for first, second in combinations(holes, 2):
        structural_count += 1
        if structural_count > 4096:
            return ()
        if (
            first.bbox.height,
            first.bbox.width,
            len(first.pixels),
        ) != (
            second.bbox.height,
            second.bbox.width,
            len(second.pixels),
        ):
            continue
        first_relative = tuple(
            sorted(
                (row - first.bbox.top, col - first.bbox.left)
                for row, col in first.pixels
            )
        )
        second_relative = tuple(
            sorted(
                (row - second.bbox.top, col - second.bbox.left)
                for row, col in second.pixels
            )
        )
        candidates: list[tuple[str, int]] = []
        if (
            second_relative
            == tuple(
                sorted(
                    (row, first.bbox.width - 1 - col)
                    for row, col in first_relative
                )
            )
            and first.bbox.top + first.bbox.bottom
            == second.bbox.top + second.bbox.bottom
        ):
            candidates.append(
                (
                    "vertical",
                    first.bbox.left
                    + first.bbox.right
                    + second.bbox.left
                    + second.bbox.right,
                )
            )
        if (
            second_relative
            == tuple(
                sorted(
                    (first.bbox.height - 1 - row, col)
                    for row, col in first_relative
                )
            )
            and first.bbox.left + first.bbox.right
            == second.bbox.left + second.bbox.right
        ):
            candidates.append(
                (
                    "horizontal",
                    first.bbox.top
                    + first.bbox.bottom
                    + second.bbox.top
                    + second.bbox.bottom,
                )
            )
        for orientation, axis_coordinate_four in candidates:
            carrier_refs: list[str] = []
            spanning_carrier_refs: list[str] = []
            pair_min_row = min(first.bbox.top, second.bbox.top)
            pair_max_row = max(first.bbox.bottom, second.bbox.bottom)
            pair_min_col = min(first.bbox.left, second.bbox.left)
            pair_max_col = max(first.bbox.right, second.bbox.right)
            for component in components:
                pixels = frozenset(component.pixels)
                if orientation == "vertical":
                    if 2 * _center_twice(component)[1] != axis_coordinate_four:
                        continue
                    reflected = {
                        (row, (axis_coordinate_four - 2 * col) // 2)
                        for row, col in component.pixels
                        if (axis_coordinate_four - 2 * col) % 2 == 0
                    }
                    spans_pair = (
                        component.bbox.top <= pair_min_row
                        and component.bbox.bottom >= pair_max_row
                    )
                else:
                    if 2 * _center_twice(component)[0] != axis_coordinate_four:
                        continue
                    reflected = {
                        ((axis_coordinate_four - 2 * row) // 2, col)
                        for row, col in component.pixels
                        if (axis_coordinate_four - 2 * row) % 2 == 0
                    }
                    spans_pair = (
                        component.bbox.left <= pair_min_col
                        and component.bbox.right >= pair_max_col
                    )
                if len(reflected) != len(pixels) or reflected != pixels:
                    continue
                carrier_refs.append(component.component_id)
                if spans_pair:
                    spanning_carrier_refs.append(component.component_id)
            if require_carrier and not carrier_refs:
                continue
            digest = stable_digest(
                (
                    orientation,
                    axis_coordinate_four,
                    first_relative,
                    first.hole_sprite_id,
                    second.hole_sprite_id,
                )
            )[:16]
            if orientation == "vertical":
                first_is_negative = (
                    first.bbox.left + first.bbox.right
                    < second.bbox.left + second.bbox.right
                )
            else:
                first_is_negative = (
                    first.bbox.top + first.bbox.bottom
                    < second.bbox.top + second.bbox.bottom
                )
            negative = first if first_is_negative else second
            positive = second if first_is_negative else first
            negative_relative = (
                first_relative if first_is_negative else second_relative
            )
            positive_relative = (
                second_relative if first_is_negative else first_relative
            )
            measurements.append(
                {
                    "cell_reflection_axis_digest": digest,
                    "cell_reflection_orientation": orientation,
                    "cell_reflection_axis_coordinate_four": axis_coordinate_four,
                    "cell_reflection_first_support_ref": first.hole_sprite_id,
                    "cell_reflection_second_support_ref": second.hole_sprite_id,
                    "cell_reflection_support_area": len(first.pixels),
                    "cell_reflection_negative_support_ref": (
                        negative.hole_sprite_id
                    ),
                    "cell_reflection_positive_support_ref": (
                        positive.hole_sprite_id
                    ),
                    "cell_reflection_negative_support_bbox": (
                        negative.bbox.top,
                        negative.bbox.left,
                        negative.bbox.bottom,
                        negative.bbox.right,
                    ),
                    "cell_reflection_positive_support_bbox": (
                        positive.bbox.top,
                        positive.bbox.left,
                        positive.bbox.bottom,
                        positive.bbox.right,
                    ),
                    "cell_reflection_negative_support_shape_digest": stable_digest(
                        negative_relative
                    )[:16],
                    "cell_reflection_positive_support_shape_digest": stable_digest(
                        positive_relative
                    )[:16],
                    "cell_reflection_supports_exact_mask_relation": True,
                    "cell_reflection_palette_equality_required": False,
                    "cell_reflection_carrier_component_refs": tuple(
                        sorted(carrier_refs)
                    ),
                    "cell_reflection_spanning_carrier_component_refs": tuple(
                        sorted(spanning_carrier_refs)
                    ),
                    "cell_reflection_zone_bbox": (
                        pair_min_row,
                        pair_min_col,
                        pair_max_row,
                        pair_max_col,
                    ),
                }
            )
            if len(measurements) >= 64:
                return tuple(measurements)
    return tuple(measurements)


def _effect_carried_reflection_axis_measurements(
    *,
    before_scene: object,
    after_scene: object,
    before_axes: tuple[dict[str, object], ...],
    after_axes: tuple[dict[str, object], ...],
    before_by_entity: dict[str, TrackedComponent],
    translations: tuple[TranslationTransition, ...],
) -> tuple[dict[str, object], ...]:
    """Join a translated identity to an equally translated cell-reflection axis."""

    witnesses: list[dict[str, object]] = []
    for translation in translations:
        before_entity = before_by_entity.get(translation.entity_ref)
        if before_entity is None:
            continue
        before_component_ref = before_entity.component.component_id
        for before_axis in before_axes:
            if before_component_ref not in before_axis[
                "cell_reflection_spanning_carrier_component_refs"
            ]:
                continue
            for after_axis in after_axes:
                if translation.component_ref not in after_axis[
                    "cell_reflection_spanning_carrier_component_refs"
                ]:
                    continue
                orientation = str(before_axis["cell_reflection_orientation"])
                if orientation != after_axis["cell_reflection_orientation"]:
                    continue
                expected_axis_delta_four = 4 * (
                    translation.delta_col
                    if orientation == "vertical"
                    else translation.delta_row
                )
                measured_axis_delta_four = int(
                    after_axis["cell_reflection_axis_coordinate_four"]
                ) - int(before_axis["cell_reflection_axis_coordinate_four"])
                if measured_axis_delta_four != expected_axis_delta_four:
                    continue
                negative_before_bbox = tuple(
                    before_axis["cell_reflection_negative_support_bbox"]
                )
                negative_after_bbox = tuple(
                    after_axis["cell_reflection_negative_support_bbox"]
                )
                positive_before_bbox = tuple(
                    before_axis["cell_reflection_positive_support_bbox"]
                )
                positive_after_bbox = tuple(
                    after_axis["cell_reflection_positive_support_bbox"]
                )

                def bbox_delta(
                    before_bbox: tuple[int, ...], after_bbox: tuple[int, ...]
                ) -> tuple[int, int] | None:
                    row_twice = (after_bbox[0] + after_bbox[2]) - (
                        before_bbox[0] + before_bbox[2]
                    )
                    col_twice = (after_bbox[1] + after_bbox[3]) - (
                        before_bbox[1] + before_bbox[3]
                    )
                    if row_twice % 2 or col_twice % 2:
                        return None
                    return row_twice // 2, col_twice // 2

                negative_delta = bbox_delta(
                    negative_before_bbox, negative_after_bbox
                )
                positive_delta = bbox_delta(
                    positive_before_bbox, positive_after_bbox
                )
                if negative_delta is None or positive_delta is None:
                    continue
                expected_reflected_delta = (
                    2 * translation.delta_row,
                    2 * translation.delta_col,
                )
                if (
                    negative_delta == expected_reflected_delta
                    and positive_delta == (0, 0)
                ):
                    moving_side = "negative"
                    moving_delta = negative_delta
                    moving_after_bbox = negative_after_bbox
                    moving_shape_digest = str(
                        after_axis[
                            "cell_reflection_negative_support_shape_digest"
                        ]
                    )
                    moving_support_ref = str(
                        after_axis["cell_reflection_negative_support_ref"]
                    )
                elif (
                    positive_delta == expected_reflected_delta
                    and negative_delta == (0, 0)
                ):
                    moving_side = "positive"
                    moving_delta = positive_delta
                    moving_after_bbox = positive_after_bbox
                    moving_shape_digest = str(
                        after_axis[
                            "cell_reflection_positive_support_shape_digest"
                        ]
                    )
                    moving_support_ref = str(
                        after_axis["cell_reflection_positive_support_ref"]
                    )
                else:
                    continue
                pair_support_refs = {
                    str(before_axis["cell_reflection_negative_support_ref"]),
                    str(before_axis["cell_reflection_positive_support_ref"]),
                }
                terminal_candidates: list[tuple[str, tuple[int, ...]]] = []
                for hole in before_scene.hole_sprites:
                    if hole.hole_sprite_id in pair_support_refs or len(hole.pixels) <= 1:
                        continue
                    relative = tuple(
                        sorted(
                            (row - hole.bbox.top, col - hole.bbox.left)
                            for row, col in hole.pixels
                        )
                    )
                    if stable_digest(relative)[:16] != moving_shape_digest:
                        continue
                    terminal_candidates.append(
                        (
                            hole.hole_sprite_id,
                            (
                                hole.bbox.top,
                                hole.bbox.left,
                                hole.bbox.bottom,
                                hole.bbox.right,
                            ),
                        )
                    )
                terminal_candidate_count = len(terminal_candidates)
                terminal_support_ref = ""
                terminal_residual_after_twice = 0
                exact_remaining_action_count = 0
                next_action_would_overshoot = False
                if terminal_candidate_count == 1:
                    terminal_support_ref, terminal_bbox = terminal_candidates[0]
                    axis_index = 1 if orientation == "vertical" else 0
                    moving_center_twice = (
                        moving_after_bbox[axis_index]
                        + moving_after_bbox[axis_index + 2]
                    )
                    terminal_center_twice = (
                        terminal_bbox[axis_index] + terminal_bbox[axis_index + 2]
                    )
                    terminal_residual_after_twice = (
                        terminal_center_twice - moving_center_twice
                    )
                    support_delta = moving_delta[axis_index]
                    support_delta_twice = 2 * support_delta
                    if (
                        support_delta_twice != 0
                        and terminal_residual_after_twice * support_delta_twice > 0
                        and abs(terminal_residual_after_twice)
                        % abs(support_delta_twice)
                        == 0
                    ):
                        exact_remaining_action_count = abs(
                            terminal_residual_after_twice
                        ) // abs(support_delta_twice)
                    next_action_would_overshoot = (
                        terminal_residual_after_twice == 0
                        or terminal_residual_after_twice * support_delta_twice <= 0
                        or abs(terminal_residual_after_twice)
                        < abs(support_delta_twice)
                    )
                witness_digest = stable_digest(
                    (
                        translation.entity_ref,
                        before_axis["cell_reflection_axis_digest"],
                        after_axis["cell_reflection_axis_digest"],
                        translation.delta_row,
                        translation.delta_col,
                    )
                )[:16]
                witnesses.append(
                    {
                        "effect_carried_reflection_axis_digest": witness_digest,
                        "effect_carried_reflection_axis_entity_ref": (
                            translation.entity_ref
                        ),
                        "effect_carried_reflection_axis_carrier_observed_value": (
                            before_entity.component.value
                        ),
                        "effect_carried_reflection_axis_action_ref": "",
                        "effect_carried_reflection_orientation": orientation,
                        "effect_carried_reflection_axis_delta_row": (
                            translation.delta_row
                        ),
                        "effect_carried_reflection_axis_delta_col": (
                            translation.delta_col
                        ),
                        "effect_carried_reflection_before_axis_coordinate_four": (
                            before_axis["cell_reflection_axis_coordinate_four"]
                        ),
                        "effect_carried_reflection_after_axis_coordinate_four": (
                            after_axis["cell_reflection_axis_coordinate_four"]
                        ),
                        "effect_carried_reflection_support_area": before_axis[
                            "cell_reflection_support_area"
                        ],
                        "effect_carried_reflection_negative_support_ref": after_axis[
                            "cell_reflection_negative_support_ref"
                        ],
                        "effect_carried_reflection_positive_support_ref": after_axis[
                            "cell_reflection_positive_support_ref"
                        ],
                        "effect_carried_reflection_moving_side": moving_side,
                        "effect_carried_reflection_moving_support_ref": (
                            moving_support_ref
                        ),
                        "effect_carried_reflection_moving_support_delta_row": (
                            moving_delta[0]
                        ),
                        "effect_carried_reflection_moving_support_delta_col": (
                            moving_delta[1]
                        ),
                        "effect_carried_reflection_terminal_candidate_count": (
                            terminal_candidate_count
                        ),
                        "effect_carried_reflection_terminal_support_ref": (
                            terminal_support_ref
                        ),
                        "effect_carried_reflection_terminal_residual_after_twice": (
                            terminal_residual_after_twice
                        ),
                        "effect_carried_reflection_exact_remaining_action_count": (
                            exact_remaining_action_count
                        ),
                        "effect_carried_reflection_next_action_would_overshoot": (
                            next_action_would_overshoot
                        ),
                        "effect_carried_reflection_relation_exact_before": True,
                        "effect_carried_reflection_relation_exact_after": True,
                        "effect_carried_reflection_palette_equality_required": False,
                    }
                )
                if len(witnesses) >= 64:
                    return tuple(witnesses)
    return tuple(witnesses)


def _is_exact_midpoint(
    derived_twice: tuple[int, int],
    first_twice: tuple[int, int],
    second_twice: tuple[int, int],
) -> bool:
    return (
        2 * derived_twice[0] == first_twice[0] + second_twice[0]
        and 2 * derived_twice[1] == first_twice[1] + second_twice[1]
    )


def _is_nearest_lattice_midpoint(
    derived_twice: tuple[int, int],
    first_twice: tuple[int, int],
    second_twice: tuple[int, int],
) -> bool:
    """Whether the observed center is a nearest grid-representable midpoint."""

    return all(
        abs(2 * derived_twice[axis] - first_twice[axis] - second_twice[axis])
        <= 2
        for axis in (0, 1)
    )


def _candidate_point(candidate_ref: str) -> tuple[int, int] | None:
    """Decode the measured point carried by a point-probe reference."""

    marker = ":at:r"
    if marker not in candidate_ref:
        return None
    suffix = candidate_ref.rsplit(marker, 1)[1]
    if ":c" not in suffix:
        return None
    row_text, col_text = suffix.split(":c", 1)
    try:
        return int(row_text), int(col_text)
    except ValueError:
        return None


def _same_component_axis_clearance(
    component: ComponentDescription,
    point: tuple[int, int],
) -> tuple[int, int]:
    pixels = frozenset(component.pixels)
    row, col = point

    def run(delta_row: int, delta_col: int) -> int:
        distance = 0
        cursor = (row + delta_row, col + delta_col)
        while cursor in pixels:
            distance += 1
            cursor = (cursor[0] + delta_row, cursor[1] + delta_col)
        return distance

    return (
        min(run(0, -1), run(0, 1)),
        min(run(-1, 0), run(1, 0)),
    )


def _reserved_nonfrequent_sprite_extents(
    scene: object,
) -> tuple[int, int]:
    components = scene.blocks.components
    components_by_ref = {item.component_id: item for item in components}
    color_pixel_counts = Counter(pixel for row in scene.frame.rows for pixel in row)
    greatest_color_pixel_count = max(color_pixel_counts.values())
    sprites = tuple(
        sprite
        for sprite in scene.sprites
        if not sprite.touches_frame_boundary
        and (
            (anchor := components_by_ref.get(sprite.anchor_component_ref)) is not None
            and color_pixel_counts[anchor.value] < greatest_color_pixel_count
        )
    )
    if not sprites:
        return 0, 0
    thickest = min(
        sprites,
        key=lambda sprite: (
            -min(sprite.bbox.height, sprite.bbox.width),
            -(sprite.bbox.height * sprite.bbox.width),
            sprite.sprite_id,
        ),
    )
    return thickest.bbox.width, thickest.bbox.height


def _stationary_empty_center_quartets(
    *,
    stationary: tuple[TrackedComponent, ...],
    derived_value: int,
    frame: object,
) -> tuple[tuple[tuple[object, ...], ...], bool]:
    """Enumerate at most 64 exact rectangular four-corner descriptions."""

    grouped: dict[tuple[int, int, int, int], list[TrackedComponent]] = {}
    for tracked in stationary:
        component = tracked.component
        if component.value != derived_value or component.touches_frame_boundary:
            continue
        signature = (
            component.value,
            component.bbox.height,
            component.bbox.width,
            component.area,
        )
        grouped.setdefault(signature, []).append(tracked)

    descriptions: list[tuple[object, ...]] = []
    truncated = False
    for signature in sorted(grouped):
        members = grouped[signature]
        if len(members) > 32:
            truncated = True
            continue
        by_center: dict[tuple[int, int], list[TrackedComponent]] = {}
        for tracked in members:
            by_center.setdefault(_center_twice(tracked.component), []).append(tracked)
        rows = sorted({row for row, _col in by_center})
        cols = sorted({col for _row, col in by_center})
        if len(rows) > 8 or len(cols) > 8:
            truncated = True
            continue
        for row_index, first_row in enumerate(rows):
            for second_row in rows[row_index + 1 :]:
                for col_index, first_col in enumerate(cols):
                    for second_col in cols[col_index + 1 :]:
                        corners = (
                            (first_row, first_col),
                            (first_row, second_col),
                            (second_row, first_col),
                            (second_row, second_col),
                        )
                        if any(corner not in by_center for corner in corners):
                            continue
                        center_row_numerator = first_row + second_row
                        center_col_numerator = first_col + second_col
                        if center_row_numerator % 2 or center_col_numerator % 2:
                            continue
                        center_twice = (
                            center_row_numerator // 2,
                            center_col_numerator // 2,
                        )
                        if center_twice[0] % 2 or center_twice[1] % 2:
                            continue
                        center = (center_twice[0] // 2, center_twice[1] // 2)
                        aliases = tuple(
                            tuple(sorted(item.entity_id for item in by_center[corner]))
                            for corner in corners
                        )
                        corner_pixels = frozenset(
                            pixel
                            for corner in corners
                            for tracked in by_center[corner]
                            for pixel in tracked.component.pixels
                        )
                        if center in corner_pixels:
                            continue
                        descriptions.append(
                            (center_twice, corners, aliases, signature)
                        )
                        if len(descriptions) > 64:
                            return tuple(descriptions[:64]), True
    return tuple(sorted(descriptions)), truncated


def analyze_controlled_transition(
    value: ControlledTransitionAnalysisInput,
) -> ControlledTransitionAnalysis:
    """Expose exact component-level causal facts without assigning roles.

    The result deliberately contains competing interpretation candidates.  A
    DRM selection, not this capability, decides whether the facts license a
    controlled transfer meaning.
    """

    before_by_component = {
        item.component.component_id: item for item in value.before_tracking.entities
    }
    before_by_entity = {
        item.entity_id: item for item in value.before_tracking.entities
    }
    after_by_entity = {item.entity_id: item for item in value.after_tracking.entities}
    clicked_component_ref = value.candidate_ref.removeprefix("probe:")
    clicked_before = before_by_component.get(clicked_component_ref)
    clicked_entity_ref = clicked_before.entity_id if clicked_before is not None else None

    extents = tuple(
        transition
        for tracked in value.after_tracking.entities
        if (transition := _extent_transition(tracked, before_by_entity)) is not None
    )
    translations = tuple(
        TranslationTransition(
            entity_ref=tracked.entity_id,
            component_ref=tracked.component.component_id,
            delta_row=tracked.delta_row,
            delta_col=tracked.delta_col,
        )
        for tracked in value.after_tracking.entities
        if tracked.identity_status == "established"
        and (tracked.delta_row != 0 or tracked.delta_col != 0)
        and tracked.match_kind == "translated_exact"
    )
    pre_action_input_aligned_bboxes = frozenset(
        value.pre_action_input_aligned_entity_bboxes
    )
    input_aligned_episode_start_bbox = value.input_aligned_episode_start_bbox
    input_aligned_episode_start_return_translations = tuple(
        transition
        for transition in translations
        if input_aligned_episode_start_bbox is not None
        and transition.entity_ref in before_by_entity
        and transition.entity_ref in after_by_entity
        and before_by_entity[transition.entity_ref].component.bbox
        in pre_action_input_aligned_bboxes
        and before_by_entity[transition.entity_ref].component.bbox
        != input_aligned_episode_start_bbox
        and after_by_entity[transition.entity_ref].component.bbox
        == input_aligned_episode_start_bbox
    )
    interface_deltas = {
        action_ref: (delta_row, delta_col)
        for action_ref, delta_row, delta_col in (
            value.interface_action_translation_deltas
        )
    }
    expected_input_delta = interface_deltas.get(value.action_ref)

    def normalized_axis_direction(delta: tuple[int, int]) -> tuple[int, int]:
        row, column = delta
        return (
            0 if row == 0 else (1 if row > 0 else -1),
            0 if column == 0 else (1 if column > 0 else -1),
        )

    def is_exact_hollow_contour(component: ComponentDescription) -> bool:
        """Measure a complete rectangular perimeter with an empty interior."""

        height = component.bbox.height
        width = component.bbox.width
        if height < 3 or width < 3:
            return False
        perimeter = frozenset(
            (row, col)
            for row in range(height)
            for col in range(width)
            if row in {0, height - 1} or col in {0, width - 1}
        )
        return frozenset(component.relative_pixels) == perimeter

    input_aligned_translations = tuple(
        transition
        for transition in translations
        if expected_input_delta is not None
        and normalized_axis_direction(
            (transition.delta_row, transition.delta_col)
        )
        == normalized_axis_direction(expected_input_delta)
    )
    adjacent_translation_rows = ()
    if value.input_aligned_group_connectivity and len(input_aligned_translations) > 1:
        groups = measure_adjacent_translation_groups(
            tracking=value.after_tracking,
            member_translations=tuple((item.entity_ref, item.delta_row, item.delta_col)
                                      for item in input_aligned_translations),
            connectivity=value.input_aligned_group_connectivity,
            max_members=64, max_pixels=4096,
        )
        deltas_by_ref = {item.entity_ref: (item.delta_row, item.delta_col)
                         for item in input_aligned_translations}
        adjacent_translation_rows = tuple(FrozenMap({
            "group_token": stable_digest(group), "member_refs": group,
            "member_count": len(group),
            "delta_row": deltas_by_ref[group[0]][0], "delta_col": deltas_by_ref[group[0]][1],
            "action_ref": value.action_ref, "transition_ref": value.transition_ref,
            "context_epoch": value.context_epoch,
        }) for group in groups if len(group) >= 2)
    input_aligned_enclosing_extents = tuple(
        transition
        for transition in input_aligned_translations
        if len(input_aligned_translations) > 1
        and all(
            (
                other.entity_ref == transition.entity_ref
                or (
                    before_by_entity[
                        transition.entity_ref
                    ].component.bbox
                    != before_by_entity[other.entity_ref].component.bbox
                    and before_by_entity[
                        transition.entity_ref
                    ].component.bbox.top
                    <= before_by_entity[other.entity_ref].component.bbox.top
                    and before_by_entity[
                        transition.entity_ref
                    ].component.bbox.left
                    <= before_by_entity[other.entity_ref].component.bbox.left
                    and before_by_entity[
                        transition.entity_ref
                    ].component.bbox.bottom
                    >= before_by_entity[other.entity_ref].component.bbox.bottom
                    and before_by_entity[
                        transition.entity_ref
                    ].component.bbox.right
                    >= before_by_entity[other.entity_ref].component.bbox.right
                )
            )
            for other in input_aligned_translations
        )
    )
    # Repeated internal marks can keep a stationary individual match, or match
    # another identical mark farther away. Their identities do not establish
    # the physical displacement of the complete observed support.
    input_aligned_enclosed_member_refs = {}
    if value.before_tracking is not None and value.after_tracking is not None:
        for transition in input_aligned_enclosing_extents:
            refs = measure_translated_enclosed_member_refs(
                before=value.before_tracking, after=value.after_tracking,
                anchor_entity_ref=transition.entity_ref, max_members=64, max_pixels=4096,
            )
            if refs is not None and all(other.entity_ref in refs for other in input_aligned_translations):
                input_aligned_enclosed_member_refs[transition.entity_ref] = refs
    input_aligned_enclosing_carriers = tuple(
        transition for transition in input_aligned_enclosing_extents
        if transition.entity_ref in input_aligned_enclosed_member_refs
    )
    if value.input_aligned_group_without_global_enclosure and input_aligned_enclosing_carriers:
        adjacent_translation_rows = ()
    input_aligned_hollow_translations = tuple(
        transition
        for transition in input_aligned_translations
        if is_exact_hollow_contour(
            before_by_entity[transition.entity_ref].component
        )
    )
    stationary_value_transitions = tuple(
        (before_by_entity[tracked.entity_id], tracked)
        for tracked in value.after_tracking.entities
        if tracked.identity_status == "established"
        and tracked.match_kind == "stationary_value_transition"
        and tracked.entity_id in before_by_entity
    )
    stationary_connector_transition_measurements = (
        _stationary_thin_connector_transition_measurements(
            before_scene=value.before_scene,
            stationary_value_transitions=stationary_value_transitions,
            transition_ref=value.transition_ref,
        )
    )
    path_substitution_transition_measurements = (
        _thin_path_substitution_transition_measurements(
            before_scene=value.before_scene,
            after_scene=value.after_scene,
            candidate_ref=value.candidate_ref,
            transition_ref=value.transition_ref,
        )
    )
    connector_transition_measurements = (
        stationary_connector_transition_measurements
        + path_substitution_transition_measurements
    )
    stationary_appearance_transitions = tuple(
        (before_by_entity[tracked.entity_id], tracked)
        for tracked in value.after_tracking.entities
        if tracked.identity_status == "established"
        and tracked.match_kind
        in {"stationary_value_transition", "stationary_appearance_transition"}
        and tracked.entity_id in before_by_entity
    )
    same_value_multicell_morphology_transitions = tuple(
        (before, after)
        for before, after in stationary_appearance_transitions
        if before.component.value == after.component.value
        and before.component.relative_pixels != after.component.relative_pixels
        and before.component.area > 1
        and after.component.area > 1
    )
    upper_left_anchor_preserved_morphology_transitions = tuple(
        (before, after)
        for before, after in same_value_multicell_morphology_transitions
        if before.component.bbox.top == after.component.bbox.top
        and before.component.bbox.left == after.component.bbox.left
    )
    boundary_aligned_monotonic_extent_decreases = tuple(
        (before, after)
        for before in value.before_scene.blocks.components
        for after in value.after_scene.blocks.components
        if before.value == after.value
        and before.touches_frame_boundary
        and after.touches_frame_boundary
        and min(before.bbox.height, before.bbox.width) == 1
        and min(after.bbox.height, after.bbox.width) == 1
        and before.bbox.top == after.bbox.top
        and before.bbox.left == after.bbox.left
        and after.area < before.area
        and frozenset(after.pixels).issubset(frozenset(before.pixels))
    )
    non_boundary_decrease_morphology_transitions = tuple(
        item
        for item in same_value_multicell_morphology_transitions
        if not (
            item[0].component.touches_frame_boundary
            and item[1].component.touches_frame_boundary
            and min(
                item[0].component.bbox.height,
                item[0].component.bbox.width,
            )
            == 1
            and min(
                item[1].component.bbox.height,
                item[1].component.bbox.width,
            )
            == 1
            and item[0].component.bbox.top == item[1].component.bbox.top
            and item[0].component.bbox.left == item[1].component.bbox.left
            and item[1].component.area < item[0].component.area
            and frozenset(item[1].component.pixels).issubset(
                frozenset(item[0].component.pixels)
            )
        )
    )
    upper_left_anchor_preserved_non_boundary_decrease_transitions = tuple(
        (before, after)
        for before, after in non_boundary_decrease_morphology_transitions
        if before.component.bbox.top == after.component.bbox.top
        and before.component.bbox.left == after.component.bbox.left
    )
    anchored_quarter_turn_match_sets = tuple(
        _exact_quarter_turn_matches(
            before.component.relative_pixels,
            after.component.relative_pixels,
        )
        for before, after in upper_left_anchor_preserved_non_boundary_decrease_transitions
    )
    before_exact_components = {
        (component.value, component.pixels)
        for component in value.before_scene.blocks.components
    }
    after_exact_components = {
        (component.value, component.pixels)
        for component in value.after_scene.blocks.components
    }
    changed_before_components = tuple(
        component
        for component in value.before_scene.blocks.components
        if not component.touches_frame_boundary
        and (component.value, component.pixels) not in after_exact_components
    )
    changed_after_components = tuple(
        component
        for component in value.after_scene.blocks.components
        if not component.touches_frame_boundary
        and (component.value, component.pixels) not in before_exact_components
    )
    before_composite_signature = Counter(
        (component.value, component.area) for component in changed_before_components
    )
    after_composite_signature = Counter(
        (component.value, component.area) for component in changed_after_components
    )
    composite_signature_is_preserved = bool(
        len(changed_before_components) >= 2
        and len(changed_after_components) >= 2
        and len({component.value for component in changed_before_components}) >= 2
        and before_composite_signature == after_composite_signature
    )
    before_composite_pixels = frozenset(
        (row, col, component.value)
        for component in changed_before_components
        for row, col in component.pixels
    )
    after_composite_pixels = frozenset(
        (row, col, component.value)
        for component in changed_after_components
        for row, col in component.pixels
    )
    before_composite_anchor = (
        (
            min(row for row, _col, _value in before_composite_pixels),
            min(col for _row, col, _value in before_composite_pixels),
        )
        if before_composite_pixels
        else None
    )
    after_composite_anchor = (
        (
            min(row for row, _col, _value in after_composite_pixels),
            min(col for _row, col, _value in after_composite_pixels),
        )
        if after_composite_pixels
        else None
    )
    rigid_multivalue_composite_morphology_transition = False
    rigid_multivalue_quarter_turn_matches: tuple[int, ...] = ()
    if (
        composite_signature_is_preserved
        and before_composite_anchor is not None
        and after_composite_anchor is not None
    ):
        before_relative = frozenset(
            (
                row - before_composite_anchor[0],
                col - before_composite_anchor[1],
                pixel_value,
            )
            for row, col, pixel_value in before_composite_pixels
        )
        after_relative = frozenset(
            (
                row - after_composite_anchor[0],
                col - after_composite_anchor[1],
                pixel_value,
            )
            for row, col, pixel_value in after_composite_pixels
        )
        height = max(row for row, _col, _value in before_relative) + 1
        width = max(col for _row, col, _value in before_relative) + 1
        rigid_rotations = (
            frozenset(
                (col, height - 1 - row, pixel_value)
                for row, col, pixel_value in before_relative
            ),
            frozenset(
                (height - 1 - row, width - 1 - col, pixel_value)
                for row, col, pixel_value in before_relative
            ),
            frozenset(
                (width - 1 - col, row, pixel_value)
                for row, col, pixel_value in before_relative
            ),
        )
        rigid_multivalue_quarter_turn_matches = tuple(
            index + 1
            for index, rotated in enumerate(rigid_rotations)
            if after_relative == rotated
        )
        rigid_multivalue_composite_morphology_transition = bool(
            rigid_multivalue_quarter_turn_matches
        )
    upper_left_anchor_preserved_rigid_multivalue_composite_transition = bool(
        rigid_multivalue_composite_morphology_transition
        and before_composite_anchor == after_composite_anchor
    )
    # Preserve a complete measured support, not a fabricated temporal identity.
    # Direction membership is evidence for DRM, never a Python role decision.
    stationary_rotation_rows = ()
    before_mask = frozenset((row, col) for row, col, _ in before_composite_pixels)
    after_mask = frozenset((row, col) for row, col, _ in after_composite_pixels)
    if (
        upper_left_anchor_preserved_rigid_multivalue_composite_transition
        and len(rigid_multivalue_quarter_turn_matches) == 1
        and before_mask == after_mask
        and before_composite_pixels != after_composite_pixels
        and value.input_aligned_group_connectivity in (4, 8)
        and 1 < len(changed_after_components) <= 64
        and 0 < len(after_mask) <= 4096
    ):
        remaining = set(after_mask)
        pending = [min(remaining)]
        offsets = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if (dr, dc) != (0, 0) and (
                            value.input_aligned_group_connectivity == 8 or abs(dr) + abs(dc) == 1))
        while pending:
            pixel = pending.pop()
            if pixel not in remaining:
                continue
            remaining.remove(pixel)
            pending.extend((pixel[0] + dr, pixel[1] + dc) for dr, dc in offsets
                           if (pixel[0] + dr, pixel[1] + dc) in remaining)
        component_ids = {component.component_id for component in changed_after_components}
        member_refs = tuple(sorted(entity.entity_id for entity in value.after_tracking.entities
                                   if entity.current_frame_ref == value.after_tracking.frame_ref
                                   and entity.component.component_id in component_ids))
        if not remaining and len(member_refs) == len(component_ids):
            stationary_rotation_rows = (FrozenMap({
                "group_token": stable_digest(member_refs), "member_refs": member_refs,
                "member_count": len(member_refs), "delta_row": 0, "delta_col": 0,
                "quarter_turns_clockwise": rigid_multivalue_quarter_turn_matches[0],
                "action_is_directional": value.action_ref in value.available_directional_action_refs,
                "support_mask_preserved": True,
                "action_ref": value.action_ref, "transition_ref": value.transition_ref,
                "context_epoch": value.context_epoch,
            }),)
    all_anchored_quarter_turn_match_sets = (
        *anchored_quarter_turn_match_sets,
        *((rigid_multivalue_quarter_turn_matches,) if (
            upper_left_anchor_preserved_rigid_multivalue_composite_transition
        ) else ()),
    )
    exact_anchored_quarter_turns = tuple(
        match_set[0]
        for match_set in all_anchored_quarter_turn_match_sets
        if len(match_set) == 1
    )
    anchored_multicell_exact_quarter_turns_clockwise = (
        exact_anchored_quarter_turns[0]
        if exact_anchored_quarter_turns
        and len(exact_anchored_quarter_turns)
        == len(all_anchored_quarter_turn_match_sets)
        and len(set(exact_anchored_quarter_turns)) == 1
        else 0
    )
    stationary_appearance_pairs = tuple(
        sorted(
            (
                stable_digest(
                    (before.component.value, before.component.relative_pixels)
                )[:16],
                stable_digest(
                    (after.component.value, after.component.relative_pixels)
                )[:16],
            )
            for before, after in stationary_appearance_transitions
        )
    )
    exclusive_two_entity_appearance_permutation = bool(
        len(stationary_appearance_transitions) == 2
        and len(
            {
                before.entity_id
                for before, _after in stationary_appearance_transitions
            }
        )
        == 2
        and sorted(
            stable_digest(
                (before.component.value, before.component.relative_pixels)
            )[:16]
            for before, _after in stationary_appearance_transitions
        )
        == sorted(
            stable_digest(
                (after.component.value, after.component.relative_pixels)
            )[:16]
            for _before, after in stationary_appearance_transitions
        )
        and all(
            (
                before.component.value,
                before.component.relative_pixels,
            )
            != (
                after.component.value,
                after.component.relative_pixels,
            )
            for before, after in stationary_appearance_transitions
        )
    )
    after_components_by_pose = {
        (
            component.bbox.top,
            component.bbox.left,
            component.bbox.bottom,
            component.bbox.right,
            component.value,
            component.relative_pixels,
        ): component
        for component in value.after_scene.blocks.components
    }
    holed_shape_counts: dict[tuple[object, ...], int] = {}
    for component in value.before_scene.blocks.components:
        if (
            component.bbox.height == component.bbox.width
            and component.bbox.height % 2 == 1
            and component.area
            == component.bbox.height * component.bbox.width - 1
        ):
            signature = (component.relative_pixels, component.area)
            holed_shape_counts[signature] = holed_shape_counts.get(signature, 0) + 1
    internal_state_transitions: list[tuple[str, int, int]] = []
    for component in value.before_scene.blocks.components:
        signature = (component.relative_pixels, component.area)
        if holed_shape_counts.get(signature, 0) < 2:
            continue
        key = (
            component.bbox.top,
            component.bbox.left,
            component.bbox.bottom,
            component.bbox.right,
            component.value,
            component.relative_pixels,
        )
        if key not in after_components_by_pose:
            continue
        center = (
            (component.bbox.top + component.bbox.bottom) // 2,
            (component.bbox.left + component.bbox.right) // 2,
        )
        if center in component.pixels:
            continue
        before_state = value.before_scene.frame.rows[center[0]][center[1]]
        after_state = value.after_scene.frame.rows[center[0]][center[1]]
        if before_state != after_state:
            internal_state_transitions.append(
                (component.component_id, before_state, after_state)
            )
    if not (
        len(internal_state_transitions) == 2
        and sorted(item[1] for item in internal_state_transitions)
        == sorted(item[2] for item in internal_state_transitions)
    ):
        internal_state_transitions = list(
            _marker_axis_internal_state_transitions(value)
        ) or internal_state_transitions
    exclusive_internal_state_permutation = bool(
        len(internal_state_transitions) == 2
        and sorted(item[1] for item in internal_state_transitions)
        == sorted(item[2] for item in internal_state_transitions)
    )
    if exclusive_internal_state_permutation:
        stationary_appearance_pairs = tuple(
            sorted(
                (stable_digest((before_state,))[:16], stable_digest((after_state,))[:16])
                for _component_ref, before_state, after_state
                in internal_state_transitions
            )
        )
        exclusive_two_entity_appearance_permutation = True
    stationary_value_pairs = tuple(
        sorted(
            (
                before.component.value,
                after.component.value,
            )
            for before, after in stationary_value_transitions
        )
    )
    exclusive_two_entity_value_permutation = bool(
        len(stationary_value_transitions) == 2
        and len({before.entity_id for before, _after in stationary_value_transitions})
        == 2
        and sorted(before.component.value for before, _after in stationary_value_transitions)
        == sorted(after.component.value for _before, after in stationary_value_transitions)
        and all(
            before.component.value != after.component.value
            for before, after in stationary_value_transitions
        )
    )
    stationary_entities = tuple(
        tracked
        for tracked in value.after_tracking.entities
        if tracked.identity_status == "established"
        and tracked.match_kind == "stationary_exact"
        and tracked.entity_id in before_by_entity
    )
    midpoint_aliases: dict[
        tuple[tuple[int, int], ...],
        list[tuple[str, str, str]],
    ] = {}
    for full in translations:
        for half in translations:
            if full.entity_ref == half.entity_ref or (
                abs(full.delta_row - 2 * half.delta_row) > 1
                or abs(full.delta_col - 2 * half.delta_col) > 1
            ):
                continue
            full_before = before_by_entity[full.entity_ref].component
            full_after = after_by_entity[full.entity_ref].component
            half_before = before_by_entity[half.entity_ref].component
            half_after = after_by_entity[half.entity_ref].component
            for fixed in stationary_entities:
                if fixed.entity_id in (full.entity_ref, half.entity_ref):
                    continue
                fixed_before = before_by_entity[fixed.entity_id].component
                fixed_after = fixed.component
                centers = (
                    _center_twice(full_before),
                    _center_twice(half_before),
                    _center_twice(fixed_before),
                    _center_twice(full_after),
                    _center_twice(half_after),
                    _center_twice(fixed_after),
                )
                if not (
                    _is_nearest_lattice_midpoint(centers[1], centers[0], centers[2])
                    and _is_nearest_lattice_midpoint(
                        centers[4], centers[3], centers[5]
                    )
                ):
                    continue
                midpoint_aliases.setdefault(centers, []).append(
                    (full.entity_ref, half.entity_ref, fixed.entity_id)
                )
    midpoint_geometry = tuple(sorted(midpoint_aliases))[:64]
    midpoint_enumeration_truncated = len(midpoint_aliases) > 64
    persistent_entity_refs = tuple(
        sorted(
            entity_ref
            for entity_ref in set(before_by_entity).intersection(after_by_entity)
            if after_by_entity[entity_ref].identity_status == "established"
        )
    )[:12]
    translation_by_ref = {item.entity_ref: item for item in translations}
    nary_affine_rows: list[dict[str, object]] = []
    nary_affine_enumeration_truncated = False
    nary_affine_structural_description_count = 0
    nary_affine_structural_description_bound = 4096
    for derived_ref in persistent_entity_refs:
        derived_translation = translation_by_ref.get(derived_ref)
        if derived_translation is None:
            continue
        others = tuple(item for item in persistent_entity_refs if item != derived_ref)
        # A midpoint is the binary projection of this same finite equal-weight
        # relation.  The compatibility path still measures that projection
        # separately, while this evidence family starts at three contributors
        # and has no semantic arity ceiling.  The current observation already
        # bounds persistent entities; this additional explicit ceiling makes
        # structural-description expansion fail closed if that bound changes.
        for contributor_count in range(3, len(others) + 1):
            for contributor_refs in combinations(others, contributor_count):
                nary_affine_structural_description_count += 1
                if (
                    nary_affine_structural_description_count
                    > nary_affine_structural_description_bound
                ):
                    nary_affine_enumeration_truncated = True
                    break
                contributor_translations = tuple(
                    translation_by_ref.get(item) for item in contributor_refs
                )
                moving_contributor_deltas = tuple(
                    (item.delta_row, item.delta_col)
                    for item in contributor_translations
                    if item is not None
                )
                moving_contributor_refs = tuple(
                    ref
                    for ref, item in zip(contributor_refs, contributor_translations)
                    if item is not None
                )
                if len(moving_contributor_refs) != 1:
                    continue
                derived_norm = abs(derived_translation.delta_row) + abs(
                    derived_translation.delta_col
                )
                if derived_norm >= max(
                    abs(delta_row) + abs(delta_col)
                    for delta_row, delta_col in moving_contributor_deltas
                ):
                    continue
                derived_before = _center_twice(
                    before_by_entity[derived_ref].component
                )
                derived_after = _center_twice(
                    after_by_entity[derived_ref].component
                )
                contributor_before = tuple(
                    _center_twice(before_by_entity[item].component)
                    for item in contributor_refs
                )
                contributor_after = tuple(
                    _center_twice(after_by_entity[item].component)
                    for item in contributor_refs
                )
                before_residual = tuple(
                    contributor_count * derived_before[axis]
                    - sum(item[axis] for item in contributor_before)
                    for axis in (0, 1)
                )
                after_residual = tuple(
                    contributor_count * derived_after[axis]
                    - sum(item[axis] for item in contributor_after)
                    for axis in (0, 1)
                )
                if not all(
                    2 * abs(item) <= contributor_count
                    for item in (*before_residual, *after_residual)
                ):
                    continue
                relation_digest = stable_digest(
                    (derived_ref, contributor_refs, contributor_count)
                )[:16]
                nary_affine_rows.append(
                    {
                        "digest": relation_digest,
                        "derived_entity_ref": derived_ref,
                        "contributor_entity_refs": contributor_refs,
                        "active_contributor_entity_ref": moving_contributor_refs[0],
                        "coefficient_numerators": (1,) * contributor_count,
                        "coefficient_denominator": contributor_count,
                        "before_residual_numerators": before_residual,
                        "after_residual_numerators": after_residual,
                        "discriminating_transition": True,
                    }
                )
                if len(nary_affine_rows) > 64:
                    nary_affine_enumeration_truncated = True
                    break
            if nary_affine_enumeration_truncated:
                break
        if nary_affine_enumeration_truncated:
            break
    if nary_affine_enumeration_truncated:
        nary_affine_rows = []
    exchanges = _exchange_candidates(
        extents=extents,
        before_by_entity=before_by_entity,
        after_by_entity=after_by_entity,
        clicked=clicked_before,
    )
    alignment = _peer_distance_change_candidates(
        translations=translations,
        before_by_entity=before_by_entity,
        after_by_entity=after_by_entity,
    )

    unique_exchange = exchanges[0] if len(exchanges) == 1 else None
    separators = (
        _separator_candidates(
            unique_exchange,
            after_tracking=value.after_tracking.entities,
            frame_enclosing_refs=frozenset(
                value.after_scene.frame_enclosing_component_refs
            ),
        )
        if unique_exchange is not None
        else ()
    )
    interfaces = (
        _interface_components(
            clicked=clicked_before,
            exchange=unique_exchange,
            separators=separators,
            before_by_entity=before_by_entity,
            after_by_entity=after_by_entity,
        )
        if unique_exchange is not None
        else (() if clicked_entity_ref is None else (clicked_entity_ref,))
    )
    exact_no_pixel_change = value.before_scene.frame.rows == value.after_scene.frame.rows
    clicked_is_known_actuator = bool(
        clicked_entity_ref
        and clicked_entity_ref in frozenset(value.known_actuator_entity_refs)
    )
    actuator_premise_refs = tuple(dict.fromkeys(
        ref
        for fact in value.canonical_known_actuator_term_facts or ()
        if clicked_entity_ref is not None
        and (fact.get("entity_ref") or fact.get("term_ref")) == clicked_entity_ref
        for ref in fact.get("role_premise_claim_refs", ())
    ))
    # Prefer distance-reducing peer observations. If several co-movers approach
    # the same morphological target, keep the closest residual after the step
    # (and break ties by smaller orthogonal gap) so a single revisable goal
    # binding remains available without hardcoding roles.
    toward_alignment = tuple(
        item for item in alignment if item.distance_change_kind == "decreased"
    )

    def _alignment_preference(
        item: PeerDistanceChangeCandidate,
    ) -> tuple[int, int, int, int, str]:
        # Same-shape / same-orientation stationary target first, then sticky
        # partial position (orthogonal residual already closed), then shortest
        # residual. Never prefer a co-moving-like long path over a shape peer.
        moving_after = after_by_entity[item.translated_entity_ref].component
        fixed = after_by_entity[item.stationary_peer_entity_ref].component
        shape_miss = (
            0
            if moving_after.relative_pixels == fixed.relative_pixels
            else 1
        )
        if item.axis == "row":
            ortho = abs(
                (moving_after.bbox.left + moving_after.bbox.right)
                - (fixed.bbox.left + fixed.bbox.right)
            )
        else:
            ortho = abs(
                (moving_after.bbox.top + moving_after.bbox.bottom)
                - (fixed.bbox.top + fixed.bbox.bottom)
            )
        # Prefer pairs that already closed the orthogonal axis (sticky 2D goal).
        sticky_partial_miss = 0 if ortho == 0 else 1
        return (
            shape_miss,
            sticky_partial_miss,
            item.after_distance_twice,
            ortho,
            item.translated_entity_ref,
        )

    unique_alignment = (
        alignment[0]
        if len(alignment) == 1
        else min(toward_alignment, key=_alignment_preference)
        if toward_alignment
        else None
    )
    translation_deltas = frozenset(
        (item.delta_row, item.delta_col) for item in translations
    )
    co_moving = (
        tuple(
            item.entity_ref
            for item in translations
            if unique_alignment is not None
            and (
                item.delta_row,
                item.delta_col,
            )
            == (
                after_by_entity[unique_alignment.translated_entity_ref].delta_row,
                after_by_entity[unique_alignment.translated_entity_ref].delta_col,
            )
        )
        if unique_alignment is not None
        else (
            # When no peer-alignment unique pair is established yet, still group
            # exact same-delta co-movers for later DRM projection.
            tuple(
                item.entity_ref
                for item in translations
                if len(translation_deltas) == 1
            )
            if translations
            else ()
        )
    )
    co_moving_partner = (
        next(
            (
                item
                for item in co_moving
                if unique_alignment is not None
                and item != unique_alignment.translated_entity_ref
                and _components_touch(
                    before_by_entity[unique_alignment.translated_entity_ref].component,
                    before_by_entity[item].component,
                )
            ),
            None,
        )
        if unique_alignment is not None
        else None
    )
    contact_translation_measurements = (
        _contact_coupled_translation_measurements(
            translations=translations,
            before_by_entity=before_by_entity,
            after_by_entity=after_by_entity,
            pre_action_input_aligned_bboxes=tuple(
                value.pre_action_input_aligned_entity_bboxes
            ),
            expected_input_delta=expected_input_delta,
        )
    )

    changed_tracked_entity_refs = frozenset(
        item.entity_ref for item in translations
    ).union(
        after.entity_id for _before, after in stationary_value_transitions
    ).union(
        after.entity_id for _before, after in stationary_appearance_transitions
    ).union(item[0] for item in internal_state_transitions)
    clicked_entity_changed = bool(
        clicked_entity_ref and clicked_entity_ref in changed_tracked_entity_refs
    )
    nonclicked_changed_entity_refs = tuple(
        sorted(
            ref
            for ref in changed_tracked_entity_refs
            if ref != clicked_entity_ref
        )
    )
    # A source-selected block/grid scene has not measured enclosed objects.
    # Do not turn that unknown into an observed absence of reflection axes.
    reflection_measured = not any(
        isinstance(scene, BlockGridSceneDescription)
        for scene in (value.before_scene, value.after_scene)
    )
    cell_reflection_axes_before: tuple[dict[str, object], ...] = ()
    cell_reflection_axes_after: tuple[dict[str, object], ...] = ()
    effect_carried_reflection_axes: tuple[dict[str, object], ...] = ()
    if reflection_measured:
        cell_reflection_axes_before = _cell_reflection_axis_measurements(
            value.before_scene
        )
        cell_reflection_axes_after = _cell_reflection_axis_measurements(
            value.after_scene
        )
        effect_carried_reflection_axes = tuple(
            {
                **item,
                "effect_carried_reflection_axis_action_ref": value.action_ref,
            }
            for item in _effect_carried_reflection_axis_measurements(
                before_scene=value.before_scene,
                after_scene=value.after_scene,
                before_axes=cell_reflection_axes_before,
                after_axes=cell_reflection_axes_after,
                before_by_entity=before_by_entity,
                translations=translations,
            )
        )

    transformation_rows, transformation_rows_truncated = measure_entity_changes(
        value.before_tracking, value.after_tracking
    )
    neutral_bindings: dict[str, object] = {
        "transformation_entity_rows": transformation_rows,
        "transformation_entity_rows_truncated": transformation_rows_truncated,
        "measurement_frame_ref": value.after_tracking.frame_ref,
        "context_epoch": value.context_epoch,
        "transition_ref": value.transition_ref,
        "candidate_ref": value.candidate_ref,
        "action_ref": value.action_ref,
        "clicked_entity_ref": clicked_entity_ref,
        "clicked_entity_persisted": bool(
            clicked_entity_ref and clicked_entity_ref in after_by_entity
        ),
        "clicked_is_known_actuator": clicked_is_known_actuator,
        "canonical_reversible_control_claim_count": len(
            value.canonical_reversible_control_claim_facts
        ),
        "canonical_provisional_control_term_count": len(
            value.canonical_provisional_control_term_facts
        ),
        "canonical_action_translation_claim_count": sum(
            1
            for fact in value.canonical_action_translation_claim_facts
            if value.action_ref in tuple(fact.get("argument_refs") or ())
            and str(fact.get("epistemic_status") or "")
            in {"supported", "established"}
            and str(fact.get("disposition") or "active") == "active"
        ),
        "canonical_periodic_transition_claim_count": len(
            value.canonical_periodic_transition_claim_facts
        ),
        "prior_context_effect_carried_reflection_axis_observed": bool(
            value.prior_context_effect_carried_reflection_axis_observed
        ),
        # A transition claim proves that a cyclic supposition was observed;
        # it does not prove that the cycle is understood.  Only an explicit
        # cycle_understood status may suppress the independent spatial
        # translation interpretation for the current action.
        "canonical_periodic_cycle_understood_transition_claim_count": sum(
            1
            for fact in value.canonical_periodic_transition_claim_facts
            if str(
                fact.get("workflow_status")
                or fact.get("cycle_status")
                or fact.get("status")
                or ""
            )
            == "cycle_understood"
        ),
        "candidate_is_discrete_probe": value.candidate_ref.startswith(
            "probe:discrete:"
        ),
        "candidate_is_point_probe": bool(
            value.candidate_ref.startswith("probe:")
            and ":at:" in value.candidate_ref
            and not value.candidate_ref.startswith("probe:discrete:")
        ),
        "available_directional_action_count": len(
            value.available_directional_action_refs
        ),
        "executed_action_is_directional_interface_action": (
            value.action_ref in value.available_directional_action_refs
        ),
        "executed_action_interface_delta_present": expected_input_delta is not None,
        "pre_action_input_aligned_bbox_tuples": tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.pre_action_input_aligned_entity_bboxes
        ),
        "post_action_input_aligned_bbox_tuples": tuple(
            (bbox.top, bbox.left, bbox.bottom, bbox.right)
            for bbox in value.post_action_input_aligned_entity_bboxes
        ),
        "input_aligned_episode_start_bbox_tuple": (
            (
                input_aligned_episode_start_bbox.top,
                input_aligned_episode_start_bbox.left,
                input_aligned_episode_start_bbox.bottom,
                input_aligned_episode_start_bbox.right,
            )
            if input_aligned_episode_start_bbox is not None
            else ()
        ),
        "input_aligned_translation_count": len(input_aligned_translations),
        "input_aligned_translation_entity_refs": tuple(
            item.entity_ref for item in input_aligned_translations
        ),
        "input_aligned_enclosing_carrier_count": len(
            input_aligned_enclosing_carriers
        ),
        "input_aligned_enclosing_carrier_entity_refs": tuple(
            item.entity_ref for item in input_aligned_enclosing_carriers
        ),
        "input_aligned_hollow_translation_count": len(
            input_aligned_hollow_translations
        ),
        "input_aligned_hollow_translation_entity_refs": tuple(
            item.entity_ref for item in input_aligned_hollow_translations
        ),
        "input_aligned_episode_start_return_translation_count": len(
            input_aligned_episode_start_return_translations
        ),
        "input_aligned_body_returned_to_episode_start_pose": bool(
            input_aligned_episode_start_return_translations
        ),
        "input_aligned_episode_start_return_bbox": (
            input_aligned_episode_start_bbox.model_dump(mode="json")
            if input_aligned_episode_start_return_translations
            and input_aligned_episode_start_bbox is not None
            else None
        ),
        "clicked_entity_changed": clicked_entity_changed,
        "changed_tracked_entity_count": len(changed_tracked_entity_refs),
        "nonclicked_changed_entity_count": len(nonclicked_changed_entity_refs),
        "nonclicked_changed_entity_refs": nonclicked_changed_entity_refs,
        "exact_no_pixel_change": exact_no_pixel_change,
        "exact_quantized_exchange_candidate_count": len(exchanges),
        "alignment_candidate_count": len(alignment),
        "separator_component_count": len(separators),
        "interface_component_count": len(interfaces),
        "translation_count": len(translations),
        "translation_entity_refs": tuple(item.entity_ref for item in translations),
        "distinct_translation_delta_count": len(translation_deltas),
        "co_moving_entity_count": len(co_moving),
        "stationary_value_transition_count": len(stationary_value_transitions),
        "stationary_value_transition_entity_refs": tuple(
            after.entity_id for _before, after in stationary_value_transitions
        ),
        "stationary_value_transition_pairs": stationary_value_pairs,
        "action_conditioned_thin_connector_transition_count": len(
            connector_transition_measurements
        ),
        "action_conditioned_thin_connector_transition_measurements": (
            connector_transition_measurements
        ),
        "action_conditioned_thin_connector_transition_enumeration_truncated": (
            len(connector_transition_measurements) >= 32
        ),
        "action_conditioned_thin_path_substitution_transition_count": len(
            path_substitution_transition_measurements
        ),
        "action_conditioned_thin_path_substitution_transition_measurements": (
            path_substitution_transition_measurements
        ),
        "action_conditioned_thin_path_substitution_transition_enumeration_truncated": (
            len(path_substitution_transition_measurements) >= 3
        ),
        "exclusive_two_entity_value_permutation": (
            exclusive_two_entity_value_permutation
        ),
        "stationary_appearance_transition_count": (
            2
            if exclusive_internal_state_permutation
            else len(stationary_appearance_transitions)
        ),
        "same_value_multicell_morphology_transition_count": len(
            same_value_multicell_morphology_transitions
        ),
        "upper_left_anchor_preserved_morphology_transition_count": len(
            upper_left_anchor_preserved_morphology_transitions
        ),
        "all_same_value_multicell_morphology_transitions_preserve_upper_left_anchor": bool(
            same_value_multicell_morphology_transitions
            and len(same_value_multicell_morphology_transitions)
            == len(upper_left_anchor_preserved_morphology_transitions)
        ),
        "boundary_aligned_monotonic_extent_decrease_count": len(
            boundary_aligned_monotonic_extent_decreases
        ),
        "non_boundary_decrease_composite_morphology_transition_count": (
            len(non_boundary_decrease_morphology_transitions)
            + int(rigid_multivalue_composite_morphology_transition)
        ),
        "all_non_boundary_decrease_composite_transitions_preserve_upper_left_anchor": bool(
            (
                non_boundary_decrease_morphology_transitions
                or rigid_multivalue_composite_morphology_transition
            )
            and len(non_boundary_decrease_morphology_transitions)
            == len(upper_left_anchor_preserved_non_boundary_decrease_transitions)
            and (
                not rigid_multivalue_composite_morphology_transition
                or upper_left_anchor_preserved_rigid_multivalue_composite_transition
            )
        ),
        "anchored_multicell_exact_quarter_turn_match_count": len(
            exact_anchored_quarter_turns
        ),
        "anchored_multicell_exact_quarter_turns_clockwise": (
            anchored_multicell_exact_quarter_turns_clockwise
        ),
        "rigid_multivalue_composite_morphology_transition_count": int(
            rigid_multivalue_composite_morphology_transition
        ),
        "upper_left_anchor_preserved_rigid_multivalue_composite_transition_count": int(
            upper_left_anchor_preserved_rigid_multivalue_composite_transition
        ),
        "all_rigid_multivalue_composite_morphology_transitions_preserve_upper_left_anchor": bool(
            upper_left_anchor_preserved_rigid_multivalue_composite_transition
        ),
        "composite_morphology_transition_count": (
            len(same_value_multicell_morphology_transitions)
            + int(rigid_multivalue_composite_morphology_transition)
        ),
        "upper_left_anchor_preserved_composite_morphology_transition_count": (
            len(upper_left_anchor_preserved_morphology_transitions)
            + int(upper_left_anchor_preserved_rigid_multivalue_composite_transition)
        ),
        "all_composite_morphology_transitions_preserve_upper_left_anchor": bool(
            (
                same_value_multicell_morphology_transitions
                or rigid_multivalue_composite_morphology_transition
            )
            and len(same_value_multicell_morphology_transitions)
            == len(upper_left_anchor_preserved_morphology_transitions)
            and (
                not rigid_multivalue_composite_morphology_transition
                or upper_left_anchor_preserved_rigid_multivalue_composite_transition
            )
        ),
        "stationary_appearance_transition_pairs": stationary_appearance_pairs,
        "stationary_appearance_transition_entity_refs": (
            tuple(item[0] for item in internal_state_transitions)
            if exclusive_internal_state_permutation
            else tuple(
                after.entity_id
                for _before, after in stationary_appearance_transitions
            )
        ),
        "exclusive_two_entity_appearance_permutation": (
            exclusive_two_entity_appearance_permutation
        ),
        "midpoint_transition_geometry_count": len(midpoint_geometry),
        "midpoint_transition_enumeration_truncated": midpoint_enumeration_truncated,
        "nary_affine_relation_measurement_count": len(nary_affine_rows),
        "nary_affine_structural_description_count": (
            nary_affine_structural_description_count
        ),
        "nary_affine_structural_description_bound": (
            nary_affine_structural_description_bound
        ),
        "nary_affine_relation_enumeration_truncated": (
            nary_affine_enumeration_truncated
        ),
        "stationary_empty_center_quartet_count": 0,
        "stationary_empty_center_quartet_enumeration_truncated": False,
    }
    if reflection_measured:
        neutral_bindings.update({
            "cell_reflection_axis_measurement_count_before": len(
                cell_reflection_axes_before
            ),
            "cell_reflection_axis_measurement_count_after": len(
                cell_reflection_axes_after
            ),
            "cell_reflection_axis_measurements_before": cell_reflection_axes_before[:3],
            "cell_reflection_axis_measurements_after": cell_reflection_axes_after[:3],
            "effect_carried_reflection_axis_witness_count": len(
                effect_carried_reflection_axes
            ),
            "effect_carried_reflection_axis_witnesses": (
                effect_carried_reflection_axes[:3]
            ),
            "effect_carried_reflection_axis_enumeration_truncated": (
                len(effect_carried_reflection_axes) >= 64
            ),
        })
    unique_input_aligned_translation = (
        input_aligned_translations[0]
        if len(input_aligned_translations) == 1
        else (
            input_aligned_enclosing_carriers[0]
            if len(input_aligned_enclosing_carriers) == 1
            else None
        )
    )
    neutral_bindings["input_aligned_adjacent_translation_groups"] = adjacent_translation_rows
    neutral_bindings["stationary_exact_rotation_groups"] = stationary_rotation_rows
    if unique_input_aligned_translation is not None:
        unique_input_aligned_component = before_by_entity[
            unique_input_aligned_translation.entity_ref
        ].component
        # These are the very same members of the unique enclosure measured
        # above, not a second grouping or a colour-based role assignment.
        input_aligned_member_refs = input_aligned_enclosed_member_refs.get(
            unique_input_aligned_translation.entity_ref,
            tuple(sorted(transition.entity_ref for transition in input_aligned_translations)),
        )
        neutral_bindings.update(
            {
                "unique_input_aligned_member_refs": (
                    input_aligned_member_refs if len(input_aligned_member_refs) <= 64 else ()
                ),
                "unique_input_aligned_member_bound_exceeded": len(input_aligned_member_refs) > 64,
                "unique_input_aligned_entity_ref": (
                    unique_input_aligned_translation.entity_ref
                ),
                "unique_input_aligned_component_ref": (
                    unique_input_aligned_component.component_id
                ),
                "unique_input_aligned_delta_row": (
                    unique_input_aligned_translation.delta_row
                ),
                "unique_input_aligned_delta_col": (
                    unique_input_aligned_translation.delta_col
                ),
                "unique_input_aligned_area": unique_input_aligned_component.area,
                "unique_input_aligned_observed_value": (
                    unique_input_aligned_component.value
                ),
                "unique_input_aligned_bbox_height": (
                    unique_input_aligned_component.bbox.height
                ),
                "unique_input_aligned_bbox_width": (
                    unique_input_aligned_component.bbox.width
                ),
                "unique_input_aligned_shape_digest": stable_digest(
                    unique_input_aligned_component.relative_pixels
                )[:16],
            }
        )
    if len(input_aligned_hollow_translations) == 1:
        unique_hollow_translation = input_aligned_hollow_translations[0]
        unique_hollow_component = before_by_entity[
            unique_hollow_translation.entity_ref
        ].component
        neutral_bindings.update(
            {
                "unique_input_aligned_hollow_entity_ref": (
                    unique_hollow_translation.entity_ref
                ),
                "unique_input_aligned_hollow_component_ref": (
                    unique_hollow_component.component_id
                ),
                "unique_input_aligned_hollow_delta_row": (
                    unique_hollow_translation.delta_row
                ),
                "unique_input_aligned_hollow_delta_col": (
                    unique_hollow_translation.delta_col
                ),
                "unique_input_aligned_hollow_area": unique_hollow_component.area,
                "unique_input_aligned_hollow_observed_value": (
                    unique_hollow_component.value
                ),
                "unique_input_aligned_hollow_bbox_height": (
                    unique_hollow_component.bbox.height
                ),
                "unique_input_aligned_hollow_bbox_width": (
                    unique_hollow_component.bbox.width
                ),
                "unique_input_aligned_hollow_shape_digest": stable_digest(
                    unique_hollow_component.relative_pixels
                )[:16],
                "unique_input_aligned_hollow_is_exact_contour": True,
            }
        )
    if len(effect_carried_reflection_axes) == 1:
        unique_effect_carried_axis = effect_carried_reflection_axes[0]
        neutral_bindings.update(
            {
                "unique_effect_carried_reflection_axis_digest": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_axis_digest"
                    ]
                ),
                "unique_effect_carried_reflection_axis_entity_ref": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_axis_entity_ref"
                    ]
                ),
                "unique_effect_carried_reflection_axis_carrier_observed_value": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_axis_carrier_observed_value"
                    ]
                ),
                "unique_effect_carried_reflection_orientation": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_orientation"
                    ]
                ),
                "unique_effect_carried_reflection_axis_delta_row": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_axis_delta_row"
                    ]
                ),
                "unique_effect_carried_reflection_axis_delta_col": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_axis_delta_col"
                    ]
                ),
                "unique_effect_carried_reflection_support_area": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_support_area"
                    ]
                ),
                "unique_effect_carried_reflection_moving_support_ref": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_moving_support_ref"
                    ]
                ),
                "unique_effect_carried_reflection_negative_support_ref": (
                    unique_effect_carried_axis["effect_carried_reflection_negative_support_ref"]
                ),
                "unique_effect_carried_reflection_positive_support_ref": (
                    unique_effect_carried_axis["effect_carried_reflection_positive_support_ref"]
                ),
                "unique_effect_carried_reflection_terminal_candidate_count": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_terminal_candidate_count"
                    ]
                ),
                "unique_effect_carried_reflection_terminal_support_ref": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_terminal_support_ref"
                    ]
                ),
                "unique_effect_carried_reflection_terminal_residual_is_zero": (
                    unique_effect_carried_axis["effect_carried_reflection_terminal_residual_after_twice"] == 0
                    if unique_effect_carried_axis["effect_carried_reflection_terminal_candidate_count"] == 1
                    else None
                ),
                "unique_effect_carried_reflection_exact_remaining_action_count": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_exact_remaining_action_count"
                    ]
                ),
                "unique_effect_carried_reflection_next_action_would_overshoot": (
                    unique_effect_carried_axis[
                        "effect_carried_reflection_next_action_would_overshoot"
                    ]
                ),
            }
        )
    neutral_bindings.update(_closed_support_permutation_facts(value))
    if stationary_value_transitions:
        neutral_bindings["stationary_value_transition_pair_digest"] = stable_digest(
            tuple(
                sorted(
                    (
                        after.entity_id,
                        before.component.value,
                        after.component.value,
                    )
                    for before, after in stationary_value_transitions
                )
            )
        )[:16]
    if stationary_appearance_transitions or exclusive_internal_state_permutation:
        neutral_bindings["stationary_appearance_transition_pair_digest"] = (
            stable_digest(
                tuple(internal_state_transitions)
                if exclusive_internal_state_permutation
                else tuple(
                    sorted(
                        (
                            after.entity_id,
                            stable_digest(
                                (
                                    before.component.value,
                                    before.component.relative_pixels,
                                )
                            )[:16],
                            stable_digest(
                                (
                                    after.component.value,
                                    after.component.relative_pixels,
                                )
                            )[:16],
                        )
                        for before, after in stationary_appearance_transitions
                    )
                )
            )[:16]
        )
    midpoint_evidence_refs: tuple[str, ...] = ()
    if len(midpoint_geometry) == 1 and not midpoint_enumeration_truncated:
        aliases = tuple(sorted(midpoint_aliases[midpoint_geometry[0]]))
        full_ref, half_ref, fixed_ref = aliases[0]
        midpoint_evidence_refs = (full_ref, half_ref, fixed_ref)
        relation_digest = stable_digest((full_ref, half_ref, fixed_ref))[:16]
        centers = midpoint_geometry[0]
        before_midpoint_residual = tuple(
            2 * centers[1][axis] - centers[0][axis] - centers[2][axis]
            for axis in (0, 1)
        )
        after_midpoint_residual = tuple(
            2 * centers[4][axis] - centers[3][axis] - centers[5][axis]
            for axis in (0, 1)
        )
        full_translation = after_by_entity[full_ref]
        half_translation = after_by_entity[half_ref]
        neutral_bindings.update(
            {
                "unique_midpoint_full_translation_entity_ref": full_ref,
                "unique_midpoint_derived_translation_entity_ref": half_ref,
                "unique_midpoint_fixed_entity_ref": fixed_ref,
                "unique_midpoint_relation_digest": relation_digest,
                "midpoint_transition_alias_count": len(aliases),
                "midpoint_equation_exact_before_and_after": all(
                    value == 0
                    for value in (*before_midpoint_residual, *after_midpoint_residual)
                ),
                "midpoint_equation_nearest_lattice_before_and_after": True,
                "midpoint_lattice_residual_max_twice": max(
                    abs(value)
                    for value in (*before_midpoint_residual, *after_midpoint_residual)
                ),
                "full_delta_is_exactly_twice_derived_delta": (
                    full_translation.delta_row == 2 * half_translation.delta_row
                    and full_translation.delta_col == 2 * half_translation.delta_col
                ),
                "full_delta_matches_nearest_lattice_half_delta": True,
            }
        )
        derived_after = after_by_entity[half_ref].component
        if any(
            isinstance(scene, BlockGridSceneDescription)
            for scene in (value.before_scene, value.after_scene)
        ):
            raise RuntimeError(
                "stationary zone enclosure measurement requires a measured zone scene"
            )
        enclosure_descriptions, enclosure_truncated = (
            _stationary_empty_center_quartets(
                stationary=stationary_entities,
                derived_value=derived_after.value,
                frame=value.after_scene.frame,
            )
        )
        before_zone_by_exact = {
            (component.value, component.pixels): component
            for component in value.before_scene.zones.components
        }
        stationary_zone_entities = tuple(
            TrackedComponent(
                entity_id=f"zone_exact.{stable_digest((component.value, component.pixels))[:16]}",
                component=component,
                first_seen_frame_ref=value.before_tracking.frame_ref,
                current_frame_ref=value.after_tracking.frame_ref,
                observation_count=2,
                identity_status="established",
                match_kind="stationary_exact",
                previous_component_ref=before_zone_by_exact[
                    (component.value, component.pixels)
                ].component_id,
            )
            for component in value.after_scene.zones.components
            if (component.value, component.pixels) in before_zone_by_exact
        )
        zone_enclosure_descriptions, zone_enclosure_truncated = (
            _stationary_empty_center_quartets(
                stationary=stationary_zone_entities,
                derived_value=derived_after.value,
                frame=value.after_scene.frame,
            )
        )
        neutral_bindings.update(
            {
                "stationary_empty_center_quartet_count": len(
                    enclosure_descriptions
                ),
                "stationary_empty_center_quartet_enumeration_truncated": (
                    enclosure_truncated
                ),
                "stationary_zone_empty_center_quartet_count": len(
                    zone_enclosure_descriptions
                ),
                "stationary_zone_empty_center_quartet_enumeration_truncated": (
                    zone_enclosure_truncated
                ),
            }
        )
        if len(zone_enclosure_descriptions) == 1 and not zone_enclosure_truncated:
            zone_center_twice = zone_enclosure_descriptions[0][0]
            neutral_bindings.update(
                {
                    "unique_zone_enclosure_center_row_twice": zone_center_twice[0],
                    "unique_zone_enclosure_center_column_twice": zone_center_twice[1],
                }
            )
        if len(enclosure_descriptions) == 1 and not enclosure_truncated:
            center_twice, _corners, member_aliases, signature = (
                enclosure_descriptions[0]
            )
            member_refs = tuple(
                member_ref
                for aliases_at_corner in member_aliases
                for member_ref in aliases_at_corner
            )
            enclosure_digest = stable_digest(
                (center_twice, member_refs, signature)
            )[:16]
            neutral_bindings.update(
                {
                    "unique_enclosure_center_row_twice": center_twice[0],
                    "unique_enclosure_center_column_twice": center_twice[1],
                    "unique_enclosure_member_entity_refs": member_refs,
                    "unique_enclosure_geometry_digest": enclosure_digest,
                    "enclosure_center_is_integral_and_unoccupied": True,
                    "derived_value_matches_enclosure_member_value": True,
                }
            )

            action_point = _candidate_point(value.candidate_ref)
            if action_point is not None:
                full_after_center = _center_twice(
                    after_by_entity[full_ref].component
                )
                fixed_after_center = _center_twice(
                    after_by_entity[fixed_ref].component
                )
                offset_twice = (
                    full_after_center[0] - 2 * action_point[0],
                    full_after_center[1] - 2 * action_point[1],
                )
                required_full_center_twice = (
                    2 * center_twice[0] - fixed_after_center[0],
                    2 * center_twice[1] - fixed_after_center[1],
                )
                action_point_twice = (
                    required_full_center_twice[0] - offset_twice[0],
                    required_full_center_twice[1] - offset_twice[1],
                )
                neutral_bindings.update(
                    {
                        "action_point_to_full_center_offset_row_twice": (
                            offset_twice[0]
                        ),
                        "action_point_to_full_center_offset_column_twice": (
                            offset_twice[1]
                        ),
                        "action_point_to_full_center_offset_measured": True,
                        "required_full_entity_center_row_twice": (
                            required_full_center_twice[0]
                        ),
                        "required_full_entity_center_column_twice": (
                            required_full_center_twice[1]
                        ),
                    }
                )
                if action_point_twice[0] % 2 == 0 and action_point_twice[1] % 2 == 0:
                    calibrated_point = (
                        action_point_twice[0] // 2,
                        action_point_twice[1] // 2,
                    )
                    destination_components = tuple(
                        component
                        for component in value.after_scene.blocks.components
                        if calibrated_point in frozenset(component.pixels)
                    )
                    if len(destination_components) == 1:
                        destination = destination_components[0]
                        horizontal_clearance, vertical_clearance = (
                            _same_component_axis_clearance(
                                destination,
                                calibrated_point,
                            )
                        )
                        reserved_width, reserved_height = (
                            _reserved_nonfrequent_sprite_extents(
                                value.after_scene
                            )
                        )
                        neutral_bindings.update(
                            {
                                "unique_calibrated_action_point_row": (
                                    calibrated_point[0]
                                ),
                                "unique_calibrated_action_point_column": (
                                    calibrated_point[1]
                                ),
                                "unique_calibrated_action_surface_component_ref": (
                                    destination.component_id
                                ),
                                "calibrated_action_horizontal_clearance": (
                                    horizontal_clearance
                                ),
                                "calibrated_action_vertical_clearance": (
                                    vertical_clearance
                                ),
                                "calibrated_action_reserved_width": reserved_width,
                                "calibrated_action_reserved_height": reserved_height,
                                "calibrated_action_horizontal_clearance_covers_reserve": bool(
                                    reserved_width
                                    and horizontal_clearance >= reserved_width
                                ),
                                "calibrated_action_vertical_clearance_covers_reserve": bool(
                                    reserved_height
                                    and vertical_clearance >= reserved_height
                                ),
                            }
                        )
        footprint_clear_enclosures: list[tuple[object, ...]] = []
        reciprocal_footprint_clear_enclosures: list[tuple[object, ...]] = []
        zone_reciprocal_footprint_clear_enclosures: list[tuple[object, ...]] = []
        symmetric_affine_witnesses: list[FrozenMap] = []
        affine_nonintegral_destination_count = 0
        affine_destination_component_mismatch_count = 0
        affine_horizontal_clearance_shortfall_count = 0
        affine_vertical_clearance_shortfall_count = 0
        if not enclosure_truncated:
            action_point = _candidate_point(value.candidate_ref)
            if action_point is not None:
                full_after_center = _center_twice(
                    after_by_entity[full_ref].component
                )
                fixed_after_center = _center_twice(
                    after_by_entity[fixed_ref].component
                )
                offset_twice = (
                    full_after_center[0] - 2 * action_point[0],
                    full_after_center[1] - 2 * action_point[1],
                )
                reserved_width, reserved_height = (
                    _reserved_nonfrequent_sprite_extents(value.after_scene)
                )
                neutral_bindings.update(
                    {
                        "action_point_to_full_center_offset_row_twice": offset_twice[0],
                        "action_point_to_full_center_offset_column_twice": offset_twice[1],
                        "action_point_to_full_center_offset_measured": True,
                        "affine_destination_reserved_width": reserved_width,
                        "affine_destination_reserved_height": reserved_height,
                    }
                )
                if (
                    len(zone_enclosure_descriptions) == 1
                    and not zone_enclosure_truncated
                    and fixed_after_center[0] % 2 == 0
                    and fixed_after_center[1] % 2 == 0
                ):
                    target_center_twice = zone_enclosure_descriptions[0][0]
                    selector_point = (
                        fixed_after_center[0] // 2,
                        fixed_after_center[1] // 2,
                    )
                    structural_directions = (
                        ("horizontal", ((0, 1),)),
                        ("vertical", ((1, 0),)),
                        ("diagonal", ((1, 1), (1, -1))),
                    )
                    maximum_magnitude = max(
                        value.after_scene.frame.height,
                        value.after_scene.frame.width,
                    )
                    for structure_kind, directions in structural_directions:
                        witness = None
                        for magnitude in range(1, maximum_magnitude + 1):
                            for direction_row, direction_col in directions:
                                first_center_twice = (
                                    target_center_twice[0]
                                    + 2 * magnitude * direction_row,
                                    target_center_twice[1]
                                    + 2 * magnitude * direction_col,
                                )
                                reflected_center_twice = (
                                    target_center_twice[0]
                                    - 2 * magnitude * direction_row,
                                    target_center_twice[1]
                                    - 2 * magnitude * direction_col,
                                )
                                if first_center_twice == full_after_center:
                                    continue
                                first_point_twice = (
                                    first_center_twice[0] - offset_twice[0],
                                    first_center_twice[1] - offset_twice[1],
                                )
                                reflected_point_twice = (
                                    reflected_center_twice[0] - offset_twice[0],
                                    reflected_center_twice[1] - offset_twice[1],
                                )
                                if any(
                                    coordinate % 2
                                    for coordinate in (
                                        *first_point_twice,
                                        *reflected_point_twice,
                                    )
                                ):
                                    continue
                                first_point = (
                                    first_point_twice[0] // 2,
                                    first_point_twice[1] // 2,
                                )
                                reflected_point = (
                                    reflected_point_twice[0] // 2,
                                    reflected_point_twice[1] // 2,
                                )
                                destination_rows: list[tuple[object, int, int]] = []
                                for point in (first_point, reflected_point):
                                    destinations = tuple(
                                        component
                                        for component in value.after_scene.blocks.components
                                        if point in frozenset(component.pixels)
                                    )
                                    if len(destinations) != 1:
                                        break
                                    horizontal, vertical = _same_component_axis_clearance(
                                        destinations[0], point
                                    )
                                    if not (
                                        reserved_width
                                        and reserved_height
                                        and horizontal >= reserved_width
                                        and vertical >= reserved_height
                                    ):
                                        break
                                    destination_rows.append(
                                        (destinations[0], horizontal, vertical)
                                    )
                                if len(destination_rows) != 2:
                                    continue
                                witness = FrozenMap(
                                    {
                                        "structure_kind": structure_kind,
                                        "offset_magnitude": magnitude,
                                        "first_action_point": first_point,
                                        "selector_point": selector_point,
                                        "reflected_action_point": reflected_point,
                                        "first_surface_component_ref": destination_rows[0][0].component_id,
                                        "selector_component_ref": after_by_entity[
                                            fixed_ref
                                        ].component.component_id,
                                        "reflected_surface_component_ref": destination_rows[1][0].component_id,
                                        "reserved_width": reserved_width,
                                        "reserved_height": reserved_height,
                                    }
                                )
                                break
                            if witness is not None:
                                break
                        if witness is not None:
                            symmetric_affine_witnesses.append(witness)
                for description in enclosure_descriptions:
                    center_twice, corners, member_aliases, signature = description
                    required_full_center_twice = (
                        2 * center_twice[0] - fixed_after_center[0],
                        2 * center_twice[1] - fixed_after_center[1],
                    )
                    action_point_twice = (
                        required_full_center_twice[0] - offset_twice[0],
                        required_full_center_twice[1] - offset_twice[1],
                    )
                    if action_point_twice[0] % 2 or action_point_twice[1] % 2:
                        affine_nonintegral_destination_count += 1
                        continue
                    calibrated_point = (
                        action_point_twice[0] // 2,
                        action_point_twice[1] // 2,
                    )
                    destinations = tuple(
                        component
                        for component in value.after_scene.blocks.components
                        if calibrated_point in frozenset(component.pixels)
                    )
                    if len(destinations) != 1:
                        affine_destination_component_mismatch_count += 1
                        continue
                    destination = destinations[0]
                    horizontal_clearance, vertical_clearance = (
                        _same_component_axis_clearance(destination, calibrated_point)
                    )
                    horizontal_shortfall = not (
                        reserved_width and horizontal_clearance >= reserved_width
                    )
                    vertical_shortfall = not (
                        reserved_height and vertical_clearance >= reserved_height
                    )
                    affine_horizontal_clearance_shortfall_count += int(
                        horizontal_shortfall
                    )
                    affine_vertical_clearance_shortfall_count += int(
                        vertical_shortfall
                    )
                    if horizontal_shortfall or vertical_shortfall:
                        continue
                    footprint_clear_enclosures.append(
                        (
                            center_twice,
                            corners,
                            member_aliases,
                            signature,
                            required_full_center_twice,
                            calibrated_point,
                            destination.component_id,
                            horizontal_clearance,
                            vertical_clearance,
                            reserved_width,
                            reserved_height,
                            offset_twice,
                        )
                    )
                if fixed_after_center[0] % 2 == 0 and fixed_after_center[1] % 2 == 0:
                    reciprocal_selector_point = (
                        fixed_after_center[0] // 2,
                        fixed_after_center[1] // 2,
                    )
                    for description in enclosure_descriptions:
                        center_twice, corners, member_aliases, signature = description
                        required_fixed_center_twice = (
                            2 * center_twice[0] - full_after_center[0],
                            2 * center_twice[1] - full_after_center[1],
                        )
                        reciprocal_point_twice = (
                            required_fixed_center_twice[0] - offset_twice[0],
                            required_fixed_center_twice[1] - offset_twice[1],
                        )
                        if reciprocal_point_twice[0] % 2 or reciprocal_point_twice[1] % 2:
                            continue
                        reciprocal_point = (
                            reciprocal_point_twice[0] // 2,
                            reciprocal_point_twice[1] // 2,
                        )
                        destinations = tuple(
                            component
                            for component in value.after_scene.blocks.components
                            if reciprocal_point in frozenset(component.pixels)
                        )
                        if len(destinations) != 1:
                            continue
                        destination = destinations[0]
                        horizontal_clearance, vertical_clearance = (
                            _same_component_axis_clearance(
                                destination, reciprocal_point
                            )
                        )
                        if not (
                            reserved_width
                            and reserved_height
                            and horizontal_clearance >= reserved_width
                            and vertical_clearance >= reserved_height
                        ):
                            continue
                        reciprocal_footprint_clear_enclosures.append(
                            (
                                center_twice,
                                corners,
                                member_aliases,
                                signature,
                                required_fixed_center_twice,
                                reciprocal_selector_point,
                                reciprocal_point,
                            )
                        )
                    for description in zone_enclosure_descriptions:
                        center_twice, corners, member_aliases, signature = description
                        required_fixed_center_twice = (
                            2 * center_twice[0] - full_after_center[0],
                            2 * center_twice[1] - full_after_center[1],
                        )
                        reciprocal_point_twice = (
                            required_fixed_center_twice[0] - offset_twice[0],
                            required_fixed_center_twice[1] - offset_twice[1],
                        )
                        if reciprocal_point_twice[0] % 2 or reciprocal_point_twice[1] % 2:
                            continue
                        reciprocal_point = (
                            reciprocal_point_twice[0] // 2,
                            reciprocal_point_twice[1] // 2,
                        )
                        destinations = tuple(
                            component
                            for component in value.after_scene.blocks.components
                            if reciprocal_point in frozenset(component.pixels)
                        )
                        if len(destinations) != 1:
                            continue
                        destination = destinations[0]
                        horizontal_clearance, vertical_clearance = (
                            _same_component_axis_clearance(
                                destination, reciprocal_point
                            )
                        )
                        if not (
                            reserved_width
                            and reserved_height
                            and horizontal_clearance >= reserved_width
                            and vertical_clearance >= reserved_height
                        ):
                            continue
                        zone_reciprocal_footprint_clear_enclosures.append(
                            (
                                center_twice,
                                corners,
                                member_aliases,
                                signature,
                                required_fixed_center_twice,
                                reciprocal_selector_point,
                                reciprocal_point,
                            )
                        )
        neutral_bindings["footprint_clear_affine_destination_count"] = len(
            footprint_clear_enclosures
        )
        neutral_bindings["bounded_symmetric_affine_witness_count"] = len(
            symmetric_affine_witnesses
        )
        neutral_bindings["bounded_symmetric_affine_witnesses"] = tuple(
            symmetric_affine_witnesses[:3]
        )
        neutral_bindings["bounded_symmetric_affine_global_limit"] = 3
        neutral_bindings["reciprocal_footprint_clear_affine_destination_count"] = len(
            reciprocal_footprint_clear_enclosures
        )
        neutral_bindings[
            "zone_reciprocal_footprint_clear_affine_destination_count"
        ] = len(zone_reciprocal_footprint_clear_enclosures)
        neutral_bindings.update(
            {
                "affine_nonintegral_destination_count": (
                    affine_nonintegral_destination_count
                ),
                "affine_destination_component_mismatch_count": (
                    affine_destination_component_mismatch_count
                ),
                "affine_horizontal_clearance_shortfall_count": (
                    affine_horizontal_clearance_shortfall_count
                ),
                "affine_vertical_clearance_shortfall_count": (
                    affine_vertical_clearance_shortfall_count
                ),
            }
        )
        if len(footprint_clear_enclosures) == 1:
            (
                center_twice,
                _corners,
                member_aliases,
                signature,
                required_full_center_twice,
                calibrated_point,
                destination_ref,
                horizontal_clearance,
                vertical_clearance,
                reserved_width,
                reserved_height,
                offset_twice,
            ) = footprint_clear_enclosures[0]
            member_refs = tuple(
                member_ref
                for aliases_at_corner in member_aliases
                for member_ref in aliases_at_corner
            )
            enclosure_digest = stable_digest(
                (center_twice, member_refs, signature)
            )[:16]
            neutral_bindings.update(
                {
                    "unique_enclosure_center_row_twice": center_twice[0],
                    "unique_enclosure_center_column_twice": center_twice[1],
                    "unique_enclosure_member_entity_refs": member_refs,
                    "unique_enclosure_geometry_digest": enclosure_digest,
                    "enclosure_center_is_integral_and_unoccupied": True,
                    "derived_value_matches_enclosure_member_value": True,
                    "action_point_to_full_center_offset_row_twice": offset_twice[0],
                    "action_point_to_full_center_offset_column_twice": offset_twice[1],
                    "action_point_to_full_center_offset_measured": True,
                    "required_full_entity_center_row_twice": required_full_center_twice[0],
                    "required_full_entity_center_column_twice": required_full_center_twice[1],
                    "unique_calibrated_action_point_row": calibrated_point[0],
                    "unique_calibrated_action_point_column": calibrated_point[1],
                    "unique_calibrated_action_surface_component_ref": destination_ref,
                    "calibrated_action_horizontal_clearance": horizontal_clearance,
                    "calibrated_action_vertical_clearance": vertical_clearance,
                    "calibrated_action_reserved_width": reserved_width,
                    "calibrated_action_reserved_height": reserved_height,
                    "calibrated_action_horizontal_clearance_covers_reserve": True,
                    "calibrated_action_vertical_clearance_covers_reserve": True,
                }
            )
        if len(reciprocal_footprint_clear_enclosures) == 1:
            (
                _center_twice_value,
                _corners,
                _member_aliases,
                _signature,
                required_fixed_center_twice,
                reciprocal_selector_point,
                reciprocal_point,
            ) = reciprocal_footprint_clear_enclosures[0]
            neutral_bindings.update(
                {
                    "unique_reciprocal_affine_entity_ref": fixed_ref,
                    "unique_reciprocal_selector_point_row": reciprocal_selector_point[0],
                    "unique_reciprocal_selector_point_column": reciprocal_selector_point[1],
                    "unique_reciprocal_required_center_row_twice": required_fixed_center_twice[0],
                    "unique_reciprocal_required_center_column_twice": required_fixed_center_twice[1],
                    "unique_reciprocal_action_point_row": reciprocal_point[0],
                    "unique_reciprocal_action_point_column": reciprocal_point[1],
                }
            )
        if len(zone_reciprocal_footprint_clear_enclosures) == 1:
            (
                zone_center_twice,
                _corners,
                zone_member_aliases,
                zone_signature,
                required_fixed_center_twice,
                reciprocal_selector_point,
                reciprocal_point,
            ) = zone_reciprocal_footprint_clear_enclosures[0]
            zone_member_refs = tuple(
                member_ref
                for aliases_at_corner in zone_member_aliases
                for member_ref in aliases_at_corner
            )
            neutral_bindings.update(
                {
                    "unique_zone_enclosure_geometry_digest": stable_digest(
                        (zone_center_twice, zone_member_refs, zone_signature)
                    )[:16],
                    "unique_zone_reciprocal_affine_entity_ref": fixed_ref,
                    "unique_zone_reciprocal_selector_point_row": reciprocal_selector_point[0],
                    "unique_zone_reciprocal_selector_point_column": reciprocal_selector_point[1],
                    "unique_zone_reciprocal_required_center_row_twice": required_fixed_center_twice[0],
                    "unique_zone_reciprocal_required_center_column_twice": required_fixed_center_twice[1],
                    "unique_zone_reciprocal_action_point_row": reciprocal_point[0],
                    "unique_zone_reciprocal_action_point_column": reciprocal_point[1],
                }
            )
    if len(nary_affine_rows) == 1 and not nary_affine_enumeration_truncated:
        unique_nary_affine = nary_affine_rows[0]
        neutral_bindings.update(
            {
                "unique_nary_affine_relation_digest": unique_nary_affine["digest"],
                "unique_nary_affine_derived_entity_ref": unique_nary_affine[
                    "derived_entity_ref"
                ],
                "unique_nary_affine_contributor_entity_refs": unique_nary_affine[
                    "contributor_entity_refs"
                ],
                "unique_nary_affine_active_contributor_entity_ref": unique_nary_affine[
                    "active_contributor_entity_ref"
                ],
                "unique_nary_affine_coefficient_numerators": unique_nary_affine[
                    "coefficient_numerators"
                ],
                "unique_nary_affine_coefficient_denominator": unique_nary_affine[
                    "coefficient_denominator"
                ],
                "unique_nary_affine_before_residual_numerators": unique_nary_affine[
                    "before_residual_numerators"
                ],
                "unique_nary_affine_after_residual_numerators": unique_nary_affine[
                    "after_residual_numerators"
                ],
                "unique_nary_affine_discriminating_transition": True,
            }
        )
    if len(translation_deltas) == 1:
        delta_row, delta_col = next(iter(translation_deltas))
        neutral_bindings["unique_translation_delta_row"] = delta_row
        neutral_bindings["unique_translation_delta_col"] = delta_col
        neutral_bindings["action_delta_frame"] = "shared_translation"
        if delta_row == 0 and delta_col != 0:
            neutral_bindings["unique_translation_quantum"] = abs(delta_col)
        elif delta_col == 0 and delta_row != 0:
            neutral_bindings["unique_translation_quantum"] = abs(delta_row)
        elif delta_row != 0 or delta_col != 0:
            # Diagonal motion: report the dominant axis length only as a
            # descriptive measurement, never as a forced mechanism law.
            neutral_bindings["unique_translation_quantum"] = max(
                abs(delta_row), abs(delta_col)
            )
        if len(translations) == 1:
            unique_translation = translations[0]
            unique_component = before_by_entity[
                unique_translation.entity_ref
            ].component
            unique_area = len(unique_component.relative_pixels)
            neutral_bindings["unique_translated_entity_ref"] = (
                unique_translation.entity_ref
            )
            neutral_bindings["unique_translated_entity_area"] = unique_area
            neutral_bindings["unique_translated_shape_digest"] = stable_digest(
                unique_component.relative_pixels
            )[:16]
            if unique_area == 1:
                neutral_bindings["single_cell_rigid_translation_observed"] = True
            elif unique_area > 1:
                neutral_bindings["multicell_rigid_translation_observed"] = True
    else:
        # Coupled opposite translations (e.g. reflection): same magnitude on one
        # axis with opposite signs still yields a shared movement quantum.
        axis_mags: list[int] = []
        pure_axis = True
        for delta_row, delta_col in translation_deltas:
            if delta_row == 0 and delta_col != 0:
                axis_mags.append(abs(delta_col))
            elif delta_col == 0 and delta_row != 0:
                axis_mags.append(abs(delta_row))
            else:
                pure_axis = False
                break
        if pure_axis and axis_mags and len(set(axis_mags)) == 1:
            neutral_bindings["unique_translation_quantum"] = axis_mags[0]
            neutral_bindings["coupled_opposite_translation_observed"] = True
            # Residual control needs a signed pure-axis delta for this action.
            # Prefer the alignment residual entity (moving peer) so residual BFS
            # works in residual-entity frame with coupling +1. Reflection is
            # already encoded in that entity's observed delta.
            residual_delta = None
            if unique_alignment is not None:
                moving_after = after_by_entity.get(
                    unique_alignment.translated_entity_ref
                )
                if moving_after is not None and (
                    moving_after.delta_row != 0 or moving_after.delta_col != 0
                ):
                    residual_delta = (moving_after.delta_row, moving_after.delta_col)
                    neutral_bindings["action_delta_frame"] = "alignment_mover"
            if residual_delta is None and translations:
                # Fallback: deterministic pure-axis representative (controller frame).
                first = min(
                    translations,
                    key=lambda item: (
                        item.entity_ref,
                        item.delta_row,
                        item.delta_col,
                    ),
                )
                residual_delta = (first.delta_row, first.delta_col)
                neutral_bindings["action_delta_frame"] = "controller_or_unspecified"
            if residual_delta is not None:
                neutral_bindings["unique_translation_delta_row"] = residual_delta[0]
                neutral_bindings["unique_translation_delta_col"] = residual_delta[1]
    pre_control_bboxes = tuple(value.pre_action_input_aligned_entity_bboxes)
    post_control_bboxes = tuple(value.post_action_input_aligned_entity_bboxes)
    if len(pre_control_bboxes) == 1 and len(post_control_bboxes) == 1:
        pre_control = pre_control_bboxes[0]
        post_control = post_control_bboxes[0]
        control_delta = (
            post_control.top - pre_control.top,
            post_control.left - pre_control.left,
        )
        same_extent = bool(
            pre_control.height == post_control.height
            and pre_control.width == post_control.width
        )
        expected_axis = (
            normalized_axis_direction(expected_input_delta)
            if expected_input_delta is not None
            else None
        )
        observed_quantum = int(neutral_bindings.get("unique_translation_quantum") or 0)
        if (
            same_extent
            and expected_axis is not None
            and normalized_axis_direction(control_delta) == expected_axis
            and max(abs(control_delta[0]), abs(control_delta[1])) > observed_quantum > 0
        ):
            neutral_bindings.update(
                {
                    "input_aligned_control_geometry_nonlocal_translation_observed": True,
                    "input_aligned_control_geometry_delta_row": control_delta[0],
                    "input_aligned_control_geometry_delta_col": control_delta[1],
                }
            )
    if co_moving:
        neutral_bindings["co_moving_entity_refs"] = co_moving
        if len(co_moving) > 1 and "co_translation_group_digest" not in neutral_bindings:
            neutral_bindings["co_translation_group_digest"] = stable_digest(
                tuple(sorted(co_moving))
            )[:12]
    if unique_exchange is not None:
        exchange_pair_digest = stable_digest(
            tuple(
                sorted(
                    (
                        unique_exchange.source_entity_ref,
                        unique_exchange.destination_entity_ref,
                    )
                )
            )
        )[:12]
        neutral_bindings.update(
            {
                "exchange_axis": unique_exchange.axis,
                "exchange_quantum": unique_exchange.quantum,
                "exchange_source_entity_ref": unique_exchange.source_entity_ref,
                "exchange_destination_entity_ref": (
                    unique_exchange.destination_entity_ref
                ),
                "aggregate_quantity_before": (
                    unique_exchange.aggregate_quantity_before
                ),
                "aggregate_quantity_after": unique_exchange.aggregate_quantity_after,
                "aggregate_conserved": (
                    unique_exchange.aggregate_quantity_before
                    == unique_exchange.aggregate_quantity_after
                ),
                "exchange_pair_digest": exchange_pair_digest,
                "boundary_assembly_digest": stable_digest(
                    separators or ("unresolved",)
                )[:12],
                "boundary_component_refs": separators,
                "local_neighborhood_digest": stable_digest(
                    (clicked_entity_ref, interfaces)
                )[:12],
                "local_neighborhood_component_refs": interfaces,
            }
        )
    if unique_alignment is not None:
        moving_before_component = before_by_entity[
            unique_alignment.translated_entity_ref
        ].component
        moving_after_component = after_by_entity[
            unique_alignment.translated_entity_ref
        ].component
        holed_envelopes = tuple(
            component
            for component in value.before_scene.blocks.components
            if component.bbox.height == component.bbox.width
            and component.bbox.height % 2 == 1
            and component.area
            == component.bbox.height * component.bbox.width - 1
        )
        envelope_shape_counts: dict[tuple[object, ...], int] = {}
        for component in holed_envelopes:
            signature = (component.relative_pixels, component.area)
            envelope_shape_counts[signature] = (
                envelope_shape_counts.get(signature, 0) + 1
            )
        def inside_repeated_envelope(component: ComponentDescription) -> bool:
            point = (
                (component.bbox.top + component.bbox.bottom) // 2,
                (component.bbox.left + component.bbox.right) // 2,
            )
            return any(
                envelope_shape_counts.get(
                    (envelope.relative_pixels, envelope.area), 0
                )
                >= 2
                and envelope.bbox.top < point[0] < envelope.bbox.bottom
                and envelope.bbox.left < point[1] < envelope.bbox.right
                and point not in envelope.pixels
                for envelope in holed_envelopes
            )
        translated_peer_is_internal_state_marker = bool(
            moving_before_component.area == 1
            and moving_after_component.area == 1
            and inside_repeated_envelope(moving_before_component)
            and inside_repeated_envelope(moving_after_component)
        )
        fixed_alignment_component = after_by_entity[
            unique_alignment.stationary_peer_entity_ref
        ].component
        translated_alignment_pair_is_exact_multicell_shape_peer = bool(
            moving_before_component.area > 1
            and moving_after_component.area > 1
            and fixed_alignment_component.area > 1
            and moving_after_component.relative_pixels
            == fixed_alignment_component.relative_pixels
        )
        neutral_bindings.update(
            {
                "unique_alignment_peer_bound": True,
                "translated_peer_entity_ref": (
                    unique_alignment.translated_entity_ref
                ),
                "stationary_peer_entity_ref": (
                    unique_alignment.stationary_peer_entity_ref
                ),
                "peer_axis": unique_alignment.axis,
                "peer_distance_change_kind": unique_alignment.distance_change_kind,
                "peer_distance_before_twice": (
                    unique_alignment.before_distance_twice
                ),
                "peer_distance_after_twice": (
                    unique_alignment.after_distance_twice
                ),
                "translated_peer_is_internal_state_marker": (
                    translated_peer_is_internal_state_marker
                ),
                "translated_alignment_pair_is_exact_multicell_shape_peer": (
                    translated_alignment_pair_is_exact_multicell_shape_peer
                ),
                "co_moving_entity_refs": co_moving,
                "peer_pair_digest": stable_digest(
                    tuple(
                        sorted(
                            (
                                unique_alignment.translated_entity_ref,
                                unique_alignment.stationary_peer_entity_ref,
                            )
                        )
                    )
                )[:12],
            }
        )
        if len(co_moving) > 1:
            # Digest is a pure measurement over co-moving identities. Contact
            # between co-movers is optional; non-touching co-movers still form
            # an exact co-translation group for later DRM projection.
            neutral_bindings["co_translation_group_digest"] = stable_digest(
                tuple(sorted(co_moving))
            )[:12]
        if co_moving_partner is not None:
            neutral_bindings["co_moving_partner_entity_ref"] = co_moving_partner

    neutral_bindings.update(
        _stationary_inverse_value_axis_focus_measurement(
            stationary_value_transitions,
            pre_action_input_aligned_bboxes=tuple(
                value.pre_action_input_aligned_entity_bboxes
            ),
        )
    )
    lattice_rows = tuple(row for row in value.canonical_relational_lattice_term_facts
        if row.get("current_frame_ref") == value.before_tracking.frame_ref)
    neutral_bindings["canonical_current_relational_lattice_count"] = len(lattice_rows)
    if len(lattice_rows) <= 1:
        unique_geometry = lattice_rows[0].get("current_lattice_geometry") if lattice_rows else None
        neutral_bindings.update(_co_translated_cell_carrier_measurements(
            before_scene=value.before_scene,
            after_scene=value.after_scene,
            before_by_entity=before_by_entity,
            after_by_entity=after_by_entity,
            co_moving_entity_refs=co_moving,
            retained_grid_geometry=unique_geometry,
        ))
    neutral_bindings.update(
        _periodic_cell_transition_measurements(
            before_scene=value.before_scene,
            after_scene=value.after_scene,
            candidate_ref=value.candidate_ref,
            action_ref=value.action_ref,
            canonical_periodic_transition_claim_facts=(
                value.canonical_periodic_transition_claim_facts
            ),
            declared_cellular_mover_pattern_refs=(
                value.declared_cellular_mover_pattern_refs
            ),
            executed_action_axis_delta=(
                normalized_axis_direction(expected_input_delta)
                if expected_input_delta is not None
                else None
            ),
        )
    )
    neutral_bindings.update(
        _translated_peer_cell_pattern_measurements(
            before_scene=value.before_scene,
            after_scene=value.after_scene,
            before_by_entity=before_by_entity,
            after_by_entity=after_by_entity,
            translations=translations,
        )
    )
    neutral_bindings.update(contact_translation_measurements)
    from .contact_raster import measure_stationary_control_contact_rasters
    neutral_bindings.update(measure_stationary_control_contact_rasters(
        before=value.before_scene.frame, after=value.after_scene.frame,
        components=tuple(item.component for item in value.after_tracking.entities),
        translations=translations, before_entities=before_by_entity,
        after_entities=after_by_entity,
        control_boxes=tuple(value.pre_action_input_aligned_entity_bboxes),
        expected_delta=expected_input_delta,
        control_bindings=value.canonical_provisional_control_term_facts,
        prior_translation_facts=value.canonical_action_translation_claim_facts))
    neutral_bindings.update(
        measure_left_to_right_outlined_successor(
            value.before_scene.frame,
            value.after_scene.frame,
        )
    )
    # Exact ordered action-event measurements are transported from the
    # interaction reconciliation.  They remain neutral facts here; DRM alone
    # decides whether they support a local-attractor interpretation.
    ordered_point_facts = value.ordered_point_trajectory.to_dict()
    after_component_ref = ordered_point_facts.get(
        "unique_point_approach_after_component_ref"
    )
    matching_current_entities = tuple(
        item.entity_id
        for item in value.after_tracking.entities
        if item.component.component_id == after_component_ref
    )
    ordered_point_facts["ordered_point_current_entity_ref_present"] = (
        len(matching_current_entities) == 1
    )
    if len(matching_current_entities) == 1:
        ordered_point_facts["ordered_point_current_entity_ref"] = (
            matching_current_entities[0]
        )
    neutral_bindings.update(ordered_point_facts)

    evidence_refs = tuple(
        dict.fromkeys(
            (
                f"evidence.transition:{value.transition_ref}",
                *((clicked_entity_ref,) if clicked_entity_ref else ()),
                *(item.entity_ref for item in extents),
                *(item.entity_ref for item in translations),
                *(
                    after.entity_id
                    for _before, after in stationary_value_transitions
                ),
                *midpoint_evidence_refs,
                *actuator_premise_refs,
            )
        )
    )
    return ControlledTransitionAnalysis(
        descriptive_facts=FrozenMap(neutral_bindings),
        evidence_refs=evidence_refs,
        transition_ref=value.transition_ref,
        candidate_ref=value.candidate_ref,
        clicked_entity_ref=clicked_entity_ref,
        extent_transitions=extents,
        exchange_candidates=exchanges,
        translations=translations,
        peer_distance_change_candidates=alignment,
        boundary_component_refs=separators,
        local_neighborhood_component_refs=interfaces,
    )


def _stationary_thin_connector_transition_measurements(
    *,
    before_scene: object,
    stationary_value_transitions: tuple[tuple[TrackedComponent, TrackedComponent], ...],
    transition_ref: str,
) -> tuple[FrozenMap, ...]:
    """Measure value-changing thin paths that touch distinct endpoint components.

    This is intentionally role-neutral.  A continuous one-cell-wide path and
    its endpoint incidence are raster facts; DRM decides whether the path is a
    connector and whether its action-conditioned value change means activation.
    """

    before_components = tuple(before_scene.blocks.components)
    records: list[FrozenMap] = []
    for before, after in stationary_value_transitions:
        pixels = frozenset(before.component.pixels)
        if (
            len(pixels) < 3
            or pixels != frozenset(after.component.pixels)
            or before.component.value == after.component.value
        ):
            continue
        if any(
            frozenset(
                {
                    (row, column),
                    (row + 1, column),
                    (row, column + 1),
                    (row + 1, column + 1),
                }
            ).issubset(pixels)
            for row, column in pixels
        ):
            continue
        degree_by_position = {
            position: sum(
                neighbor in pixels
                for neighbor in (
                    (position[0] - 1, position[1]),
                    (position[0] + 1, position[1]),
                    (position[0], position[1] - 1),
                    (position[0], position[1] + 1),
                )
            )
            for position in pixels
        }
        if any(degree > 2 for degree in sorted(degree_by_position.values())):
            continue
        endpoints = tuple(
            sorted(
                position
                for position, degree in degree_by_position.items()
                if degree == 1
            )
        )
        if len(endpoints) != 2:
            continue
        endpoint_adjacency_sets: list[frozenset[str]] = []
        endpoint_supported = True
        for endpoint in endpoints:
            adjacent_refs = frozenset(
                component.component_id
                for component in before_components
                if component.component_id != before.component.component_id
                and any(
                    neighbor in frozenset(component.pixels)
                    for neighbor in (
                        (endpoint[0] - 1, endpoint[1]),
                        (endpoint[0] + 1, endpoint[1]),
                        (endpoint[0], endpoint[1] - 1),
                        (endpoint[0], endpoint[1] + 1),
                    )
                )
            )
            if not adjacent_refs:
                endpoint_supported = False
                break
            endpoint_adjacency_sets.append(adjacent_refs)
        if not endpoint_supported:
            continue
        shared_adjacent_refs = set.intersection(
            *(set(items) for items in endpoint_adjacency_sets)
        )
        endpoint_exclusive_refs = tuple(
            tuple(sorted(set(items) - shared_adjacent_refs))
            for items in endpoint_adjacency_sets
        )
        if any(not items for items in endpoint_exclusive_refs):
            continue
        endpoint_component_refs = (
            tuple(items[0] for items in endpoint_exclusive_refs)
            if all(len(items) == 1 for items in endpoint_exclusive_refs)
            else ()
        )
        shape_digest = stable_digest(before.component.relative_pixels)[:12]
        measurement_digest = stable_digest(
            (
                transition_ref,
                after.entity_id,
                endpoint_exclusive_refs,
                before.component.value,
                after.component.value,
            )
        )[:12]
        records.append(
            FrozenMap(
                {
                    "connector_measurement_digest": measurement_digest,
                    "connector_shape_digest": shape_digest,
                    "connector_entity_ref": after.entity_id,
                    "endpoint_component_refs": endpoint_component_refs,
                    "endpoint_component_candidate_ref_sets": endpoint_exclusive_refs,
                    "endpoint_component_candidate_count": sum(
                        len(items) for items in endpoint_exclusive_refs
                    ),
                    "endpoint_binding_exact": bool(endpoint_component_refs),
                    "endpoint_component_count": 2,
                    "before_attribute_value": before.component.value,
                    "after_attribute_value": after.component.value,
                    "transition_ref": transition_ref,
                }
            )
        )
        if len(records) >= 32:
            break
    return tuple(records)


def _thin_path_substitution_transition_measurements(
    *,
    before_scene: object,
    after_scene: object,
    candidate_ref: str,
    transition_ref: str,
) -> tuple[FrozenMap, ...]:
    """Measure a local thin path whose pixels all change by the same value pair.

    The measurement deliberately does not call the path a circuit, a control, or
    an active state.  DRM may interpret the exact substitution only after the
    controlled action and while preserving the opposite binary meaning.
    """

    before_rows = before_scene.frame.rows
    after_rows = after_scene.frame.rows
    if (
        len(before_rows) != len(after_rows)
        or any(len(left) != len(right) for left, right in zip(before_rows, after_rows))
    ):
        return ()
    point = _candidate_point(candidate_ref)
    if point is None:
        return ()
    positions_by_value_pair: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row, (before_row, after_row) in enumerate(zip(before_rows, after_rows)):
        for column, (before_value, after_value) in enumerate(
            zip(before_row, after_row)
        ):
            if before_value == after_value:
                continue
            positions_by_value_pair.setdefault(
                (before_value, after_value), set()
            ).add((row, column))

    before_components = tuple(before_scene.blocks.components)
    after_components = tuple(after_scene.blocks.components)
    records: list[FrozenMap] = []
    for (before_value, after_value), raw_positions in sorted(
        positions_by_value_pair.items()
    ):
        remaining = set(raw_positions)
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            frontier = [seed]
            connected: set[tuple[int, int]] = {seed}
            while frontier:
                current = frontier.pop()
                for neighbor in (
                    (current[0] - 1, current[1]),
                    (current[0] + 1, current[1]),
                    (current[0], current[1] - 1),
                    (current[0], current[1] + 1),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        connected.add(neighbor)
                        frontier.append(neighbor)
            if len(connected) < 3:
                continue
            if min(
                abs(row - point[0]) + abs(column - point[1])
                for row, column in connected
            ) > 6:
                continue
            if any(
                frozenset(component.pixels) == frozenset(connected)
                for component in before_components
            ) and any(
                frozenset(component.pixels) == frozenset(connected)
                for component in after_components
            ):
                # The established whole-component measurement already covers
                # this case; path substitution is for a local subpath only.
                continue
            if any(
                {
                    (row, column),
                    (row + 1, column),
                    (row, column + 1),
                    (row + 1, column + 1),
                }.issubset(connected)
                for row, column in connected
            ):
                continue
            degrees = {
                position: sum(
                    neighbor in connected
                    for neighbor in (
                        (position[0] - 1, position[1]),
                        (position[0] + 1, position[1]),
                        (position[0], position[1] - 1),
                        (position[0], position[1] + 1),
                    )
                )
                for position in connected
            }
            if any(degree > 2 for degree in sorted(degrees.values())):
                continue
            endpoints = tuple(
                sorted(position for position, degree in degrees.items() if degree == 1)
            )
            if len(endpoints) != 2:
                continue
            path_component_refs = frozenset(
                component.component_id
                for component in after_components
                if connected.intersection(component.pixels)
            )
            endpoint_candidate_sets: list[tuple[str, ...]] = []
            for endpoint in endpoints:
                adjacent = tuple(
                    sorted(
                        component.component_id
                        for component in after_components
                        if component.component_id not in path_component_refs
                        and any(
                            neighbor in frozenset(component.pixels)
                            for neighbor in (
                                (endpoint[0] - 1, endpoint[1]),
                                (endpoint[0] + 1, endpoint[1]),
                                (endpoint[0], endpoint[1] - 1),
                                (endpoint[0], endpoint[1] + 1),
                            )
                        )
                    )
                )
                if not adjacent:
                    endpoint_candidate_sets = []
                    break
                endpoint_candidate_sets.append(adjacent)
            if len(endpoint_candidate_sets) != 2:
                continue
            shared = set(endpoint_candidate_sets[0]).intersection(
                endpoint_candidate_sets[1]
            )
            exclusive = tuple(
                tuple(item for item in candidates if item not in shared)
                for candidates in endpoint_candidate_sets
            )
            if any(not candidates for candidates in exclusive):
                continue
            endpoint_refs = (
                tuple(candidates[0] for candidates in exclusive)
                if all(len(candidates) == 1 for candidates in exclusive)
                else ()
            )
            top = min(row for row, _column in connected)
            left = min(column for _row, column in connected)
            relative = tuple(sorted((row - top, column - left) for row, column in connected))
            measurement_digest = stable_digest(
                (
                    transition_ref,
                    tuple(sorted(connected)),
                    before_value,
                    after_value,
                    exclusive,
                )
            )[:12]
            records.append(
                FrozenMap(
                    {
                        "connector_measurement_digest": measurement_digest,
                        "connector_shape_digest": stable_digest(relative)[:12],
                        "connector_entity_ref": (
                            f"measurement.path-substitution.{measurement_digest}"
                        ),
                        "endpoint_component_refs": endpoint_refs,
                        "endpoint_component_candidate_ref_sets": exclusive,
                        "endpoint_component_candidate_count": sum(
                            len(candidates) for candidates in exclusive
                        ),
                        "endpoint_binding_exact": bool(endpoint_refs),
                        "endpoint_component_count": 2,
                        "before_attribute_value": before_value,
                        "after_attribute_value": after_value,
                        "path_substitution_pixel_count": len(connected),
                        "transition_ref": transition_ref,
                    }
                )
            )
            if len(records) >= 3:
                return tuple(records)
    return tuple(records)


def _co_translated_cell_carrier_measurements(
    *,
    before_scene: object,
    after_scene: object,
    before_by_entity: dict[str, TrackedComponent],
    after_by_entity: dict[str, TrackedComponent],
    co_moving_entity_refs: tuple[str, ...],
    retained_grid_geometry: tuple[object, ...] | None = None,
) -> dict[str, object]:
    """Measure one logical cell carrying an exact co-translation group.

    Every compatible periodic description must agree on the same witness.
    The output remains role-neutral: DRM alone may propose mover and traversable
    roles from the measured carrier and its vacated substrate.
    """

    if len(co_moving_entity_refs) < 2:
        return {}
    try:
        before_components = tuple(
            before_by_entity[entity_ref].component
            for entity_ref in co_moving_entity_refs
        )
        after_components = tuple(
            after_by_entity[entity_ref].component
            for entity_ref in co_moving_entity_refs
        )
        translation_deltas = {
            (
                int(after_by_entity[entity_ref].delta_row),
                int(after_by_entity[entity_ref].delta_col),
            )
            for entity_ref in co_moving_entity_refs
        }
    except KeyError:
        return {}
    if len(translation_deltas) != 1:
        return {}
    pixel_delta_row, pixel_delta_col = next(iter(translation_deltas))
    if (pixel_delta_row == 0) == (pixel_delta_col == 0):
        return {}

    def containing_cell(grid: object, components: tuple[object, ...]) -> object | None:
        """Return the unique logical cell containing the complete sprite."""

        strict_carriers = tuple(
            cell
            for cell in grid.cells
            if all(
                cell.bbox.top <= component.bbox.top
                and component.bbox.bottom <= cell.bbox.bottom
                and cell.bbox.left <= component.bbox.left
                and component.bbox.right <= cell.bbox.right
                for component in components
            )
        )
        if len(strict_carriers) == 1:
            return strict_carriers[0]

        # A logical cell core may carry a border or shadow outside its own
        # pixels. Require one exact core, not the bounding box of that drawing.
        # Two co-translated cores remain two bodies; nearby fragments alone
        # never create a cell. The shared measured delta is checked above.
        exact_cores = tuple(
            cell for cell in grid.cells
            if any(component.bbox == cell.bbox for component in components)
        )
        if len(exact_cores) > 1:
            return None
        if len(exact_cores) == 1:
            core = exact_cores[0]
            if all(
                core.bbox.top - int(grid.row_pitch) <= component.bbox.top
                and component.bbox.bottom <= core.bbox.bottom + int(grid.row_pitch)
                and core.bbox.left - int(grid.col_pitch) <= component.bbox.left
                and component.bbox.right <= core.bbox.right + int(grid.col_pitch)
                for component in components
            ):
                return core
            return None

        sprite_top = min(component.bbox.top for component in components)
        sprite_bottom = max(component.bbox.bottom for component in components)
        sprite_left = min(component.bbox.left for component in components)
        sprite_right = max(component.bbox.right for component in components)
        if (
            sprite_bottom - sprite_top + 1 != int(grid.row_pitch)
            or sprite_right - sprite_left + 1 != int(grid.col_pitch)
        ):
            return None

        pitch_carriers = tuple(
            cell
            for cell in grid.cells
            if cell.bbox.top <= sprite_top
            and sprite_bottom <= cell.bbox.top + int(grid.row_pitch) - 1
            and cell.bbox.left <= sprite_left
            and sprite_right <= cell.bbox.left + int(grid.col_pitch) - 1
        )
        return pitch_carriers[0] if len(pitch_carriers) == 1 else None

    before_grids = before_scene.periodic_cell_grids
    after_grids = after_scene.periodic_cell_grids
    if retained_grid_geometry is not None:
        before_grids = materialize_retained_periodic_cell_grid(before_scene.frame, tuple(retained_grid_geometry))
        after_grids = materialize_retained_periodic_cell_grid(after_scene.frame, tuple(retained_grid_geometry))

    def complete_pitch_pattern_grid(scene, grid, components):
        # A q-by-q co-translated footprint must be compared using all q-by-q
        # pixels, not different crops of that same footprint. This preserves
        # the measured phase and pitch; it neither selects a lattice nor
        # assigns a role. Smaller sprites and decorative overhangs retain
        # their original carrier representation.
        frame = getattr(scene, "frame", None)
        top = min(component.bbox.top for component in components)
        bottom = max(component.bbox.bottom for component in components)
        left = min(component.bbox.left for component in components)
        right = max(component.bbox.right for component in components)
        if (
            frame is None
            or bottom - top + 1 != grid.row_pitch
            or right - left + 1 != grid.col_pitch
            or (grid.cell_height == grid.row_pitch and grid.cell_width == grid.col_pitch)
            or (top - grid.row_offset) % grid.row_pitch
            or (left - grid.col_offset) % grid.col_pitch
        ):
            return grid
        complete = materialize_retained_periodic_cell_grid(
            frame,
            (grid.candidate_ref, grid.row_offset, grid.col_offset,
             grid.row_pitch, grid.col_pitch, grid.row_pitch, grid.col_pitch,
             min(grid.logical_rows, (frame.height - grid.row_offset) // grid.row_pitch),
             min(grid.logical_columns, (frame.width - grid.col_offset) // grid.col_pitch)),
        )
        return complete.candidates[0] if len(complete.candidates) == 1 else grid

    before_candidates = tuple(complete_pitch_pattern_grid(before_scene, grid, before_components)
                              for grid in before_grids.candidates)
    after_candidates = tuple(complete_pitch_pattern_grid(after_scene, grid, after_components)
                             for grid in after_grids.candidates)
    witnesses: set[tuple[str, str, int, int]] = set()
    for before_grid in before_candidates:
        for after_grid in after_candidates:
            if (
                before_grid.cell_height,
                before_grid.cell_width,
                before_grid.row_gap,
                before_grid.col_gap,
                before_grid.row_pitch,
                before_grid.col_pitch,
            ) != (
                after_grid.cell_height,
                after_grid.cell_width,
                after_grid.row_gap,
                after_grid.col_gap,
                after_grid.row_pitch,
                after_grid.col_pitch,
            ):
                continue
            before_carrier = containing_cell(before_grid, before_components)
            after_carrier = containing_cell(after_grid, after_components)
            if before_carrier is None or after_carrier is None:
                continue
            delta_row = int(after_carrier.row) - int(before_carrier.row)
            delta_col = int(after_carrier.col) - int(before_carrier.col)
            if abs(delta_row) + abs(delta_col) != 1:
                continue
            if (
                pixel_delta_row != delta_row * int(after_grid.row_pitch)
                or pixel_delta_col != delta_col * int(after_grid.col_pitch)
            ):
                continue
            vacated = next(
                (
                    cell
                    for cell in after_grid.cells
                    if (cell.row, cell.col)
                    == (before_carrier.row, before_carrier.col)
                ),
                None,
            )
            if vacated is None or after_carrier.pattern_occurrence_count != 1:
                continue
            witnesses.add(
                (
                    str(after_carrier.pattern_ref),
                    str(vacated.pattern_ref),
                    delta_row,
                    delta_col,
                )
            )
    if len(witnesses) != 1:
        return {}
    pattern_ref, vacated_ref, delta_row, delta_col = next(iter(witnesses))
    return {
        "unique_co_translated_cell_carrier_present": True,
        "unique_co_translated_cell_pattern_ref": pattern_ref,
        "co_translated_cell_vacated_after_pattern_ref": vacated_ref,
        "co_translated_cell_delta_row": delta_row,
        "co_translated_cell_delta_col": delta_col,
    }


def _translated_peer_cell_pattern_measurements(
    *,
    before_scene: object,
    after_scene: object,
    before_by_entity: dict[str, TrackedComponent],
    after_by_entity: dict[str, TrackedComponent],
    translations: tuple[TranslationTransition, ...],
) -> dict[str, object]:
    """Measure the current periodic-cell carrier of a translated peer group.

    Whole-cell hashes may change when the same visible overlay crosses a new
    substrate.  For two or three exactly tracked translated peers, bind each
    before/after body to one compatible logical cell and expose the common
    current pattern plus the vacated pattern set.  The measurement assigns no
    role; transport DRM decides whether these facts support mover/traversable
    suppositions.
    """

    entity_refs = tuple(sorted(item.entity_ref for item in translations))
    if not 2 <= len(entity_refs) <= 3:
        return {}
    try:
        before_components = tuple(
            before_by_entity[entity_ref].component for entity_ref in entity_refs
        )
        after_components = tuple(
            after_by_entity[entity_ref].component for entity_ref in entity_refs
        )
    except KeyError:
        return {}
    if len(
        {
            (component.value, component.relative_pixels)
            for component in after_components
        }
    ) != 1:
        return {}

    def carrier(grid: object, component: ComponentDescription) -> object | None:
        matches = tuple(
            cell
            for cell in grid.cells
            if cell.bbox.top <= component.bbox.top
            and component.bbox.bottom <= cell.bbox.bottom
            and cell.bbox.left <= component.bbox.left
            and component.bbox.right <= cell.bbox.right
        )
        return matches[0] if len(matches) == 1 else None

    witnesses: set[tuple[str, tuple[str, ...]]] = set()
    for before_grid in before_scene.periodic_cell_grids.candidates[:8]:
        for after_grid in after_scene.periodic_cell_grids.candidates[:8]:
            if (
                before_grid.cell_height,
                before_grid.cell_width,
                before_grid.row_gap,
                before_grid.col_gap,
                before_grid.row_pitch,
                before_grid.col_pitch,
            ) != (
                after_grid.cell_height,
                after_grid.cell_width,
                after_grid.row_gap,
                after_grid.col_gap,
                after_grid.row_pitch,
                after_grid.col_pitch,
            ):
                continue
            before_cells = tuple(
                carrier(before_grid, component) for component in before_components
            )
            after_cells = tuple(
                carrier(after_grid, component) for component in after_components
            )
            if any(cell is None for cell in (*before_cells, *after_cells)):
                continue
            after_by_position = {
                (int(cell.row), int(cell.col)): cell for cell in after_grid.cells
            }
            carrier_refs = {
                str(cell.pattern_ref) for cell in after_cells if cell is not None
            }
            if len(carrier_refs) != 1:
                continue
            vacated_refs = tuple(
                sorted(
                    {
                        str(vacated.pattern_ref)
                        for cell in before_cells
                        if cell is not None
                        and (
                            vacated := after_by_position.get(
                                (int(cell.row), int(cell.col))
                            )
                        )
                        is not None
                    }
                )
            )
            witnesses.add((next(iter(carrier_refs)), vacated_refs))
    if len(witnesses) != 1:
        return {}
    pattern_ref, vacated_refs = next(iter(witnesses))
    facts: dict[str, object] = {
        "translated_peer_cell_pattern_present": True,
        "translated_peer_cell_occurrence_count": len(entity_refs),
        "unique_moved_cell_pattern_ref": pattern_ref,
        "translated_peer_vacated_cell_pattern_refs": vacated_refs,
    }
    if len(vacated_refs) == 1:
        facts["moved_cell_vacated_after_pattern_ref"] = vacated_refs[0]
    return facts


def _periodic_cell_transition_measurements(
    *,
    before_scene: object,
    after_scene: object,
    candidate_ref: str,
    action_ref: str,
    canonical_periodic_transition_claim_facts: tuple[FrozenMap, ...] = (),
    canonical_periodic_cycle_workflow_facts: tuple[FrozenMap, ...] = (),
    declared_cellular_mover_pattern_refs: tuple[str, ...] = (),
    executed_action_axis_delta: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Measure exact logical-cell overlap under bounded grid translations."""

    before_grids = before_scene.periodic_cell_grids.candidates
    after_grids = after_scene.periodic_cell_grids.candidates
    # A source-selected lattice may refine the cell footprint while retaining
    # its measured pitch and origin (e.g. remove a spurious separator). Compare
    # the original rasters on that same lattice, instead of requiring another
    # physical action merely to refresh pattern identities at the new extent.
    if len(after_grids) == 1:
        current = after_grids[0]
        lattice = (current.row_pitch, current.col_pitch, current.row_offset, current.col_offset)
        same_lattice = tuple(grid for grid in before_grids if (
            grid.row_pitch, grid.col_pitch, grid.row_offset, grid.col_offset
        ) == lattice)
        if same_lattice and not any(
            (grid.cell_height, grid.cell_width, grid.row_gap, grid.col_gap) ==
            (current.cell_height, current.cell_width, current.row_gap, current.col_gap)
            for grid in same_lattice
        ):
            from agents.yf_arc3_v5.capabilities.frame import materialize_retained_periodic_cell_grid
            recounted = materialize_retained_periodic_cell_grid(
                before_scene.frame,
                (current.candidate_ref, current.row_offset, current.col_offset,
                 current.cell_height, current.cell_width, current.row_pitch, current.col_pitch,
                 current.logical_rows, current.logical_columns),
                separator_consistency_ppm=current.separator_consistency_ppm,
            )
            before_grids = recounted.candidates
    shift_measurements: list[dict[str, object]] = []
    for before_grid in before_grids:
        before_by_position = {
            (cell.row, cell.col): cell.pattern_ref for cell in before_grid.cells
        }
        ordered_before_positions = tuple(sorted(before_by_position.items()))
        for after_grid in after_grids:
            if (
                before_grid.cell_height,
                before_grid.cell_width,
                before_grid.row_gap,
                before_grid.col_gap,
            ) != (
                after_grid.cell_height,
                after_grid.cell_width,
                after_grid.row_gap,
                after_grid.col_gap,
            ):
                continue
            after_by_position = {
                (cell.row, cell.col): cell.pattern_ref for cell in after_grid.cells
            }
            for delta_row in range(-4, 5):
                for delta_col in range(-4, 5):
                    pairs = tuple(
                        (before_pattern, after_by_position[after_position])
                        for (row, col), before_pattern in ordered_before_positions
                        if (
                            after_position := (row + delta_row, col + delta_col)
                        )
                        in after_by_position
                    )
                    if len(pairs) < 4:
                        continue
                    match_count = sum(left == right for left, right in pairs)
                    match_ratio_ppm = match_count * 1_000_000 // len(pairs)
                    shift_measurements.append(
                        {
                            "delta_row": delta_row,
                            "delta_col": delta_col,
                            "overlap_count": len(pairs),
                            "match_count": match_count,
                            "match_ratio_ppm": match_ratio_ppm,
                            "change_count": len(pairs) - match_count,
                            "before_grid_ref": before_grid.candidate_ref,
                            "after_grid_ref": after_grid.candidate_ref,
                            "combined_separator_consistency_ppm": (
                                before_grid.separator_consistency_ppm
                                + after_grid.separator_consistency_ppm
                            ),
                            "combined_separator_axis_count": sum(
                                (
                                    before_grid.row_gap > 0,
                                    before_grid.col_gap > 0,
                                    after_grid.row_gap > 0,
                                    after_grid.col_gap > 0,
                                )
                            ),
                        }
                    )
    point_cell_probe_requested = candidate_ref.startswith("probe:cell:")
    if not shift_measurements:
        return {
            "periodic_grid_pair_count": 0,
            "periodic_grid_shift_candidate_count": 0,
            "point_cell_probe_requested": point_cell_probe_requested,
            "periodic_cycle_transition_observed": False,
            "periodic_cycle_closure_observed": False,
        }
    best_key = max(
        (
            int(item["combined_separator_axis_count"]),
            (
                int(item["match_count"])
                if int(item["combined_separator_axis_count"]) > 0
                else 0
            ),
            int(item["match_ratio_ppm"]),
            int(item["match_count"]),
            -int(item["change_count"]),
            int(item["combined_separator_consistency_ppm"]),
        )
        for item in shift_measurements
    )
    maxima = tuple(
        item
        for item in shift_measurements
        if (
            int(item["combined_separator_axis_count"]),
            (
                int(item["match_count"])
                if int(item["combined_separator_axis_count"]) > 0
                else 0
            ),
            int(item["match_ratio_ppm"]),
            int(item["match_count"]),
            -int(item["change_count"]),
            int(item["combined_separator_consistency_ppm"]),
        )
        == best_key
    )
    facts: dict[str, object] = {
        "periodic_grid_pair_count": len(
            {
                (str(item["before_grid_ref"]), str(item["after_grid_ref"]))
                for item in shift_measurements
            }
        ),
        "periodic_grid_shift_candidate_count": len(shift_measurements),
        "point_cell_probe_requested": point_cell_probe_requested,
    }
    if len(maxima) != 1:
        return facts
    unique = maxima[0]
    before_grid = next(
        grid
        for grid in before_grids
        if grid.candidate_ref == unique["before_grid_ref"]
    )
    after_grid = next(
        grid
        for grid in after_grids
        if grid.candidate_ref == unique["after_grid_ref"]
    )
    periodic_cycle_ref = str(after_grid.candidate_ref)
    before_positions: dict[str, list[tuple[int, int]]] = {}
    after_positions: dict[str, list[tuple[int, int]]] = {}
    after_pattern_by_position = {
        (cell.row, cell.col): cell.pattern_ref for cell in after_grid.cells
    }
    for cell in before_grid.cells:
        before_positions.setdefault(cell.pattern_ref, []).append((cell.row, cell.col))
    for cell in after_grid.cells:
        after_positions.setdefault(cell.pattern_ref, []).append((cell.row, cell.col))
    moved_patterns: list[tuple[str, int, int]] = []
    repeated_moved_patterns: list[
        tuple[str, tuple[tuple[int, int], ...], tuple[str, ...]]
    ] = []
    moved_cell_vacated_after_pattern_ref: str | None = None
    shift_row = int(unique["delta_row"])
    shift_col = int(unique["delta_col"])
    for pattern_ref in sorted(set(before_positions).intersection(after_positions)):
        if (
            len(before_positions[pattern_ref]) != 1
            or len(after_positions[pattern_ref]) != 1
        ):
            continue
        before_row, before_col = before_positions[pattern_ref][0]
        after_row, after_col = after_positions[pattern_ref][0]
        relative_row = after_row - (before_row + shift_row)
        relative_col = after_col - (before_col + shift_col)
        if relative_row or relative_col:
            moved_patterns.append((pattern_ref, relative_row, relative_col))
            try:
                vacated_after = next(
                    cell
                    for cell in after_grid.cells
                    if (cell.row, cell.col)
                    == (before_row + shift_row, before_col + shift_col)
                )
                moved_cell_vacated_after_pattern_ref = vacated_after.pattern_ref
            except StopIteration:
                pass
    # A coupled interface may move two or three occurrences of the same exact
    # cell pattern in different directions.  Measure that bounded repeated
    # carrier class by mutual unique nearest correspondence.  This names no
    # mover or terrain role; DRM interprets the exact pattern/delta/vacancy
    # facts after reconciliation.
    for pattern_ref in sorted(set(before_positions).intersection(after_positions)):
        before_group = tuple(sorted(before_positions[pattern_ref]))
        after_group = tuple(sorted(after_positions[pattern_ref]))
        if not (
            2 <= len(before_group) <= 3
            and len(before_group) == len(after_group)
            and before_group != after_group
        ):
            continue
        shifted_before = tuple(
            (row + shift_row, col + shift_col) for row, col in before_group
        )

        def cell_distance(
            left: tuple[int, int], right: tuple[int, int]
        ) -> int:
            return abs(left[0] - right[0]) + abs(left[1] - right[1])

        after_to_before: dict[tuple[int, int], tuple[int, int]] = {}
        for after_position in after_group:
            distances = tuple(
                (cell_distance(before_position, after_position), before_position)
                for before_position in shifted_before
            )
            minimum = min(item[0] for item in distances)
            nearest = tuple(
                before_position
                for candidate_distance, before_position in distances
                if candidate_distance == minimum
            )
            if minimum <= 0 or len(nearest) != 1:
                after_to_before.clear()
                break
            after_to_before[after_position] = nearest[0]
        if len(after_to_before) != len(after_group) or len(
            set(after_to_before.values())
        ) != len(shifted_before):
            continue
        mutual = True
        for before_position in shifted_before:
            distances = tuple(
                (cell_distance(before_position, after_position), after_position)
                for after_position in after_group
            )
            minimum = min(item[0] for item in distances)
            nearest = tuple(
                after_position
                for candidate_distance, after_position in distances
                if candidate_distance == minimum
            )
            if (
                len(nearest) != 1
                or after_to_before.get(nearest[0]) != before_position
            ):
                mutual = False
                break
        if not mutual:
            continue
        deltas = tuple(
            sorted(
                (
                    after_position[0] - before_position[0],
                    after_position[1] - before_position[1],
                )
                for after_position, before_position in sorted(after_to_before.items())
            )
        )
        vacated_refs = tuple(
            sorted(
                {
                    after_pattern_by_position[before_position]
                    for before_position in shifted_before
                    if before_position in after_pattern_by_position
                }
            )
        )
        repeated_moved_patterns.append((pattern_ref, deltas, vacated_refs))
    # A mobile overlay can acquire a different whole-cell hash when it crosses
    # a different substrate.  Preserve the purely morphological alternative:
    # one uniquely most-complex cell before and after a discrete action.
    if not moved_patterns and action_ref != "ACTION6":
        before_singletons = tuple(
            cell for cell in before_grid.cells if cell.pattern_occurrence_count == 1
        )
        after_singletons = tuple(
            cell for cell in after_grid.cells if cell.pattern_occurrence_count == 1
        )
        before_complexity = max(
            (cell.palette_value_count for cell in before_singletons), default=0
        )
        after_complexity = max(
            (cell.palette_value_count for cell in after_singletons), default=0
        )
        before_complex = tuple(
            cell
            for cell in before_singletons
            if cell.palette_value_count == before_complexity
        )
        after_complex = tuple(
            cell
            for cell in after_singletons
            if cell.palette_value_count == after_complexity
        )
        if len(before_complex) == 1 and len(after_complex) == 1:
            before_cell = before_complex[0]
            after_cell = after_complex[0]
            relative_row = after_cell.row - (before_cell.row + shift_row)
            relative_col = after_cell.col - (before_cell.col + shift_col)
            if relative_row or relative_col:
                moved_patterns.append(
                    (after_cell.pattern_ref, relative_row, relative_col)
                )
                try:
                    vacated_after = next(
                        cell
                        for cell in after_grid.cells
                        if (cell.row, cell.col)
                        == (
                            before_cell.row + shift_row,
                            before_cell.col + shift_col,
                        )
                    )
                    moved_cell_vacated_after_pattern_ref = (
                        vacated_after.pattern_ref
                    )
                except StopIteration:
                    pass
    # An already-declared mover can change its interior orientation without
    # changing its multiset of pixels.  Keep only a unique, action-consistent
    # correspondence; a matching histogram alone never names an identity.
    if declared_cellular_mover_pattern_refs and action_ref != "ACTION6":
        def cell_histogram(scene: object, grid: object, cell: object) -> tuple:
            top = grid.row_offset + cell.row * grid.row_pitch
            left = grid.col_offset + cell.col * grid.col_pitch
            return tuple(sorted(Counter(
                scene.frame.rows[row][col]
                for row in range(top, top + grid.cell_height)
                for col in range(left, left + grid.cell_width)
            ).items()))

        known_before = tuple(
            cell for cell in before_grid.cells
            if cell.pattern_ref in declared_cellular_mover_pattern_refs
            and cell.pattern_occurrence_count == 1
            and cell.palette_value_count > 1
        )
        newly_drawn = tuple(
            cell for cell in after_grid.cells
            if cell.pattern_ref not in before_positions
            and cell.pattern_occurrence_count == 1
            and cell.palette_value_count > 1
        )
        continuations = tuple(
            (prior, current)
            for prior in known_before
            for current in newly_drawn
            if cell_histogram(before_scene, before_grid, prior)
            == cell_histogram(after_scene, after_grid, current)
            and (
                (current.row - prior.row - shift_row,
                 current.col - prior.col - shift_col) == (0, 0)
                or (
                    executed_action_axis_delta is not None
                    and (
                        (executed_action_axis_delta[0] != 0
                         and current.col - prior.col - shift_col == 0
                         and 0 < (current.row - prior.row - shift_row)
                         * executed_action_axis_delta[0] <= 3)
                        or
                        (executed_action_axis_delta[1] != 0
                         and current.row - prior.row - shift_row == 0
                         and 0 < (current.col - prior.col - shift_col)
                         * executed_action_axis_delta[1] <= 3)
                    )
                )
            )
        )
        if len(continuations) == 1:
            prior, current = continuations[0]
            facts["unique_reoriented_cell_pattern_ref"] = current.pattern_ref
            facts["prior_reoriented_cell_pattern_ref"] = prior.pattern_ref
            relative = (current.row - prior.row - shift_row,
                        current.col - prior.col - shift_col)
            if relative != (0, 0):
                measured = (current.pattern_ref, *relative)
                if measured not in moved_patterns:
                    moved_patterns.append(measured)
                vacated = next((cell for cell in after_grid.cells
                                if (cell.row, cell.col) ==
                                (prior.row + shift_row, prior.col + shift_col)), None)
                if vacated is not None:
                    moved_cell_vacated_after_pattern_ref = vacated.pattern_ref
    new_patterns = tuple(
        pattern_ref
        for pattern_ref in sorted(set(after_positions) - set(before_positions))
        if len(after_positions[pattern_ref]) == 1
    )
    moved_pattern_refs = {pattern_ref for pattern_ref, _, _ in moved_patterns}
    new_patterns = tuple(
        pattern_ref
        for pattern_ref in new_patterns
        if pattern_ref not in moved_pattern_refs
        and pattern_ref != facts.get("unique_reoriented_cell_pattern_ref")
    )
    current_pose_ref = (
        "measurement.periodic_pose."
        + stable_digest(
            tuple(
                sorted(
                    (int(cell.row), int(cell.col), str(cell.pattern_ref))
                    for cell in after_grid.cells
                )
            )
        )[:16]
    )
    current_transition_signature_ref = (
        "measurement.periodic_transition."
        + stable_digest(
            (
                periodic_cycle_ref,
                int(unique["delta_row"]),
                int(unique["delta_col"]),
                current_pose_ref,
                tuple(sorted(moved_patterns)),
            )
        )[:16]
    )
    prior_pose_refs: set[str] = set()
    prior_transition_refs: set[str] = set()
    for prior_fact in (
        *canonical_periodic_transition_claim_facts,
        *canonical_periodic_cycle_workflow_facts,
    ):
        prior_cycle_ref = prior_fact.get("periodic_cycle_ref") or prior_fact.get(
            "cycle_ref"
        )
        if str(prior_cycle_ref or "") != periodic_cycle_ref:
            continue
        prior_pose_ref = prior_fact.get("periodic_pose_ref")
        if prior_pose_ref:
            prior_pose_refs.add(str(prior_pose_ref))
        prior_transition_ref = prior_fact.get("periodic_transition_signature_ref")
        if prior_transition_ref:
            prior_transition_refs.add(str(prior_transition_ref))
    observed_pose_refs = prior_pose_refs | {current_pose_ref}
    observed_transition_refs = prior_transition_refs | {
        current_transition_signature_ref
    }
    cycle_closure_observed = bool(
        current_transition_signature_ref in prior_transition_refs
        and len(observed_pose_refs) >= 2
    )
    facts.update(
        {
            "unique_periodic_grid_shift": True,
            "periodic_grid_delta_row": int(unique["delta_row"]),
            "periodic_grid_delta_col": int(unique["delta_col"]),
            "periodic_grid_overlap_count": int(unique["overlap_count"]),
            "periodic_grid_match_count": int(unique["match_count"]),
            "periodic_grid_change_count": int(unique["change_count"]),
            "nonzero_periodic_grid_shift": bool(
                int(unique["delta_row"]) or int(unique["delta_col"])
            ),
            "point_conditioned_periodic_cell_change": bool(
                point_cell_probe_requested
                and int(unique["change_count"]) > 0
            ),
            "discrete_conditioned_periodic_grid_change": bool(
                action_ref != "ACTION6" and int(unique["change_count"]) > 0
            ),
            "periodic_unique_moved_pattern_count": len(moved_patterns)
            + len(repeated_moved_patterns),
            "periodic_unique_new_pattern_count": len(new_patterns),
            # These are exact cycle-coverage measurements.  DRM derives the
            # workflow status from them; the capability never names a status.
            "periodic_cycle_ref": periodic_cycle_ref,
            "periodic_pose_ref": current_pose_ref,
            "periodic_transition_signature_ref": current_transition_signature_ref,
            "periodic_distinct_pose_count": len(observed_pose_refs),
            "periodic_distinct_transition_count": len(observed_transition_refs),
            "periodic_cycle_transition_observed": True,
            "periodic_cycle_closure_observed": cycle_closure_observed,
        }
    )
    if point_cell_probe_requested:
        try:
            _, row_text, col_text = candidate_ref.rsplit(":", 2)
            requested_row, requested_col = int(row_text), int(col_text)
            facts["requested_cell_row"] = requested_row
            facts["requested_cell_col"] = requested_col
            requested_before = next(
                cell
                for cell in before_grid.cells
                if (cell.row, cell.col) == (requested_row, requested_col)
            )
            requested_after = next(
                cell
                for cell in after_grid.cells
                if (cell.row, cell.col)
                == (requested_row + shift_row, requested_col + shift_col)
            )
            facts["requested_cell_before_pattern_ref"] = requested_before.pattern_ref
            facts["requested_cell_after_pattern_ref"] = requested_after.pattern_ref
            # A directly replaced clicked cell is causal effect evidence, not a
            # newly revealed contact objective. Preserve other novel cells.
            new_patterns = tuple(
                pattern_ref
                for pattern_ref in new_patterns
                if pattern_ref != requested_after.pattern_ref
            )
            facts["periodic_unique_new_pattern_count"] = len(new_patterns)
        except (ValueError, StopIteration):
            pass
    if len(moved_patterns) == 1:
        pattern_ref, relative_row, relative_col = moved_patterns[0]
        facts.update(
            {
                "unique_moved_cell_pattern_ref": pattern_ref,
                "moved_cell_delta_row": relative_row,
                "moved_cell_delta_col": relative_col,
            }
        )
        if moved_cell_vacated_after_pattern_ref:
            facts["moved_cell_vacated_after_pattern_ref"] = (
                moved_cell_vacated_after_pattern_ref
            )
    elif not moved_patterns and len(repeated_moved_patterns) == 1:
        pattern_ref, deltas, vacated_refs = repeated_moved_patterns[0]
        facts["unique_moved_cell_pattern_ref"] = pattern_ref
        facts["repeated_moved_cell_occurrence_count"] = len(deltas)
        facts["repeated_moved_cell_translation_deltas"] = deltas
        facts["repeated_moved_cell_vacated_pattern_refs"] = vacated_refs
        if len(set(deltas)) == 1:
            facts["moved_cell_delta_row"] = deltas[0][0]
            facts["moved_cell_delta_col"] = deltas[0][1]
        if len(vacated_refs) == 1:
            facts["moved_cell_vacated_after_pattern_ref"] = vacated_refs[0]
    facts["unique_new_pattern_matches_unique_moved_pattern"] = bool(
        len(new_patterns) == 1
        and len(moved_patterns) == 1
        and new_patterns[0] == moved_patterns[0][0]
    )
    facts["unique_new_pattern_matches_known_mover_pattern"] = bool(
        len(new_patterns) == 1
        and (
            new_patterns[0]
            in frozenset(str(ref) for ref in declared_cellular_mover_pattern_refs)
            or (
                len(moved_patterns) == 1
                and new_patterns[0] == moved_patterns[0][0]
            )
        )
    )
    if len(new_patterns) == 1:
        pattern_ref = new_patterns[0]
        row, col = after_positions[pattern_ref][0]
        facts.update(
            {
                "unique_new_cell_pattern_ref": pattern_ref,
                "unique_new_cell_row": row,
                "unique_new_cell_col": col,
            }
        )
    return facts


def analyze_goal_completion(
    value: GoalCompletionAnalysisInput,
) -> GoalCompletionAnalysis:
    """Expose exact completion evidence at a frame/level boundary."""

    terminal_objective_contract_attested = bool(
        value.after_score > value.before_score
        and value.terminal_objective_contract_ref
        and value.terminal_objective_measure_ref
        and value.terminal_objective_comparator == "equals"
        and value.terminal_objective_target_value is not None
        and value.terminal_objective_predicted_value_after_plan
        == value.terminal_objective_target_value
    )
    retrospective_target_coverage_boundary_evidence_present = bool(
        value.after_score > value.before_score
        and value.retrospective_coverage_target_cell_count_before_winning_action
        is not None
        and value.retrospective_coverage_target_cell_count_before_winning_action
        > 0
        and value.retrospective_uncovered_target_cell_count_before_winning_action
        is not None
    )
    shared = {
        "transition_ref": value.transition_ref,
        "goal_ref": value.goal_ref,
        "goal_ref_present": value.goal_ref is not None,
        "terminal_access_candidate_ref": value.terminal_access_candidate_ref,
        "terminal_access_candidate_ref_present": (
            value.terminal_access_candidate_ref is not None
        ),
        "terminal_objective_contract_ref": (
            value.terminal_objective_contract_ref
            if terminal_objective_contract_attested
            else None
        ),
        "terminal_objective_contract_ref_present": (
            terminal_objective_contract_attested
        ),
        "terminal_objective_measure_ref": (
            value.terminal_objective_measure_ref
            if terminal_objective_contract_attested
            else None
        ),
        "terminal_objective_comparator": (
            value.terminal_objective_comparator
            if terminal_objective_contract_attested
            else None
        ),
        "terminal_objective_target_value": (
            value.terminal_objective_target_value
            if terminal_objective_contract_attested
            else None
        ),
        "terminal_objective_predicted_value_after_plan": (
            value.terminal_objective_predicted_value_after_plan
            if terminal_objective_contract_attested
            else None
        ),
        "terminal_objective_contract_attested_by_official_boundary_and_predicate_match": (
            terminal_objective_contract_attested
        ),
        "retrospective_target_coverage_boundary_evidence_present": (
            retrospective_target_coverage_boundary_evidence_present
        ),
        "retrospective_coverage_target_cell_count_before_winning_action": (
            value.retrospective_coverage_target_cell_count_before_winning_action
            if retrospective_target_coverage_boundary_evidence_present
            else None
        ),
        "retrospective_uncovered_target_cell_count_before_winning_action": (
            value.retrospective_uncovered_target_cell_count_before_winning_action
            if retrospective_target_coverage_boundary_evidence_present
            else None
        ),
        "retrospective_stationary_peer_palette_nonboundary_cell_count_before_winning_action": (
            value.retrospective_stationary_peer_palette_nonboundary_cell_count_before_winning_action
        ),
        "retrospective_predicted_stationary_peer_palette_remaining_cell_count_after_exact_coincidence": (
            value.retrospective_predicted_stationary_peer_palette_remaining_cell_count_after_exact_coincidence
        ),
        "retrospective_exact_stationary_peer_morphology_match": (
            value.retrospective_exact_stationary_peer_morphology_match
        ),
        "retrospective_hot_workflow_ref": value.retrospective_hot_workflow_ref,
        "retrospective_hot_workflow_ref_present": (
            value.retrospective_hot_workflow_ref is not None
        ),
        "retrospective_hot_workflow_activity_refs": (
            value.retrospective_hot_workflow_activity_refs
        ),
        "retrospective_hot_workflow_activity_count": len(
            value.retrospective_hot_workflow_activity_refs
        ),
        "retrospective_hot_workflow_observation_count": (
            value.retrospective_hot_workflow_observation_count
        ),
        "moving_entity_ref": value.moving_entity_ref,
        "fixed_entity_ref": value.fixed_entity_ref,
        "alignment_axis": value.alignment_axis,
        "expected_remaining_step_count": value.expected_remaining_step_count,
        "before_score": value.before_score,
        "after_score": value.after_score,
        "score_increased": value.after_score > value.before_score,
        "context_epoch": value.context_epoch,
        "score_delta": value.after_score - value.before_score,
        "action_ref": value.action_ref,
        "candidate_ref": value.candidate_ref,
        "distance_is_quantized": value.distance_is_quantized,
        "exact_terminal_relation_step_expected": (
            value.exact_terminal_relation_step_expected
        ),
        "lifecycle_revision_group_ref": value.lifecycle_revision_group_ref,
        "lifecycle_revision_group_ref_present": (
            value.lifecycle_revision_group_ref is not None
        ),
        "official_success_observed": value.after_score > value.before_score,
        "causally_effective_action_count": sum(
            value.causally_effective_action_deltas
        ),
        "effective_action_ledger_entry_count": len(
            value.causally_effective_action_deltas
        ),
        "depth_claim_ref": value.depth_claim_ref or "",
        "depth_claim_ref_nonempty": bool(value.depth_claim_ref),
        "completion_event_ref": value.transition_ref,
        "declared_minimum_effective_action_count": (
            value.declared_minimum_effective_action_count
        ),
    }
    evidence_refs = tuple(
        item
        for item in (
            f"evidence.transition:{value.transition_ref}",
            value.goal_ref,
            (
                value.terminal_objective_contract_ref
                if terminal_objective_contract_attested
                else None
            ),
            value.retrospective_hot_workflow_ref,
            f"evidence.score_before:{value.before_score}",
            f"evidence.score_after:{value.after_score}",
        )
        if item
    )
    return GoalCompletionAnalysis(
        descriptive_facts=FrozenMap(shared),
        evidence_refs=evidence_refs,
        transition_ref=value.transition_ref,
    )


def _extent_transition(
    tracked: TrackedComponent,
    before_by_entity: dict[str, TrackedComponent],
) -> ExtentTransition | None:
    if tracked.match_kind != "extent_transition" or tracked.changed_extent_axis is None:
        return None
    before = before_by_entity.get(tracked.entity_id)
    if before is None:
        return None
    if tracked.changed_extent_axis == "column":
        stable_edge = (
            "left"
            if before.component.bbox.left == tracked.component.bbox.left
            else "right"
        )
        before_extent = before.component.bbox.width
        after_extent = tracked.component.bbox.width
    else:
        stable_edge = (
            "top"
            if before.component.bbox.top == tracked.component.bbox.top
            else "bottom"
        )
        before_extent = before.component.bbox.height
        after_extent = tracked.component.bbox.height
    return ExtentTransition(
        entity_ref=tracked.entity_id,
        before_component_ref=before.component.component_id,
        after_component_ref=tracked.component.component_id,
        axis=tracked.changed_extent_axis,
        before_extent=before_extent,
        after_extent=after_extent,
        delta_extent=after_extent - before_extent,
        stable_edge=stable_edge,  # type: ignore[arg-type]
    )


def _exchange_candidates(
    *,
    extents: tuple[ExtentTransition, ...],
    before_by_entity: dict[str, TrackedComponent],
    after_by_entity: dict[str, TrackedComponent],
    clicked: TrackedComponent | None,
) -> tuple[QuantizedExchangeCandidate, ...]:
    if clicked is None:
        return ()
    candidates: list[QuantizedExchangeCandidate] = []
    for source in extents:
        if source.delta_extent >= 0:
            continue
        for destination in extents:
            if destination.delta_extent <= 0 or source.axis != destination.axis:
                continue
            quantum = -source.delta_extent
            expected_quantum = (
                clicked.component.bbox.width
                if source.axis == "column"
                else clicked.component.bbox.height
            )
            source_before = before_by_entity[source.entity_ref].component
            destination_before = before_by_entity[destination.entity_ref].component
            if (
                destination.delta_extent != quantum
                or quantum != expected_quantum
                or source_before.value != destination_before.value
            ):
                continue
            source_after = after_by_entity[source.entity_ref].component
            destination_after = after_by_entity[destination.entity_ref].component
            source_before_quantity = (
                source_before.bbox.width
                if source.axis == "column"
                else source_before.bbox.height
            )
            source_after_quantity = (
                source_after.bbox.width
                if source.axis == "column"
                else source_after.bbox.height
            )
            destination_before_quantity = (
                destination_before.bbox.width
                if source.axis == "column"
                else destination_before.bbox.height
            )
            destination_after_quantity = (
                destination_after.bbox.width
                if source.axis == "column"
                else destination_after.bbox.height
            )
            candidates.append(
                QuantizedExchangeCandidate(
                    source_entity_ref=source.entity_ref,
                    destination_entity_ref=destination.entity_ref,
                    axis=source.axis,
                    quantum=quantum,
                    source_quantity_before=source_before_quantity,
                    source_quantity_after=source_after_quantity,
                    destination_quantity_before=destination_before_quantity,
                    destination_quantity_after=destination_after_quantity,
                    aggregate_quantity_before=(
                        source_before_quantity + destination_before_quantity
                    ),
                    aggregate_quantity_after=(
                        source_after_quantity + destination_after_quantity
                    ),
                )
            )
    return tuple(candidates)


def _peer_distance_change_candidates(
    *,
    translations: tuple[TranslationTransition, ...],
    before_by_entity: dict[str, TrackedComponent],
    after_by_entity: dict[str, TrackedComponent],
) -> tuple[PeerDistanceChangeCandidate, ...]:
    candidates: list[PeerDistanceChangeCandidate] = []
    for motion in translations:
        if bool(motion.delta_row) == bool(motion.delta_col):
            continue
        moving_before = before_by_entity[motion.entity_ref].component
        moving_after = after_by_entity[motion.entity_ref].component
        axis = "row" if motion.delta_row else "column"
        stationary_peers = tuple(
            (fixed_ref, fixed_after_tracked)
            for fixed_ref, fixed_after_tracked in sorted(after_by_entity.items())
            if fixed_ref != motion.entity_ref
            and fixed_after_tracked.match_kind == "stationary_exact"
        )
        same_value_stationary = tuple(
            item
            for item in stationary_peers
            if item[1].component.value == moving_before.value
        )
        same_shape_same_value = tuple(
            item
            for item in same_value_stationary
            if item[1].component.relative_pixels == moving_before.relative_pixels
        )
        # Exact silhouette peers even when palette differs (color alone is not
        # meaning; DRM still decides targethood). Prefer shape+value, then shape,
        # then a unique same-value peer.
        same_shape_any_value = tuple(
            item
            for item in stationary_peers
            if item[1].component.relative_pixels == moving_before.relative_pixels
        )
        fixed_candidates = (
            same_shape_same_value
            if same_shape_same_value
            else same_shape_any_value
            if same_shape_any_value
            else same_value_stationary
            if len(same_value_stationary) == 1
            else ()
        )
        for fixed_ref, fixed_after_tracked in fixed_candidates:
            fixed = fixed_after_tracked.component
            if axis == "column":
                fixed_center = fixed.bbox.left + fixed.bbox.right
                before_center = moving_before.bbox.left + moving_before.bbox.right
                after_center = moving_after.bbox.left + moving_after.bbox.right
            else:
                fixed_center = fixed.bbox.top + fixed.bbox.bottom
                before_center = moving_before.bbox.top + moving_before.bbox.bottom
                after_center = moving_after.bbox.top + moving_after.bbox.bottom
            before_distance = abs(fixed_center - before_center)
            after_distance = abs(fixed_center - after_center)
            distance_change_kind = (
                "zero"
                if after_distance == 0
                else "decreased"
                if after_distance < before_distance
                else "increased"
                if after_distance > before_distance
                else "unchanged"
            )
            candidates.append(
                PeerDistanceChangeCandidate(
                    translated_entity_ref=motion.entity_ref,
                    stationary_peer_entity_ref=fixed_ref,
                    axis=axis,
                    before_distance_twice=before_distance,
                    after_distance_twice=after_distance,
                    distance_change_kind=distance_change_kind,
                )
            )
    return tuple(candidates)


def _contact_coupled_translation_measurements(
    *,
    translations: tuple[TranslationTransition, ...],
    before_by_entity: dict[str, TrackedComponent],
    after_by_entity: dict[str, TrackedComponent],
    pre_action_input_aligned_bboxes: tuple[BoundingBox, ...] = (),
    expected_input_delta: tuple[int, int] | None = None,
) -> FrozenMap:
    """Measure bounded pre-action contact/delta relations without assigning roles.

    Contact is the defining observable boundary for a possible push.  A remote
    displacement remains a different mechanism even when it is collinear with
    another body.  DRM, not this capability, assigns pusher/pushed meanings to
    these neutral pair measurements.
    """

    def axis_direction(delta: tuple[int, int]) -> tuple[int, int]:
        return (
            0 if delta[0] == 0 else (1 if delta[0] > 0 else -1),
            0 if delta[1] == 0 else (1 if delta[1] > 0 else -1),
        )

    maximum_pairs = 64
    moving = tuple(translations[:32])
    moving_refs = frozenset(item.entity_ref for item in moving)
    stationary = tuple(
        tracked
        for entity_ref, tracked in sorted(after_by_entity.items())
        if entity_ref not in moving_refs
        and tracked.match_kind == "stationary_exact"
        and entity_ref in before_by_entity
        and not before_by_entity[entity_ref].component.touches_frame_boundary
    )[:32]
    stationary_mobile_pairs: list[tuple[str, str]] = []
    equal_delta_contact_preserved_pairs: list[tuple[str, str]] = []
    unequal_collinear_pairs: list[tuple[str, str]] = []
    other_moving_pairs: list[tuple[str, str]] = []
    input_aligned_contact_nonlocal_pairs: list[tuple[str, str]] = []
    enumerated = 0
    truncated = False

    for motion in moving:
        moving_before = before_by_entity.get(motion.entity_ref)
        if moving_before is None:
            continue
        for fixed in stationary:
            if enumerated >= maximum_pairs:
                truncated = True
                break
            if not _components_touch(moving_before.component, fixed.component):
                continue
            enumerated += 1
            stationary_mobile_pairs.append((fixed.entity_id, motion.entity_ref))
        if truncated:
            break

    if not truncated:
        for first, second in combinations(moving, 2):
            if enumerated >= maximum_pairs:
                truncated = True
                break
            first_before = before_by_entity.get(first.entity_ref)
            second_before = before_by_entity.get(second.entity_ref)
            first_after = after_by_entity.get(first.entity_ref)
            second_after = after_by_entity.get(second.entity_ref)
            if (
                first_before is None
                or second_before is None
                or first_after is None
                or second_after is None
                or not _components_touch(
                    first_before.component, second_before.component
                )
            ):
                continue
            enumerated += 1
            pair = tuple(sorted((first.entity_ref, second.entity_ref)))
            first_delta = (first.delta_row, first.delta_col)
            second_delta = (second.delta_row, second.delta_col)
            if first_delta == second_delta and _components_touch(
                first_after.component, second_after.component
            ):
                equal_delta_contact_preserved_pairs.append(pair)
                continue
            first_axis = bool(first.delta_row) != bool(first.delta_col)
            second_axis = bool(second.delta_row) != bool(second.delta_col)
            same_direction = bool(
                first_axis
                and second_axis
                and axis_direction(first_delta) == axis_direction(second_delta)
            )
            if same_direction and first_delta != second_delta:
                unequal_collinear_pairs.append(pair)
            else:
                other_moving_pairs.append(pair)

    if expected_input_delta is not None and expected_input_delta != (0, 0):
        expected_axis = axis_direction(expected_input_delta)
        expected_magnitude = max(
            abs(expected_input_delta[0]), abs(expected_input_delta[1])
        )
        for control_bbox in pre_action_input_aligned_bboxes[:3]:
            control_components = tuple(
                tracked.component
                for tracked in sorted(
                    before_by_entity.values(), key=lambda item: item.entity_id
                )
                if tracked.component.bbox == control_bbox
            )
            for motion in moving:
                moving_before = before_by_entity.get(motion.entity_ref)
                if (
                    moving_before is None
                    or moving_before.component.bbox == control_bbox
                    or not any(
                        _components_touch(component, moving_before.component)
                        for component in control_components
                    )
                ):
                    continue
                motion_delta = (motion.delta_row, motion.delta_col)
                if (
                    axis_direction(motion_delta) == expected_axis
                    and max(abs(motion.delta_row), abs(motion.delta_col))
                    > expected_magnitude
                ):
                    input_aligned_contact_nonlocal_pairs.append(
                        (
                            ":".join(
                                str(value)
                                for value in (
                                    control_bbox.top,
                                    control_bbox.left,
                                    control_bbox.bottom,
                                    control_bbox.right,
                                )
                            ),
                            motion.entity_ref,
                        )
                    )

    return FrozenMap(
        {
            "pre_action_contact_translation_pair_count": enumerated,
            "pre_action_contact_translation_pair_enumeration_truncated": truncated,
            "stationary_mobile_pre_action_contact_pair_count": len(
                stationary_mobile_pairs
            ),
            "equal_delta_contact_preserved_pair_count": len(
                equal_delta_contact_preserved_pairs
            ),
            "unequal_collinear_pre_action_contact_pair_count": len(
                unequal_collinear_pairs
            ),
            "other_pre_action_contact_coupling_pair_count": len(
                other_moving_pairs
            ),
            "input_aligned_contact_nonlocal_translation_count": len(
                input_aligned_contact_nonlocal_pairs
            ),
            "stationary_mobile_pre_action_contact_pairs": tuple(
                stationary_mobile_pairs
            ),
            "equal_delta_contact_preserved_pairs": tuple(
                equal_delta_contact_preserved_pairs
            ),
            "unequal_collinear_pre_action_contact_pairs": tuple(
                unequal_collinear_pairs
            ),
            "other_pre_action_contact_coupling_pairs": tuple(other_moving_pairs),
            "input_aligned_contact_nonlocal_translation_pairs": tuple(
                input_aligned_contact_nonlocal_pairs
            ),
        }
    )


def _separator_candidates(
    exchange: QuantizedExchangeCandidate,
    *,
    after_tracking: tuple[TrackedComponent, ...],
    frame_enclosing_refs: frozenset[str],
) -> tuple[str, ...]:
    by_entity = {item.entity_id: item for item in after_tracking}
    source = by_entity[exchange.source_entity_ref].component.bbox
    destination = by_entity[exchange.destination_entity_ref].component.bbox
    separated_by_rows = source.bottom < destination.top or destination.bottom < source.top
    separated_by_columns = source.right < destination.left or destination.right < source.left
    candidates: list[str] = []
    for tracked in after_tracking:
        component = tracked.component
        if (
            tracked.match_kind != "stationary_exact"
            or component.component_id in frame_enclosing_refs
            or tracked.entity_id in {
                exchange.source_entity_ref,
                exchange.destination_entity_ref,
            }
        ):
            continue
        if separated_by_rows:
            upper_bottom = min(source.bottom, destination.bottom)
            lower_top = max(source.top, destination.top)
            between = upper_bottom < component.bbox.top and component.bbox.bottom < lower_top
            elongated = component.bbox.width > component.bbox.height
        elif separated_by_columns:
            left_right = min(source.right, destination.right)
            right_left = max(source.left, destination.left)
            between = left_right < component.bbox.left and component.bbox.right < right_left
            elongated = component.bbox.height > component.bbox.width
        else:
            continue
        if between and elongated:
            candidates.append(tracked.entity_id)
    return tuple(candidates)


def _interface_components(
    *,
    clicked: TrackedComponent | None,
    exchange: QuantizedExchangeCandidate,
    separators: tuple[str, ...],
    before_by_entity: dict[str, TrackedComponent],
    after_by_entity: dict[str, TrackedComponent],
) -> tuple[str, ...]:
    if clicked is None:
        return ()
    candidates = [clicked.entity_id]
    for entity_ref in (
        exchange.source_entity_ref,
        exchange.destination_entity_ref,
        *separators,
    ):
        tracked = before_by_entity.get(entity_ref) or after_by_entity.get(entity_ref)
        if tracked is not None and _components_touch(clicked.component, tracked.component):
            candidates.append(entity_ref)
    return tuple(dict.fromkeys(candidates))


def _components_touch(
    first: object,
    second: object,
) -> bool:
    if not hasattr(first, "pixels") or not hasattr(second, "pixels"):
        return False
    second_pixels = frozenset(second.pixels)
    return any(
        (row + delta_row, col + delta_col) in second_pixels
        for row, col in first.pixels
        for delta_row, delta_col in ((-1, 0), (0, -1), (0, 1), (1, 0))
    )


def _scene_transition_effect_signature_ref(
    *,
    before: FrameGrid,
    intermediate_frames: tuple[FrameGrid, ...],
    after: FrameGrid,
    action_ref: str | None,
    candidate_ref: str | None,
) -> str:
    """Return the exact locally indexed signature for one observed packet."""

    return stable_digest(
        (
            before.rows,
            after.rows,
            intermediate_frames,
            action_ref,
            candidate_ref,
        )
    )[:16]


def _measurements(
    *,
    action_ref: str | None,
    measurement_input: SceneTransitionMeasurementInput | None = None,
    pixel_delta: PixelTransitionDelta,
    potential_nonzero_translation_count: int,
    shared_facts: dict[str, object],
    ordered_packet_facts: dict[str, object],
) -> SceneTransitionMeasurements:
    alternatives = (SCENE_EQUIVALENT_CANDIDATE, SCENE_DIFFERENT_CANDIDATE)
    signature_ref = (
        _scene_transition_effect_signature_ref(
            before=measurement_input.before,
            intermediate_frames=measurement_input.intermediate_frames,
            after=measurement_input.after,
            action_ref=action_ref,
            candidate_ref=measurement_input.candidate_ref,
        )
        if measurement_input
        else None
    )
    changed = pixel_delta.final_changed_count > 0
    declared_indicator_decrease_changed_pixel_count = (
        measurement_input.declared_boundary_indicator_decrease_changed_pixel_count_measurement
        if measurement_input
        else 0
    )
    established_indicator_decrease = bool(
        measurement_input
        and measurement_input.boundary_indicator_quantum_delta_measurement < 0
        and measurement_input.established_boundary_indicator_decrement_changed_pixel_count_measurement
        > 0
    )
    indicator_only_decrease = bool(
        measurement_input
        and (
            (
                measurement_input.boundary_indicator_quantum_delta_measurement < 0
                and measurement_input.established_boundary_indicator_decrement_changed_pixel_count_measurement
                > 0
                and pixel_delta.final_changed_count
                == measurement_input.established_boundary_indicator_decrement_changed_pixel_count_measurement
            )
            or (
                declared_indicator_decrease_changed_pixel_count > 0
                and pixel_delta.final_changed_count
                == declared_indicator_decrease_changed_pixel_count
            )
        )
    )
    if not changed:
        novelty = EffectSignatureNovelty.NONE
        scope = ObservableChangeScope.NONE
    elif indicator_only_decrease:
        novelty = EffectSignatureNovelty.NOVEL
        scope = ObservableChangeScope.RESOURCE_ONLY
    elif measurement_input and (
        measurement_input.current_effect_signature_known
        or measurement_input.current_effect_signature_known_cycle
    ):
        novelty = EffectSignatureNovelty.KNOWN
        scope = (
            ObservableChangeScope.KNOWN_CYCLIC_POSE_ONLY
            if measurement_input.current_effect_signature_known_cycle
            else ObservableChangeScope.WORLD_SPATIAL_CONFIGURATION
        )
    else:
        novelty = EffectSignatureNovelty.NOVEL
        scope = ObservableChangeScope.WORLD_SPATIAL_CONFIGURATION
    return SceneTransitionMeasurements(
        alternative_refs=alternatives,
        ordered_packet_facts=FrozenMap(ordered_packet_facts),
        alternative_facts=FrozenMap(
            {
                alternative: FrozenMap(
                    {
                        **shared_facts,
                        "effect_signature_novelty_measurement": novelty.value,
                        "observable_change_scope_measurement": scope.value,
                        "declared_boundary_indicator_decrease_observed_measurement": (
                            declared_indicator_decrease_changed_pixel_count > 0
                        ),
                        "boundary_indicator_decrement_unit_established_measurement": (
                            established_indicator_decrease
                        ),
                        "measured_delta_signature_ref": signature_ref,
                        "known_effect_signature_count": (
                            measurement_input.known_effect_signature_count
                            if measurement_input
                            else 0
                        ),
                        "known_cycle_transition_signature_count": (
                            measurement_input.known_cycle_transition_signature_count
                            if measurement_input
                            else 0
                        ),
                        "classification_candidate_kind": (
                            "equivalent"
                            if alternative == SCENE_EQUIVALENT_CANDIDATE
                            else "different"
                        ),
                    }
                )
                for alternative in alternatives
            }
        ),
        changed_count=pixel_delta.final_changed_count,
        potential_nonzero_translation_count=potential_nonzero_translation_count,
        pixel_delta=pixel_delta,
        action_ref=action_ref,
        candidate_ref=(measurement_input.candidate_ref if measurement_input else None),
        before_configuration_digest=(
            measurement_input.before_configuration_digest if measurement_input else None
        ),
        after_configuration_digest=(
            measurement_input.after_configuration_digest if measurement_input else None
        ),
        before_available_action_set_digest=(
            measurement_input.before_available_action_set_digest
            if measurement_input
            else None
        ),
        measured_delta_signature_ref=(
            stable_digest(
                (
                    measurement_input.before.rows,
                    measurement_input.after.rows,
                    measurement_input.intermediate_frames,
                    action_ref,
                    measurement_input.candidate_ref,
                )
            )[:16]
            if measurement_input
            else None
        ),
        configuration_transition_changed=(
            measurement_input.before_configuration_digest
            != measurement_input.after_configuration_digest
            if measurement_input
            and measurement_input.before_configuration_digest is not None
            and measurement_input.after_configuration_digest is not None
            else None
        ),
        score_delta=(
            measurement_input.after_score - measurement_input.before_score
            if measurement_input and measurement_input.after_score is not None
            else None
        ),
        effect_signature_novelty_measurement=novelty,
        observable_change_scope_measurement=scope,
    )


def _measure_pixel_packet(
    value: SceneTransitionMeasurementInput,
    *,
    changed_positions_sink: set[tuple[int, int]] | None = None,
) -> PixelTransitionDelta:
    """Scan an ordered packet once; retain positions only for an explicit caller."""

    frames = (value.before, *value.intermediate_frames, value.after)
    intermediate_count = len(value.intermediate_frames)
    consecutive_counts = [0] * (intermediate_count + 1)
    from_before_counts = [0] * intermediate_count
    to_after_counts = [0] * intermediate_count
    changed_before_values: set[int] = set()
    changed_after_values: set[int] = set()
    transition_pairs: set[tuple[int, int]] = set()
    changed_count = 0
    top: int | None = None
    left: int | None = None
    bottom: int | None = None
    right: int | None = None
    dimensions = {(frame.height, frame.width) for frame in frames}
    dimensions_consistent = len(dimensions) == 1
    maximum_height = max(frame.height for frame in frames)
    maximum_width = max(frame.width for frame in frames)
    missing = object()

    for row in range(maximum_height):
        packet_rows = tuple(
            frame.rows[row] if row < frame.height else None for frame in frames
        )
        for col in range(maximum_width):
            before_pixel = (
                packet_rows[0][col]
                if packet_rows[0] is not None and col < len(packet_rows[0])
                else missing
            )
            after_pixel = (
                packet_rows[-1][col]
                if packet_rows[-1] is not None and col < len(packet_rows[-1])
                else missing
            )
            previous_pixel = before_pixel
            for index in range(intermediate_count):
                packet_row = packet_rows[index + 1]
                pixel = (
                    packet_row[col]
                    if packet_row is not None and col < len(packet_row)
                    else missing
                )
                if previous_pixel != pixel:
                    consecutive_counts[index] += 1
                if before_pixel != pixel:
                    from_before_counts[index] += 1
                if pixel != after_pixel:
                    to_after_counts[index] += 1
                previous_pixel = pixel
            if previous_pixel != after_pixel:
                consecutive_counts[-1] += 1
            if before_pixel == after_pixel:
                continue

            changed_count += 1
            if changed_positions_sink is not None:
                changed_positions_sink.add((row, col))
            top = row if top is None else min(top, row)
            left = col if left is None else min(left, col)
            bottom = row if bottom is None else max(bottom, row)
            right = col if right is None else max(right, col)
            if before_pixel is not missing and after_pixel is not missing:
                before_value = int(before_pixel)
                after_value = int(after_pixel)
                changed_before_values.add(before_value)
                changed_after_values.add(after_value)
                transition_pairs.add((before_value, after_value))

    bbox = (
        None
        if top is None or left is None or bottom is None or right is None
        else BoundingBox(top=top, left=left, bottom=bottom, right=right)
    )
    union_is_collinear = bool(
        bbox is not None and (bbox.height == 1 or bbox.width == 1)
    )
    union_touches_boundary = bool(
        bbox is not None
        and dimensions_consistent
        and (
            bbox.top == 0
            or bbox.left == 0
            or bbox.bottom == value.before.height - 1
            or bbox.right == value.before.width - 1
        )
    )
    potential_translation_count = (
        len(changed_before_values.intersection(changed_after_values))
        if dimensions_consistent
        else 0
    )
    return PixelTransitionDelta(
        transition_ref=value.transition_ref,
        action_ref=value.action_ref,
        packet_dimensions_consistent=dimensions_consistent,
        intermediate_frame_count=intermediate_count,
        consecutive_changed_counts=tuple(consecutive_counts),
        intermediate_changed_counts_from_before=tuple(from_before_counts),
        intermediate_changed_counts_to_after=tuple(to_after_counts),
        final_changed_count=changed_count,
        changed_bbox=bbox,
        changed_edges=sum(count > 0 for count in consecutive_counts),
        changed_transition_pair_count=len(transition_pairs),
        potential_nonzero_translation_count=potential_translation_count,
        changed_union_is_collinear=union_is_collinear,
        changed_union_touches_boundary=union_touches_boundary,
        final_scene_equivalent_with_intermediate_change=bool(
            changed_count == 0 and any(from_before_counts)
        ),
    )


def measure_pixel_packet_with_groups(
    value: SceneTransitionMeasurementInput,
    *,
    max_groups: int = 32,
    max_pairs: int = 3,
) -> tuple[PixelTransitionDelta, PixelDeltaGroups]:
    """Reuse one packet scan for exact transition facts and bounded change groups.

    SRC may request this pure measurement when change geometry is needed. The
    existing transition path remains unchanged until that demand is declared.
    """

    from agents.yf_arc3_v5.capabilities.pixel_delta import (
        PixelDeltaGroups,
        summarize_changed_positions,
    )

    if not (1 <= max_groups <= 32 and 1 <= max_pairs <= 3):
        raise ValueError("packet group bounds must stay within hard limits")
    changed_positions: set[tuple[int, int]] = set()
    delta = _measure_pixel_packet(
        value, changed_positions_sink=changed_positions
    )
    if not delta.packet_dimensions_consistent:
        return delta, PixelDeltaGroups(
            changed_pixel_count=delta.final_changed_count,
            changed_bbox=delta.changed_bbox,
            group_count=0,
            groups=(),
            equal_shape_pairs=(),
            enumeration_truncated=True,
        )
    groups = summarize_changed_positions(
        changed_positions,
        frame_height=value.before.height,
        frame_width=value.before.width,
        max_groups=max_groups,
        max_pairs=max_pairs,
    )
    if (
        groups.changed_pixel_count != delta.final_changed_count
        or groups.changed_bbox != delta.changed_bbox
    ):
        raise RuntimeError("one-pass pixel change groups disagree with packet delta")
    return delta, groups


def _pixel_delta_facts(value: PixelTransitionDelta) -> dict[str, object]:
    return {
        "transition_ref": value.transition_ref,
        "action_ref": value.action_ref,
        "packet_dimensions_consistent": value.packet_dimensions_consistent,
        "intermediate_frame_count": value.intermediate_frame_count,
        "consecutive_changed_counts": value.consecutive_changed_counts,
        "intermediate_changed_counts_from_before": (
            value.intermediate_changed_counts_from_before
        ),
        "intermediate_changed_counts_to_after": (
            value.intermediate_changed_counts_to_after
        ),
        "final_changed_count": value.final_changed_count,
        "changed_bbox": (
            None
            if value.changed_bbox is None
            else value.changed_bbox.model_dump(mode="json")
        ),
        "changed_transition_pair_count": value.changed_transition_pair_count,
        "changed_edges": value.changed_edges,
        "potential_nonzero_translation_count": (
            value.potential_nonzero_translation_count
        ),
        "changed_union_is_collinear": value.changed_union_is_collinear,
        "changed_union_touches_boundary": value.changed_union_touches_boundary,
        "final_scene_equivalent_with_intermediate_change": (
            value.final_scene_equivalent_with_intermediate_change
        ),
    }


def _closed_support_permutation_facts(
    value: ControlledTransitionAnalysisInput,
) -> dict[str, object]:
    """Compare anonymous values on one exact closed support in slot order."""

    before = measure_orthogonal_perimeter_assemblies(
        OrthogonalPerimeterAssemblyInput(components=value.before_scene.blocks)
    )
    after = measure_orthogonal_perimeter_assemblies(
        OrthogonalPerimeterAssemblyInput(components=value.after_scene.blocks)
    )
    facts: dict[str, object] = {
        "before_orthogonal_perimeter_candidate_count": len(
            before.perimeter_candidates
        ),
        "after_orthogonal_perimeter_candidate_count": len(after.perimeter_candidates),
        "closed_support_permutation_match_count": 0,
    }
    if len(before.perimeter_candidates) != 1 or len(after.perimeter_candidates) != 1:
        from .joint_slots import measure_joint_slot_geometry, measure_supported_slot_rotation

        before_joint = measure_joint_slot_geometry(value.before_scene.blocks.components)
        after_joint = measure_joint_slot_geometry(value.after_scene.blocks.components)
        if before_joint and after_joint:
            facts.update(measure_supported_slot_rotation(before_joint, after_joint))
        return facts
    before_support = before.perimeter_candidates[0]
    after_support = after.perimeter_candidates[0]
    if before_support.slot_centers_twice != after_support.slot_centers_twice:
        return facts
    before_values = before_support.slot_values
    after_values = after_support.slot_values
    slot_count = len(before_values)
    signed_matches: list[int] = []
    for raw_shift in range(slot_count):
        rotated = tuple(
            before_values[(index - raw_shift) % slot_count]
            for index in range(slot_count)
        )
        if rotated != after_values:
            continue
        signed_shift = (
            raw_shift
            if raw_shift <= slot_count // 2
            else raw_shift - slot_count
        )
        signed_matches.append(signed_shift)
    facts.update(
        {
            "closed_support_slot_count": slot_count,
            "closed_support_slot_centers_twice": before_support.slot_centers_twice,
            "closed_support_before_value_sequence": before_values,
            "closed_support_after_value_sequence": after_values,
            "closed_support_before_state_digest": stable_digest(before_values)[:16],
            "closed_support_after_state_digest": stable_digest(after_values)[:16],
            "closed_support_permutation_match_count": len(signed_matches),
            "closed_support_signed_shift_matches": tuple(signed_matches[:3]),
            "closed_support_permutation_matches_truncated": len(signed_matches) > 3,
        }
    )
    if len(signed_matches) == 1:
        signed_shift = signed_matches[0]
        facts.update(
            {
                "unique_closed_support_permutation": True,
                "unique_closed_support_signed_slot_quantum": signed_shift,
                "closed_support_one_slot_quantum": abs(signed_shift) == 1,
                "closed_support_nonzero_permutation": signed_shift != 0,
            }
        )
        before_ids = {entity.component.component_id: entity.entity_id
                      for entity in value.before_tracking.entities}
        after_ids = {entity.component.component_id: entity.entity_id
                     for entity in value.after_tracking.entities}
        before_members = tuple(before_ids.get(ref) for ref in before_support.component_refs)
        after_members = tuple(after_ids.get(ref) for ref in after_support.component_refs)
        if (all(before_members) and all(after_members)
                and len(set(before_members)) == slot_count
                and set(before_members) == set(after_members)):
            facts["closed_support_collective_measurements"] = (FrozenMap({
                "member_refs": tuple(sorted(before_members)),
                "before_slot_member_refs": before_members,
                "after_slot_member_refs": after_members,
                "slot_centers_twice": before_support.slot_centers_twice,
                "before_values": before_values, "after_values": after_values,
                "signed_slot_quantum": signed_shift, "slot_count": slot_count,
            }),)
    return facts
