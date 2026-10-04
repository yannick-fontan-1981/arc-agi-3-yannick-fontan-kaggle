"""Pure exhaustive accounting of ordered frame-transition phenomena."""

from __future__ import annotations

from collections import deque

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    OrderedAnimationExtentChange,
    OrderedAnimationGlyphCandidate,
    OrderedAnimationHomologousOccurrence,
    TransitionPhenomenonGroup,
    TransitionPhenomenonInventoryInput,
    TransitionPhenomenonInventoryMeasurements,
    TransitionPhenomenonRelation,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.capabilities.affine_glyphs import (
    measure_ordered_glyph_activation_episodes,
)


Position = tuple[int, int]
Grid = tuple[tuple[int, ...], ...]


def _grid(frame: object, *, height: int, width: int) -> Grid:
    rows = getattr(frame, "rows")
    return tuple(
        tuple(
            int(rows[row][col]) if row < len(rows) and col < len(rows[row]) else -1
            for col in range(width)
        )
        for row in range(height)
    )


def _changed_positions(before: Grid, after: Grid) -> frozenset[Position]:
    return frozenset(
        (row, col)
        for row in range(len(before))
        for col in range(len(before[row]))
        if before[row][col] != after[row][col]
    )


def _orthogonal_component_count(positions: frozenset[Position]) -> int:
    """Count exact four-neighbour components without assigning object meaning."""

    remaining = set(positions)
    count = 0
    while remaining:
        count += 1
        start = min(remaining)
        remaining.remove(start)
        queue = deque((start,))
        while queue:
            row, col = queue.popleft()
            for neighbor in (
                (row - 1, col),
                (row, col - 1),
                (row, col + 1),
                (row + 1, col),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return count


def _ordered_palette_delta_facts(frames: tuple[Grid, ...]) -> FrozenMap:
    """Measure bounded palette growth, branching and later recolours in order."""

    transitions: list[FrozenMap] = []
    added_by_value: dict[int, dict[int, set[Position]]] = {}
    removed_by_value: dict[int, int] = {}
    maximum_records = 512
    overflow_count = 0
    for pair_index, (before, after) in enumerate(zip(frames, frames[1:])):
        grouped: dict[tuple[int, int], set[Position]] = {}
        for row in range(len(before)):
            for col in range(len(before[row])):
                old_value = int(before[row][col])
                new_value = int(after[row][col])
                if old_value == new_value:
                    continue
                grouped.setdefault((old_value, new_value), set()).add((row, col))
                added_by_value.setdefault(new_value, {}).setdefault(
                    pair_index, set()
                ).add((row, col))
                removed_by_value[old_value] = removed_by_value.get(old_value, 0) + 1
        for (old_value, new_value), raw_positions in sorted(grouped.items()):
            positions = frozenset(raw_positions)
            if len(transitions) >= maximum_records:
                overflow_count += 1
                continue
            box = _box(tuple(sorted(positions)))
            transitions.append(
                FrozenMap(
                    {
                        "pair_index": pair_index,
                        "from_value": old_value,
                        "to_value": new_value,
                        "pixel_count": len(positions),
                        "component_count": _orthogonal_component_count(positions),
                        "bounding_box": (box.top, box.left, box.bottom, box.right),
                    }
                )
            )

    candidates: list[FrozenMap] = []
    for value, per_pair in sorted(added_by_value.items()):
        if removed_by_value.get(value, 0) or len(per_pair) < 3:
            continue
        ordered_pairs = tuple(sorted(per_pair))
        component_counts = tuple(
            _orthogonal_component_count(frozenset(per_pair[index]))
            for index in ordered_pairs
        )
        candidates.append(
            FrozenMap(
                {
                    "palette_value": value,
                    "first_pair_index": ordered_pairs[0],
                    "last_pair_index": ordered_pairs[-1],
                    "added_pair_indices": ordered_pairs,
                    "added_pair_count": len(ordered_pairs),
                    "added_pixel_count": sum(len(per_pair[index]) for index in ordered_pairs),
                    "split_pair_indices": tuple(
                        index
                        for index, component_count in zip(ordered_pairs, component_counts)
                        if component_count > 1
                    ),
                    "maximum_added_component_count": max(component_counts),
                }
            )
        )

    unique = candidates[0] if len(candidates) == 1 else FrozenMap()
    first_growth_pair_value = unique.get("first_pair_index")
    first_growth_pair = (
        int(first_growth_pair_value)
        if isinstance(first_growth_pair_value, int)
        and not isinstance(first_growth_pair_value, bool)
        else -1
    )
    growth_value = unique.get("palette_value")
    non_growth_transitions = tuple(
        transition
        for transition in transitions
        if first_growth_pair >= 0
        and int(transition["pair_index"]) >= first_growth_pair
        and int(transition["to_value"]) != growth_value
        and int(transition["from_value"]) != growth_value
    )
    terminal = frames[-1]
    first = frames[0]
    bottom_edge_change_count = sum(
        int(first[-1][col] != terminal[-1][col]) for col in range(len(first[-1]))
    )
    changed_pair_counts = tuple(
        len(_changed_positions(before, after))
        for before, after in zip(frames, frames[1:])
    )
    return FrozenMap(
        {
            "ordered_palette_transition_records": tuple(transitions),
            "ordered_palette_transition_record_count": len(transitions),
            "ordered_palette_transition_overflow_count": overflow_count,
            "ordered_palette_transition_inventory_complete": overflow_count == 0,
            "monotone_palette_growth_candidates": tuple(candidates),
            "monotone_palette_growth_candidate_count": len(candidates),
            "unique_monotone_palette_growth_value": growth_value,
            "unique_monotone_palette_growth_added_pair_count": int(
                unique.get("added_pair_count") or 0
            ),
            "unique_monotone_palette_growth_split_pair_count": len(
                tuple(unique.get("split_pair_indices") or ())
            ),
            "unique_monotone_palette_growth_maximum_added_component_count": int(
                unique.get("maximum_added_component_count") or 0
            ),
            "non_growth_palette_transition_after_growth_start_count": len(
                non_growth_transitions
            ),
            "non_growth_palette_transition_after_growth_start_pair_count": len(
                {int(item["pair_index"]) for item in non_growth_transitions}
            ),
            "non_growth_palette_transition_after_growth_start_records": (
                non_growth_transitions
            ),
            "terminal_pair_is_stable": bool(
                changed_pair_counts and changed_pair_counts[-1] == 0
            ),
            "final_bottom_edge_change_count": bottom_edge_change_count,
        }
    )


def _near_components(positions: frozenset[Position]) -> tuple[tuple[Position, ...], ...]:
    """Group one animated morphology even when its ink has one-cell gaps."""

    remaining = set(positions)
    components: list[tuple[Position, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque((start,))
        component: list[Position] = []
        while queue:
            row, col = queue.popleft()
            component.append((row, col))
            neighbors = tuple(
                position
                for position in remaining
                if abs(position[0] - row) <= 1 and abs(position[1] - col) <= 1
            )
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: item[0]))


def _box(positions: tuple[Position, ...]) -> BoundingBox:
    return BoundingBox(
        top=min(row for row, _ in positions),
        left=min(col for _, col in positions),
        bottom=max(row for row, _ in positions),
        right=max(col for _, col in positions),
    )


def _canonical_pattern(
    positions: tuple[Position, ...], grid: Grid
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    box = _box(positions)
    value_index: dict[int, int] = {}
    palette: list[int] = []
    pattern: list[tuple[int, int, int]] = []
    for row, col in positions:
        value = int(grid[row][col])
        if value not in value_index:
            value_index[value] = len(value_index)
            palette.append(value)
        pattern.append((row - box.top, col - box.left, value_index[value]))
    return tuple(pattern), tuple(palette)


def _homologous_occurrences(
    *,
    before: Grid,
    positions: tuple[Position, ...],
    restored: Grid,
    morphology_digest: str,
    maximum_occurrences: int = 128,
) -> tuple[OrderedAnimationHomologousOccurrence, ...]:
    box = _box(positions)
    pattern, _ = _canonical_pattern(positions, restored)
    height = box.bottom - box.top + 1
    width = box.right - box.left + 1
    mask = {(row, col) for row, col, _ in pattern}
    expected_classes = tuple(item[2] for item in pattern)
    occurrences: list[OrderedAnimationHomologousOccurrence] = []
    for top in range(0, len(before) - height + 1):
        for left in range(0, len(before[0]) - width + 1):
            candidate_values = tuple(
                int(before[top + row][left + col]) for row, col, _ in pattern
            )
            value_classes: dict[int, int] = {}
            candidate_classes = tuple(
                value_classes.setdefault(value, len(value_classes))
                for value in candidate_values
            )
            if candidate_classes != expected_classes:
                continue
            candidate_palette = tuple(dict.fromkeys(candidate_values))
            if any(
                int(before[top + row][left + col]) in candidate_palette
                for row in range(height)
                for col in range(width)
                if (row, col) not in mask
            ):
                continue
            halo_same_value = False
            for row in range(max(0, top - 1), min(len(before), top + height + 1)):
                for col in range(max(0, left - 1), min(len(before[0]), left + width + 1)):
                    if top <= row < top + height and left <= col < left + width:
                        continue
                    if int(before[row][col]) in candidate_palette:
                        halo_same_value = True
                        break
                if halo_same_value:
                    break
            if halo_same_value:
                continue
            occurrence_box = BoundingBox(
                top=top,
                left=left,
                bottom=top + height - 1,
                right=left + width - 1,
            )
            occurrence_ref = (
                "measurement.ordered_animation_homologue."
                f"{stable_digest((morphology_digest, occurrence_box, candidate_palette))[:16]}"
            )
            occurrences.append(
                OrderedAnimationHomologousOccurrence(
                    occurrence_ref=occurrence_ref,
                    bounding_box=occurrence_box,
                    observed_palette_values=candidate_palette,
                )
            )
            if len(occurrences) >= maximum_occurrences:
                return tuple(occurrences)
    return tuple(occurrences)


def _runs(values: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    runs: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return tuple(runs)


def _extent_for_palette(grid: Grid, box: BoundingBox, palette_value: int) -> int:
    rows = tuple(
        row
        for row in range(len(grid))
        if any(
            int(grid[row][col]) == palette_value
            for col in range(box.left, box.right + 1)
        )
    )
    overlapping = tuple(
        run
        for run in _runs(rows)
        if run[0] <= box.bottom + 1 and run[1] + 1 >= box.top
    )
    if not overlapping:
        return 0
    start, end = max(overlapping, key=lambda run: run[1] - run[0])
    return end - start + 1


def _bound_extent_candidates(
    grid: Grid, box: BoundingBox, palette_value: int
) -> tuple[int, ...]:
    rows_with_palette = tuple(
        row
        for row in range(len(grid))
        if any(
            int(grid[row][col]) == palette_value
            for col in range(box.left, box.right + 1)
        )
    )
    if not rows_with_palette or box.right - box.left + 1 < 3:
        return ()
    bottom = max(rows_with_palette)
    marker_rows: list[int] = []
    for row in range(len(grid)):
        values = tuple(int(grid[row][col]) for col in range(box.left, box.right + 1))
        even = values[0::2]
        odd = values[1::2]
        if odd and len(set(even)) == 1 and len(set(odd)) == 1 and even[0] != odd[0]:
            marker_rows.append(row)
    return tuple(sorted({bottom - row + 1 for row in marker_rows if row <= bottom}))[:8]


def _extent_changes(
    *,
    before: Grid,
    after: Grid,
    effect_pair_index: int,
    transition_ref: str,
) -> tuple[OrderedAnimationExtentChange, ...]:
    changed = _changed_positions(before, after)
    results: list[OrderedAnimationExtentChange] = []
    for positions in _near_components(changed):
        box = _box(positions)
        palette_candidates = tuple(
            sorted(
                {
                    int(before[row][col])
                    for row, col in positions
                }
                | {
                    int(after[row][col])
                    for row, col in positions
                }
            )
        )
        measured: list[tuple[int, int, int, tuple[int, ...]]] = []
        for palette_value in palette_candidates:
            before_extent = _extent_for_palette(before, box, palette_value)
            after_extent = _extent_for_palette(after, box, palette_value)
            if before_extent != after_extent:
                measured.append(
                    (
                        palette_value,
                        before_extent,
                        after_extent,
                        _bound_extent_candidates(before, box, palette_value),
                    )
                )
        if not measured:
            continue
        measured.sort(
            key=lambda item: (
                -int(len(item[3]) == 2),
                -abs(item[2] - item[1]),
                item[0],
            )
        )
        palette_value, before_extent, after_extent, bound_extents = measured[0]
        extent_ref = (
            "measurement.ordered_animation_extent_change."
            f"{stable_digest((transition_ref, effect_pair_index, positions, palette_value))[:16]}"
        )
        results.append(
            OrderedAnimationExtentChange(
                extent_change_ref=extent_ref,
                effect_pair_index=effect_pair_index,
                bounding_box=box,
                changed_pixel_count=len(positions),
                palette_value=palette_value,
                signed_extent_delta=after_extent - before_extent,
                before_extent=before_extent,
                after_extent=after_extent,
                bound_extent_candidates=bound_extents,
                full_rectangular_change=(
                    len(positions)
                    == (box.bottom - box.top + 1) * (box.right - box.left + 1)
                ),
            )
        )
    return tuple(results)


def _is_bracketed_extent_change(change: OrderedAnimationExtentChange) -> bool:
    if len(change.bound_extent_candidates) != 2:
        return False
    lower, upper = sorted(change.bound_extent_candidates)
    return (
        lower <= change.before_extent <= upper
        and lower <= change.after_extent <= upper
        and change.signed_extent_delta != 0
    )


def _bounded_quantity_key(change: OrderedAnimationExtentChange) -> tuple[object, ...]:
    box = change.bounding_box
    height = box.bottom - box.top + 1
    width = box.right - box.left + 1
    changed_extent = abs(change.signed_extent_delta)
    if height == changed_extent:
        return ("vertical_extent", change.palette_value, box.left, box.right)
    if width == changed_extent:
        return ("horizontal_extent", change.palette_value, box.top, box.bottom)
    return (
        "unresolved_extent_axis",
        change.palette_value,
        box.top,
        box.left,
        box.bottom,
        box.right,
    )


def _ordered_animation_candidates(
    *, frames: tuple[Grid, ...], transition_ref: str
) -> tuple[OrderedAnimationGlyphCandidate, ...]:
    pair_changes = tuple(
        _changed_positions(before, after) for before, after in zip(frames, frames[1:])
    )
    activation_starts = tuple(
        index
        for index in range(max(0, len(pair_changes) - 2))
        if pair_changes[index]
        and not pair_changes[index + 1]
        and pair_changes[index] == pair_changes[index + 2]
    )
    grouped: dict[tuple[int, int, int, int, str], dict[str, object]] = {}
    for pair_index in activation_starts:
        restored_frame = frames[pair_index + 3]
        for positions in _near_components(pair_changes[pair_index]):
            box = _box(positions)
            pattern, palette = _canonical_pattern(positions, restored_frame)
            morphology_digest = stable_digest(pattern)[:16]
            key = (box.top, box.left, box.bottom, box.right, morphology_digest)
            bucket = grouped.setdefault(
                key,
                {
                    "positions": positions,
                    "box": box,
                    "pattern": pattern,
                    "palette": palette,
                    "activation_indices": [],
                    "extent_changes": [],
                    "last_local_frame": restored_frame,
                },
            )
            bucket["activation_indices"].append(pair_index)  # type: ignore[union-attr]
            next_pair = pair_index + 3
            if next_pair >= len(pair_changes) or not pair_changes[next_pair]:
                continue
            if next_pair in activation_starts:
                continue
            stable_after_index = next_pair + 1
            if (
                next_pair + 1 < len(pair_changes)
                and pair_changes[next_pair + 1]
                and next_pair + 1 not in activation_starts
            ):
                stable_after_index = next_pair + 2
            stable_after_index = min(stable_after_index, len(frames) - 1)
            stable_after = frames[stable_after_index]
            bucket["extent_changes"].extend(  # type: ignore[union-attr]
                _extent_changes(
                    before=restored_frame,
                    after=stable_after,
                    effect_pair_index=next_pair,
                    transition_ref=transition_ref,
                )
            )
            bucket["last_local_frame"] = stable_after

    results: list[OrderedAnimationGlyphCandidate] = []
    before_action = frames[0]
    for key, bucket in sorted(grouped.items()):
        if len(bucket["extent_changes"]) > 64:  # type: ignore[arg-type]
            # The hypothesis exceeds the declared local structural bound.  Do
            # not truncate causal evidence or crash the whole retrospection.
            continue
        positions = bucket["positions"]
        box = bucket["box"]
        palette = bucket["palette"]
        morphology_digest = key[-1]
        last_local_frame = bucket["last_local_frame"]
        restored_values = tuple(
            int(frames[bucket["activation_indices"][0] + 3][row][col])  # type: ignore[index]
            for row, col in positions  # type: ignore[union-attr]
        )
        present_before = all(
            int(before_action[row][col]) == value
            for (row, col), value in zip(positions, restored_values)  # type: ignore[arg-type]
        )
        persists_locally = all(
            int(last_local_frame[row][col]) == value  # type: ignore[index]
            for (row, col), value in zip(positions, restored_values)  # type: ignore[arg-type]
        )
        glyph_ref = (
            "measurement.ordered_animation_glyph."
            f"{stable_digest((transition_ref, key))[:16]}"
        )
        results.append(
            OrderedAnimationGlyphCandidate(
                glyph_candidate_ref=glyph_ref,
                morphology_digest=morphology_digest,
                destination_bounding_box=box,  # type: ignore[arg-type]
                observed_palette_values=palette,  # type: ignore[arg-type]
                activation_pair_indices=tuple(bucket["activation_indices"]),  # type: ignore[arg-type]
                newly_present_after_first_activation=not present_before,
                present_before_action=present_before,
                persists_after_last_local_effect=persists_locally,
                homologous_occurrences=_homologous_occurrences(
                    before=before_action,
                    positions=positions,  # type: ignore[arg-type]
                    restored=frames[bucket["activation_indices"][0] + 3],  # type: ignore[index]
                    morphology_digest=morphology_digest,
                ),
                extent_changes=tuple(bucket["extent_changes"]),  # type: ignore[arg-type]
            )
        )
    return tuple(results[:64])


def measure_transition_phenomenon_inventory(
    value: TransitionPhenomenonInventoryInput,
) -> TransitionPhenomenonInventoryMeasurements:
    """Account for every changed cell without assigning causal meaning in Python."""

    frames = (value.before, *value.intermediate_frames, value.after)
    height = max(frame.height for frame in frames)
    width = max(frame.width for frame in frames)
    ordered_grids = tuple(_grid(frame, height=height, width=width) for frame in frames)
    ordered_effectful_pair_indices = tuple(
        pair_index
        for pair_index, (pair_before, pair_after) in enumerate(
            zip(ordered_grids, ordered_grids[1:])
        )
        if pair_before != pair_after
    )
    ordered_effectful_pair_count = len(ordered_effectful_pair_indices)
    ordered_last_effectful_pair_index = (
        ordered_effectful_pair_indices[-1]
        if ordered_effectful_pair_indices
        else -1
    )
    ordered_pair_count = max(0, len(ordered_grids) - 1)
    ordered_stable_suffix_pair_count = (
        ordered_pair_count - ordered_last_effectful_pair_index - 1
        if ordered_effectful_pair_indices
        else ordered_pair_count
    )
    prior_effectful_pair_count = int(
        value.prior_same_declared_interaction_effectful_pair_count
    )
    terminal_effectful_pair_deficit = max(
        0,
        prior_effectful_pair_count - ordered_effectful_pair_count,
    )
    official_terminal_effectful_prefix_shorter = bool(
        value.official_terminal_success
        and value.terminal_packet_excludes_next_level_scene
        and ordered_effectful_pair_count > 0
        and terminal_effectful_pair_deficit > 0
    )
    ordered_animation_glyph_candidates = _ordered_animation_candidates(
        frames=ordered_grids,
        transition_ref=value.transition_ref,
    )
    ordered_glyph_activation_context = measure_ordered_glyph_activation_episodes(
        before=value.before,
        intermediate_frames=value.intermediate_frames,
        after=value.after,
    )
    ordered_palette_delta_facts = _ordered_palette_delta_facts(ordered_grids)

    trace_by_position: dict[tuple[int, int], tuple[int, ...]] = {}
    unchanged_pixel_count = 0
    for row in range(height):
        for col in range(width):
            trace = tuple(
                (
                    int(frame.rows[row][col])
                    if row < frame.height and col < frame.width
                    else None
                )
                for frame in frames
            )
            if len(set(trace)) == 1:
                unchanged_pixel_count += 1
            else:
                trace_by_position[(row, col)] = trace

    components: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    remaining = set(trace_by_position)
    while remaining:
        start = min(remaining)
        signature = trace_by_position[start]
        queue = deque((start,))
        remaining.remove(start)
        positions: list[tuple[int, int]] = []
        while queue:
            current = queue.popleft()
            positions.append(current)
            row, col = current
            for neighbor in (
                (row - 1, col),
                (row, col - 1),
                (row, col + 1),
                (row + 1, col),
            ):
                if neighbor in remaining and trace_by_position[neighbor] == signature:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append((signature, tuple(sorted(positions))))

    components.sort(key=lambda item: (item[1][0], len(item[1]), item[0]))
    retained_components = components[: value.maximum_phenomenon_groups]
    overflow_components = components[value.maximum_phenomenon_groups :]
    groups: list[TransitionPhenomenonGroup] = []
    for signature, positions in retained_components:
        changed_pairs = tuple(
            index
            for index, (before_value, after_value) in enumerate(
                zip(signature, signature[1:])
            )
            if before_value != after_value
        )
        top = min(row for row, _ in positions)
        left = min(col for _, col in positions)
        bottom = max(row for row, _ in positions)
        right = max(col for _, col in positions)
        trace_digest = stable_digest(signature)[:16]
        phenomenon_ref = (
            "measurement.transition_phenomenon."
            f"{stable_digest((value.transition_ref, trace_digest, positions))[:16]}"
        )
        groups.append(
            TransitionPhenomenonGroup(
                phenomenon_ref=phenomenon_ref,
                trace_digest=trace_digest,
                pixel_count=len(positions),
                changed_pair_indices=changed_pairs,
                first_changed_pair_index=changed_pairs[0],
                last_changed_pair_index=changed_pairs[-1],
                returns_to_initial=signature[-1] == signature[0],
                differs_in_settled_frame=signature[-1] != signature[0],
                bounding_box=BoundingBox(
                    top=top,
                    left=left,
                    bottom=bottom,
                    right=right,
                ),
            )
        )

    retained_refs = {group.phenomenon_ref for group in groups}
    relations: list[TransitionPhenomenonRelation] = []
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            first_box = first.bounding_box
            second_box = second.bounding_box
            row_overlap = not (
                first_box.bottom < second_box.top
                or second_box.bottom < first_box.top
            )
            col_overlap = not (
                first_box.right < second_box.left
                or second_box.right < first_box.left
            )
            relations.append(
                TransitionPhenomenonRelation(
                    first_phenomenon_ref=first.phenomenon_ref,
                    second_phenomenon_ref=second.phenomenon_ref,
                    shares_changed_pair=bool(
                        set(first.changed_pair_indices)
                        & set(second.changed_pair_indices)
                    ),
                    first_finishes_before_second_starts=(
                        first.last_changed_pair_index
                        < second.first_changed_pair_index
                    ),
                    second_finishes_before_first_starts=(
                        second.last_changed_pair_index
                        < first.first_changed_pair_index
                    ),
                    bounding_boxes_overlap=row_overlap and col_overlap,
                    bounding_boxes_touch_orthogonally=bool(
                        (
                            row_overlap
                            and (
                                first_box.right + 1 == second_box.left
                                or second_box.right + 1 == first_box.left
                            )
                        )
                        or (
                            col_overlap
                            and (
                                first_box.bottom + 1 == second_box.top
                                or second_box.bottom + 1 == first_box.top
                            )
                        )
                    ),
                )
            )
    links_by_group: dict[str, list[object]] = {}
    for link in value.explanation_links:
        if link.phenomenon_ref not in retained_refs:
            raise ValueError("explanation link names an uninventoried phenomenon")
        links_by_group.setdefault(link.phenomenon_ref, []).append(link)
    linked_refs = tuple(sorted(links_by_group))
    explained_refs = tuple(
        sorted(
            group_ref
            for group_ref, links in links_by_group.items()
            if any(link.evidence_refs and link.falsifier_ref for link in links)
        )
    )
    unexplained_refs = tuple(
        group.phenomenon_ref
        for group in groups
        if group.phenomenon_ref not in set(explained_refs)
    )
    transient_refs = tuple(
        group.phenomenon_ref for group in groups if group.returns_to_initial
    )
    settled_refs = tuple(
        group.phenomenon_ref for group in groups if group.differs_in_settled_frame
    )
    overflow_pixel_count = sum(len(positions) for _, positions in overflow_components)
    changed_pixel_count = len(trace_by_position)
    phenomenon_group_count = len(components)
    inventory_complete = not overflow_components
    explanatory_animation_candidates = tuple(
        candidate
        for candidate in ordered_animation_glyph_candidates
        if any(_is_bracketed_extent_change(change) for change in candidate.extent_changes)
    )
    ordered_animation_candidate_facts = tuple(
        FrozenMap(
            {
                "glyph_candidate_ref": candidate.glyph_candidate_ref,
                "morphology_digest": candidate.morphology_digest,
                "destination_bounding_box": (
                    candidate.destination_bounding_box.top,
                    candidate.destination_bounding_box.left,
                    candidate.destination_bounding_box.bottom,
                    candidate.destination_bounding_box.right,
                ),
                "observed_palette_values": candidate.observed_palette_values,
                "activation_pair_indices": tuple(
                    pair_index
                    for pair_index in candidate.activation_pair_indices
                    if pair_index
                    <= max(
                        change.effect_pair_index
                        for change in candidate.extent_changes
                        if _is_bracketed_extent_change(change)
                    )
                ),
                "newly_present_after_first_activation": (
                    candidate.newly_present_after_first_activation
                ),
                "present_before_action": candidate.present_before_action,
                "persists_after_last_local_effect": (
                    candidate.persists_after_last_local_effect
                ),
                "homologous_occurrences": tuple(
                    FrozenMap(
                        {
                            "occurrence_ref": occurrence.occurrence_ref,
                            "bounding_box": (
                                occurrence.bounding_box.top,
                                occurrence.bounding_box.left,
                                occurrence.bounding_box.bottom,
                                occurrence.bounding_box.right,
                            ),
                            "observed_palette_values": occurrence.observed_palette_values,
                        }
                    )
                    for occurrence in candidate.homologous_occurrences
                ),
                "extent_changes": tuple(
                    FrozenMap(
                        {
                            "extent_change_ref": change.extent_change_ref,
                            "effect_pair_index": change.effect_pair_index,
                            "bounding_box": (
                                change.bounding_box.top,
                                change.bounding_box.left,
                                change.bounding_box.bottom,
                                change.bounding_box.right,
                            ),
                            "palette_value": change.palette_value,
                            "signed_extent_delta": change.signed_extent_delta,
                            "before_extent": change.before_extent,
                            "after_extent": change.after_extent,
                            "bound_extent_candidates": change.bound_extent_candidates,
                            "full_rectangular_change": change.full_rectangular_change,
                        }
                    )
                    for change in candidate.extent_changes
                    if _is_bracketed_extent_change(change)
                ),
            }
        )
        for candidate in explanatory_animation_candidates
    )
    marked_extent_changes = tuple(
        change
        for candidate in ordered_animation_glyph_candidates
        for change in candidate.extent_changes
        if _is_bracketed_extent_change(change)
    )
    one_activation_one_marked_extent_absent_after_refs = tuple(
        candidate.glyph_candidate_ref
        for candidate in ordered_animation_glyph_candidates
        if (
        len(candidate.activation_pair_indices) == 1
        and not candidate.persists_after_last_local_effect
        and sum(
            _is_bracketed_extent_change(change)
            for change in candidate.extent_changes
        )
        == 1
        )
    )
    replayed_one_marked_extent_persistent_refs = tuple(
        candidate.glyph_candidate_ref
        for candidate in ordered_animation_glyph_candidates
        if (
            len(candidate.activation_pair_indices) >= 2
            or candidate.present_before_action
        )
        and candidate.persists_after_last_local_effect
        and sum(
            _is_bracketed_extent_change(change)
            for change in candidate.extent_changes
        )
        == 1
    )
    replayed_opposite_marked_extent_pair_persistent_refs = tuple(
        candidate.glyph_candidate_ref
        for candidate in ordered_animation_glyph_candidates
        if (
            len(candidate.activation_pair_indices) >= 2
            or candidate.present_before_action
        )
        and candidate.persists_after_last_local_effect
        and (
            lambda deltas: len(deltas) == 2 and deltas[0] * deltas[1] < 0
        )(
            tuple(
                change.signed_extent_delta
                for change in candidate.extent_changes
                if _is_bracketed_extent_change(change)
            )
        )
    )
    bounded_quantity_keys = {
        _bounded_quantity_key(change) for change in marked_extent_changes
    }
    descriptive = FrozenMap(
        {
            **ordered_glyph_activation_context,
            **ordered_palette_delta_facts,
            "transition_ref": value.transition_ref,
            "action_ref": value.action_ref,
            "packet_frame_count": len(frames),
            "ordered_effectful_pair_count": ordered_effectful_pair_count,
            "ordered_last_effectful_pair_index": (
                ordered_last_effectful_pair_index
            ),
            "ordered_stable_suffix_pair_count": (
                ordered_stable_suffix_pair_count
            ),
            "prior_same_declared_interaction_effectful_pair_count": (
                prior_effectful_pair_count
            ),
            "same_declared_interaction_effectful_history_present": (
                prior_effectful_pair_count > 0
            ),
            "terminal_effectful_pair_deficit": terminal_effectful_pair_deficit,
            "official_terminal_effectful_prefix_shorter_than_prior_packet": (
                official_terminal_effectful_prefix_shorter
            ),
            "changed_pixel_count": changed_pixel_count,
            "unchanged_pixel_count": unchanged_pixel_count,
            "phenomenon_group_count": phenomenon_group_count,
            "phenomenon_relation_count": len(relations),
            "simultaneous_relation_count": sum(
                relation.shares_changed_pair for relation in relations
            ),
            "ordered_relation_count": sum(
                relation.first_finishes_before_second_starts
                or relation.second_finishes_before_first_starts
                for relation in relations
            ),
            "retained_group_count": len(groups),
            "temporally_described_group_count": len(groups),
            "transient_group_count": len(transient_refs),
            "settled_change_group_count": len(settled_refs),
            "action_conditioned_linked_group_count": (
                len(groups) if value.action_ref is not None else 0
            ),
            "causally_explained_group_count": len(explained_refs),
            "unexplained_group_count": (
                len(unexplained_refs) + len(overflow_components)
            ),
            "unexplained_group_refs": unexplained_refs,
            "transient_group_refs": transient_refs,
            "settled_change_group_refs": settled_refs,
            "overflow_group_count": len(overflow_components),
            "overflow_pixel_count": overflow_pixel_count,
            "inventory_complete": inventory_complete,
            "has_observed_change": changed_pixel_count > 0,
            "has_causal_explanation": bool(explained_refs),
            "causal_explanation_complete": bool(
                inventory_complete and len(explained_refs) == len(groups)
            ),
            "ordered_animation_glyph_candidate_count": len(
                ordered_animation_glyph_candidates
            ),
            "ordered_animation_explanatory_candidate_count": len(
                explanatory_animation_candidates
            ),
            "ordered_animation_glyph_candidates": ordered_animation_candidate_facts,
            "ordered_animation_marked_extent_change_count": len(marked_extent_changes),
            "ordered_animation_marked_extent_refs": tuple(
                change.extent_change_ref for change in marked_extent_changes
            ),
            "one_activation_one_marked_extent_absent_after_count": (
                len(one_activation_one_marked_extent_absent_after_refs)
            ),
            "one_activation_one_marked_extent_absent_after_refs": (
                one_activation_one_marked_extent_absent_after_refs
            ),
            "replayed_one_marked_extent_count": len(
                replayed_one_marked_extent_persistent_refs
            ),
            "replayed_one_marked_extent_persistent_refs": (
                replayed_one_marked_extent_persistent_refs
            ),
            "replayed_opposite_marked_extent_pair_count": (
                len(replayed_opposite_marked_extent_pair_persistent_refs)
            ),
            "replayed_opposite_marked_extent_pair_persistent_refs": (
                replayed_opposite_marked_extent_pair_persistent_refs
            ),
            "ordered_animation_source_compartment_count": len(
                explanatory_animation_candidates
            ),
            "ordered_animation_bounded_quantity_count": len(bounded_quantity_keys),
            "ordered_animation_has_homologous_source_occurrences": any(
                candidate.homologous_occurrences
                for candidate in ordered_animation_glyph_candidates
            ),
        }
    )
    return TransitionPhenomenonInventoryMeasurements(
        transition_ref=value.transition_ref,
        action_ref=value.action_ref,
        packet_frame_count=len(frames),
        changed_pixel_count=changed_pixel_count,
        unchanged_pixel_count=unchanged_pixel_count,
        phenomenon_groups=tuple(groups),
        phenomenon_relations=tuple(relations),
        ordered_animation_glyph_candidates=ordered_animation_glyph_candidates,
        phenomenon_group_count=phenomenon_group_count,
        transient_group_refs=transient_refs,
        settled_change_group_refs=settled_refs,
        linked_group_refs=linked_refs,
        explained_group_refs=explained_refs,
        unexplained_group_refs=unexplained_refs,
        overflow_group_count=len(overflow_components),
        overflow_pixel_count=overflow_pixel_count,
        inventory_complete=inventory_complete,
        descriptive_facts=descriptive,
    )
