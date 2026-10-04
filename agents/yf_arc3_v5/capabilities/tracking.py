"""Pure temporal identity tracking over descriptive visual components.

The tracker deliberately uses a short ordered proof system instead of a
weighted assignment score. An identity is preserved only by an exact witness,
by a coherent group of exact translations, or by one unambiguous stationary /
one-axis extent transition. Ambiguity remains explicit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from functools import lru_cache
from itertools import combinations

from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentDescription,
    ExactMulticellPeerRelationInput,
    ExactMulticellPeerRelationMeasurement,
    ExactMulticellPeerRelationMeasurements,
    TemporalConcurrencyInput,
    TemporalConcurrencyMeasurements,
    TemporalReplaySequenceInput,
    TemporalReplaySequenceMeasurements,
    TemporalPairChangeMeasurement,
    TemporalTrackingInput,
    TemporalTrackingResult,
    TimelineEntityMeasurement,
    TimelineSample,
    TrackedComponent,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


_EXACT_PEER_PRIORITY_CRITERION_REFS = (
    "criterion.completion_proximity",
    "criterion.screen_space_importance",
    "criterion.observation_frequency",
    "criterion.first_observation_sequence",
    "criterion.maximal_collective_scope",
    "criterion.screen_reading_order",
)


def _first_observation_sequence(entity_refs: tuple[str, ...]) -> int:
    """Return the explicit tracker sequence; never derive order from a hash."""

    sequences: list[int] = []
    for entity_ref in entity_refs:
        prefix, separator, raw_sequence = entity_ref.rpartition(".")
        if not separator or not prefix.startswith("entity.visual"):
            raise ValueError(
                "exact peer priority requires an explicit visual entity sequence"
            )
        sequences.append(int(raw_sequence))
    return min(sequences)


def _cochange_present(ordered_by_entity: dict[str, dict[int, TimelineSample]]) -> bool:
    """Answer the declared existence question without materializing every pair.

    A common observation schedule lets one interval count all co-changing
    entities at once. Sparse, differing schedules retain the exact pairwise
    definition and stop at the first witness.
    """

    schedules: dict[tuple[int, ...], list[dict[int, TimelineSample]]] = defaultdict(list)
    schedule_by_entity: dict[str, tuple[int, ...]] = {}
    for entity_ref in sorted(ordered_by_entity):
        timeline = ordered_by_entity[entity_ref]
        schedule = tuple(timeline)
        schedules[schedule].append(timeline)
        schedule_by_entity[entity_ref] = schedule
    for schedule in sorted(schedules):
        timelines = schedules[schedule]
        if len(timelines) < 2:
            continue
        for before_index, after_index in zip(schedule, schedule[1:]):
            changed = 0
            for timeline in timelines:
                changed += timeline[before_index].state_ref != timeline[after_index].state_ref
                if changed > 1:
                    return True
    for left_ref, right_ref in combinations(sorted(ordered_by_entity), 2):
        if schedule_by_entity[left_ref] == schedule_by_entity[right_ref]:
            continue
        left = ordered_by_entity[left_ref]
        right = ordered_by_entity[right_ref]
        shared_frames = sorted(set(left).intersection(right))
        for before_index, after_index in zip(shared_frames, shared_frames[1:]):
            if (left[before_index].state_ref != left[after_index].state_ref
                    and right[before_index].state_ref != right[after_index].state_ref):
                return True
    return False


def _ordered_rendered_reversal_facts(recorded_frames, reverse_frames):
    """Exact immutable content-reference comparison; keep every repetition."""
    available = bool(recorded_frames and reverse_frames)
    same_count = available and len(recorded_frames) == len(reverse_frames)
    return {
        "rendered_history_comparison_available": available,
        "recorded_rendered_frame_count": len(recorded_frames),
        "candidate_reverse_rendered_frame_count": len(reverse_frames),
        "same_rendered_frame_count": same_count,
        "reverse_rendered_frame_order_exact": same_count and all(
            recorded_frames[len(recorded_frames) - index - 1] == frame_ref
            for index, frame_ref in enumerate(reverse_frames)
        ),
    }


def measure_temporal_concurrency(
    value: TemporalConcurrencyInput,
) -> TemporalConcurrencyMeasurements:
    """Measure ordered timelines and co-change without naming temporal mechanisms."""

    samples_by_entity: dict[str, list[TimelineSample]] = defaultdict(list)
    samples_by_frame: dict[int, list[TimelineSample]] = defaultdict(list)
    from agents.yf_arc3_v5.capabilities.temporal_cells import samples_from_current_observed_scene
    for sample in samples_from_current_observed_scene(value.samples):
        samples_by_entity[sample.entity_ref].append(sample)
        samples_by_frame[sample.frame_index].append(sample)

    timelines: list[TimelineEntityMeasurement] = []
    total_transitions = 0
    total_returns = 0
    duplicated_frames = 0
    action_conditioned_transitions = 0
    ordered_by_entity: dict[str, dict[int, TimelineSample]] = {}
    for entity_ref in sorted(samples_by_entity):
        ordered = sorted(
            samples_by_entity[entity_ref], key=lambda item: item.frame_index
        )
        ordered_by_entity[entity_ref] = {item.frame_index: item for item in ordered}
        transition_count = max(0, len(ordered) - 1)
        seen_states: set[str] = set()
        return_count = 0
        duplication_count = 0
        for index, sample in enumerate(ordered):
            if sample.state_ref in seen_states:
                return_count += 1
            seen_states.add(sample.state_ref)
            duplication_count += int(len(sample.instance_refs) > 1)
            if index > 0 and sample.preceding_action_ref is not None:
                action_conditioned_transitions += 1
        total_transitions += transition_count
        total_returns += return_count
        duplicated_frames += duplication_count
        timelines.append(
            TimelineEntityMeasurement(
                entity_ref=entity_ref,
                observed_frame_indices=tuple(item.frame_index for item in ordered),
                ordered_transition_count=transition_count,
                state_return_count=return_count,
                duplicated_instance_frame_count=duplication_count,
            )
        )

    pair_changes: list[TemporalPairChangeMeasurement] = []
    simultaneous_change_count = 0
    if value.requested_detail == "presence":
        simultaneous_change_present = _cochange_present(ordered_by_entity)
    else:
        for left_ref, right_ref in combinations(sorted(ordered_by_entity), 2):
            left = ordered_by_entity[left_ref]
            right = ordered_by_entity[right_ref]
            shared_frames = sorted(set(left).intersection(right))
            shared_intervals = 0
            simultaneous = 0
            for before_index, after_index in zip(shared_frames, shared_frames[1:]):
                shared_intervals += 1
                left_changed = left[before_index].state_ref != left[after_index].state_ref
                right_changed = (
                    right[before_index].state_ref != right[after_index].state_ref
                )
                simultaneous += int(left_changed and right_changed)
            if shared_intervals:
                simultaneous_change_count += simultaneous
                pair_changes.append(
                    TemporalPairChangeMeasurement(
                        left_entity_ref=left_ref,
                        right_entity_ref=right_ref,
                        shared_transition_interval_count=shared_intervals,
                        simultaneous_state_change_count=simultaneous,
                    )
                )
        simultaneous_change_present = simultaneous_change_count > 0

    occupancy_conflict_count = 0
    for frame_samples in (samples_by_frame[__yf_order_key] for __yf_order_key in sorted(samples_by_frame)):
        slot_entities: dict[str, set[str]] = defaultdict(set)
        for sample in frame_samples:
            for slot_ref in sample.occupied_slot_refs:
                slot_entities[slot_ref].add(sample.entity_ref)
        occupancy_conflict_count += sum(
            len(entity_refs) > 1 for entity_refs in (slot_entities[__yf_order_key] for __yf_order_key in sorted(slot_entities))
        )

    descriptive_facts = {
        "measurement_context_ref": value.measurement_context_ref,
        "entity_count": len(samples_by_entity),
        "frame_count": len(samples_by_frame),
        "sample_count": len(value.samples),
        "ordered_transition_count": total_transitions,
        "state_return_count": total_returns,
        "duplicated_instance_frame_count": duplicated_frames,
        "action_conditioned_transition_count": action_conditioned_transitions,
        "simultaneous_state_change_present": simultaneous_change_present,
        "occupancy_conflict_count": occupancy_conflict_count,
        "source_authorized_reset_observed": value.source_authorized_reset_observed,
        "observation_evidence_refs": value.observation_evidence_refs,
        "observation_revision": value.observation_revision,
        "replay_transition_count": 0,
        "measured_return_action_refs": (),
    }
    descriptive_facts.update(_ordered_rendered_reversal_facts(
        value.recorded_rendered_frame_refs, value.candidate_reverse_rendered_frame_refs
    ))
    descriptive_facts.update(_observed_instance_sequence_facts(value, samples_by_entity))
    descriptive_facts["return_recording_span_packs"] = _measure_current_return_spans(value, samples_by_entity)
    from agents.yf_arc3_v5.capabilities.temporal_cells import measure_disjoint_cell_changes
    cell_groups = tuple(tuple(sorted(rows, key=lambda sample: sample.frame_index))
                        for _, rows in sorted(samples_by_entity.items()) if any(sample.observed_cell_facts for sample in rows))
    descriptive_facts["contact_inventory_measurement_available"] = len(cell_groups) == 1
    if len(cell_groups) == 1:
        cell_rows = cell_groups[0]
        descriptive_facts.update(measure_disjoint_cell_changes(cell_rows, value.measured_translation_action_refs))
        descriptive_facts["contact_inventory_observation_frame_ref"] = cell_rows[-1].frame_ref
        descriptive_facts["current_observed_cell_facts"] = cell_rows[-1].observed_cell_facts
        descriptive_facts["contact_remote_cell_row_count"] = len(descriptive_facts.get("contact_remote_cell_rows", ()))
        descriptive_facts["contact_source_pattern_refs"] = tuple(sorted({row['source_pre_entry_pattern_ref']
            for row in descriptive_facts.get("contact_remote_cell_rows", ()) if row.get('source_pre_entry_pattern_ref')}))
        descriptive_facts["contact_source_pattern_orbit_refs"] = tuple(sorted({row['source_pre_entry_pattern_orbit_ref']
            for row in descriptive_facts.get("contact_remote_cell_rows", ()) if row.get('source_pre_entry_pattern_orbit_ref')}))
    if value.requested_detail == "full":
        descriptive_facts["simultaneous_state_change_count"] = simultaneous_change_count
    return TemporalConcurrencyMeasurements(
        entity_timelines=tuple(timelines),
        pair_changes=tuple(pair_changes),
        descriptive_facts=FrozenMap(descriptive_facts),
    )


def _measure_current_return_spans(value, grouped):
    """Exact observed spans at a fresh return; no invented replay instances.

    Keep several morphology measurements for later exact cell-core matching.
    A return command is just a measured discriminator here, never a reset rule.
    """
    if value.source_authorized_reset_observed:
        return ()
    packs = []
    for morphology_ref, samples in sorted(grouped.items()):
        ordered = sorted(samples, key=lambda row: row.frame_index)
        if not 4 <= len(ordered) <= 65 or len(ordered[0].instance_refs) != 1:
            continue
        if any(right.frame_index != left.frame_index + 1 for left, right in zip(ordered, ordered[1:])):
            continue
        original = ordered[0].instance_refs[0]
        if any(row.frame_ref is None or original not in row.instance_refs or original not in row.instance_bounds for row in ordered):
            continue
        boundaries, operator = [], None
        for index in range(2, len(ordered)):
            row, prior = ordered[index], ordered[index - 1]
            action = row.preceding_action_ref
            if (row.instance_pose_refs[original] != ordered[0].instance_pose_refs[original]
                    or prior.instance_pose_refs[original] == ordered[0].instance_pose_refs[original]
                    or action is None or action == value.source_reset_action_ref
                    or (operator is not None and action != operator)
                    or (operator is None and action in {r.preceding_action_ref for r in ordered[1:index]})):
                continue
            operator = action
            boundaries.append(index)
        if not boundaries or boundaries[-1] != len(ordered) - 1 or len(boundaries) > 3:
            continue
        spans, start = [], 0
        for end in boundaries:
            spans.append(FrozenMap({
                "recording_frame_ref": ordered[start].frame_ref,
                "recording_frame_index": ordered[start].frame_index,
                "recorded_instance_bounds": tuple(row.instance_bounds[original]
                    for row in ordered[start:end]),
            }))
            start = end
        packs.append(FrozenMap({
            "morphology_ref": morphology_ref,
            "observation_frame_ref": ordered[-1].frame_ref,
            "phase_frame_index": ordered[-1].frame_index,
            "phase_action_ref": operator,
            "current_original_bounds": ordered[-1].instance_bounds[original],
            "recording_span_rows": tuple(spans),
        }))
        if len(packs) > 4:
            return ()
    return tuple(packs)


def _measure_unique_recorded_pose_traces(segments, after, original, phase):
    """Exact recorded-pose correspondence, independent of tracker labels.

    Return no witness on missing, overlapping or non-unique visible matches.
    Keep a constant-label divergent trace in the ordinary comparator instead.
    """
    if not after or not 1 <= len(segments) <= 3 or len(after) > 65:
        return ()
    traces = []
    for start_index, recorded in segments:
        trace, first_value = [], None
        for index, sample in enumerate(after):
            expected_pose = recorded[min(index,len(recorded)-1)].instance_pose_refs[original]
            matches = tuple(ref for ref in sample.instance_refs if ref != original
                and sample.instance_pose_refs[ref] == expected_pose
                and sample.instance_value_refs[ref] != phase.instance_value_refs[original]
                and (first_value is None or sample.instance_value_refs[ref] == first_value))
            if len(matches) != 1:
                return ()
            ref = matches[0]
            first_value = sample.instance_value_refs[ref]
            trace.append(ref)
        traces.append((start_index, recorded, trace[-1], tuple(trace)))
    if any(len(frozenset(trace[index] for _,_,_,trace in traces)) != len(traces)
           for index in range(len(after))):
        return ()
    return tuple(traces)


def _observed_instance_sequence_facts(value, grouped):
    """Compare bounded recording segments with current instances, preserving ambiguity.

    Phase boundaries are measured returns under a different command, not reset
    meanings. Multiple segments require a unique exact prefix assignment; colors
    only preserve the measured instance distinction, never assign its role.
    """
    requests = []
    group_count = 0
    for morphology_ref in sorted(grouped):
        ordered = sorted(grouped[morphology_ref], key=lambda sample: sample.frame_index)
        if len(ordered) > 65:
            return {"observed_instance_sequence_comparison_available": False,
                    "observed_instance_sequence_candidate_count": 0,
                    "observed_instance_sequence_bound_reached": True}
        if len(ordered) < 4 or any(not sample.instance_pose_refs or not sample.instance_value_refs for sample in ordered):
            continue
        first = ordered[0]
        if len(first.instance_refs) != 1:
            continue
        original = first.instance_refs[0]
        phase_indices = []
        phase_action = None
        for phase_index in range(2, len(ordered) - 1):
            phase = ordered[phase_index]
            prior = ordered[phase_index - 1]
            if (original not in phase.instance_refs or original not in prior.instance_refs
                    or phase.instance_pose_refs[original] != first.instance_pose_refs[original]
                    or prior.instance_pose_refs[original] == first.instance_pose_refs[original]
                    or phase.preceding_action_ref is None
                    or (phase_action is not None and phase.preceding_action_ref != phase_action)
                    or (phase_action is None and phase.preceding_action_ref in {
                        sample.preceding_action_ref for sample in ordered[1:phase_index]})):
                continue
            phase_action = phase.preceding_action_ref
            phase_indices.append(phase_index)
        if not phase_indices:
            continue
        if len(phase_indices) > 3:
            return {"observed_instance_sequence_comparison_available": False,
                    "observed_instance_sequence_candidate_count": 4,
                    "observed_instance_sequence_bound_reached": True}
        phase_index = phase_indices[-1]
        phase = ordered[phase_index]
        after = ordered[phase_index + 1:]
        if (any(right.frame_index != left.frame_index + 1 for left, right in zip(ordered, ordered[1:]))
                or any(original not in sample.instance_refs for sample in after)):
            continue
        segments = []
        start_index = 0
        for end_index in phase_indices:
            recorded = ordered[start_index + 1:end_index]
            if not recorded or any(original not in sample.instance_refs for sample in recorded):
                segments = []
                break
            segments.append((start_index, recorded))
            start_index = end_index
        if not segments:
            continue
        candidates = tuple(ref for ref in after[0].instance_refs if ref != original
                           and all(ref in sample.instance_refs for sample in after)
                           and after[0].instance_value_refs[ref] != phase.instance_value_refs[original]
                           and all(sample.instance_value_refs[ref] == after[0].instance_value_refs[ref]
                                   for sample in after))
        pairs = []
        if len(candidates) == len(segments):
            for start_index, recorded in segments:
                compatible = tuple(ref for ref in candidates if len(candidates) == 1 or all(
                    prior.instance_pose_refs[original] == current.instance_pose_refs[ref]
                    for prior, current in zip(recorded, after)))
                if len(compatible) != 1:
                    pairs = []
                    break
                pairs.append((start_index, recorded, compatible[0], tuple(compatible[0] for _ in after)))
        if not pairs:
            pairs = _measure_unique_recorded_pose_traces(segments, after, original, phase)
        if not pairs or len({ref for _, _, ref, _ in pairs}) != len(pairs):
            continue
        group_count += 1
        for start_index, recorded, other, trace in pairs:
            recorded_bounds = (
                (ordered[start_index].instance_bounds[original],
                 *(sample.instance_bounds[original] for sample in recorded))
                if ordered[start_index].instance_bounds and all(sample.instance_bounds for sample in recorded)
                else ()
            )
            candidate_bounds = tuple(sample.instance_bounds[ref] for sample,ref in zip(after,trace)) if all(
                ref in sample.instance_bounds for sample,ref in zip(after,trace)) else ()
            request = TemporalReplaySequenceInput(
                observation_frame_ref=ordered[-1].frame_ref,
                measurement_context_ref=f"{value.measurement_context_ref}:instance-sequence:{morphology_ref}:{ordered[start_index].frame_index}:{phase.frame_index}",
                sequence_measurement_kind="ordered_observed_states",
                recorded_observed_state_refs=tuple(sample.instance_pose_refs[original] for sample in recorded),
                candidate_observed_state_refs=tuple(sample.instance_pose_refs[ref] for sample,ref in zip(after,trace)),
                recorded_instance_bounds=recorded_bounds,
                candidate_instance_bounds=candidate_bounds,
                recorded_action_refs=tuple(sample.preceding_action_ref for sample in recorded)
                    if all(sample.preceding_action_ref for sample in recorded) else (),
                recording_frame_index=ordered[start_index].frame_index,
                replay_phase_frame_index=phase.frame_index,
                phase_action_ref=phase.preceding_action_ref,
                recorded_entity_ref=original, candidate_entity_ref=other,
                candidate_identifier_change_count=sum(left != right for left,right in zip(trace,trace[1:])),
                phase_action_conditioned=True, state_return_observed=True,
                duplicate_instance_observed=True,
                independent_command_observed=any(
                    current.preceding_action_ref is not None
                    and prior.preceding_action_ref is not None
                    and current.preceding_action_ref != prior.preceding_action_ref
                    for prior, current in zip(recorded, after)
                ),
                source_authorized_reset_observed=(
                    phase.preceding_action_ref == value.source_reset_action_ref
                    if value.source_reset_action_ref else None
                ),
            )
            requests.append(request)
            if len(requests) > 3:
                return {"observed_instance_sequence_comparison_available": False,
                        "observed_instance_sequence_candidate_count": 4,
                        "observed_instance_sequence_bound_reached": True}
    facts = {"observed_instance_sequence_comparison_available": group_count == 1 and bool(requests),
             "observed_instance_sequence_candidate_count": len(requests),
             "observed_instance_sequence_pack_count": len(requests) if group_count == 1 else 0,
             "observed_instance_sequence_bound_reached": False}
    if group_count == 1:
        facts['measured_return_action_refs'] = tuple(dict.fromkeys(request.phase_action_ref for request in requests
                                                                if request.phase_action_ref is not None))
        for index, request in enumerate(requests):
            key = "observed_instance_sequence_facts" if index == 0 else f"observed_instance_sequence_facts_{index}"
            facts[key] = measure_temporal_replay_sequence(request).descriptive_facts
    return facts


def measure_temporal_replay_sequence(
    value: TemporalReplaySequenceInput,
) -> TemporalReplaySequenceMeasurements:
    """Compare one explicit ordered projection without inferring a hidden origin."""

    observed_states = value.sequence_measurement_kind == "ordered_observed_states"
    recorded_refs = (value.recorded_observed_state_refs if observed_states
                     else value.recorded_transition_delta_refs)
    candidate_refs = (value.candidate_observed_state_refs if observed_states
                      else value.replay_transition_delta_refs)
    comparable = min(
        len(recorded_refs),
        len(candidate_refs),
    )
    first_divergence: int | None = None
    for index in range(comparable):
        if (
            recorded_refs[index] != candidate_refs[index]
        ):
            first_divergence = index
            break
    matching = first_divergence if first_divergence is not None else comparable
    exhaustion_index = len(recorded_refs)
    exhausted = len(candidate_refs) >= exhaustion_index
    endpoint_states = value.candidate_observed_state_refs if observed_states else value.replay_state_refs
    endpoint_index = exhaustion_index - 1 if observed_states else exhaustion_index
    post_states = (
        endpoint_states[endpoint_index + 1 :] if exhausted else ()
    )
    endpoint_state = endpoint_states[endpoint_index] if exhausted else None
    persistent = sum(state_ref == endpoint_state for state_ref in post_states)
    recorded_frames = value.recorded_rendered_frame_refs
    reverse_frames = value.candidate_reverse_rendered_frame_refs
    rendered_facts = _ordered_rendered_reversal_facts(recorded_frames, reverse_frames)
    facts = FrozenMap(
        {
            "measurement_context_ref": value.measurement_context_ref,
            "sequence_measurement_kind": value.sequence_measurement_kind,
            "observation_frame_ref": value.observation_frame_ref,
            "remaining_recorded_state_count": max(0, len(recorded_refs) - matching),
            "recorded_observed_state_count": len(value.recorded_observed_state_refs),
            "candidate_observed_state_count": len(value.candidate_observed_state_refs),
            "independent_command_observed": value.independent_command_observed,
            "recorded_entity_ref": value.recorded_entity_ref,
            "candidate_entity_ref": value.candidate_entity_ref,
            "candidate_identifier_change_count": value.candidate_identifier_change_count,
            "recorded_instance_bounds": tuple(box.model_dump(mode='python') for box in value.recorded_instance_bounds),
            "candidate_instance_bounds": tuple(box.model_dump(mode='python') for box in value.candidate_instance_bounds),
            "recorded_action_refs": value.recorded_action_refs,
            "recording_frame_index": value.recording_frame_index,
            "replay_phase_frame_index": value.replay_phase_frame_index,
            "phase_action_ref": value.phase_action_ref,
            "recorded_transition_count": len(value.recorded_transition_delta_refs),
            "replay_transition_count": len(value.replay_transition_delta_refs),
            "comparable_prefix_count": comparable,
            "matching_prefix_count": matching,
            "prefix_exact": (
                comparable == len(recorded_refs)
                and first_divergence is None
            ),
            "first_divergence_index": first_divergence,
            "has_divergence": first_divergence is not None,
            "recorded_sequence_exhausted": exhausted,
            "phase_action_conditioned": value.phase_action_conditioned,
            "state_return_observed": value.state_return_observed,
            "duplicate_instance_observed": value.duplicate_instance_observed,
            "post_exhaustion_observation_count": len(post_states),
            "persistent_post_exhaustion_count": persistent,
            "endpoint_persistence_exact": bool(post_states)
            and persistent == len(post_states),
            **rendered_facts,
            "source_authorized_reset_observed": value.source_authorized_reset_observed,
        }
    )
    return TemporalReplaySequenceMeasurements(
        comparable_prefix_count=comparable,
        matching_prefix_count=matching,
        first_divergence_index=first_divergence,
        replay_exhaustion_index=exhaustion_index,
        post_exhaustion_observation_count=len(post_states),
        persistent_post_exhaustion_count=persistent,
        descriptive_facts=facts,
    )


def track_components(value: TemporalTrackingInput) -> TemporalTrackingResult:
    """Preserve visual identity through stationary, moving, and resizing states."""

    if value.previous is None or value.reset_correspondence:
        first_sequence = (
            1 if value.previous is None else value.previous.next_entity_sequence
        )
        return TemporalTrackingResult(
            frame_ref=value.frame_ref,
            entities=tuple(
                TrackedComponent(
                    entity_id=f"entity.visual.{index:04d}",
                    component=component,
                    first_seen_frame_ref=value.frame_ref,
                    current_frame_ref=value.frame_ref,
                    observation_count=1,
                    identity_status="new",
                    match_kind=(
                        "initial" if value.previous is None else "appeared"
                    ),
                )
                for index, component in enumerate(
                    value.current.components, start=first_sequence
                )
            ),
            next_entity_sequence=first_sequence + len(value.current.components),
            action_ref=value.action_ref,
        )

    previous = value.previous
    unmatched_previous = {item.entity_id: item for item in previous.entities}
    unmatched_current = {
        item.component_id: item for item in value.current.components
    }
    resolved: dict[str, TrackedComponent] = {}

    _resolve_unique_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        predicate=_stationary_exact,
        match_kind="stationary_exact",
        frame_ref=value.frame_ref,
        resolved=resolved,
    )
    # When several same-shape components share one exact translation, keep
    # that coherent motion together before considering a same-locus value
    # transition.  A lone translated match remains ambiguous and is handled
    # by the existing tiers below; this preserves fixed-position value
    # permutations while retaining identity for a moving material pair.
    _resolve_coherent_translation_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        frame_ref=value.frame_ref,
        resolved=resolved,
    )
    _resolve_mutual_nearest_exact_translation_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        frame_ref=value.frame_ref,
        resolved=resolved,
    )
    _resolve_unique_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        predicate=_stationary_value_transition,
        match_kind="stationary_value_transition",
        frame_ref=value.frame_ref,
        resolved=resolved,
    )
    _resolve_unique_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        predicate=_stationary_appearance_transition,
        match_kind="stationary_appearance_transition",
        frame_ref=value.frame_ref,
        resolved=resolved,
    )
    _resolve_unique_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        predicate=_translated_exact,
        match_kind="translated_exact",
        frame_ref=value.frame_ref,
        resolved=resolved,
    )
    _resolve_unique_tier(
        previous=unmatched_previous,
        current=unmatched_current,
        predicate=_one_axis_extent_transition,
        match_kind="extent_transition",
        frame_ref=value.frame_ref,
        resolved=resolved,
    )

    next_sequence = previous.next_entity_sequence
    ambiguous_refs: list[str] = []
    for component_id in tuple(unmatched_current):
        component = unmatched_current.pop(component_id)
        predecessors = tuple(
            sorted(
                entity_id
                for entity_id, prior in sorted(unmatched_previous.items())
                if _translated_exact(prior.component, component)
                or _stationary_value_transition(prior.component, component)
                or _stationary_appearance_transition(prior.component, component)
                or _one_axis_extent_transition(prior.component, component)
            )
        )
        if predecessors:
            ambiguous_refs.append(component.component_id)
            entity_id = f"entity.visual.provisional.{next_sequence:04d}"
            next_sequence += 1
            resolved[component.component_id] = TrackedComponent(
                entity_id=entity_id,
                component=component,
                first_seen_frame_ref=value.frame_ref,
                current_frame_ref=value.frame_ref,
                observation_count=1,
                identity_status="special",
                match_kind="ambiguous",
                possible_predecessor_entity_refs=predecessors,
            )
            continue
        entity_id = f"entity.visual.{next_sequence:04d}"
        next_sequence += 1
        resolved[component.component_id] = TrackedComponent(
            entity_id=entity_id,
            component=component,
            first_seen_frame_ref=value.frame_ref,
            current_frame_ref=value.frame_ref,
            observation_count=1,
            identity_status="new",
            match_kind="appeared",
        )

    inherited_ids = {
        item.entity_id
        for item in (resolved[__yf_order_key] for __yf_order_key in sorted(resolved))
        if item.previous_component_ref is not None
    }
    possible_predecessors = {
        predecessor
        for item in (resolved[__yf_order_key] for __yf_order_key in sorted(resolved))
        for predecessor in item.possible_predecessor_entity_refs
    }
    disappeared = tuple(
        sorted(set(unmatched_previous) - inherited_ids - possible_predecessors)
    )
    ordered = tuple(
        resolved[component.component_id]
        for component in value.current.components
    )
    return TemporalTrackingResult(
        frame_ref=value.frame_ref,
        entities=ordered,
        disappeared_entity_refs=disappeared,
        ambiguous_current_component_refs=tuple(ambiguous_refs),
        next_entity_sequence=next_sequence,
        action_ref=value.action_ref,
        )


def _resolve_coherent_translation_tier(
    *,
    previous: dict[str, TrackedComponent],
    current: dict[str, ComponentDescription],
    frame_ref: str,
    resolved: dict[str, TrackedComponent],
) -> None:
    """Resolve a uniquely assigned group moving by one common delta.

    This is a mechanical correspondence rule, not a semantic choice: two or
    more one-to-one exact-shape matches with the same row/column delta are a
    stronger identity witness than an isolated same-locus value transition.
    A group of one is deliberately left to the historical tier ordering.
    """

    def exact_signature(component: ComponentDescription) -> tuple[object, object]:
        return (component.value, component.relative_pixels)

    current_refs_by_signature: dict[tuple[object, object], set[str]] = defaultdict(
        set
    )
    previous_refs_by_signature: dict[tuple[object, object], set[str]] = defaultdict(
        set
    )
    for current_ref, component in sorted(current.items()):
        current_refs_by_signature[exact_signature(component)].add(current_ref)
    for entity_ref, prior in sorted(previous.items()):
        previous_refs_by_signature[exact_signature(prior.component)].add(entity_ref)

    by_delta: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for current_ref, component in sorted(current.items()):
        for entity_ref, prior in sorted(previous.items()):
            if not _translated_exact(prior.component, component):
                continue
            delta = (
                component.bbox.top - prior.component.bbox.top,
                component.bbox.left - prior.component.bbox.left,
            )
            by_delta.setdefault(delta, []).append((current_ref, entity_ref))

    coherent_groups = tuple(
        sorted(
            (
                (delta, tuple(sorted(pairs)))
                for delta, pairs in sorted(by_delta.items())
                if len(pairs) >= 2
                and len({current_ref for current_ref, _entity_ref in pairs})
                == len(pairs)
                and len({entity_ref for _current_ref, entity_ref in pairs})
                == len(pairs)
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )
    )
    for delta, pairs in coherent_groups:
        if any(
            current_ref not in current or entity_ref not in previous
            for current_ref, entity_ref in pairs
        ):
            continue
        # A repeated exact morphology is evidence for one coherent translation
        # only when the delta covers its whole unmatched class. Accepting a
        # proper subset manufactures identity in cyclic marker permutations:
        # two cells can appear to share a delta while the remaining identical
        # cell wraps by a different amount. Complete-class coverage retains
        # genuine co-moving assemblies without converting that alias into
        # several controlled bodies.
        represented_signatures = {
            exact_signature(current[current_ref])
            for current_ref, _entity_ref in pairs
        }
        if any(
            (
                len(current_refs_by_signature[signature]) > 1
                or len(previous_refs_by_signature[signature]) > 1
            )
            and (
                {
                    current_ref
                    for current_ref, _entity_ref in pairs
                    if exact_signature(current[current_ref]) == signature
                }
                != current_refs_by_signature[signature]
                or {
                    entity_ref
                    for _current_ref, entity_ref in pairs
                    if exact_signature(previous[entity_ref].component) == signature
                }
                != previous_refs_by_signature[signature]
            )
            for signature in represented_signatures
        ):
            continue
        # Equal-support overlapping groups are genuinely ambiguous.  Keep
        # them unresolved for the later bounded tracking tiers instead of
        # manufacturing identity from deterministic iteration order.
        pair_current_refs = {current_ref for current_ref, _entity_ref in pairs}
        pair_entity_refs = {entity_ref for _current_ref, entity_ref in pairs}
        if any(
            len(other_pairs) == len(pairs)
            and other_delta != delta
            and (
                pair_current_refs.intersection(
                    current_ref for current_ref, _entity_ref in other_pairs
                )
                or pair_entity_refs.intersection(
                    entity_ref for _current_ref, entity_ref in other_pairs
                )
            )
            for other_delta, other_pairs in coherent_groups
        ):
            continue
        for current_ref, entity_ref in pairs:
            component = current.pop(current_ref)
            prior = previous.pop(entity_ref)
            resolved[current_ref] = TrackedComponent(
                entity_id=prior.entity_id,
                component=component,
                first_seen_frame_ref=prior.first_seen_frame_ref,
                current_frame_ref=frame_ref,
                observation_count=prior.observation_count + 1,
                identity_status="established",
                match_kind="translated_exact",
                previous_component_ref=prior.component.component_id,
                delta_row=component.bbox.top - prior.component.bbox.top,
                delta_col=component.bbox.left - prior.component.bbox.left,
                changed_extent_axis=None,
            )


@lru_cache(maxsize=256)
def _relative_pixel_shape_digest(pixels: tuple[tuple[int, int], ...]) -> str:
    """Reuse an exact shape digest across pairs, without selecting peers."""
    return stable_digest(pixels)[:20]


def measure_exact_multicell_peer_relations(
    value: ExactMulticellPeerRelationInput,
) -> ExactMulticellPeerRelationMeasurements:
    """Enumerate exact same-orientation peers over tracked blocks and 8-zones.

    The returned objects are geometry only.  An 8-connected zone becomes a
    stable measured assembly reference when it is the exact union of at least
    two tracked 4-connected members.  Common non-zero member translation is
    reported as causal confirmation; DRM decides what either observation means.
    """

    tracked_by_component = {
        item.component.component_id: item for item in value.tracking.entities
    }
    measured_entities: list[dict[str, object]] = []
    for tracked in value.tracking.entities:
        component = tracked.component
        measured_entities.append(
            {
                "entity_ref": tracked.entity_id,
                "component_ref": component.component_id,
                "member_entity_refs": (tracked.entity_id,),
                "member_component_refs": (component.component_id,),
                "value": component.value,
                "pixels": frozenset(component.pixels),
                "relative_pixels": component.relative_pixels,
                "bbox": component.bbox,
                "area": component.area,
                "observation_count": tracked.observation_count,
                "first_observation_sequence": _first_observation_sequence(
                    (tracked.entity_id,)
                ),
                "is_eight_connected_assembly": False,
                "assembly_cotranslation_observed": False,
            }
        )

    assembly_count = 0
    confirmed_assembly_count = 0
    for zone in value.zones.components:
        zone_pixels = frozenset(zone.pixels)
        members = tuple(
            sorted(
                (
                    tracked_by_component[component.component_id]
                    for component in (
                        item.component for item in value.tracking.entities
                    )
                    if component.value == zone.value
                    and frozenset(component.pixels).issubset(zone_pixels)
                ),
                key=lambda item: item.entity_id,
            )
        )
        if len(members) < 2:
            continue
        member_union = frozenset(
            pixel for member in members for pixel in member.component.pixels
        )
        if member_union != zone_pixels:
            continue
        member_refs = tuple(item.entity_id for item in members)
        member_component_refs = tuple(
            item.component.component_id for item in members
        )
        member_deltas = {(item.delta_row, item.delta_col) for item in members}
        common_nonzero_translation = bool(
            len(member_deltas) == 1 and next(iter(member_deltas)) != (0, 0)
        )
        assembly_ref = (
            "measurement.tracked_eight_connected_assembly:"
            f"{stable_digest(member_refs)[:20]}"
        )
        measured_entities.append(
            {
                "entity_ref": assembly_ref,
                "component_ref": zone.component_id,
                "member_entity_refs": member_refs,
                "member_component_refs": member_component_refs,
                "value": zone.value,
                "pixels": zone_pixels,
                "relative_pixels": zone.relative_pixels,
                "bbox": zone.bbox,
                "area": zone.area,
                "observation_count": min(
                    item.observation_count for item in members
                ),
                "first_observation_sequence": _first_observation_sequence(
                    member_refs
                ),
                "is_eight_connected_assembly": True,
                "assembly_cotranslation_observed": common_nonzero_translation,
            }
        )
        assembly_count += 1
        confirmed_assembly_count += int(common_nonzero_translation)

    relations: list[ExactMulticellPeerRelationMeasurement] = []
    ordered_entities = tuple(
        sorted(measured_entities, key=lambda item: str(item["entity_ref"]))
    )
    exact_shape_occurrence_counts = Counter(
        (int(item["area"]), tuple(item["relative_pixels"]))
        for item in ordered_entities
        if int(item["area"]) >= 2
    )
    for first, second in combinations(ordered_entities, 2):
        first_area = int(first["area"])
        second_area = int(second["area"])
        if first_area < 2 or first_area != second_area:
            continue
        if first["relative_pixels"] != second["relative_pixels"]:
            continue
        first_ref = str(first["entity_ref"])
        second_ref = str(second["entity_ref"])
        first_bbox = first["bbox"]
        second_bbox = second["bbox"]
        residual_row_twice = (
            int(second_bbox.top)
            + int(second_bbox.bottom)
            - int(first_bbox.top)
            - int(first_bbox.bottom)
        )
        residual_column_twice = (
            int(second_bbox.left)
            + int(second_bbox.right)
            - int(first_bbox.left)
            - int(first_bbox.right)
        )
        relation_ref = (
            "measurement.exact_multicell_peer_relation:"
            f"{stable_digest((first_ref, second_ref))[:20]}"
        )
        relations.append(
            ExactMulticellPeerRelationMeasurement(
                relation_ref=relation_ref,
                first_entity_ref=first_ref,
                second_entity_ref=second_ref,
                first_component_ref=str(first["component_ref"]),
                second_component_ref=str(second["component_ref"]),
                first_member_entity_refs=tuple(first["member_entity_refs"]),
                second_member_entity_refs=tuple(second["member_entity_refs"]),
                first_member_component_refs=tuple(first["member_component_refs"]),
                second_member_component_refs=tuple(second["member_component_refs"]),
                relative_shape_ref=(
                    "measurement.relative_shape:"
                    f"{_relative_pixel_shape_digest(first['relative_pixels'])}"
                ),
                area=first_area,
                exact_shape_occurrence_count=int(
                    exact_shape_occurrence_counts[
                        (first_area, tuple(first["relative_pixels"]))
                    ]
                ),
                first_value=int(first["value"]),
                second_value=int(second["value"]),
                residual_row_twice=residual_row_twice,
                residual_column_twice=residual_column_twice,
                residual_manhattan_twice=(
                    abs(residual_row_twice) + abs(residual_column_twice)
                ),
                joint_observation_count=min(
                    int(first["observation_count"]),
                    int(second["observation_count"]),
                ),
                first_observation_sequence=int(
                    first["first_observation_sequence"]
                ),
                second_observation_sequence=int(
                    second["first_observation_sequence"]
                ),
                first_bbox_top=int(first_bbox.top),
                first_bbox_left=int(first_bbox.left),
                first_bbox_bottom=int(first_bbox.bottom),
                first_bbox_right=int(first_bbox.right),
                second_bbox_top=int(second_bbox.top),
                second_bbox_left=int(second_bbox.left),
                second_bbox_bottom=int(second_bbox.bottom),
                second_bbox_right=int(second_bbox.right),
                first_is_eight_connected_assembly=bool(
                    first["is_eight_connected_assembly"]
                ),
                second_is_eight_connected_assembly=bool(
                    second["is_eight_connected_assembly"]
                ),
                first_assembly_cotranslation_observed=bool(
                    first["assembly_cotranslation_observed"]
                ),
                second_assembly_cotranslation_observed=bool(
                    second["assembly_cotranslation_observed"]
                ),
            )
        )

    # Keep every exact pair as an independent durable observation while
    # measuring which pairs are strict sub-relations of a larger exact pair.
    # This lets DRM prefer the maximal collective body without deleting the
    # smaller relations from the agenda.  Endpoint order is canonical rather
    # than semantic, so both direct and crossed containment are accepted.
    relation_members = {
        item.relation_ref: (
            frozenset(item.first_member_entity_refs),
            frozenset(item.second_member_entity_refs),
        )
        for item in relations
    }
    classified_relations: list[ExactMulticellPeerRelationMeasurement] = []
    # Both endpoints must occur in a containing relation. Intersect incidence
    # indexes before exact subset checks instead of comparing every pair of
    # relations (quadratic in the already quadratic shape-pair population).
    relation_indexes_by_member: dict[str, set[int]] = {}
    for ordinal, relation in enumerate(relations):
        first_members, second_members = relation_members[relation.relation_ref]
        for member in sorted(first_members | second_members):
            relation_indexes_by_member.setdefault(member, set()).add(ordinal)
    for item in relations:
        item_first, item_second = relation_members[item.relation_ref]
        containing_refs: list[str] = []
        internal_endpoint_refs: list[str] = []
        candidate_indexes = (
            relation_indexes_by_member[min(item_first)]
            & relation_indexes_by_member[min(item_second)]
        )
        for ordinal in sorted(candidate_indexes):
            larger = relations[ordinal]
            if larger.area <= item.area:
                continue
            larger_first, larger_second = relation_members[larger.relation_ref]
            if (
                item_first.issubset(larger_first)
                and item_second.issubset(larger_second)
            ) or (
                item_first.issubset(larger_second)
                and item_second.issubset(larger_first)
            ):
                containing_refs.append(larger.relation_ref)
            if (
                item_first.issubset(larger_first)
                and item_second.issubset(larger_first)
            ) or (
                item_first.issubset(larger_second)
                and item_second.issubset(larger_second)
            ):
                internal_endpoint_refs.append(larger.relation_ref)
        containing = tuple(sorted(containing_refs))
        internal = tuple(sorted(internal_endpoint_refs))
        classified_relations.append(
            item.model_copy(
                update={
                    "containing_exact_peer_relation_refs": containing,
                    "strictly_contained_in_larger_exact_peer_relation": bool(
                        containing
                    ),
                    "internal_to_larger_exact_peer_endpoint_refs": internal,
                    "internal_to_larger_exact_peer_endpoint": bool(internal),
                }
            )
        )
    relations = classified_relations

    def priority_key(item: ExactMulticellPeerRelationMeasurement) -> tuple[int, ...]:
        return (
            item.residual_manhattan_twice,
            -item.area,
            -item.joint_observation_count,
            item.first_observation_sequence,
            item.second_observation_sequence,
            int(item.internal_to_larger_exact_peer_endpoint),
            int(item.strictly_contained_in_larger_exact_peer_relation),
            -int(item.first_is_eight_connected_assembly),
            -int(item.second_is_eight_connected_assembly),
            item.first_bbox_top,
            item.first_bbox_left,
            item.first_bbox_bottom,
            item.first_bbox_right,
            item.second_bbox_top,
            item.second_bbox_left,
            item.second_bbox_bottom,
            item.second_bbox_right,
            item.first_value,
            item.second_value,
        )

    relation_priority_keys = tuple(priority_key(item) for item in relations)
    if len(set(relation_priority_keys)) != len(relation_priority_keys):
        raise RuntimeError(
            "exact peer declared priority criteria did not produce a unique order"
        )
    relations.sort(key=priority_key)
    total_relation_count = len(relations)
    retained = tuple(relations[: value.maximum_relations])
    deferred_relation_count = total_relation_count - len(retained)
    observation_ref = value.tracking.frame_ref
    relation_facts = tuple(
        FrozenMap(
            {
                **item.model_dump(mode="python"),
                "observation_ref": observation_ref,
                "lifecycle_revision_family_ref": (
                    "measurement.exact_multicell_peer_lifecycle_family:"
                    f"{stable_digest(item.relation_ref)[:20]}"
                ),
                "relation_residual_open": bool(item.residual_manhattan_twice),
                "exact_multicell_peer_relation_observed": True,
            }
        )
        for item in retained
    )
    descriptive_facts = FrozenMap(
        {
            "observation_ref": observation_ref,
            "exact_multicell_peer_relation_measurements": relation_facts,
            "exact_multicell_peer_relation_count": len(retained),
            "exact_multicell_peer_total_relation_count": total_relation_count,
            "exact_multicell_peer_deferred_relation_count": deferred_relation_count,
            "exact_multicell_peer_priority_frontier_complete": True,
            "exact_multicell_peer_priority_criterion_refs": (
                _EXACT_PEER_PRIORITY_CRITERION_REFS
            ),
            "exact_multicell_peer_enumeration_truncated": False,
            "tracked_component_count": len(value.tracking.entities),
            "eight_connected_assembly_count": assembly_count,
            "confirmed_cotranslating_assembly_count": confirmed_assembly_count,
        }
    )
    return ExactMulticellPeerRelationMeasurements(
        observation_ref=observation_ref,
        relations=retained,
        total_relation_count=total_relation_count,
        deferred_relation_count=deferred_relation_count,
        priority_frontier_complete=True,
        priority_criterion_refs=_EXACT_PEER_PRIORITY_CRITERION_REFS,
        enumeration_truncated=False,
        tracked_component_count=len(value.tracking.entities),
        eight_connected_assembly_count=assembly_count,
        confirmed_cotranslating_assembly_count=confirmed_assembly_count,
        descriptive_facts=descriptive_facts,
    )
def _resolve_mutual_nearest_exact_translation_tier(
    *,
    previous: dict[str, TrackedComponent],
    current: dict[str, ComponentDescription],
    frame_ref: str,
    resolved: dict[str, TrackedComponent],
) -> None:
    """Resolve a small exact-morphology group by mutual unique proximity.

    Coupled bodies may translate in opposite directions, so a common-delta
    witness is unavailable.  This tier is deliberately stricter than a global
    assignment: groups are capped at three, every current component must have
    one strictly nearest exact predecessor, every predecessor must name that
    same current as its strictly nearest successor, and the resulting mapping
    must be bijective.  The measured mapping must also contain a non-zero
    exact opposite delta pair; unrelated unequal translations remain
    ambiguous.  Ties remain ambiguous.
    """

    signatures = {
        (component.value, component.relative_pixels)
        for component in (current[__yf_order_key] for __yf_order_key in sorted(current))
    }
    for value, relative_pixels in sorted(signatures):
        current_group = tuple(
            sorted(
                (
                    (component_ref, component)
                    for component_ref, component in sorted(current.items())
                    if component.value == value
                    and component.relative_pixels == relative_pixels
                ),
                key=lambda item: item[0],
            )
        )
        previous_group = tuple(
            sorted(
                (
                    (entity_ref, tracked)
                    for entity_ref, tracked in sorted(previous.items())
                    if tracked.component.value == value
                    and tracked.component.relative_pixels == relative_pixels
                ),
                key=lambda item: item[0],
            )
        )
        if not (
            2 <= len(current_group) <= 3
            and len(current_group) == len(previous_group)
        ):
            continue

        def distance(
            before: ComponentDescription,
            after: ComponentDescription,
        ) -> int:
            return abs(after.bbox.top - before.bbox.top) + abs(
                after.bbox.left - before.bbox.left
            )

        current_to_previous: dict[str, str] = {}
        for current_ref, component in current_group:
            distances = tuple(
                (distance(tracked.component, component), entity_ref)
                for entity_ref, tracked in previous_group
            )
            minimum = min(item[0] for item in distances)
            nearest = tuple(
                entity_ref
                for candidate_distance, entity_ref in distances
                if candidate_distance == minimum
            )
            if minimum <= 0 or len(nearest) != 1:
                current_to_previous.clear()
                break
            current_to_previous[current_ref] = nearest[0]
        if len(current_to_previous) != len(current_group) or len(
            set(current_to_previous.values())
        ) != len(previous_group):
            continue

        mutual = True
        for entity_ref, tracked in previous_group:
            distances = tuple(
                (distance(tracked.component, component), current_ref)
                for current_ref, component in current_group
            )
            minimum = min(item[0] for item in distances)
            nearest = tuple(
                current_ref
                for candidate_distance, current_ref in distances
                if candidate_distance == minimum
            )
            if (
                len(nearest) != 1
                or current_to_previous.get(nearest[0]) != entity_ref
            ):
                mutual = False
                break
        if not mutual:
            continue

        measured_deltas = tuple(
            (
                current[current_ref].bbox.top
                - previous[entity_ref].component.bbox.top,
                current[current_ref].bbox.left
                - previous[entity_ref].component.bbox.left,
            )
            for current_ref, entity_ref in sorted(current_to_previous.items())
        )
        if not any(
            (delta_row != 0 or delta_col != 0)
            and (-delta_row, -delta_col) in measured_deltas
            for delta_row, delta_col in measured_deltas
        ):
            continue

        for current_ref, entity_ref in sorted(current_to_previous.items()):
            if current_ref not in current or entity_ref not in previous:
                continue
            component = current.pop(current_ref)
            prior = previous.pop(entity_ref)
            resolved[current_ref] = TrackedComponent(
                entity_id=prior.entity_id,
                component=component,
                first_seen_frame_ref=prior.first_seen_frame_ref,
                current_frame_ref=frame_ref,
                observation_count=prior.observation_count + 1,
                identity_status="established",
                match_kind="translated_exact",
                previous_component_ref=prior.component.component_id,
                delta_row=component.bbox.top - prior.component.bbox.top,
                delta_col=component.bbox.left - prior.component.bbox.left,
                changed_extent_axis=None,
            )


def _resolve_unique_tier(
    *,
    previous: dict[str, TrackedComponent],
    current: dict[str, ComponentDescription],
    predicate: Callable[[ComponentDescription, ComponentDescription], bool],
    match_kind: str,
    frame_ref: str,
    resolved: dict[str, TrackedComponent],
) -> None:
    candidates = {
        current_ref: tuple(
            entity_ref
            for entity_ref, prior in sorted(previous.items())
            if predicate(prior.component, component)
        )
        for current_ref, component in sorted(current.items())
    }
    prior_counts = {
        entity_ref: sum(entity_ref in refs for refs in (candidates[__yf_order_key] for __yf_order_key in sorted(candidates)))
        for entity_ref in previous
    }
    pairs = tuple(
        (current_ref, refs[0])
        for current_ref, refs in sorted(candidates.items())
        if len(refs) == 1 and prior_counts[refs[0]] == 1
    )
    for current_ref, entity_ref in pairs:
        if current_ref not in current or entity_ref not in previous:
            continue
        component = current.pop(current_ref)
        prior = previous.pop(entity_ref)
        axis = _changed_extent_axis(prior.component, component)
        resolved[current_ref] = TrackedComponent(
            entity_id=prior.entity_id,
            component=component,
            first_seen_frame_ref=prior.first_seen_frame_ref,
            current_frame_ref=frame_ref,
            observation_count=prior.observation_count + 1,
            identity_status="established",
            match_kind=match_kind,  # type: ignore[arg-type]
            previous_component_ref=prior.component.component_id,
            delta_row=component.bbox.top - prior.component.bbox.top,
            delta_col=component.bbox.left - prior.component.bbox.left,
            changed_extent_axis=axis,
        )


def _stationary_exact(
    before: ComponentDescription,
    after: ComponentDescription,
) -> bool:
    return before.value == after.value and before.pixels == after.pixels


def _stationary_value_transition(
    before: ComponentDescription,
    after: ComponentDescription,
) -> bool:
    """Measure an exact fixed-locus morphology whose scalar value changed."""

    return (
        before.value != after.value
        and before.pixels == after.pixels
        and before.relative_pixels == after.relative_pixels
    )


def _translated_exact(
    before: ComponentDescription,
    after: ComponentDescription,
) -> bool:
    return (
        before.value == after.value
        and before.relative_pixels == after.relative_pixels
    )


def _stationary_appearance_transition(
    before: ComponentDescription,
    after: ComponentDescription,
) -> bool:
    """Preserve a compact fixed bbox while its exact appearance changes."""

    return bool(
        not before.touches_frame_boundary
        and not after.touches_frame_boundary
        and before.bbox == after.bbox
        and before.area == after.area
        and (
            before.value != after.value
            or before.relative_pixels != after.relative_pixels
        )
    )


def _one_axis_extent_transition(
    before: ComponentDescription,
    after: ComponentDescription,
) -> bool:
    if before.value != after.value or before.area == after.area:
        return False
    same_rows = (
        before.bbox.top == after.bbox.top
        and before.bbox.bottom == after.bbox.bottom
    )
    one_horizontal_edge = (
        before.bbox.left == after.bbox.left
    ) != (before.bbox.right == after.bbox.right)
    horizontal_overlap = not (
        before.bbox.right < after.bbox.left or after.bbox.right < before.bbox.left
    )
    same_columns = (
        before.bbox.left == after.bbox.left
        and before.bbox.right == after.bbox.right
    )
    one_vertical_edge = (
        before.bbox.top == after.bbox.top
    ) != (before.bbox.bottom == after.bbox.bottom)
    vertical_overlap = not (
        before.bbox.bottom < after.bbox.top or after.bbox.bottom < before.bbox.top
    )
    return bool(
        (same_rows and one_horizontal_edge and horizontal_overlap)
        or (same_columns and one_vertical_edge and vertical_overlap)
    )


def _changed_extent_axis(
    before: ComponentDescription,
    after: ComponentDescription,
) -> str | None:
    if before.bbox.width != after.bbox.width and before.bbox.height == after.bbox.height:
        return "column"
    if before.bbox.height != after.bbox.height and before.bbox.width == after.bbox.width:
        return "row"
    return None


__all__ = ("track_components",)
