"""Pure bounded geometry for declared rod, piercing, withdrawal and clearance questions."""

from agents.yf_arc3_v5.capabilities.contracts import (
    PiercingClearanceInput,
    PiercingClearanceMeasurements,
    PiercingIntersectionInput,
    PiercingIntersectionMeasurements,
    PiercingPierceSuccessorInput,
    PiercingPierceSuccessorMeasurements,
    PiercingPieceState,
    PiercingPushSuccessorInput,
    PiercingPushSuccessorMeasurements,
    PiercingRodIntersection,
    PiercingWithdrawSuccessorInput,
    PiercingWithdrawSuccessorMeasurements,
)


def _translated(positions: tuple[tuple[int, int], ...], delta: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple((row + delta[0], column + delta[1]) for row, column in positions)


def _support_map(rods, pieces) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for piece in pieces:
        occupied = set(piece.occupied_positions)
        result[piece.piece_ref] = tuple(
            rod.rod_ref for rod in rods if occupied & set(rod.root_to_tip_positions)
        )
    return result


def measure_piercing_intersections(value: PiercingIntersectionInput) -> PiercingIntersectionMeasurements:
    intersections: list[PiercingRodIntersection] = []
    ordered_by_rod: list[tuple[str, tuple[str, ...]]] = []
    pieces_by_ref = {piece.piece_ref: piece for piece in value.pieces}
    for rod in value.rods:
        indexes = {position: index for index, position in enumerate(rod.root_to_tip_positions)}
        intersected: list[tuple[int, str]] = []
        for piece in value.pieces:
            positions = tuple(position for position in rod.root_to_tip_positions if position in set(piece.occupied_positions))
            if not positions:
                continue
            first_index = min(indexes[position] for position in positions)
            intersections.append(PiercingRodIntersection(rod_ref=rod.rod_ref, piece_ref=piece.piece_ref, intersection_positions=positions, first_root_index=first_index))
            intersected.append((first_index, piece.piece_ref))
        ordered_by_rod.append((rod.rod_ref, tuple(piece_ref for _, piece_ref in sorted(intersected))))
    requested_rod = next((rod for rod in value.rods if rod.root_ref == value.requested_root_ref), None)
    requested_refs = next((refs for rod_ref, refs in ordered_by_rod if requested_rod and rod_ref == requested_rod.rod_ref), ())
    requested_types = tuple(pieces_by_ref[ref].type_ref for ref in requested_refs)
    constrained = set(requested_refs)
    return PiercingIntersectionMeasurements(
        intersections=tuple(intersections), ordered_piece_refs_by_rod=tuple(ordered_by_rod),
        requested_support_ref=requested_rod.rod_ref if requested_rod else None,
        requested_support_piece_refs=requested_refs, requested_support_type_order=requested_types,
        rooted_match=requested_rod is not None and requested_types == value.requested_type_order,
        remaining_unconstrained_piece_refs=tuple(piece.piece_ref for piece in value.pieces if piece.piece_ref not in constrained),
    )


def measure_piercing_push_successor(value: PiercingPushSuccessorInput) -> PiercingPushSuccessorMeasurements:
    contacted, movable = set(value.contacted_piece_refs), set(value.movable_piece_refs)
    translated_refs = tuple(piece.piece_ref for piece in value.pieces if piece.piece_ref in contacted & movable)
    blocked_refs = tuple(piece.piece_ref for piece in value.pieces if piece.piece_ref in contacted - movable)
    resulting = tuple(
        piece.model_copy(update={"occupied_positions": _translated(piece.occupied_positions, value.translation_delta)})
        if piece.piece_ref in set(translated_refs) else piece
        for piece in value.pieces
    )
    return PiercingPushSuccessorMeasurements(resulting_pieces=resulting, translated_piece_refs=translated_refs, blocked_piece_refs=blocked_refs)


def measure_piercing_pierce_successor(value: PiercingPierceSuccessorInput) -> PiercingPierceSuccessorMeasurements:
    candidate = next(piece for piece in value.pieces if piece.piece_ref == value.candidate_piece_ref)
    intersections = tuple(position for position in value.supplied_after_rod.root_to_tip_positions if position in set(candidate.occupied_positions))
    pierced = bool(intersections) and value.observed_entry_face == value.permitted_entry_face and candidate.piece_ref in set(value.immobilized_piece_refs)
    supports = candidate.support_refs
    if pierced and value.supplied_after_rod.rod_ref not in supports:
        supports = (*supports, value.supplied_after_rod.rod_ref)
    resulting = tuple(piece.model_copy(update={"support_refs": supports}) if piece.piece_ref == candidate.piece_ref else piece for piece in value.pieces)
    return PiercingPierceSuccessorMeasurements(resulting_pieces=resulting, pierced=pierced, intersection_positions=intersections, resulting_support_refs=supports)


def measure_piercing_withdraw_successor(value: PiercingWithdrawSuccessorInput) -> PiercingWithdrawSuccessorMeasurements:
    before = _support_map(value.before_rods, value.pieces)
    after = _support_map(value.supplied_after_rods, value.pieces)
    retained: list[tuple[str, str]] = []
    released: list[tuple[str, str]] = []
    resulting: list[PiercingPieceState] = []
    for piece in value.pieces:
        current = after[piece.piece_ref]
        retained.extend((piece.piece_ref, support) for support in current)
        released.extend((piece.piece_ref, support) for support in before[piece.piece_ref] if support not in current)
        resulting.append(piece.model_copy(update={"support_refs": current}))
    return PiercingWithdrawSuccessorMeasurements(
        resulting_pieces=tuple(resulting), retained_relations=tuple(retained), released_relations=tuple(released),
        multi_support_piece_refs=tuple(piece.piece_ref for piece in value.pieces if len(after[piece.piece_ref]) > 1),
        world_positions_preserved=all(a.occupied_positions == b.occupied_positions for a, b in zip(value.pieces, resulting)),
    )


def measure_piercing_clearance(value: PiercingClearanceInput) -> PiercingClearanceMeasurements:
    translated = _translated(value.tool_and_load_positions, value.requested_delta)
    translated_set = set(translated)
    obstacles = tuple(region.obstacle_ref for region in value.obstacles if translated_set & set(region.occupied_positions))
    pieces = tuple(piece.piece_ref for piece in value.free_pieces if translated_set & set(piece.occupied_positions))
    return PiercingClearanceMeasurements(translated_positions=translated, blocking_obstacle_refs=obstacles, blocking_free_piece_refs=pieces, clear=not obstacles and not pieces)
