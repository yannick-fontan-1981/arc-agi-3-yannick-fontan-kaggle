"""Bounded geometric checks of a remembered calibration in the current frame.

DRM supplies the admitted memory, role and proof contracts. The result retains
all calibration alternatives and non-tiled objects; it is never a walkability
classification, a replacement-grid search or an action permission.
"""
from collections import Counter

from agents.yf_arc3_v5.capabilities import FrameGrid
from agents.yf_arc3_v5.capabilities.cell_recount import recount_known_grid_cells
from agents.yf_arc3_v5.capabilities.contracts import KnownGridCellRecountInput
from agents.yf_arc3_v5.logos.operations import EnvironmentActionDispatchedRecord
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current


def measure_recalled_navigation_lattices(*, snapshot, tracking, repository, store, profile):
    empty = FrozenMap({"grid_rows": (), "has_inventory_delta": False,
        "evidence_refs": (), "state_revision": snapshot.revision})
    world = repository.get_by_frame_id(tracking.frame_ref) if repository else None
    if world is None:
        return empty
    memories = snapshot.terms_for_projection_contract(profile["memory_contract"])
    proofs = snapshot.claims_for_predicate(profile["memory_predicate"])
    current = snapshot.terms_for_literal_attribute("operational_scene_ref", tracking.frame_ref)
    bindings = tuple(t for t in current if t.attributes.get("projection_contract") == profile["binding_contract"]
        and canonical_reference_bindings_current(snapshot, t, frame_ref=tracking.frame_ref))
    if max(len(memories), len(proofs), len(bindings), len(tracking.entities)) > profile["maximum_records"]:
        raise ValueError("recalled navigation lattice record bound exceeded")
    frame = FrameGrid(rows=world.frame)
    if frame.height * frame.width > profile["maximum_pixels"]:
        raise ValueError("recalled navigation lattice pixel bound exceeded")
    entities = {e.entity_id: e for e in tracking.entities if e.current_frame_ref == tracking.frame_ref}
    records = store.artifacts_of_type(EnvironmentActionDispatchedRecord)
    if len(records) > profile["maximum_history_inputs"]:
        raise ValueError("recalled navigation lattice dispatch bound exceeded")
    preceding = tuple(r for r in records if r.raw_input_ref == world.raw_input_ref)
    same_level_dispatch = False
    if len(preceding) == 1:
        before = repository.get_by_frame_id(preceding[0].frame_id)
        same_level_dispatch = before is not None and before.score == world.score

    def admitted(claim):
        return (claim is not None and claim.epistemic_status.value in profile["proof_statuses"]
            and claim.disposition.value == "active" and claim.polarity.value == "positive"
            and not claim.active_contradictions and not claim.defeated_by)

    rows, evidence = [], set()
    for memory in memories:
        links = tuple(c for c in proofs if c.arguments[0] == memory.id
            and c.proof_rule == profile["memory_rule"] and admitted(c)
            and all(admitted(snapshot.claim(ref)) for ref in c.grounds))
        if not links:
            continue
        calibrations = memory.attributes.get("observed_lattice_calibration_rows", ())
        if len(calibrations) > profile["maximum_candidates"]:
            raise ValueError("recalled navigation lattice alternatives exceed bound")
        role_groups = []
        matched_bindings = []
        for field in profile["association_fields"]:
            refs = memory.attributes.get(field, ())
            matching = tuple(b for b in bindings if b.attributes.get("association_ref") in refs
                and b.attributes.get("match_code") in profile["exact_match_codes"])
            matched_bindings.extend(matching)
            role_groups.append(tuple(sorted({r for b in matching
                for r in b.attributes.get("current_member_entity_refs", ()) if r in entities})))
        role_refs = tuple(r for group in role_groups for r in group)
        dependencies = tuple(FrozenMap({"term_ref": b.id,
            "revision": b.last_changed_state_revision}) for b in matched_bindings)
        link_refs = tuple(sorted({c.id for c in links}
            | {ref for b in matched_bindings for ref in b.attributes.get("role_premise_claim_refs", ())}))
        for calibration in calibrations:
            h, w, rp, cp, ro, co = (calibration[k] for k in
                ("cell_height", "cell_width", "row_pitch", "col_pitch", "row_phase", "col_phase"))
            if min(h, w, rp, cp) <= 0 or max(h, w, rp, cp) > profile["maximum_cell_extent"]:
                raise ValueError("invalid remembered calibration extent")
            nr, nc = (frame.height-ro)//rp, (frame.width-co)//cp
            if min(nr, nc) <= 0 or h > rp or w > cp or not (0 <= ro < rp and 0 <= co < cp):
                raise ValueError("invalid remembered calibration geometry")
            token = stable_digest((memory.id, calibration))[:20]
            frame_token = stable_digest(tracking.frame_ref)[:20]
            grid_ref = profile["grid_ref_pattern"].format(grid_token=token)
            geometry = (grid_ref, ro, co, h, w, rp, cp, nr, nc)
            recount = recount_known_grid_cells(KnownGridCellRecountInput(frame=frame,
                grid_ref=grid_ref, row_offset=ro, col_offset=co, cell_height=h, cell_width=w,
                row_pitch=rp, col_pitch=cp, logical_rows=nr, logical_columns=nc))
            tiled, untiled, supports = set(), [], []
            for ref, entity in sorted(entities.items()):
                counts, outside = Counter(), False
                for r, c in entity.component.pixels:
                    lr, ir = divmod(r-ro, rp); lc, ic = divmod(c-co, cp)
                    outside |= not (0 <= lr < nr and 0 <= lc < nc and ir < h and ic < w)
                    counts[(lr, lc)] += 1
                if outside or not counts or any(counts[cell] != h*w for cell in sorted(counts)):
                    untiled.append(ref)
                    continue
                tiled.add(ref)
                top, left = min(r for r,c in counts), min(c for r,c in counts)
                types = tuple(recount.cell_type_ids[r*nc+c] for r,c in sorted(counts))
                supports.append(FrozenMap({"support_ref": ref, "top": top, "left": left,
                    "height": max(r for r,c in counts)-top+1, "width": max(c for r,c in counts)-left+1,
                    "normalized_cells": tuple((r-top,c-left) for r,c in sorted(counts)),
                    "cell_type_ids": types, "cell_type_counts": tuple(sorted(Counter(types).items()))}))
            if len(supports) > profile["maximum_supports"]:
                raise ValueError("recalled navigation lattice support bound exceeded")
            observed = tuple((entities[r].delta_row, entities[r].delta_col) for r in role_groups[0]
                if entities[r].match_kind == "translated_exact" and same_level_dispatch)
            expected = tuple(tuple(delta) for row in memory.attributes.get("observed_command_translation_rows", ())
                if same_level_dispatch and row["action_ref"] == preceding[0].action_ref
                for delta in row["observed_delta_rows"])
            row = FrozenMap({"frame_ref": tracking.frame_ref, "frame_token": frame_token, "grid_token": token,
                "grid_geometry": geometry, "grid_claim_ref": profile["claim_pattern"].format(frame_token=frame_token, grid_token=token),
                "row_claim_refs": link_refs, "source_memory_ref": memory.id,
                "historical_calibration_unique": len(calibrations) == 1,
                "historical_calibration_complete": calibration["homogeneous_pose_count"] == calibration["observed_pose_count"]
                    and calibration["compatible_translation_count"] == calibration["translation_count"] > 0,
                "current_role_bearers_unique": all(len(g) == 1 for g in role_groups),
                "current_role_bearers_tiled": bool(role_refs) and all(r in tiled for r in role_refs),
                "current_motion_observed": bool(observed),
                "current_motion_confirms": bool(observed) and bool(expected) and all(d in expected for d in observed),
                "current_motion_contradicts": bool(observed) and bool(expected) and any(d not in expected for d in observed),
                "type_patterns": recount.type_patterns, "cell_type_ids": recount.cell_type_ids,
                "cell_support_rows": tuple(supports), "unbound_entity_refs": tuple(untiled),
                "uncovered_bottom_extent": (frame.height-ro)%rp, "uncovered_right_extent": (frame.width-co)%cp,
                "canonical_term_dependency_revisions": (*dependencies, FrozenMap({"term_ref":memory.id,"revision":memory.last_changed_state_revision})),
                "canonical_claim_dependency_digests": tuple(FrozenMap({"claim_ref":r,"content_digest":stable_digest(snapshot.claim(r))}) for r in link_refs)})
            term_ref = profile["term_pattern"].format(frame_token=frame_token, grid_token=token)
            if snapshot.term(term_ref) is None:
                rows.append(row)
                evidence.update(link_refs)
            if len(rows) > profile["maximum_candidates"]:
                raise ValueError("recalled navigation lattice output bound exceeded")
    return FrozenMap({"grid_rows": tuple(rows), "has_inventory_delta": bool(rows),
        "evidence_refs": tuple(sorted(evidence)), "state_revision": snapshot.revision})
