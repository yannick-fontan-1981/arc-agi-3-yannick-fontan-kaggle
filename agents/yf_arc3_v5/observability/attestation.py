"""Bounded source-attestation and action-lineage queries over canonical events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    LineagedRecord,
    OperatorInvocation,
    OperatorResult,
)
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import FrozenMap
from agents.yf_arc3_v5.observability.models import (
    AttestationQuery,
    AttestationResult,
    AttestedRecord,
)
from agents.yf_arc3_v5.src.compiler import CompiledModule
from agents.yf_arc3_v5.state import CognitiveEvent, PutClaim, PutTerm, RecordArtifact


def query_attestation(
    events: Iterable[CognitiveEvent],
    query: AttestationQuery,
    *,
    compiled_modules: tuple[CompiledModule, ...] = (),
) -> AttestationResult:
    records = collect_attested_records(events, compiled_modules=compiled_modules)
    connected_ids = (
        None
        if query.reference is None
        else _connected_record_ids(records, query.reference, limit=query.limit)
    )
    matches = tuple(
        record
        for record in records
        if _matches(record, query, connected_ids=connected_ids)
    )
    return AttestationResult(
        query=query,
        records=matches[: query.limit],
        matched_count=len(matches),
        truncated=len(matches) > query.limit,
    )


def collect_attested_records(
    events: Iterable[CognitiveEvent],
    *,
    compiled_modules: tuple[CompiledModule, ...] = (),
) -> tuple[AttestedRecord, ...]:
    source_locations = _source_location_index(compiled_modules)
    records: list[AttestedRecord] = []
    for event in events:
        mutation = event.mutation
        if isinstance(mutation, PutTerm):
            records.append(
                _state_record(
                    event,
                    record=mutation.after,
                    record_kind="term",
                    premise_refs=mutation.after.provenance,
                )
            )
        elif isinstance(mutation, PutClaim):
            claim = mutation.after
            records.append(
                _state_record(
                    event,
                    record=claim,
                    record_kind="claim",
                    premise_refs=(
                        *claim.conditions,
                        *claim.grounds,
                        *claim.source_operator_results,
                        *claim.active_contradictions,
                    ),
                    authority_ref=claim.proof_rule,
                )
            )
        elif isinstance(mutation, RecordArtifact):
            records.append(
                _artifact_record(
                    event,
                    mutation.artifact,
                    source_locations=source_locations,
                )
            )
    return tuple(records)


def project_action_lineage(
    events: Iterable[CognitiveEvent],
    *,
    action_intent_id: str | None = None,
    compiled_modules: tuple[CompiledModule, ...] = (),
) -> FrozenMap:
    records = collect_attested_records(events, compiled_modules=compiled_modules)
    intents = tuple(
        record for record in records if record.record_kind == "action_intent"
    )
    selected_intent = action_intent_id or (intents[-1].id if intents else None)
    if selected_intent is None:
        return FrozenMap(
            {
                "status": "absent",
                "reason": "no_action_intent_recorded",
                "complete": False,
            }
        )
    connected = _connected_record_ids(records, selected_intent, limit=100)
    linked = tuple(record for record in records if record.id in connected)
    by_kind: dict[str, list[AttestedRecord]] = {}
    for record in linked:
        by_kind.setdefault(record.record_kind, []).append(record)
    expected_kinds = (
        "viability_assessment",
        "commitment_decision",
        "action_intent",
        "action_release_permit",
        "environment_action_dispatched",
        "action_observed",
    )
    missing = tuple(kind for kind in expected_kinds if kind not in by_kind)
    return FrozenMap(
        {
            "status": "complete" if not missing else "incomplete",
            "complete": not missing,
            "action_intent_id": selected_intent,
            "missing_record_kinds": missing,
            "records": tuple(
                {
                    "id": record.id,
                    "record_kind": record.record_kind,
                    "event_ref": record.event_ref,
                    "source_hash": record.source_hash,
                    "state_revision": record.state_revision,
                    "workflow_definition_id": record.workflow_definition_id,
                    "workflow_instance_id": record.workflow_instance_id,
                    "workflow_step_id": record.workflow_step_id,
                    "frame_id": record.frame_id,
                    "action_lineage": record.action_lineage,
                }
                for record in linked
                if record.record_kind in expected_kinds
            ),
        }
    )


def _state_record(
    event: CognitiveEvent,
    *,
    record: Term | Claim,
    record_kind: str,
    premise_refs: tuple[str, ...],
    authority_ref: str | None = None,
) -> AttestedRecord:
    return AttestedRecord(
        id=record.id,
        record_kind=record_kind,
        event_ref=event.event_id,
        event_sequence=event.sequence,
        state_revision=event.state_revision_after,
        source_hash=event.source_hash,
        frame_id=event.frame_id,
        authority_ref=authority_ref,
        premise_refs=tuple(dict.fromkeys(premise_refs)),
        provenance=record.provenance if isinstance(record, Term) else record.grounds,
    )


def _artifact_record(
    event: CognitiveEvent,
    artifact: LineagedRecord,
    *,
    source_locations: Mapping[str, FrozenMap],
) -> AttestedRecord:
    # Only serialize the references consumed below. Workflow locals and operator
    # payloads can retain large immutable contexts; expanding them here neither
    # contributes evidence nor changes the journal-derived lineage.
    values = artifact.model_dump(mode="python", include=_ARTIFACT_REFERENCE_FIELDS)
    premise_refs = _refs_from_fields(
        values,
        (
            "input_refs",
            "provenance",
            "reason_refs",
            "evidence_refs",
            "source_claims",
            "supporting_evidence",
            "pending_expectation_refs",
        ),
    )
    authority_ref = _first_ref(
        values,
        "authority_ref",
        "acquisition_model_ref",
    )
    policy_ref = _first_ref(values, "policy_ref", "update_policy_ref")
    change_ref = _first_ref(
        values,
        "change_ref",
        "source_change_id",
        "proposal_id",
    )
    action_fields = (
        "action_intent_id",
        "action_release_permit_id",
        "commitment_decision_id",
        "viability_assessment_id",
        "environment_action_event_ref",
        "observation_record_id",
        "pre_action_snapshot_digest",
        "request_digest",
        "response_digest",
        "raw_input_ref",
        "action_ref",
    )
    action_lineage = FrozenMap(
        {
            field: values[field]
            for field in action_fields
            if values.get(field) is not None
        }
    )
    operator = (
        artifact.operator.value
        if isinstance(artifact, (OperatorInvocation, OperatorResult))
        else None
    )
    workflow_step_id = getattr(artifact, "workflow_step_id", None)
    location = source_locations.get(workflow_step_id or "", FrozenMap())
    return AttestedRecord(
        id=artifact.id,
        record_kind=artifact.record_kind,
        event_ref=event.event_id,
        event_sequence=event.sequence,
        state_revision=artifact.state_revision,
        source_unit=artifact.source_unit,
        source_hash=artifact.source_hash,
        workflow_definition_id=artifact.workflow_definition_id,
        workflow_instance_id=artifact.workflow_instance_id,
        workflow_step_id=workflow_step_id,
        frame_id=artifact.frame_id,
        operator=operator,
        authority_ref=authority_ref,
        premise_refs=premise_refs,
        policy_ref=policy_ref,
        change_ref=change_ref,
        action_lineage=action_lineage,
        provenance=getattr(artifact, "provenance", ()),
        source_location=location,
    )


_ARTIFACT_REFERENCE_FIELDS = frozenset({
    "input_refs", "provenance", "reason_refs", "evidence_refs", "source_claims",
    "supporting_evidence", "pending_expectation_refs", "authority_ref",
    "acquisition_model_ref", "policy_ref", "update_policy_ref", "change_ref",
    "source_change_id", "proposal_id", "action_intent_id", "action_release_permit_id",
    "commitment_decision_id", "viability_assessment_id", "environment_action_event_ref",
    "observation_record_id", "pre_action_snapshot_digest", "request_digest",
    "response_digest", "raw_input_ref", "action_ref",
})


def _source_location_index(
    modules: tuple[CompiledModule, ...],
) -> dict[str, FrozenMap]:
    result: dict[str, FrozenMap] = {}
    for module in modules:
        for entry in module.source_map:
            result[entry.node_id] = FrozenMap(
                {
                    "module_id": module.module_id,
                    "source_name": entry.span.source_name,
                    "construct_kind": entry.construct_kind,
                    "start_line": entry.span.start.line,
                    "start_column": entry.span.start.column,
                    "end_line": entry.span.end.line,
                    "end_column": entry.span.end.column,
                }
            )
    return result


def _matches(
    record: AttestedRecord,
    query: AttestationQuery,
    *,
    connected_ids: set[str] | None,
) -> bool:
    if connected_ids is not None and record.id not in connected_ids:
        return False
    checks = (
        (
            query.workflow_definition_id,
            record.workflow_definition_id,
        ),
        (query.workflow_instance_id, record.workflow_instance_id),
        (query.workflow_step_id, record.workflow_step_id),
        (query.operator, record.operator),
        (query.record_kind, record.record_kind),
    )
    return all(expected is None or actual == expected for expected, actual in checks)


def _connected_record_ids(
    records: tuple[AttestedRecord, ...],
    reference: str,
    *,
    limit: int,
) -> set[str]:
    edges = {record.id: _relation_refs(record) for record in records}
    active = {reference}
    matched: set[str] = set()
    changed = True
    while changed and len(matched) < limit:
        changed = False
        for record_id, relations in sorted(edges.items()):
            if record_id in matched:
                continue
            if record_id in active or active.intersection(relations):
                matched.add(record_id)
                active.update(relations)
                changed = True
                if len(matched) >= limit:
                    break
    return matched


def _relation_refs(record: AttestedRecord) -> set[str]:
    result = {
        *record.premise_refs,
        *record.provenance,
    }
    for value in (
        record.authority_ref,
        record.policy_ref,
        record.change_ref,
    ):
        if value:
            result.add(value)
    result.update(str(value) for value in (record.action_lineage[__yf_order_key] for __yf_order_key in sorted(record.action_lineage)))
    return result


def _refs_from_fields(
    values: Mapping[str, object],
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    result: list[str] = []
    for field in fields:
        value = values.get(field)
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, (list, tuple)):
            result.extend(str(item) for item in value)
    return tuple(dict.fromkeys(result))


def _first_ref(values: Mapping[str, object], *fields: str) -> str | None:
    return next(
        (
            str(values[field])
            for field in fields
            if isinstance(values.get(field), str) and values.get(field)
        ),
        None,
    )
