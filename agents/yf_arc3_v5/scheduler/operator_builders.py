"""Mechanical clause-to-request builders for the nine Logos primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agents.yf_arc3_v5.logos.operations import ActionIntent, ActionReleasePermit
from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    OperatorName,
    RelationFamily,
)
from agents.yf_arc3_v5.operators.contracts import (
    ActRequest,
    ClaimSeed,
    CompareRequest,
    DeriveRequest,
    DistinguishRequest,
    DominanceWitness,
    ObserveRequest,
    ProposeRequest,
    RelateRequest,
    SelectRequest,
    UpdateItem,
    UpdateRequest,
)
from agents.yf_arc3_v5.scheduler.contracts import OperatorBuildContext
from agents.yf_arc3_v5.scheduler.registry import SchedulerRuntimeRegistry


def register_operator_builders(runtime: SchedulerRuntimeRegistry) -> None:
    runtime.register_operator_builder(OperatorName.OBSERVE, _observe)
    runtime.register_operator_builder(OperatorName.ACT, _act)
    runtime.register_operator_builder(OperatorName.DISTINGUISH, _distinguish)
    runtime.register_operator_builder(OperatorName.RELATE, _relate)
    runtime.register_operator_builder(OperatorName.PROPOSE, _propose)
    runtime.register_operator_builder(OperatorName.DERIVE, _derive)
    runtime.register_operator_builder(OperatorName.COMPARE, _compare)
    runtime.register_operator_builder(OperatorName.UPDATE, _update)
    runtime.register_operator_builder(OperatorName.SELECT, _select)


def _base(context: OperatorBuildContext) -> dict[str, Any]:
    return {
        "request_id": context.request_id,
        "run_id": context.run_id,
        "timeline_id": context.timeline_id,
        "frame_id": context.frame_id,
        "expected_state_revision": context.expected_state_revision,
        "source_unit": context.source_unit,
        "source_hash": context.source_hash,
        "workflow_definition_id": context.workflow_definition_id,
        "workflow_instance_id": context.workflow_instance_id,
        "workflow_step_id": context.workflow_step_id,
    }


def _refs(value: object, description: str) -> tuple[str, ...]:
    values: tuple[str, ...]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{description} requires one reference or a finite sequence")
    if not values or any(not item.strip() for item in values):
        raise ValueError(f"{description} requires non-empty references")
    return values


def _optional_refs(value: object, description: str) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(str(item) for item in value)
        if any(not item.strip() for item in values):
            raise ValueError(f"{description} contains an empty reference")
        return values
    return _refs(value, description)


def _observe(context: OperatorBuildContext) -> ObserveRequest:
    return ObserveRequest(
        **_base(context),
        raw_input_ref=str(context.clauses["FROM"]),
        observation_ref=f"observation:{context.frame_id or context.request_id}",
        acquisition_model_ref=str(context.clauses["USING"]),
    )


def _act(context: OperatorBuildContext) -> ActRequest:
    return ActRequest(
        **_base(context),
        intent=ActionIntent.model_validate(context.clauses["FROM"]),
        release_permit=ActionReleasePermit.model_validate(context.clauses["RELEASE"]),
        world_context=FrozenMap(context.clauses["CONTEXT"]),
    )


def _distinguish(context: OperatorBuildContext) -> DistinguishRequest:
    return DistinguishRequest(
        **_base(context),
        source_refs=_refs(context.clauses["FROM"], "DISTINGUISH FROM"),
        model_ref=str(context.clauses["USING"]),
    )


def _relate(context: OperatorBuildContext) -> RelateRequest:
    scope = FrozenMap(context.clauses["SCOPE"])
    family_value = scope.get("relation_family")
    if family_value is None:
        raise ValueError("RELATE SCOPE must declare relation_family")
    predicate = str(context.clauses["PREDICATE"])
    arguments = _refs(context.clauses["ARGUMENTS"], "RELATE ARGUMENTS")
    basis = _refs(context.clauses["BASIS"], "RELATE BASIS")
    return RelateRequest(
        **_base(context),
        relation=ClaimSeed(
            id=f"{context.request_id}:claim",
            predicate=predicate,
            family=RelationFamily(str(family_value)),
            arguments=arguments,
            scope=scope,
        ),
        predicate_ref=predicate,
        basis_refs=basis,
    )


def _propose(context: OperatorBuildContext) -> ProposeRequest:
    raw_seeds = context.clauses["KIND"]
    if not isinstance(raw_seeds, Sequence) or isinstance(
        raw_seeds, (str, bytes, bytearray)
    ):
        raise ValueError("PROPOSE KIND requires explicit claim seeds")
    declared_scope = FrozenMap(context.clauses["SCOPE"])
    seeds: list[ClaimSeed] = []
    for item in raw_seeds:
        seed = ClaimSeed.model_validate(item)
        if seed.scope and seed.scope != declared_scope:
            raise ValueError("PROPOSE seed scope conflicts with the SRC SCOPE clause")
        seeds.append(seed.model_copy(update={"scope": declared_scope}))
    return ProposeRequest(
        **_base(context),
        proposals=tuple(seeds),
        basis_refs=_refs(context.clauses["BASIS"], "PROPOSE BASIS"),
        live_alternatives=_refs(
            context.clauses["ALTERNATIVES"], "PROPOSE ALTERNATIVES"
        ),
        missing_evidence=tuple(str(item) for item in context.clauses["MISSING"]),
        falsifier_refs=tuple(str(item) for item in context.clauses["FALSIFIERS"]),
    )


def _derive(context: OperatorBuildContext) -> DeriveRequest:
    raw_context = context.clauses.get("CONTEXT")
    if not isinstance(raw_context, Mapping):
        raise ValueError("DERIVE CONTEXT must declare requested conclusions")
    raw_conclusions = raw_context.get("conclusions")
    if not isinstance(raw_conclusions, Sequence) or isinstance(
        raw_conclusions, (str, bytes, bytearray)
    ):
        raise ValueError("DERIVE CONTEXT requires a finite conclusions collection")
    return DeriveRequest(
        **_base(context),
        conclusions=tuple(ClaimSeed.model_validate(item) for item in raw_conclusions),
        premise_refs=_refs(context.clauses["FROM"], "DERIVE FROM"),
        rule_or_model_ref=str(context.clauses["USING"]),
        context_refs=tuple(str(item) for item in raw_context.get("context_refs", ())),
    )


def _compare(context: OperatorBuildContext) -> CompareRequest:
    return CompareRequest(
        **_base(context),
        left_refs=_refs(context.clauses["LEFT"], "COMPARE LEFT"),
        right_refs=_refs(context.clauses["RIGHT"], "COMPARE RIGHT"),
        criterion_ref=str(context.clauses["USING"]),
    )


def _update(context: OperatorBuildContext) -> UpdateRequest:
    raw_updates = context.clauses["TARGET"]
    if not isinstance(raw_updates, Sequence) or isinstance(
        raw_updates, (str, bytes, bytearray)
    ):
        raise ValueError("UPDATE TARGET requires explicit update items")
    before_revision = int(context.clauses["BEFORE"])
    if before_revision != context.expected_state_revision:
        raise ValueError("UPDATE BEFORE does not match the current state revision")
    invalidates = _optional_refs(context.clauses["INVALIDATE"], "UPDATE INVALIDATE")
    recalculates = _optional_refs(context.clauses["RECALCULATE"], "UPDATE RECALCULATE")
    preserves = set(_optional_refs(context.clauses["PRESERVE"], "UPDATE PRESERVE"))
    if preserves & set(invalidates):
        raise ValueError(
            "UPDATE cannot both preserve and invalidate the same reference"
        )
    merged_updates: list[UpdateItem] = []
    for item in raw_updates:
        update = UpdateItem.model_validate(item)
        merged_updates.append(
            update.model_copy(
                update={
                    "invalidates": tuple(
                        dict.fromkeys((*update.invalidates, *invalidates))
                    ),
                    "recalculates": tuple(
                        dict.fromkeys((*update.recalculates, *recalculates))
                    ),
                }
            )
        )
    unique_updates: list[UpdateItem] = []
    update_by_target_id: dict[str, UpdateItem] = {}
    for update in merged_updates:
        target_id = update.mutation.after.id
        existing = update_by_target_id.get(target_id)
        if existing is None:
            update_by_target_id[target_id] = update
            unique_updates.append(update)
            continue
        if existing != update:
            raise ValueError(
                "UPDATE TARGET contains conflicting mutations for canonical ref: "
                f"{target_id}"
            )
    return UpdateRequest(
        **_base(context),
        source_change_id=f"{context.request_id}:change",
        updates=tuple(unique_updates),
        evidence_refs=_refs(context.clauses["FROM"], "UPDATE FROM"),
        update_policy_ref=str(context.clauses["USING"]),
    )


def _select(context: OperatorBuildContext) -> SelectRequest:
    raw_witnesses = context.clauses.get("DOMINANCE", ())
    if not isinstance(raw_witnesses, Sequence) or isinstance(
        raw_witnesses, (str, bytes, bytearray)
    ):
        raise ValueError("SELECT DOMINANCE requires explicit comparison witnesses")
    raw_facts = context.clauses.get("FACTS", {})
    if not isinstance(raw_facts, Mapping):
        raise ValueError("SELECT FACTS requires a mapping indexed by alternative")
    raw_context = context.clauses.get("CONTEXT", {})
    if not isinstance(raw_context, Mapping):
        raise ValueError("SELECT CONTEXT requires a shared fact mapping")
    return SelectRequest(
        **_base(context),
        goal_ref=str(context.clauses["FOR"]),
        alternative_refs=_refs(context.clauses["FROM"], "SELECT FROM"),
        selection_policy_ref=str(context.clauses["UNDER"]),
        requirement_refs=_refs(context.clauses["REQUIRE"], "SELECT REQUIRE"),
        preserve_refs=_refs(context.clauses["PRESERVE"], "SELECT PRESERVE"),
        dominance_witnesses=tuple(
            DominanceWitness.model_validate(item) for item in raw_witnesses
        ),
        alternative_facts=FrozenMap(raw_facts),
        context_facts=FrozenMap(raw_context),
    )
