"""Bounded run reports derived from V5 journal, registry, and world inputs."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    EnvironmentActionAttemptRecord,
    EnvironmentActionDispatchedRecord,
    WorkflowInstanceRecord,
)
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import FrozenMap
from agents.yf_arc3_v5.observability.attestation import project_action_lineage
from agents.yf_arc3_v5.observability.models import CompactSrlLine, RunReport
from agents.yf_arc3_v5.observability.projection import project_compact_srl
from agents.yf_arc3_v5.observability.trace_diff import diff_compact_traces
from agents.yf_arc3_v5.runtime.acquisition import WorldInputRepository
from agents.yf_arc3_v5.src.compiler import CompiledModule
from agents.yf_arc3_v5.src.registry import RegistryEntry
from agents.yf_arc3_v5.state import EventStore

_RESOURCE_WORDS = frozenset(
    {"resource", "quantity", "capacity", "conservation", "reserve", "cost"}
)


def build_run_report(
    *,
    event_store: EventStore,
    registry_entries: tuple[RegistryEntry, ...],
    world_inputs: WorldInputRepository,
    current_raw_input_ref: str | None = None,
    compiled_modules: tuple[CompiledModule, ...] = (),
    identity: FrozenMap | None = None,
    started_at: float | None = None,
    clock: Callable[[], float] = time.time,
    expected_trace: Sequence[CompactSrlLine | str] | None = None,
) -> RunReport:
    events = event_store.event_view
    artifacts = event_store.artifacts
    latest_workflows: dict[str, WorkflowInstanceRecord] = {}
    for artifact in artifacts:
        if isinstance(artifact, WorkflowInstanceRecord):
            latest_workflows[artifact.workflow_instance_id or ""] = artifact
    # Repository order is lexical identity order, not observation chronology.
    # A controller may also have a current manual observation after dispatch.
    latest_dispatch = event_store.latest_artifact(EnvironmentActionDispatchedRecord)
    raw_ref = current_raw_input_ref or (latest_dispatch.raw_input_ref if latest_dispatch else None)
    if raw_ref is not None:
        latest_world = world_inputs.get(raw_ref)
    else:
        inputs = world_inputs.values
        latest_world = inputs[0] if len(inputs) == 1 else None
    dispatch_count = sum(
        isinstance(item, EnvironmentActionDispatchedRecord) for item in artifacts
    )
    attempted_intents = {
        item.action_intent_id
        for item in artifacts
        if isinstance(item, EnvironmentActionAttemptRecord)
    }
    dispatched_intents = {
        item.action_intent_id
        for item in artifacts
        if isinstance(item, EnvironmentActionDispatchedRecord)
    }
    uncertain_dispatch_count = len(attempted_intents - dispatched_intents)
    elapsed = None if started_at is None else max(0.0, clock() - started_at)
    actual_trace = project_compact_srl(events, limit=2000)
    if expected_trace is None:
        divergence = FrozenMap(
            {
                "status": "not_compared",
                "reason": "no_expected_compact_trace_supplied",
            }
        )
    else:
        diff = diff_compact_traces(expected_trace, actual_trace)
        divergence = FrozenMap(
            {
                "status": "equivalent" if diff.equivalent else "diverged",
                "matched_prefix_count": diff.matched_prefix_count,
                "expected_count": diff.expected_count,
                "actual_count": diff.actual_count,
                "first_divergence": (
                    None
                    if diff.first_divergence is None
                    else diff.first_divergence.model_dump(mode="python")
                ),
            }
        )
    statuses: dict[str, int] = {}
    for workflow in (latest_workflows[__yf_order_key] for __yf_order_key in sorted(latest_workflows)):
        statuses[workflow.status.value] = statuses.get(workflow.status.value, 0) + 1
    return RunReport(
        identity=identity or FrozenMap({"agent": "yf-arc3-v5", "version": 5}),
        result=FrozenMap(
            {
                "run_id": event_store.run_id,
                "world_state": latest_world.state if latest_world else "unobserved",
                "score": latest_world.score if latest_world else 0,
                "latest_frame_id": latest_world.frame_id if latest_world else None,
                "workflow_status_counts": statuses,
                "dispatch_attempt_count": len(attempted_intents),
                "dispatch_count": dispatch_count,
                "uncertain_dispatch_count": uncertain_dispatch_count,
            }
        ),
        source_hashes=tuple(
            FrozenMap(
                {
                    "module_id": entry.module_id,
                    "source_hash": entry.source_hash,
                    "semantic_hash": entry.semantic_hash,
                    "transitive_source_hash": entry.transitive_source_hash,
                }
            )
            for entry in registry_entries
        ),
        first_divergence=divergence,
        resource_outcome=_resource_outcome(
            event_store.snapshot.terms,
            event_store.snapshot.current_claims,
        ),
        performance=FrozenMap(
            {
                "elapsed_seconds": elapsed,
                "event_count": len(events),
                "compact_srl_line_count": len(actual_trace),
                "dispatch_attempt_count": len(attempted_intents),
                "dispatch_count": dispatch_count,
                "uncertain_dispatch_count": uncertain_dispatch_count,
                "events_per_second": (
                    None
                    if elapsed is None or elapsed <= 0
                    else round(len(events) / elapsed, 3)
                ),
            }
        ),
        action_lineage=project_action_lineage(
            events,
            compiled_modules=compiled_modules,
        ),
        state=FrozenMap(
            {
                "revision": event_store.snapshot.revision,
                "prediction_revision": event_store.snapshot.prediction_revision,
                "state_hash": event_store.snapshot.state_hash,
                "term_count": len(event_store.snapshot.terms),
                "claim_count": len(event_store.snapshot.current_claims),
            }
        ),
    )


def _resource_outcome(
    terms: tuple[Term, ...],
    claims: tuple[Claim, ...],
) -> FrozenMap:
    term_refs = tuple(item.id for item in terms if _has_resource_word(item.label))
    claim_refs = tuple(item.id for item in claims if _has_resource_word(item.predicate))
    return FrozenMap(
        {
            "status": (
                "explicit_resource_state_present"
                if term_refs or claim_refs
                else "no_explicit_resource_state"
            ),
            "basis": "explicit_symbolic_labels_only",
            "term_refs": term_refs[:40],
            "claim_refs": claim_refs[:40],
            "truncated": len(term_refs) > 40 or len(claim_refs) > 40,
        }
    )


def _has_resource_word(value: str) -> bool:
    words = set(filter(None, re.split(r"[^a-z0-9]+", value.lower())))
    return bool(words & _RESOURCE_WORDS)
