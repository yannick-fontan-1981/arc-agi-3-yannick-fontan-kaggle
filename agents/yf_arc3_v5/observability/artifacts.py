"""Atomic persistence of bounded, non-authoritative M8 debug projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.yf_arc3_v5.logos.operations import OperatorInvocation, OperatorResult
from agents.yf_arc3_v5.drm.registry import DrmRegistry
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.observability.models import RunReport
from agents.yf_arc3_v5.observability.projection import (
    project_active_state,
    project_compact_srl,
    render_compact_srl,
)
from agents.yf_arc3_v5.runtime.acquisition import WorldInputRepository
from agents.yf_arc3_v5.src.compiler import CompiledModule
from agents.yf_arc3_v5.src.registry import RegistryEntry
from agents.yf_arc3_v5.state import CognitiveEvent, EventStore, RecordArtifact


def write_observability_bundle(
    target_dir: str | Path,
    *,
    event_store: EventStore,
    registry_entries: tuple[RegistryEntry, ...],
    world_inputs: WorldInputRepository,
    compiled_modules: tuple[CompiledModule, ...],
    report: RunReport,
    drm_registry: DrmRegistry | None = None,
    max_events: int = 500,
    max_state_items: int = 50,
) -> FrozenMap:
    if max_events < 1 or max_events > 5000:
        raise ValueError("max_events must be between 1 and 5000")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    events = event_store.event_view
    lines = project_compact_srl(events, limit=max_events)
    state = project_active_state(
        event_store.snapshot,
        events,
        max_items=max_state_items,
        drm=drm_registry,
    )
    bounded_events = events[-max_events:]
    compact_jsonl = "\n".join(
        json.dumps(_compact_event(event), ensure_ascii=True, sort_keys=True)
        for event in bounded_events
    )
    payloads = {
        "visible.srl": render_compact_srl(lines),
        "active_state.json": state.model_dump_json(indent=2),
        "events.compact.jsonl": compact_jsonl,
        "run_report.json": report.model_dump_json(indent=2),
        "source_registry.json": json.dumps(
            [entry.model_dump(mode="json") for entry in registry_entries],
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ),
    }
    files: list[dict[str, Any]] = []
    for name, content in sorted(payloads.items()):
        path = target / name
        _atomic_write(
            path, content + ("\n" if content and not content.endswith("\n") else "")
        )
        files.append(
            {
                "name": name,
                "path": str(path),
                "content_hash": stable_digest(content),
                "bytes": len(content.encode("utf-8")),
            }
        )
    manifest_values = {
        "run_id": event_store.run_id,
        "state_hash": event_store.snapshot.state_hash,
        "event_count": len(events),
        "bounded_event_count": len(bounded_events),
        "world_input_count": len(world_inputs.values),
        "compiled_module_count": len(compiled_modules),
        "files": tuple(files),
        "authoritative": False,
    }
    manifest = FrozenMap(
        {
            **manifest_values,
            "content_hash": stable_digest(manifest_values),
        }
    )
    _atomic_write(
        target / "manifest.json",
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True, sort_keys=True)
        + "\n",
    )
    return manifest


def _compact_event(event: CognitiveEvent) -> dict[str, Any]:
    mutation = event.mutation
    value: dict[str, Any] = {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "transaction_id": event.transaction_id,
        "frame_id": event.frame_id,
        "state_revision_before": event.state_revision_before,
        "state_revision_after": event.state_revision_after,
        "source_hash": event.source_hash,
        "mutation_kind": mutation.kind,
    }
    if isinstance(mutation, RecordArtifact):
        artifact = mutation.artifact
        value.update(
            {
                "record_id": artifact.id,
                "record_kind": artifact.record_kind,
                "workflow_definition_id": artifact.workflow_definition_id,
                "workflow_instance_id": artifact.workflow_instance_id,
                "workflow_step_id": getattr(artifact, "workflow_step_id", None),
            }
        )
        if isinstance(artifact, (OperatorInvocation, OperatorResult)):
            value["operator"] = artifact.operator.value
        if isinstance(artifact, OperatorResult):
            value["operation_status"] = artifact.status.value
    else:
        record = mutation.after
        value.update(
            {
                "record_id": record.id,
                "record_kind": "term" if mutation.kind == "put_term" else "claim",
            }
        )
    return value


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
