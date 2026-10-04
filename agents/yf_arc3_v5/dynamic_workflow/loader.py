"""Strict loader for the source-attested production cold-component document."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.dynamic_workflow.contracts import ColdWorkflowComponent
from agents.yf_arc3_v5.dynamic_workflow.registry import ColdComponentRegistry
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref, require_unique, stable_digest, canonical_json
from agents.yf_arc3_v5.src.registry import WorkflowRegistry

DEFAULT_COMPONENT_DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "drm"
    / "dynamic_workflow_components.json"
)


class ColdComponentMigrationControl(FrozenModel):
    workflow_ref: Ref
    component_ref: Ref
    migration_state: Ref
    action_authority: Literal["none", "legacy_only", "dynamic_only"]
    shadow_equivalence: Ref
    shadow_proof_refs: tuple[Ref, ...] = ()
    removal_authorization: Ref
    removal_proof_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.cold_component_migration_control.v1"

    @model_validator(mode="after")
    def validate_control(self) -> "ColdComponentMigrationControl":
        require_unique(self.shadow_proof_refs, "shadow proof refs")
        require_unique(self.removal_proof_refs, "removal proof refs")
        active = self.migration_state in {
            "single_node_active",
            "composed_candidate",
            "composed_active",
            "legacy_retirement_candidate",
            "legacy_retired",
        }
        if active and (
            self.action_authority != "dynamic_only"
            or self.shadow_equivalence != "passed"
            or not self.shadow_proof_refs
        ):
            raise ValueError(
                "active cold component requires dynamic-only authority and shadow proof"
            )
        if self.migration_state == "legacy_retired" and (
            self.removal_authorization != "authorized_after_proof"
            or not self.removal_proof_refs
        ):
            raise ValueError(
                "retired legacy authority requires explicit authorization and proof"
            )
        return self


class ColdComponentDocument(FrozenModel):
    authority: FrozenMap
    component_count: int = Field(strict=True, ge=1)
    components: tuple[ColdWorkflowComponent, ...] = Field(min_length=1)
    migration_controls: tuple[ColdComponentMigrationControl, ...] = Field(
        min_length=1
    )
    declaration_source_hashes: FrozenMap
    global_bounds: FrozenMap
    schema_version: Ref = "yf_arc3_v5.cold_workflow_component_document.v1"

    @property
    def synthesis_policy_ref(self) -> Ref:
        value = self.authority.get("synthesis_policy_ref")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("cold component document lacks declared synthesis policy")
        return value

    @model_validator(mode="after")
    def validate_document(self) -> "ColdComponentDocument":
        if self.component_count != len(self.components):
            raise ValueError("cold component count mismatch")
        if self.component_count != len(self.migration_controls):
            raise ValueError("cold component migration-control count mismatch")
        require_unique(
            tuple(item.component_ref for item in self.components),
            "cold component refs",
        )
        require_unique(
            tuple(item.workflow_definition_ref or "" for item in self.components),
            "cold component workflow definitions",
        )
        if self.components != tuple(
            sorted(self.components, key=lambda item: item.component_ref)
        ):
            raise ValueError("cold components must use canonical reference order")
        require_unique(
            tuple(item.workflow_ref for item in self.migration_controls),
            "migration-control workflow refs",
        )
        if self.migration_controls != tuple(
            sorted(self.migration_controls, key=lambda item: item.workflow_ref)
        ):
            raise ValueError("migration controls must use canonical workflow order")
        component_refs = {item.component_ref for item in self.components}
        if {item.component_ref for item in self.migration_controls} != component_refs:
            raise ValueError("migration controls and cold components differ")
        return self


class LoadedColdComponentDocument(FrozenModel):
    document: ColdComponentDocument
    source_unit: Ref
    source_hash: Ref
    semantic_hash: Ref
    registry_hash: Ref
    schema_version: Ref = "yf_arc3_v5.loaded_cold_workflow_components.v1"


class ColdComponentLoadError(ValueError):
    """The cold-component document or a referenced workflow failed attestation."""


def load_cold_component_document(
    path: Path = DEFAULT_COMPONENT_DOCUMENT,
    *,
    workflow_registry: WorkflowRegistry | None = None,
    runtime_profile_ref: Ref | None = None,
    document_override: ColdComponentDocument | None = None,
) -> tuple[LoadedColdComponentDocument, ColdComponentRegistry]:
    try:
        source = (path.read_bytes() if document_override is None
                  else canonical_json(document_override).encode("utf-8"))
        raw: Any = json.loads(source.decode("utf-8"))
        document = ColdComponentDocument.model_validate(raw)
        registry = ColdComponentRegistry()
        for component in document.components:
            requires_runtime_attestation = component.runtime_dispatchable and (
                runtime_profile_ref is None
                or runtime_profile_ref in component.runtime_profile_refs
            )
            if workflow_registry is not None and requires_runtime_attestation:
                if component.workflow_definition_ref is None:
                    raise ValueError("production component lacks a workflow definition")
                workflow_registry.resolve_workflow(
                    component.workflow_definition_ref,
                    expected_transitive_source_hash=component.transitive_source_hash,
                )
            registry.register(component)
    except Exception as error:
        raise ColdComponentLoadError(
            f"invalid cold component document {path}: {error}"
        ) from error
    loaded = LoadedColdComponentDocument(
        document=document,
        source_unit="drm.dynamic_workflow_components",
        source_hash=hashlib.sha256(source).hexdigest(),
        semantic_hash=stable_digest(document),
        registry_hash=registry.registry_hash,
    )
    return loaded, registry
