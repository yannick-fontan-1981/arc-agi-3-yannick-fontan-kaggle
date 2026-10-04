"""Exact source-hashed registries for cold components and dynamic IR values."""

from __future__ import annotations

from agents.yf_arc3_v5.dynamic_workflow.contracts import (
    ColdWorkflowComponent,
    DynamicWorkflowIR,
)
from agents.yf_arc3_v5.logos.types import Ref, stable_digest


class DynamicWorkflowRegistryError(ValueError):
    """A source identity or semantic hash did not resolve exactly."""


class ColdComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, ColdWorkflowComponent] = {}
        self._components_cache: tuple[ColdWorkflowComponent, ...] | None = None
        self._registry_hash_cache: str | None = None

    @property
    def components(self) -> tuple[ColdWorkflowComponent, ...]:
        if self._components_cache is None:
            self._components_cache = tuple(self._components[ref] for ref in sorted(self._components))
        return self._components_cache

    @property
    def registry_hash(self) -> str:
        if self._registry_hash_cache is None:
            self._registry_hash_cache = stable_digest(
                tuple(
                    (
                        component.component_ref,
                        component.transitive_source_hash,
                        component.semantic_hash,
                    )
                    for component in self.components
                )
            )
        return self._registry_hash_cache

    def register(self, component: ColdWorkflowComponent) -> ColdWorkflowComponent:
        existing = self._components.get(component.component_ref)
        if existing is not None:
            if existing == component:
                return existing
            raise DynamicWorkflowRegistryError(
                f"cold component source collision: {component.component_ref}"
            )
        self._components[component.component_ref] = component
        self._components_cache = None
        self._registry_hash_cache = None
        return component

    def resolve(
        self,
        component_ref: Ref,
        *,
        expected_transitive_source_hash: Ref | None = None,
    ) -> ColdWorkflowComponent:
        component = self._components.get(component_ref)
        if component is None:
            raise DynamicWorkflowRegistryError(
                f"unregistered cold component: {component_ref}"
            )
        if (
            expected_transitive_source_hash is not None
            and component.transitive_source_hash != expected_transitive_source_hash
        ):
            raise DynamicWorkflowRegistryError(
                f"stale cold component source hash: {component_ref}"
            )
        return component


class DynamicWorkflowRegistry:
    def __init__(self) -> None:
        self._workflows: dict[str, DynamicWorkflowIR] = {}

    def register(self, workflow: DynamicWorkflowIR) -> DynamicWorkflowIR:
        existing = self._workflows.get(workflow.semantic_hash)
        if existing is not None:
            if existing == workflow:
                return existing
            raise DynamicWorkflowRegistryError(
                f"dynamic workflow semantic hash collision: {workflow.semantic_hash}"
            )
        self._workflows[workflow.semantic_hash] = workflow
        return workflow

    def resolve(self, semantic_hash: Ref) -> DynamicWorkflowIR:
        workflow = self._workflows.get(semantic_hash)
        if workflow is None:
            raise DynamicWorkflowRegistryError(
                f"unregistered dynamic workflow: {semantic_hash}"
            )
        return workflow
