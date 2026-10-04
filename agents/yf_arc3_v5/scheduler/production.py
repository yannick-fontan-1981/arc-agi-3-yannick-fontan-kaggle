"""Construction of the M6 offline production workflow runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from agents.yf_arc3_v5.capabilities import (
    CapabilityRegistry,
    build_default_capability_registry,
)
from agents.yf_arc3_v5.drm import (
    DrmRegistry,
    build_declared_static_knowledge_graph,
    build_drm_authority_registry,
    build_production_drm_registry,
    load_production_drm,
)
from agents.yf_arc3_v5.operators import OperatorAuthorityRegistry, OperatorRuntime
from agents.yf_arc3_v5.drm.contracts import LoadedDrmDocument
from agents.yf_arc3_v5.runtime.acquisition import (
    WorldInputRepository,
    register_frame_acquisition,
)
from agents.yf_arc3_v5.runtime.boundary import ActionBoundary
from agents.yf_arc3_v5.runtime.contracts import EnvironmentTransport
from agents.yf_arc3_v5.scheduler.capability_calls import register_capability_calls
from agents.yf_arc3_v5.scheduler.control_calls import SourceControlledActionCalls
from agents.yf_arc3_v5.scheduler.dynamic_workflow_calls import (
    register_dynamic_workflow_calls,
)
from agents.yf_arc3_v5.scheduler.dynamic_workflow_shadow import (
    DynamicWorkflowShadowObserver,
)
from agents.yf_arc3_v5.scheduler.dynamic_workflow_runtime import (
    DynamicWorkflowRuntimeGate,
)
from agents.yf_arc3_v5.dynamic_workflow.loader import load_cold_component_document, ColdComponentDocument
from agents.yf_arc3_v5.scheduler.operator_builders import register_operator_builders
from agents.yf_arc3_v5.scheduler.registry import SchedulerRuntimeRegistry
from agents.yf_arc3_v5.scheduler.runtime import WorkflowScheduler
from agents.yf_arc3_v5.src.compiler import CompiledModule
from agents.yf_arc3_v5.src.production_contracts import (
    ProductionWorkflowContract,
    load_production_workflow_contracts,
)
from agents.yf_arc3_v5.src.production import compile_production_workflows, production_environment
from agents.yf_arc3_v5.src.registry import RegistryEntry, WorkflowRegistry
from agents.yf_arc3_v5.src.runtime_sources import compile_m7_workflows, m7_environment
from agents.yf_arc3_v5.state import EventStore
from agents.yf_arc3_v5.capabilities import StaticKnowledgeGraph


@dataclass(frozen=True)
class M6ProductionRuntime:
    scheduler: WorkflowScheduler
    workflow_registry: WorkflowRegistry
    runtime_registry: SchedulerRuntimeRegistry
    authority_registry: OperatorAuthorityRegistry
    capability_registry: CapabilityRegistry
    drm_registry: DrmRegistry
    static_knowledge_graph: StaticKnowledgeGraph
    compiled_modules: tuple[CompiledModule, ...]
    registry_entries: tuple[RegistryEntry, ...]


@dataclass(frozen=True)
class M7ProductionRuntime:
    scheduler: WorkflowScheduler
    workflow_registry: WorkflowRegistry
    runtime_registry: SchedulerRuntimeRegistry
    authority_registry: OperatorAuthorityRegistry
    capability_registry: CapabilityRegistry
    drm_registry: DrmRegistry
    static_knowledge_graph: StaticKnowledgeGraph
    world_inputs: WorldInputRepository
    compiled_modules: tuple[CompiledModule, ...]
    registry_entries: tuple[RegistryEntry, ...]
    workflow_execution_contracts: Mapping[str, ProductionWorkflowContract]
    workflow_execution_contract_source_hash: str
    _action_boundaries: dict[int, tuple[EnvironmentTransport, ActionBoundary]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def action_boundary(self, transport: EnvironmentTransport) -> ActionBoundary:
        key = id(transport)
        cached = self._action_boundaries.get(key)
        if cached is not None and cached[0] is transport:
            return cached[1]
        boundary = ActionBoundary(
            scheduler=self.scheduler,
            event_store=self.scheduler.event_store,
            world_inputs=self.world_inputs,
            transport=transport,
        )
        self._action_boundaries[key] = (transport, boundary)
        return boundary


def build_m6_production_runtime(
    store: EventStore, *, enable_dynamic_workflow_shadow: bool = False,
    enable_pure_operator_audit: bool = False,
) -> M6ProductionRuntime:
    capabilities = build_default_capability_registry()
    drm = build_production_drm_registry(load_production_drm())
    modules = compile_production_workflows()
    workflows = WorkflowRegistry()
    entries = workflows.register_graph(modules)
    loaded_components, cold_components = load_cold_component_document(
        workflow_registry=workflows,
        runtime_profile_ref="runtime.m6",
    )
    runtime = SchedulerRuntimeRegistry()
    register_capability_calls(runtime, capabilities)
    register_dynamic_workflow_calls(runtime, cold_components)
    register_operator_builders(runtime)
    authorities = build_drm_authority_registry(drm)
    scheduler = WorkflowScheduler(
        workflow_registry=workflows,
        runtime_registry=runtime,
        operator_runtime=OperatorRuntime(
            store, authorities,
            persist_pure_operator_audit=enable_pure_operator_audit,
        ),
        dynamic_workflow_shadow=(
            DynamicWorkflowShadowObserver(loaded_components, cold_components)
            if enable_dynamic_workflow_shadow else None
        ),
        dynamic_workflow_runtime=DynamicWorkflowRuntimeGate(
            loaded_components,
            cold_components,
        ),
    )
    # M6 workflows include declarative proposal preparation calls as well as
    # pure capability calls.  Keep the same source-controlled call surface as
    # the M7 runtime; otherwise a migrated workflow fails at dispatch with an
    # unregistered call even though its source compiled successfully.
    SourceControlledActionCalls(
        scheduler=scheduler,
        workflow_registry=workflows,
        event_store=store,
        drm_registry=drm,
        capability_registry=capabilities,
    ).register(runtime)
    return M6ProductionRuntime(
        scheduler=scheduler,
        workflow_registry=workflows,
        runtime_registry=runtime,
        authority_registry=authorities,
        capability_registry=capabilities,
        drm_registry=drm,
        static_knowledge_graph=build_declared_static_knowledge_graph(
            drm, "knowledge.graph.generic"
        ),
        compiled_modules=modules,
        registry_entries=entries,
    )


def build_m7_production_runtime(
    store: EventStore,
    *,
    world_inputs: WorldInputRepository | None = None,
    drm_documents: tuple[LoadedDrmDocument, ...] | None = None,
    src_modules: tuple[CompiledModule, ...] | None = None,
    cold_component_document: ColdComponentDocument | None = None,
    control_source_hashes: dict[str, str] | None = None,
    enable_dynamic_workflow_shadow: bool = False,
    enable_pure_operator_audit: bool = False,
) -> M7ProductionRuntime:
    capabilities = build_default_capability_registry()
    drm = build_production_drm_registry(load_production_drm() if drm_documents is None else drm_documents)
    modules = (compile_m7_workflows(environment=m7_environment(
        base_environment=production_environment(capabilities=capabilities, drm=drm)
    )) if src_modules is None else src_modules)
    workflows = WorkflowRegistry()
    entries = workflows.register_graph(modules)
    loaded_components, cold_components = load_cold_component_document(
        workflow_registry=workflows,
        runtime_profile_ref="runtime.m7",
        document_override=cold_component_document,
    )
    runtime = SchedulerRuntimeRegistry()
    register_capability_calls(runtime, capabilities)
    register_dynamic_workflow_calls(runtime, cold_components, control_source_hashes=control_source_hashes)
    register_operator_builders(runtime)
    authorities = build_drm_authority_registry(drm)
    active_world_inputs = world_inputs or WorldInputRepository()
    register_frame_acquisition(authorities, active_world_inputs)
    scheduler = WorkflowScheduler(
        workflow_registry=workflows,
        runtime_registry=runtime,
        operator_runtime=OperatorRuntime(
            store, authorities,
            persist_pure_operator_audit=enable_pure_operator_audit,
        ),
        dynamic_workflow_shadow=(
            DynamicWorkflowShadowObserver(loaded_components, cold_components)
            if enable_dynamic_workflow_shadow else None
        ),
        dynamic_workflow_runtime=DynamicWorkflowRuntimeGate(
            loaded_components,
            cold_components,
        ),
    )
    SourceControlledActionCalls(
        scheduler=scheduler,
        workflow_registry=workflows,
        event_store=store,
        drm_registry=drm,
        capability_registry=capabilities,
        world_inputs=active_world_inputs,
    ).register(runtime, control_source_hashes=control_source_hashes)
    execution_contracts, execution_contract_hash = load_production_workflow_contracts()
    return M7ProductionRuntime(
        scheduler=scheduler,
        workflow_registry=workflows,
        runtime_registry=runtime,
        authority_registry=authorities,
        capability_registry=capabilities,
        drm_registry=drm,
        static_knowledge_graph=build_declared_static_knowledge_graph(
            drm, "knowledge.graph.generic"
        ),
        world_inputs=active_world_inputs,
        compiled_modules=modules,
        registry_entries=entries,
        workflow_execution_contracts=MappingProxyType({
            contract.workflow_id: contract for contract in execution_contracts.contracts
        }),
        workflow_execution_contract_source_hash=execution_contract_hash,
    )
