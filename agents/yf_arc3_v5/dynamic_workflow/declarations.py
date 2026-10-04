"""Typed projection of DRM-authored dynamic workflow composition declarations."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from agents.yf_arc3_v5.drm.loader import load_drm_document
from agents.yf_arc3_v5.dynamic_workflow.contracts import (
    WorkflowBounds,
    WorkflowCompletionContract,
)
from agents.yf_arc3_v5.dynamic_workflow.synthesis import (
    DeclaredCompositionOrder,
    DeclaredPredicateProducerOrder,
    DynamicWorkflowSynthesisRequest,
    WorkflowSynthesisSnapshot,
)
from agents.yf_arc3_v5.logos.types import FrozenModel, Ref, stable_digest


_DYNAMIC_WORKFLOW_DRM_PATH = (
    Path(__file__).resolve().parents[1] / "drm" / "dynamic_workflow.drm"
)


class DeclaredCompositionFamily(FrozenModel):
    family_ref: Ref
    root_goal_ref: Ref
    synthesis_policy_ref: Ref
    drm_declaration_hash: Ref
    orders: tuple[DeclaredCompositionOrder, ...] = Field(min_length=1, max_length=3)
    completion_contract: WorkflowCompletionContract
    preserved_invariant_refs: tuple[Ref, ...] = ()
    global_falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.declared_composition_family.v1"


class DeclaredRuntimeCompositionEdge(FrozenModel):
    edge_ref: Ref
    parent_workflow_refs: tuple[Ref, ...] = Field(min_length=1)
    child_workflow_refs: tuple[Ref, ...] = Field(min_length=1)
    composition_principle_ref: Ref
    falsifier_ref: Ref
    maximum_occurrences_per_parent: int = Field(strict=True, ge=1, le=16)
    schema_version: Ref = "yf_arc3_v5.declared_runtime_composition_edge.v1"


def load_declared_runtime_composition_edges() -> tuple[DeclaredRuntimeCompositionEdge, ...]:
    loaded = load_drm_document(_DYNAMIC_WORKFLOW_DRM_PATH)
    policies = tuple(
        declaration
        for declaration in loaded.document.declarations
        if declaration.kind.value == "selection_policy"
        and declaration.configuration.get("runtime_composition_edges")
    )
    if len(policies) != 1:
        raise ValueError("dynamic workflow DRM must declare runtime composition edges")
    values = policies[0].configuration.get("runtime_composition_edges", ())
    return tuple(
        sorted(
            (
                DeclaredRuntimeCompositionEdge(
                    edge_ref=value["edge_ref"],
                    parent_workflow_refs=tuple(value["parent_workflow_refs"]),
                    child_workflow_refs=tuple(value["child_workflow_refs"]),
                    composition_principle_ref=value["composition_principle_ref"],
                    falsifier_ref=value["falsifier_ref"],
                    maximum_occurrences_per_parent=int(
                        value["maximum_occurrences_per_parent"]
                    ),
                )
                for value in values
            ),
            key=lambda item: item.edge_ref,
        )
    )


def load_declared_composition_families() -> tuple[DeclaredCompositionFamily, ...]:
    loaded = load_drm_document(_DYNAMIC_WORKFLOW_DRM_PATH)
    synthesis_policies = tuple(
        declaration
        for declaration in loaded.document.declarations
        if declaration.kind.value == "selection_policy"
        and declaration.configuration.get("composition_families")
    )
    if len(synthesis_policies) != 1:
        raise ValueError("dynamic workflow DRM must declare one synthesis policy")
    policy = synthesis_policies[0]
    law_hash_by_ref = {
        declaration.id: stable_digest(declaration)
        for declaration in loaded.document.declarations
    }
    family_values = policy.configuration.get("composition_families", ())
    families: list[DeclaredCompositionFamily] = []
    for family_value in family_values:
        orders: list[DeclaredCompositionOrder] = []
        for order_value in family_value["orders"]:
            producer_orders = tuple(
                DeclaredPredicateProducerOrder(
                    required_predicate_ref=item["required_predicate_ref"],
                    eligible_component_refs=tuple(item["eligible_component_refs"]),
                    composition_principle_ref=item["composition_principle_ref"],
                    falsifier_ref=item["falsifier_ref"],
                    composition_law_hash=law_hash_by_ref[
                        item["composition_principle_ref"]
                    ],
                )
                for item in order_value.get("producer_orders", ())
            )
            orders.append(
                DeclaredCompositionOrder(
                    order_ref=order_value["order_ref"],
                    root_component_ref=order_value["root_component_ref"],
                    producer_orders=producer_orders,
                )
            )
        families.append(
            DeclaredCompositionFamily(
                family_ref=family_value["family_ref"],
                root_goal_ref=family_value["root_goal_ref"],
                synthesis_policy_ref=policy.id,
                drm_declaration_hash=loaded.semantic_hash,
                orders=tuple(orders),
                completion_contract=WorkflowCompletionContract.model_validate(
                    family_value["completion_contract"]
                ),
                preserved_invariant_refs=tuple(
                    family_value.get("preserved_invariant_refs", ())
                ),
                global_falsifier_refs=tuple(
                    family_value["global_falsifier_refs"]
                ),
            )
        )
    return tuple(sorted(families, key=lambda item: item.family_ref))


def build_declared_family_synthesis_request(
    family: DeclaredCompositionFamily,
    *,
    snapshot: WorkflowSynthesisSnapshot,
    cold_registry_hash: Ref,
    bounds: WorkflowBounds,
) -> DynamicWorkflowSynthesisRequest:
    if snapshot.root_goal_ref != family.root_goal_ref:
        raise ValueError("workflow synthesis snapshot names another declared family goal")
    return DynamicWorkflowSynthesisRequest(
        cold_registry_hash=cold_registry_hash,
        drm_declaration_hash=family.drm_declaration_hash,
        synthesis_policy_ref=family.synthesis_policy_ref,
        snapshot=snapshot,
        declared_orders=family.orders,
        global_resource_bounds=bounds,
        completion_contract=family.completion_contract,
        preserved_invariant_refs=family.preserved_invariant_refs,
        global_falsifier_refs=family.global_falsifier_refs,
    )
