"""Mechanical projection of DRM-declared reusable knowledge."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    StaticKnowledgeGraph,
    StaticKnowledgeNode,
    StaticKnowledgePriority,
)
from agents.yf_arc3_v5.drm.registry import DrmRegistry
from agents.yf_arc3_v5.logos.types import (
    Ref,
    StaticKnowledgeNodeKind,
)


def build_declared_static_knowledge_graph(
    drm: DrmRegistry,
    knowledge_graph_ref: Ref,
) -> StaticKnowledgeGraph:
    """Project only DRM-marked declarations; never invent a knowledge node."""

    declarations = tuple(
        declaration
        for declaration_id in drm.declaration_ids
        for declaration, _loaded in (drm.resolve(declaration_id),)
        if declaration.static_knowledge_node_kind is not None
    )
    nodes_by_ref: dict[str, StaticKnowledgeNode] = {
        declaration.id: StaticKnowledgeNode(
            knowledge_ref=declaration.id,
            node_kind=declaration.static_knowledge_node_kind,
            parent_knowledge_refs=declaration.static_knowledge_parent_refs,
            activation_condition_refs=declaration.applicability,
            expected_effect_refs=declaration.invariants,
            falsifier_refs=declaration.falsifiers,
            dependency_refs=declaration.premises,
        )
        for declaration in declarations
    }

    def add_leaf(
        ref: str,
        kind: StaticKnowledgeNodeKind,
        parent_ref: str,
    ) -> None:
        existing = nodes_by_ref.get(ref)
        if existing is not None:
            if existing.node_kind is not kind:
                raise ValueError(
                    f"DRM static knowledge ref has conflicting kinds: {ref}"
                )
            return
        nodes_by_ref[ref] = StaticKnowledgeNode(
            knowledge_ref=ref,
            node_kind=kind,
            parent_knowledge_refs=(parent_ref,),
        )

    for declaration in declarations:
        root_ref = declaration.id
        for ref in declaration.applicability:
            add_leaf(ref, StaticKnowledgeNodeKind.ACTIVATION_CONDITION, root_ref)
        for ref in declaration.invariants:
            add_leaf(ref, StaticKnowledgeNodeKind.EXPECTED_EFFECT, root_ref)
        for ref in declaration.falsifiers:
            add_leaf(ref, StaticKnowledgeNodeKind.FALSIFIER, root_ref)
        for ref in declaration.premises:
            add_leaf(ref, StaticKnowledgeNodeKind.DEPENDENCY, root_ref)

    nodes = tuple(nodes_by_ref[ref] for ref in sorted(nodes_by_ref))
    priorities = tuple(
        StaticKnowledgePriority(
            priority_ref=declaration.id,
            ordered_alternative_refs=declaration.declared_alternative_refs,
            activation_condition_refs=declaration.applicability,
            expected_effect_refs=declaration.invariants,
            falsifier_refs=declaration.falsifiers,
            dependency_refs=declaration.premises,
        )
        for declaration in declarations
        if declaration.static_knowledge_node_kind is StaticKnowledgeNodeKind.PRIORITY
    )
    node_refs = {node.knowledge_ref for node in nodes}
    roots = tuple(
        node.knowledge_ref
        for node in nodes
        if not node.parent_knowledge_refs
    )
    if not nodes or not roots or any(
        parent_ref not in node_refs
        for node in nodes
        for parent_ref in node.parent_knowledge_refs
    ):
        raise ValueError("DRM static knowledge declarations must form a non-empty forest")
    return StaticKnowledgeGraph(
        knowledge_graph_ref=knowledge_graph_ref,
        nodes=nodes,
        edges=(),
        priorities=priorities,
        root_knowledge_refs=roots,
    )
