"""Mechanical bounded graph operations for DRM-declared goal relations.

This module never opens, ranks, enables, or selects a goal.  It only condenses
the declared active relation graph so SRC/DRM can reason over a finite witness.
"""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    DeclaredGoalPlanAlternatives,
    DeclaredGoalPlanInput,
    GoalGraph,
    GoalGraphAnalysis,
    GoalGraphComponent,
    GoalPlanRequest,
    GoalPlanResult,
    GoalPlanWitness,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def analyze_goal_graph(value: GoalGraph) -> GoalGraphAnalysis:
    """Return deterministic strongly-connected components of active edges."""

    nodes = tuple(sorted(node.identity_view.goal_ref for node in value.nodes))
    adjacency: dict[str, tuple[str, ...]] = {node: () for node in nodes}
    for edge in sorted(value.edges, key=lambda item: item.edge_ref):
        if not edge.active:
            continue
        adjacency[edge.source_ref] = tuple(
            sorted(set(adjacency[edge.source_ref]) | {edge.target_ref})
        )

    # Iterative Tarjan is unnecessary at the global bound (256 nodes), but the
    # recursive implementation is easier to audit and remains bounded by the
    # declared graph size.
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adjacency[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        member_refs: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            member_refs.append(member)
            if member == node:
                break
        components.append(tuple(sorted(member_refs)))

    for node in nodes:
        if node not in indices:
            visit(node)

    components.sort(key=lambda refs: refs[0])
    component_by_node = {
        node_ref: f"goal_component:{component_index}"
        for component_index, members in enumerate(components)
        for node_ref in members
    }
    component_models = []
    for component_index, members in enumerate(components):
        component_ref = f"goal_component:{component_index}"
        outgoing = sorted(
            {
                component_by_node[edge.target_ref]
                for edge in value.edges
                if edge.active
                and component_by_node[edge.source_ref] == component_ref
                and component_by_node[edge.target_ref] != component_ref
            }
        )
        component_models.append(
            GoalGraphComponent(
                component_ref=component_ref,
                node_refs=members,
                outgoing_component_refs=tuple(outgoing),
            )
        )
    return GoalGraphAnalysis(
        graph_revision=value.revision,
        static_knowledge_graph_ref=value.static_knowledge_graph_ref,
        components=tuple(component_models),
        node_component_refs=tuple(
            (node_ref, component_by_node[node_ref]) for node_ref in nodes
        ),
        active_edge_refs=tuple(
            edge.edge_ref for edge in value.edges if edge.active
        ),
    )


def enumerate_goal_plan_witnesses(value: GoalPlanRequest) -> GoalPlanResult:
    """Enumerate at most three mechanical topological plan witnesses.

    The function preserves every action alternative as a structural branch and
    never ranks or selects a witness.  DRM/SRC own commitment and release.
    """

    by_ref = {step.subgoal_ref: step for step in value.steps}
    witnesses: list[GoalPlanWitness] = []

    def extend(
        ordered_steps: tuple[str, ...],
        remaining: set[str],
        actions: tuple[str, ...],
    ) -> None:
        if len(witnesses) >= value.maximum_witnesses:
            return
        if not remaining:
            addressed_dimensions = tuple(
                sorted(
                    {
                        dimension
                        for subgoal_ref in ordered_steps
                        for dimension in by_ref[
                            subgoal_ref
                        ].addressed_gap_dimensions
                    },
                    key=lambda item: item.value,
                )
            )
            plan_ref = "goal_plan_witness:" + stable_digest(
                FrozenMap(
                    {
                        "goal_ref": value.goal_ref,
                        "subgoal_refs": ordered_steps,
                        "primitive_action_refs": actions,
                        "addressed_gap_dimensions": tuple(
                            item.value for item in addressed_dimensions
                        ),
                    }
                )
            )[:24]
            witnesses.append(
                GoalPlanWitness(
                    plan_ref=plan_ref,
                    goal_ref=value.goal_ref,
                    subgoal_refs=ordered_steps,
                    primitive_action_refs=actions,
                    addressed_gap_dimensions=addressed_dimensions,
                    remaining_action_bound=len(actions),
                )
            )
            return
        available = sorted(
            ref
            for ref in remaining
            if set(by_ref[ref].prerequisite_subgoal_refs) <= set(ordered_steps)
        )
        if not available:
            return
        for ref in available:
            step = by_ref[ref]
            # The first structural action is sufficient for a witness.  Other
            # alternatives remain available to a subsequent bounded branch.
            extend(
                (*ordered_steps, ref),
                remaining - {ref},
                actions
                + (step.primitive_action_refs[0],) * step.action_bound,
            )
            if len(witnesses) >= value.maximum_witnesses:
                return

    extend((), set(by_ref), ())
    return GoalPlanResult(goal_ref=value.goal_ref, witnesses=tuple(witnesses))


def project_declared_goal_plan_alternatives(
    value: DeclaredGoalPlanInput,
) -> DeclaredGoalPlanAlternatives:
    """Project exact gap coverage facts without ranking or selecting a plan."""

    result = enumerate_goal_plan_witnesses(value.plan_request)
    measurements = value.gap_measurements.measurements
    goal_scope_matches = bool(measurements) and all(
        item.goal_ref == result.goal_ref for item in measurements
    )
    open_dimensions = tuple(
        sorted(
            {item.dimension for item in measurements if not item.satisfied},
            key=lambda item: item.value,
        )
    )
    has_open_goal_gap = goal_scope_matches and bool(open_dimensions)
    facts = FrozenMap.from_frozen_items(
        {
            witness.plan_ref: FrozenMap(
                {
                    "goal_ref": witness.goal_ref,
                    "goal_scope_matches": goal_scope_matches,
                    "has_open_goal_gap": has_open_goal_gap,
                    "covers_all_open_goal_gaps": (
                        has_open_goal_gap
                        and set(open_dimensions)
                        <= set(witness.addressed_gap_dimensions)
                    ),
                    "open_gap_dimensions": tuple(
                        item.value for item in open_dimensions
                    ),
                    "addressed_gap_dimensions": tuple(
                        item.value for item in witness.addressed_gap_dimensions
                    ),
                    "subgoal_refs": witness.subgoal_refs,
                    "primitive_action_refs": witness.primitive_action_refs,
                    "remaining_action_bound": witness.remaining_action_bound,
                }
            )
            for witness in result.witnesses
        }
    )
    return DeclaredGoalPlanAlternatives(
        goal_ref=result.goal_ref,
        witnesses=result.witnesses,
        alternative_refs=tuple(item.plan_ref for item in result.witnesses),
        alternative_facts=facts,
        has_candidates=bool(result.witnesses),
    )
