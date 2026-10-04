"""Source-attested mechanical executor for the noninitial SRC scene branch.

Compilation fails closed if the declared normalization/decomposition order or
the branch topology changes.  No morphology meaning or action is selected here.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.capabilities.contracts import (
    BlockGridSceneDescription,
    FrameNormalizationInput,
    FrameNormalizationResult,
    RevisionedCapabilityRequest,
    VisualSceneDescription,
    VisualSceneInput,
)
from agents.yf_arc3_v5.capabilities.registry import CapabilityRegistry
from agents.yf_arc3_v5.scheduler.evaluator import RuntimeModelMap
from agents.yf_arc3_v5.src.ast import ExpressionKind, StatementKind
from agents.yf_arc3_v5.src.compiler import CompiledModule, IrExpression, IrStep


class PartialSceneRuntimeMap(RuntimeModelMap):
    """Preserve a partial scene without presenting unmeasured families as absent."""

    __slots__ = ()
    _unmeasured_fields = frozenset({
        "zones",
        "sprites",
        "hole_sprites",
        "hole_sprite_measurement_count",
        "hole_sprite_enumeration_truncated",
    })

    @classmethod
    def from_projection(cls, value: RuntimeModelMap) -> "PartialSceneRuntimeMap":
        if not isinstance(value.source_model, BlockGridSceneDescription):
            raise TypeError("partial scene requires a typed block-grid measurement")
        guarded = object.__new__(cls)
        guarded._data = value._data
        guarded._items = value._items
        guarded._hash = value._hash
        guarded._digest = value._digest
        guarded.source_model = value.source_model
        return guarded

    def __getitem__(self, key: str) -> object:
        if key in self._unmeasured_fields:
            raise RuntimeError(f"scene family {key} was not measured")
        return super().__getitem__(key)


def _source_value(expression: IrExpression | None) -> object:
    if expression is None or expression.kind not in {
        ExpressionKind.REFERENCE,
        ExpressionKind.STRING,
        ExpressionKind.BOOLEAN,
    }:
        raise ValueError("compiled assimilation has an unsupported source expression")
    return expression.value


def _next_ref(step: IrStep, statement_index: int, workflow_ref: str) -> str:
    transition = step.statements[statement_index]
    if transition.kind is not StatementKind.NEXT:
        raise ValueError("compiled assimilation requires a declared NEXT")
    return f"{workflow_ref}.{_source_value(transition.expression)}"


def _bool_arms(step: IrStep, expression_ref: str, workflow_ref: str) -> dict[bool, object]:
    if len(step.statements) != 1:
        raise ValueError("compiled assimilation MATCH has extra work")
    match = step.statements[0]
    if match.kind is not StatementKind.MATCH or _source_value(match.expression) != expression_ref:
        raise ValueError("compiled assimilation MATCH changed its declared input")
    if len(match.match_arms) != 2:
        raise ValueError("compiled assimilation expects exactly two branches")
    arms: dict[bool, object] = {}
    for arm in match.match_arms:
        if len(arm.labels) != 1 or arm.labels[0].kind is not ExpressionKind.BOOLEAN:
            raise ValueError("compiled assimilation requires Boolean branch labels")
        label = bool(_source_value(arm.labels[0]))
        if label in arms:
            raise ValueError("compiled assimilation repeats a branch label")
        transition = arm.transition
        if transition.kind is StatementKind.NEXT:
            arms[label] = f"{workflow_ref}.{_source_value(transition.expression)}"
        elif transition.kind is StatementKind.STOP:
            arms[label] = ("STOP", _source_value(transition.expression))
        else:
            raise ValueError("compiled assimilation has an unsupported transition")
    if set(arms) != {False, True}:
        raise ValueError("compiled assimilation omits a Boolean branch")
    return arms


@dataclass(frozen=True, slots=True)
class CompiledSceneAssimilation:
    workflow_ref: str
    src_source_hash: str
    entry_step_id: str
    branch_step_ids: tuple[str, str]
    join_step_id: str

    def run(
        self,
        *,
        frames: tuple[tuple[tuple[int, ...], ...], ...],
        action_conditioned_grid_evidence: bool,
        input_revision: int,
        capabilities: CapabilityRegistry,
        retained_grid_geometry: tuple[int, ...] | None = None,
        retained_grid_separator_consistency_ppm: int = 0,
    ) -> VisualSceneDescription:
        normalized = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.frame_normalization",
                input_revision=input_revision,
                payload=FrameNormalizationInput.from_payload(frames),
            )
        ).output
        if not isinstance(normalized, FrameNormalizationResult):
            raise RuntimeError("compiled assimilation normalization type changed")
        scene = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.visual_scene_decomposition",
                input_revision=input_revision,
                payload=VisualSceneInput(
                    frame=normalized.frame,
                    action_conditioned_grid_evidence=action_conditioned_grid_evidence,
                    retained_grid_geometry=retained_grid_geometry,
                    retained_grid_separator_consistency_ppm=retained_grid_separator_consistency_ppm,
                ),
            )
        ).output
        if not isinstance(scene, VisualSceneDescription):
            raise RuntimeError("compiled assimilation scene type changed")
        return scene


def compile_scene_assimilation(
    module: CompiledModule, workflow_ref: str
) -> CompiledSceneAssimilation:
    workflow = next((item for item in module.workflows if item.id == workflow_ref), None)
    if workflow is None:
        raise ValueError("compiled assimilation workflow is absent")
    by_id = {step.id: step for step in workflow.steps}
    if set(binding.name for binding in workflow.inputs) != {
        "raw_frame", "raw_input_ref", "available_action_refs",
        "is_initial_observation", "action_conditioned_grid_evidence",
        "before_revision", "retained_grid_geometry",
        "retained_grid_separator_consistency_ppm",
    }:
        raise ValueError("compiled assimilation inputs changed")
    entry = by_id[workflow.entry_step_id]
    arms = _bool_arms(entry, "action_conditioned_grid_evidence", workflow_ref)
    branch_ids = (arms[False], arms[True])
    if not all(isinstance(step_id, str) and step_id in by_id for step_id in branch_ids):
        raise ValueError("compiled assimilation branch leaves the workflow")
    join_ids: set[str] = set()
    for flag, step_id in ((False, branch_ids[0]), (True, branch_ids[1])):
        step = by_id[step_id]
        if len(step.statements) != 3:
            raise ValueError("compiled assimilation branch is not two calls and NEXT")
        normalization, decomposition = step.statements[:2]
        if normalization.kind is not StatementKind.CALL or decomposition.kind is not StatementKind.CALL:
            raise ValueError("compiled assimilation call order changed")
        if normalization.expression is None or normalization.expression.value != "capability.frame_normalization":
            raise ValueError("compiled assimilation normalization source changed")
        if decomposition.expression is None or decomposition.expression.value != "capability.visual_scene_decomposition":
            raise ValueError("compiled assimilation scene source changed")
        normal_args = {arg.name: _source_value(arg.value) for arg in normalization.expression.arguments}
        scene_args = {arg.name: _source_value(arg.value) for arg in decomposition.expression.arguments}
        if normal_args != {"payload": "raw_frame"} or scene_args != {
            "frame": f"{normalization.result_binding}.frame",
            "action_conditioned_grid_evidence": flag,
            "retained_grid_geometry": "retained_grid_geometry",
            "retained_grid_separator_consistency_ppm": "retained_grid_separator_consistency_ppm",
        } or decomposition.result_binding != "result":
            raise ValueError("compiled assimilation call binding changed")
        join_ids.add(_next_ref(step, 2, workflow_ref))
    if len(join_ids) != 1:
        raise ValueError("compiled assimilation branches do not rejoin")
    join_id = next(iter(join_ids))
    if join_id not in by_id:
        raise ValueError("compiled assimilation join is absent")
    final_arms = _bool_arms(by_id[join_id], "is_initial_observation", workflow_ref)
    if final_arms[False] != ("STOP", "result"):
        raise ValueError("compiled assimilation noninitial branch no longer returns scene")
    if not isinstance(final_arms[True], str) or final_arms[True] not in by_id:
        raise ValueError("compiled assimilation initial branch is malformed")
    return CompiledSceneAssimilation(
        workflow_ref=workflow_ref,
        src_source_hash=module.source_hash,
        entry_step_id=entry.id,
        branch_step_ids=branch_ids,
        join_step_id=join_id,
    )


@dataclass(frozen=True, slots=True)
class CompiledBlockGridAssimilation:
    workflow_ref: str
    src_source_hash: str
    path: tuple[str, str, str]

    def run(
        self,
        *,
        frames: tuple[tuple[tuple[int, ...], ...], ...],
        action_conditioned_grid_evidence: bool,
        input_revision: int,
        capabilities: CapabilityRegistry,
        retained_grid_geometry: tuple[int, ...] | None = None,
        retained_grid_separator_consistency_ppm: int = 0,
    ) -> BlockGridSceneDescription:
        normalized = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.frame_normalization",
                input_revision=input_revision,
                payload=FrameNormalizationInput.from_payload(frames),
            )
        ).output
        if not isinstance(normalized, FrameNormalizationResult):
            raise RuntimeError("compiled partial assimilation normalization type changed")
        scene = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.block_grid_scene_decomposition",
                input_revision=input_revision,
                payload=VisualSceneInput(
                    frame=normalized.frame,
                    action_conditioned_grid_evidence=action_conditioned_grid_evidence,
                    retained_grid_geometry=retained_grid_geometry,
                    retained_grid_separator_consistency_ppm=(
                        retained_grid_separator_consistency_ppm
                    ),
                ),
            )
        ).output
        if not isinstance(scene, BlockGridSceneDescription):
            raise RuntimeError("compiled partial assimilation scene type changed")
        return scene


def compile_block_grid_assimilation(
    module: CompiledModule, workflow_ref: str
) -> CompiledBlockGridAssimilation:
    workflow = next((item for item in module.workflows if item.id == workflow_ref), None)
    if workflow is None:
        raise ValueError("compiled partial assimilation workflow is absent")
    if {binding.name for binding in workflow.inputs} != {
        "raw_frame", "raw_input_ref", "action_conditioned_grid_evidence",
        "before_revision", "retained_grid_geometry",
        "retained_grid_separator_consistency_ppm",
    }:
        raise ValueError("compiled partial assimilation inputs changed")
    by_id = {step.id: step for step in workflow.steps}
    if len(by_id) != 3:
        raise ValueError("compiled partial assimilation must have three steps")
    normalize = by_id[workflow.entry_step_id]
    if len(normalize.statements) != 2:
        raise ValueError("compiled partial normalization has extra work")
    normalize_call, normalize_next = normalize.statements
    if (
        normalize_call.kind is not StatementKind.CALL
        or normalize_call.expression is None
        or normalize_call.expression.value != "capability.frame_normalization"
        or {arg.name: _source_value(arg.value) for arg in normalize_call.expression.arguments}
        != {"payload": "raw_frame"}
        or normalize_call.result_binding != "normalized"
        or normalize_next.kind is not StatementKind.NEXT
    ):
        raise ValueError("compiled partial normalization source changed")
    measure_id = _next_ref(normalize, 1, workflow_ref)
    measure = by_id[measure_id]
    if len(measure.statements) != 2:
        raise ValueError("compiled partial measurement has extra work")
    measure_call, measure_next = measure.statements
    if (
        measure_call.kind is not StatementKind.CALL
        or measure_call.expression is None
        or measure_call.expression.value != "capability.block_grid_scene_decomposition"
        or {arg.name: _source_value(arg.value) for arg in measure_call.expression.arguments}
        != {
            "frame": "normalized.frame",
            "action_conditioned_grid_evidence": "action_conditioned_grid_evidence",
            "retained_grid_geometry": "retained_grid_geometry",
            "retained_grid_separator_consistency_ppm": (
                "retained_grid_separator_consistency_ppm"
            ),
        }
        or measure_call.result_binding != "result"
        or measure_next.kind is not StatementKind.NEXT
    ):
        raise ValueError("compiled partial measurement source changed")
    finish_id = _next_ref(measure, 1, workflow_ref)
    finish = by_id[finish_id]
    if (
        len(finish.statements) != 1
        or finish.statements[0].kind is not StatementKind.STOP
        or _source_value(finish.statements[0].expression) != "result"
    ):
        raise ValueError("compiled partial finish source changed")
    return CompiledBlockGridAssimilation(
        workflow_ref=workflow_ref,
        src_source_hash=module.source_hash,
        path=(normalize.id, measure.id, finish.id),
    )
