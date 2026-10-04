"""Source-attested pure scheduler adapter for bounded dynamic-workflow synthesis."""

from __future__ import annotations

from agents.yf_arc3_v5.dynamic_workflow.registry import ColdComponentRegistry
from agents.yf_arc3_v5.dynamic_workflow.synthesis import (
    DynamicWorkflowSynthesisRequest,
    DynamicWorkflowSynthesisResult,
    reduce_dynamic_workflow_synthesis_facts,
    synthesize_declared_workflow_candidates,
)
from agents.yf_arc3_v5.scheduler.contracts import FunctionCallContext
from agents.yf_arc3_v5.scheduler.evaluator import restore_runtime_model_refs
from agents.yf_arc3_v5.scheduler.registry import SchedulerRuntimeRegistry
from agents.yf_arc3_v5.src.production import production_control_symbol_definitions

_SYNTHESIS_CALL_REF = "control.synthesize_declared_dynamic_workflow_candidates"
_REDUCE_SYNTHESIS_FACTS_CALL_REF = "control.reduce_dynamic_workflow_synthesis_facts"


def register_dynamic_workflow_calls(
    runtime: SchedulerRuntimeRegistry,
    components: ColdComponentRegistry,
    *,
    control_source_hashes: dict[str, str] | None = None,
) -> None:
    definitions = {
        definition.id: (definition if definition.source_unit not in (control_source_hashes or {}) else
                        definition.model_copy(update={"source_hash": control_source_hashes[definition.source_unit]}))
        for definition in production_control_symbol_definitions()
    }
    def synthesize(context: FunctionCallContext) -> object:
        arguments = restore_runtime_model_refs(context.arguments)
        request = DynamicWorkflowSynthesisRequest.model_validate(
            arguments["synthesis_request"]
        )
        return synthesize_declared_workflow_candidates(request, components)

    def reduce_facts(context: FunctionCallContext) -> object:
        arguments = restore_runtime_model_refs(context.arguments)
        result = DynamicWorkflowSynthesisResult.model_validate(
            arguments["synthesis_result"]
        )
        return reduce_dynamic_workflow_synthesis_facts(result)

    runtime.register_call(definitions[_SYNTHESIS_CALL_REF], synthesize)
    runtime.register_call(
        definitions[_REDUCE_SYNTHESIS_FACTS_CALL_REF],
        reduce_facts,
    )
