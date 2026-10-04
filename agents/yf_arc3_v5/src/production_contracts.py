"""Declarative required-lineage contracts for production SRC workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenModel, Ref, require_unique
from agents.yf_arc3_v5.src.ast import StatementKind
from agents.yf_arc3_v5.src.compiler import CompiledModule, IrStatement, IrWorkflow


class ProductionWorkflowContractError(ValueError):
    """Compiled production source is missing required visible lineage."""


class ProductionWorkflowContract(FrozenModel):
    workflow_id: Ref
    required_sequence: tuple[Ref, ...] = Field(min_length=1)
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    governing_principle_refs: tuple[Ref, ...] = ()
    instruction_quantum: int = Field(default=1000, ge=1, le=8192)
    maximum_instructions: int = Field(default=1000, ge=1, le=8192)

    @model_validator(mode="after")
    def validate_contract(self) -> "ProductionWorkflowContract":
        if self.instruction_quantum > self.maximum_instructions:
            raise ValueError("instruction quantum exceeds the total execution bound")
        require_unique(self.reason_refs, "production contract reasons")
        require_unique(
            self.governing_principle_refs,
            "production contract governing principles",
        )
        return self


class ProductionWorkflowContractSet(FrozenModel):
    contracts: tuple[ProductionWorkflowContract, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.production_workflow_contracts.v1"

    @model_validator(mode="after")
    def validate_contracts(self) -> "ProductionWorkflowContractSet":
        require_unique(
            tuple(item.workflow_id for item in self.contracts),
            "production workflow contracts",
        )
        return self


def production_contract_path() -> Path:
    return Path(__file__).with_name("production_workflow_contracts.json")


def load_production_workflow_contracts() -> tuple[ProductionWorkflowContractSet, str]:
    path = production_contract_path()
    source = path.read_bytes()
    contracts = ProductionWorkflowContractSet.model_validate(
        json.loads(source.decode("utf-8"))
    )
    return contracts, hashlib.sha256(source).hexdigest()


def validate_production_workflow_contracts(
    modules: tuple[CompiledModule, ...],
    contracts: ProductionWorkflowContractSet,
) -> None:
    workflows = {
        workflow.id: workflow for module in modules for workflow in module.workflows
    }
    expected_ids = {item.workflow_id for item in contracts.contracts}
    actual_ids = set(workflows)
    if actual_ids != expected_ids:
        raise ProductionWorkflowContractError(
            "production workflow identity mismatch; "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    for contract in contracts.contracts:
        observed = _workflow_sequence(workflows[contract.workflow_id])
        if not _is_subsequence(contract.required_sequence, observed):
            raise ProductionWorkflowContractError(
                f"{contract.workflow_id} lacks required source lineage "
                f"{list(contract.required_sequence)}; observed={list(observed)}"
            )


def _workflow_sequence(workflow: IrWorkflow) -> tuple[str, ...]:
    return tuple(
        token
        for step in workflow.steps
        for statement in step.statements
        for token in _statement_tokens(statement)
    )


def _statement_tokens(statement: IrStatement) -> tuple[str, ...]:
    if statement.kind is StatementKind.CALL:
        assert statement.expression is not None
        own = (f"CALL:{statement.expression.value}",)
    else:
        own = (statement.kind.value,)
    nested = tuple(
        token
        for child in (*statement.body, *statement.else_body)
        for token in _statement_tokens(child)
    )
    arms = tuple(
        token
        for arm in statement.match_arms
        for token in _statement_tokens(arm.transition)
    )
    return (*own, *nested, *arms)


def _is_subsequence(required: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    index = 0
    for token in observed:
        if token == required[index]:
            index += 1
            if index == len(required):
                return True
    return False
