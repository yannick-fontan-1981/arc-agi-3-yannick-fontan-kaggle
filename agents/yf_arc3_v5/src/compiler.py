"""Deterministic SRC v0.1 compiler and typed JSON-compatible IR."""

from __future__ import annotations

import hashlib

from pydantic import Field, ValidationError, model_validator

from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    FrozenModel,
    Ref,
    canonical_data,
    require_unique,
    stable_digest,
)
from agents.yf_arc3_v5.src.ast import (
    BindingDecl,
    Expression,
    ExpressionKind,
    LoopContract,
    ModuleAst,
    Statement,
    StatementKind,
    WorkflowDecl,
)
from agents.yf_arc3_v5.src.errors import SrcValidationError
from agents.yf_arc3_v5.src.parser import parse
from agents.yf_arc3_v5.src.symbols import (
    CompilerEnvironment,
    ParameterSpec,
    SymbolDefinition,
    SymbolKind,
    TypeSpec,
    core_environment,
)
from agents.yf_arc3_v5.src.tokens import SourceSpan
from agents.yf_arc3_v5.src.validation import ValidationResult, validate_module


class IrCallArgument(FrozenModel):
    name: Ref | None = None
    value: "IrExpression"


class IrExpression(FrozenModel):
    kind: ExpressionKind
    value: str | int | bool | None = None
    items: tuple["IrExpression", ...] = ()
    arguments: tuple[IrCallArgument, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> "IrExpression":
        if self.kind in {
            ExpressionKind.REFERENCE,
            ExpressionKind.STRING,
            ExpressionKind.INTEGER,
            ExpressionKind.BOOLEAN,
        }:
            if self.value is None or self.items or self.arguments:
                raise ValueError(f"invalid IR scalar expression: {self.kind.value}")
        elif self.kind in {ExpressionKind.NULL, ExpressionKind.EMPTY_SET}:
            if self.value is not None or self.items or self.arguments:
                raise ValueError(f"invalid IR empty expression: {self.kind.value}")
        elif self.kind is ExpressionKind.LIST:
            if self.value is not None or self.arguments:
                raise ValueError("invalid IR list expression")
        elif self.kind is ExpressionKind.CALL:
            if not isinstance(self.value, str) or self.items:
                raise ValueError("invalid IR call expression")
            named = tuple(argument.name for argument in self.arguments if argument.name)
            require_unique(named, "IR named call arguments")
            saw_named = False
            for argument in self.arguments:
                if argument.name:
                    saw_named = True
                elif saw_named:
                    raise ValueError(
                        "IR positional call arguments must precede named arguments"
                    )
        return self


class IrImport(FrozenModel):
    module_id: Ref
    alias: Ref | None = None


class IrBinding(FrozenModel):
    name: Ref
    type_spec: TypeSpec
    default: IrExpression | None = None


class IrClause(FrozenModel):
    name: Ref
    value: IrExpression


class IrMatchArm(FrozenModel):
    labels: tuple[IrExpression, ...] = Field(min_length=1)
    transition: "IrStatement"


class IrLoopContract(FrozenModel):
    progress_witness: IrExpression
    stop_condition: IrExpression
    maximum_symbolic_depth: int = Field(strict=True, ge=1)
    limit_result: IrExpression


class IrStatement(FrozenModel):
    id: Ref
    kind: StatementKind
    target: Ref | None = None
    expression: IrExpression | None = None
    type_annotation: TypeSpec | None = None
    result_binding: Ref | None = None
    clauses: tuple[IrClause, ...] = ()
    body: tuple["IrStatement", ...] = ()
    else_body: tuple["IrStatement", ...] = ()
    match_arms: tuple[IrMatchArm, ...] = ()
    iterator: Ref | None = None
    loop_contract: IrLoopContract | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "IrStatement":
        require_unique(
            tuple(clause.name for clause in self.clauses), "IR statement clauses"
        )
        if self.kind is StatementKind.LET:
            if not self.target or self.expression is None:
                raise ValueError("IR LET requires target and expression")
        elif self.kind is StatementKind.CALL:
            if (
                self.expression is None
                or self.expression.kind is not ExpressionKind.CALL
            ):
                raise ValueError("IR CALL requires call expression")
        elif self.kind in {
            StatementKind.OBSERVE,
            StatementKind.ACT,
            StatementKind.DISTINGUISH,
            StatementKind.RELATE,
            StatementKind.PROPOSE,
            StatementKind.DERIVE,
            StatementKind.COMPARE,
            StatementKind.UPDATE,
            StatementKind.SELECT,
            StatementKind.EXPECT,
            StatementKind.TEST,
            StatementKind.TRACK,
            StatementKind.MEASURE,
            StatementKind.PROPAGATE,
            StatementKind.OPERATIONAL_WISDOM,
        }:
            if not self.target:
                raise ValueError(f"IR {self.kind.value} requires target")
        elif self.kind is StatementKind.IF:
            if self.expression is None or not self.body:
                raise ValueError("IR IF requires condition and body")
        elif self.kind is StatementKind.MATCH:
            if self.expression is None or not self.match_arms:
                raise ValueError("IR MATCH requires subject and arms")
        elif self.kind is StatementKind.FOR_EACH:
            if not self.iterator or self.expression is None or not self.body:
                raise ValueError("IR FOR EACH requires iterator, collection, and body")
        elif self.kind is StatementKind.WHILE:
            if self.expression is None or self.loop_contract is None or not self.body:
                raise ValueError("IR WHILE requires condition, contract, and body")
        elif self.kind is StatementKind.TRANSACTION:
            if not self.target or not self.body:
                raise ValueError("IR TRANSACTION requires identity and body")
        elif self.kind in {
            StatementKind.REQUIRE,
            StatementKind.AWAIT_FRAME,
            StatementKind.NEXT,
            StatementKind.STOP,
            StatementKind.BLOCK,
            StatementKind.DEFER,
            StatementKind.FAIL,
        }:
            if self.expression is None:
                raise ValueError(f"IR {self.kind.value} requires expression")
        return self


class IrStep(FrozenModel):
    id: Ref
    statements: tuple[IrStatement, ...] = Field(min_length=1)


class IrWorkflow(FrozenModel):
    id: Ref
    entry_step_id: Ref
    inputs: tuple[IrBinding, ...]
    outputs: tuple[IrBinding, ...] = Field(min_length=1)
    state: tuple[IrBinding, ...]
    steps: tuple[IrStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> "IrWorkflow":
        binding_names = tuple(
            binding.name for binding in (*self.inputs, *self.outputs, *self.state)
        )
        require_unique(binding_names, "IR workflow bindings")
        step_ids = tuple(step.id for step in self.steps)
        require_unique(step_ids, "IR workflow steps")
        if self.entry_step_id != self.steps[0].id:
            raise ValueError("IR workflow entry must be its first declared step")
        statement_ids: list[str] = []

        def collect(statement: IrStatement) -> None:
            statement_ids.append(statement.id)
            for arm in statement.match_arms:
                collect(arm.transition)
            for nested in (*statement.body, *statement.else_body):
                collect(nested)

        for step in self.steps:
            for statement in step.statements:
                collect(statement)
        require_unique(tuple(statement_ids), "IR statement identities")
        return self


class SourceMapEntry(FrozenModel):
    node_id: Ref
    construct_kind: Ref
    span: SourceSpan


class AuthorityAttestation(FrozenModel):
    authority_id: Ref
    kind: SymbolKind
    effect_class: EffectClass
    source_unit: Ref
    source_hash: Ref


class CompiledModule(FrozenModel):
    module_id: Ref
    source_name: Ref
    source_hash: Ref
    semantic_hash: Ref
    imports: tuple[IrImport, ...]
    workflows: tuple[IrWorkflow, ...] = Field(min_length=1)
    authority_attestations: tuple[AuthorityAttestation, ...] = ()
    source_map: tuple[SourceMapEntry, ...] = ()
    schema_version: Ref = "yf_arc3_v5.src_ir.v0_1"

    @model_validator(mode="after")
    def validate_semantic_hash(self) -> "CompiledModule":
        require_unique(
            tuple(item.module_id for item in self.imports), "compiled imports"
        )
        require_unique(
            tuple(item.alias or item.module_id for item in self.imports),
            "compiled import aliases",
        )
        require_unique(tuple(item.id for item in self.workflows), "compiled workflows")
        require_unique(
            tuple(item.authority_id for item in self.authority_attestations),
            "authority attestations",
        )
        require_unique(
            tuple(item.node_id for item in self.source_map),
            "source-map node identities",
        )
        expected = stable_digest(self.semantic_payload())
        if self.semantic_hash != expected:
            raise ValueError("compiled SRC semantic hash mismatch")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "imports": self.imports,
            "workflows": self.workflows,
            "authority_attestations": self.authority_attestations,
            "schema_version": self.schema_version,
        }

    def semantic_ir(self) -> dict[str, object]:
        value = canonical_data(self.semantic_payload())
        assert isinstance(value, dict)
        return value


def _lower_expression(expression: Expression) -> IrExpression:
    return IrExpression(
        kind=expression.kind,
        value=expression.value,
        items=tuple(_lower_expression(item) for item in expression.items),
        arguments=tuple(
            IrCallArgument(name=argument.name, value=_lower_expression(argument.value))
            for argument in expression.arguments
        ),
    )


def _lower_binding(binding: BindingDecl) -> IrBinding:
    return IrBinding(
        name=binding.name,
        type_spec=TypeSpec.from_ast(binding.type_expression),
        default=None if binding.default is None else _lower_expression(binding.default),
    )


class _Lowerer:
    def __init__(self, ast: ModuleAst) -> None:
        self.ast = ast
        self.source_map: list[SourceMapEntry] = []

    def add_source(self, node_id: str, construct: str, span: SourceSpan) -> None:
        self.source_map.append(
            SourceMapEntry(node_id=node_id, construct_kind=construct, span=span)
        )

    def lower(self) -> tuple[tuple[IrWorkflow, ...], tuple[SourceMapEntry, ...]]:
        self.add_source(self.ast.module_id, "MODULE", self.ast.span)
        for index, item in enumerate(self.ast.imports):
            self.add_source(
                f"{self.ast.module_id}.import.{index:04d}",
                "IMPORT",
                item.span,
            )
        workflows: list[IrWorkflow] = []
        for workflow in self.ast.workflows:
            workflow_id = f"{self.ast.module_id}.{workflow.name}"
            self.add_source(workflow_id, "WORKFLOW", workflow.span)
            for group_name, bindings in (
                ("input", workflow.inputs),
                ("output", workflow.outputs),
                ("state", workflow.state),
            ):
                for binding in bindings:
                    self.add_source(
                        f"{workflow_id}.{group_name}.{binding.name}",
                        group_name.upper(),
                        binding.span,
                    )
            steps: list[IrStep] = []
            for step in workflow.steps:
                step_id = f"{workflow_id}.{step.name}"
                self.add_source(step_id, "STEP", step.span)
                statements = tuple(
                    self.lower_statement(
                        statement,
                        f"{step_id}.s{index:04d}",
                    )
                    for index, statement in enumerate(step.statements)
                )
                steps.append(IrStep(id=step_id, statements=statements))
            workflows.append(
                IrWorkflow(
                    id=workflow_id,
                    entry_step_id=steps[0].id,
                    inputs=tuple(
                        _lower_binding(binding) for binding in workflow.inputs
                    ),
                    outputs=tuple(
                        _lower_binding(binding) for binding in workflow.outputs
                    ),
                    state=tuple(_lower_binding(binding) for binding in workflow.state),
                    steps=tuple(steps),
                )
            )
        return tuple(workflows), tuple(self.source_map)

    def lower_statement(self, statement: Statement, node_id: str) -> IrStatement:
        self.add_source(node_id, statement.kind.value, statement.span)
        clauses: list[IrClause] = []
        for clause in sorted(statement.clauses, key=lambda item: item.name):
            self.add_source(
                f"{node_id}.clause.{clause.name.lower()}",
                clause.name,
                clause.span,
            )
            clauses.append(
                IrClause(name=clause.name, value=_lower_expression(clause.value))
            )
        body = tuple(
            self.lower_statement(item, f"{node_id}.body.s{index:04d}")
            for index, item in enumerate(statement.body)
        )
        else_body = tuple(
            self.lower_statement(item, f"{node_id}.else.s{index:04d}")
            for index, item in enumerate(statement.else_body)
        )
        match_arms: list[IrMatchArm] = []
        for index, arm in enumerate(statement.match_arms):
            arm_id = f"{node_id}.arm.{index:04d}"
            self.add_source(arm_id, "MATCH_ARM", arm.span)
            match_arms.append(
                IrMatchArm(
                    labels=tuple(_lower_expression(label) for label in arm.labels),
                    transition=self.lower_statement(
                        arm.transition, f"{arm_id}.transition"
                    ),
                )
            )
        loop_contract = None
        if statement.loop_contract:
            loop_contract = self.lower_loop_contract(statement.loop_contract, node_id)
        return IrStatement(
            id=node_id,
            kind=statement.kind,
            target=statement.target,
            expression=(
                None
                if statement.expression is None
                else _lower_expression(statement.expression)
            ),
            type_annotation=(
                None
                if statement.type_annotation is None
                else TypeSpec.from_ast(statement.type_annotation)
            ),
            result_binding=statement.result_binding,
            clauses=tuple(clauses),
            body=body,
            else_body=else_body,
            match_arms=tuple(match_arms),
            iterator=statement.iterator,
            loop_contract=loop_contract,
        )

    def lower_loop_contract(
        self,
        contract: LoopContract,
        node_id: str,
    ) -> IrLoopContract:
        contract_id = f"{node_id}.loop_contract"
        self.add_source(contract_id, "LOOP_CONTRACT", contract.span)
        return IrLoopContract(
            progress_witness=_lower_expression(contract.progress_witness),
            stop_condition=_lower_expression(contract.stop_condition),
            maximum_symbolic_depth=contract.maximum_symbolic_depth,
            limit_result=_lower_expression(contract.limit_result),
        )


def _workflow_symbol(
    module_id: str,
    workflow: IrWorkflow,
    *,
    symbol_id: str,
    source_hash: str,
) -> SymbolDefinition:
    return SymbolDefinition(
        id=symbol_id,
        kind=SymbolKind.WORKFLOW,
        parameters=tuple(
            ParameterSpec(name=binding.name, type_spec=binding.type_spec)
            for binding in workflow.inputs
        ),
        output_type=(
            workflow.outputs[0].type_spec
            if len(workflow.outputs) == 1
            else TypeSpec(name="WorkflowResult")
        ),
        source_unit=f"src.{module_id}",
        source_hash=source_hash,
    )


def _local_workflow_source_hash(module_id: str, workflow: WorkflowDecl) -> str:
    return stable_digest(
        {
            "module_id": module_id,
            "workflow": workflow.name,
            "inputs": tuple(
                (binding.name, binding.type_expression.semantic_data())
                for binding in workflow.inputs
            ),
            "outputs": tuple(
                (binding.name, binding.type_expression.semantic_data())
                for binding in workflow.outputs
            ),
        }
    )


def _effective_environment(
    ast: ModuleAst,
    environment: CompilerEnvironment,
    imported_modules: tuple[CompiledModule, ...],
) -> CompilerEnvironment:
    symbols = list(environment.symbols)
    for workflow in ast.workflows:
        symbols.append(
            SymbolDefinition(
                id=f"{ast.module_id}.{workflow.name}",
                kind=SymbolKind.WORKFLOW,
                parameters=tuple(
                    ParameterSpec(
                        name=binding.name,
                        type_spec=TypeSpec.from_ast(binding.type_expression),
                    )
                    for binding in workflow.inputs
                ),
                output_type=(
                    TypeSpec.from_ast(workflow.outputs[0].type_expression)
                    if len(workflow.outputs) == 1
                    else TypeSpec(name="WorkflowResult")
                ),
                source_unit=f"src.{ast.module_id}",
                source_hash=_local_workflow_source_hash(ast.module_id, workflow),
            )
        )
    import_ast_index = {item.module: item for item in ast.imports}
    for module in imported_modules:
        declaration = import_ast_index.get(module.module_id)
        if declaration is None:
            continue
        prefix = declaration.alias or module.module_id
        for imported_workflow in module.workflows:
            short_name = imported_workflow.id.rsplit(".", 1)[-1]
            symbols.append(
                _workflow_symbol(
                    module.module_id,
                    imported_workflow,
                    symbol_id=f"{prefix}.{short_name}",
                    source_hash=module.semantic_hash,
                )
            )
    return CompilerEnvironment(types=environment.types, symbols=tuple(symbols))


def _authority_attestations(
    validation: ValidationResult,
    environment: CompilerEnvironment,
) -> tuple[AuthorityAttestation, ...]:
    return tuple(
        AuthorityAttestation(
            authority_id=definition.id,
            kind=definition.kind,
            effect_class=definition.effect_class,
            source_unit=definition.source_unit,
            source_hash=definition.source_hash,
        )
        for definition in (
            environment.symbol_index[authority_id]
            for authority_id in validation.authority_ids
        )
    )


def compile_source(
    source: str,
    *,
    source_name: str = "<memory>",
    resolved_imports: frozenset[str] = frozenset(),
    imported_modules: tuple[CompiledModule, ...] = (),
    environment: CompilerEnvironment | None = None,
) -> CompiledModule:
    ast = parse(source, source_name=source_name)
    workflow_names = tuple(workflow.name for workflow in ast.workflows)
    if len(workflow_names) != len(set(workflow_names)):
        raise SrcValidationError("duplicate workflow identity", ast.span)
    declared_imports = tuple(item.module for item in ast.imports)
    if len(declared_imports) != len(set(declared_imports)):
        raise SrcValidationError("duplicate imported module", ast.span)
    import_aliases = tuple(item.alias or item.module for item in ast.imports)
    if len(import_aliases) != len(set(import_aliases)):
        raise SrcValidationError("duplicate import alias", ast.span)
    supplied_imports = tuple(module.module_id for module in imported_modules)
    if len(supplied_imports) != len(set(supplied_imports)):
        raise SrcValidationError("duplicate compiled import supplied", ast.span)
    imported_ids = {module.module_id for module in imported_modules}
    available_imports = set(resolved_imports) | imported_ids
    unresolved = tuple(
        item.module for item in ast.imports if item.module not in available_imports
    )
    if unresolved:
        raise SrcValidationError(f"unresolved imports: {list(unresolved)}", ast.span)
    undeclared_modules = sorted(imported_ids - {item.module for item in ast.imports})
    if undeclared_modules:
        raise SrcValidationError(
            f"compiled imports were supplied but not declared: {undeclared_modules}",
            ast.span,
        )

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    base_environment = core_environment() if environment is None else environment
    try:
        effective_environment = _effective_environment(
            ast,
            base_environment,
            imported_modules,
        )
    except ValidationError as error:
        raise SrcValidationError(
            f"invalid compiler symbol environment: {error.errors(include_url=False)[0]['msg']}",
            ast.span,
        ) from error
    validation = validate_module(ast, effective_environment)
    lowerer = _Lowerer(ast)
    workflows, source_map = lowerer.lower()
    imports = tuple(
        IrImport(module_id=item.module, alias=item.alias)
        for item in sorted(
            ast.imports, key=lambda value: (value.module, value.alias or "")
        )
    )
    attestations = _authority_attestations(validation, effective_environment)
    semantic_payload = {
        "module_id": ast.module_id,
        "imports": imports,
        "workflows": workflows,
        "authority_attestations": attestations,
        "schema_version": "yf_arc3_v5.src_ir.v0_1",
    }
    return CompiledModule(
        module_id=ast.module_id,
        source_name=source_name,
        source_hash=source_hash,
        semantic_hash=stable_digest(semantic_payload),
        imports=imports,
        workflows=workflows,
        authority_attestations=attestations,
        source_map=source_map,
    )


IrCallArgument.model_rebuild()
IrExpression.model_rebuild()
IrMatchArm.model_rebuild()
IrStatement.model_rebuild()
