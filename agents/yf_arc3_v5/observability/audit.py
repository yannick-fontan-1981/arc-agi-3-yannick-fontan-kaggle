"""Fail-closed static and runtime architecture audits for V5."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

from agents.yf_arc3_v5.capabilities import CapabilityRegistry
from agents.yf_arc3_v5.logos.types import OperatorName
from agents.yf_arc3_v5.observability.models import (
    ArchitectureAuditFinding,
    ArchitectureAuditReport,
)
from agents.yf_arc3_v5.operators import OperatorAuthorityRegistry
from agents.yf_arc3_v5.runtime.environment import environment_symbol_definition
from agents.yf_arc3_v5.scheduler.registry import SchedulerRuntimeRegistry
from agents.yf_arc3_v5.src.compiler import CompiledModule
from agents.yf_arc3_v5.src.registry import RegistryEntry

_COMMIT_WRITERS = frozenset(
    {
        "operators/runtime.py",
        "runtime/boundary.py",
        "scheduler/control_calls.py",
        "scheduler/propagation.py",
        "scheduler/runtime.py",
    }
)
_STATE_CONTAINER_OWNER = "state/event_store.py"
_ACTION_TRANSPORT_OWNER = "runtime/boundary.py"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GAME_MARKERS = tuple(
    re.compile(rf"\b{left}{right}\b", re.IGNORECASE)
    for left, right in (("VC", "33"), ("FT", "09"), ("LS", "20"), ("LP", "85"))
)


def audit_v5_architecture(
    package_root: str | Path,
    *,
    compiled_modules: tuple[CompiledModule, ...] = (),
    registry_entries: tuple[RegistryEntry, ...] = (),
    runtime_registry: SchedulerRuntimeRegistry | None = None,
    authority_registry: OperatorAuthorityRegistry | None = None,
    capability_registry: CapabilityRegistry | None = None,
) -> ArchitectureAuditReport:
    """Audit source boundaries and exact executable authority registrations."""

    root = Path(package_root).resolve()
    python_files = tuple(sorted(root.rglob("*.py")))
    parsed: list[tuple[str, ast.Module]] = []
    syntax_errors: list[str] = []
    for path in python_files:
        relative = path.relative_to(root).as_posix()
        try:
            parsed.append(
                (
                    relative,
                    ast.parse(path.read_text(encoding="utf-8"), filename=relative),
                )
            )
        except SyntaxError as error:
            syntax_errors.append(f"{relative}:{error.lineno}:{error.msg}")

    findings = [
        _finding("python_syntax", syntax_errors),
        _finding("direct_cognitive_writes", _direct_write_violations(parsed)),
        _finding("action_boundary_bypass", _action_bypass_violations(parsed)),
        _finding("v4_or_oracle_runtime_import", _forbidden_imports(parsed)),
        _finding("game_specific_runtime_constant", _game_constants(parsed)),
        _finding(
            "source_hash_completeness",
            _hash_violations(compiled_modules, registry_entries),
        ),
        _finding(
            "runtime_authority_registration",
            _authority_violations(
                compiled_modules,
                runtime_registry=runtime_registry,
                authority_registry=authority_registry,
                capability_registry=capability_registry,
            ),
        ),
        _finding(
            "primitive_operator_registration",
            _operator_registration_violations(runtime_registry),
        ),
    ]
    return ArchitectureAuditReport(
        passed=all(item.status == "passed" for item in findings),
        findings=tuple(findings),
        scanned_files=len(python_files),
    )


def _finding(check: str, violations: Iterable[str]) -> ArchitectureAuditFinding:
    details = tuple(sorted(set(violations)))
    return ArchitectureAuditFinding(
        check=check,
        status="passed" if not details else "failed",
        details=details[:100],
    )


def _direct_write_violations(
    parsed: list[tuple[str, ast.Module]],
) -> tuple[str, ...]:
    violations: list[str] = []
    for relative, tree in parsed:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "commit" and relative not in _COMMIT_WRITERS:
                    violations.append(f"{relative}:{node.lineno}:commit")
                if (
                    node.func.attr in {"append", "extend"}
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "_events"
                    and relative != _STATE_CONTAINER_OWNER
                ):
                    violations.append(
                        f"{relative}:{node.lineno}:event_container_mutation"
                    )
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    tuple(node.targets)
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                )
                if relative != _STATE_CONTAINER_OWNER and any(
                    isinstance(target, ast.Attribute)
                    and target.attr in {"_snapshot", "_events"}
                    for target in targets
                ):
                    violations.append(f"{relative}:{node.lineno}:state_container_write")
    return tuple(violations)


def _action_bypass_violations(
    parsed: list[tuple[str, ast.Module]],
) -> tuple[str, ...]:
    violations: list[str] = []
    for relative, tree in parsed:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            if (
                node.func.attr == "execute"
                and isinstance(owner, ast.Attribute)
                and owner.attr == "_transport"
                and relative != _ACTION_TRANSPORT_OWNER
            ):
                violations.append(f"{relative}:{node.lineno}:transport_execute")
    return tuple(violations)


def _forbidden_imports(parsed: list[tuple[str, ast.Module]]) -> tuple[str, ...]:
    violations: list[str] = []
    forbidden = (
        ".".join(("agents", "yf_arc3_" + "v4")),
        ".".join(("docs", "oracles")),
    )
    for relative, tree in parsed:
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            for name in names:
                if any(
                    name == item or name.startswith(f"{item}.") for item in forbidden
                ):
                    violations.append(f"{relative}:{getattr(node, 'lineno', 0)}:{name}")
    return tuple(violations)


def _game_constants(parsed: list[tuple[str, ast.Module]]) -> tuple[str, ...]:
    violations: list[str] = []
    for relative, tree in parsed:
        if relative == "observability/audit.py":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if any(pattern.search(node.value) for pattern in _GAME_MARKERS):
                violations.append(f"{relative}:{node.lineno}:{node.value[:60]}")
    return tuple(violations)


def _hash_violations(
    modules: tuple[CompiledModule, ...],
    entries: tuple[RegistryEntry, ...],
) -> tuple[str, ...]:
    values: list[tuple[str, str]] = []
    for module in modules:
        values.extend(
            (
                (f"module:{module.module_id}:source", module.source_hash),
                (f"module:{module.module_id}:semantic", module.semantic_hash),
            )
        )
        values.extend(
            (f"authority:{item.authority_id}", item.source_hash)
            for item in module.authority_attestations
        )
    for entry in entries:
        values.extend(
            (
                (f"entry:{entry.module_id}:source", entry.source_hash),
                (f"entry:{entry.module_id}:semantic", entry.semantic_hash),
                (f"entry:{entry.module_id}:transitive", entry.transitive_source_hash),
            )
        )
    return tuple(label for label, value in values if not _HASH_RE.fullmatch(value))


def _authority_violations(
    modules: tuple[CompiledModule, ...],
    *,
    runtime_registry: SchedulerRuntimeRegistry | None,
    authority_registry: OperatorAuthorityRegistry | None,
    capability_registry: CapabilityRegistry | None,
) -> tuple[str, ...]:
    if not modules:
        return ()
    if (
        runtime_registry is None
        or authority_registry is None
        or capability_registry is None
    ):
        return ("runtime registries unavailable",)
    registered: dict[str, tuple[str, str, str]] = {}
    for call_definition in runtime_registry.call_definitions:
        registered[call_definition.id] = (
            call_definition.effect_class.value,
            call_definition.source_unit,
            call_definition.source_hash,
        )
    for authority_definition in authority_registry.definitions:
        registered[authority_definition.id] = (
            authority_definition.effect_class.value,
            authority_definition.source_unit,
            authority_definition.source_hash,
        )
    for entry in capability_registry.entries:
        capability_definition = entry.definition
        registered[capability_definition.id] = (
            capability_definition.effect_class.value,
            capability_definition.source_unit,
            capability_definition.source_hash,
        )
    environment = environment_symbol_definition()
    registered[environment.id] = (
        environment.effect_class.value,
        environment.source_unit,
        environment.source_hash,
    )

    violations: list[str] = []
    for module in modules:
        for item in module.authority_attestations:
            actual = registered.get(item.authority_id)
            expected = (
                item.effect_class.value,
                item.source_unit,
                item.source_hash,
            )
            if actual is None:
                violations.append(
                    f"{module.module_id}:{item.authority_id}:unregistered"
                )
            elif actual != expected:
                violations.append(f"{module.module_id}:{item.authority_id}:mismatch")
    return tuple(violations)


def _operator_registration_violations(
    runtime_registry: SchedulerRuntimeRegistry | None,
) -> tuple[str, ...]:
    if runtime_registry is None:
        return ("runtime registry unavailable",)
    missing = set(OperatorName) - set(runtime_registry.operator_names)
    return tuple(
        f"missing:{item.value}" for item in sorted(missing, key=lambda x: x.value)
    )
