"""M7 source set layered on the M6 production cognitive workflows."""

from __future__ import annotations

from pathlib import Path

from agents.yf_arc3_v5.src.compiler import CompiledModule, compile_source
from agents.yf_arc3_v5.src.production import (
    compile_production_workflows,
    production_environment,
)
from agents.yf_arc3_v5.src.symbols import CompilerEnvironment


def m7_environment(*, base_environment: CompilerEnvironment | None = None) -> CompilerEnvironment:
    from agents.yf_arc3_v5.runtime.acquisition import acquisition_symbol_definition
    from agents.yf_arc3_v5.runtime.environment import environment_symbol_definition

    base = production_environment() if base_environment is None else base_environment
    overrides = {
        item.id: item
        for item in (
            acquisition_symbol_definition(),
            environment_symbol_definition(),
        )
    }
    return base.model_copy(
        update={
            "symbols": tuple(
                overrides.get(symbol.id, symbol) for symbol in base.symbols
            )
        }
    )


def m7_src_paths() -> tuple[Path, ...]:
    root = Path(__file__).parent / "core"
    return (
        root / "release_action.src",
        root / "released_action_cycle.src",
    )


def compile_m7_workflows(
    *,
    environment: CompilerEnvironment | None = None,
) -> tuple[CompiledModule, ...]:
    active_environment = environment or m7_environment()
    base = compile_production_workflows(environment=active_environment)
    runtime_modules = tuple(
        compile_source(
            path.read_text(encoding="utf-8"),
            source_name=str(path),
            environment=active_environment,
        )
        for path in m7_src_paths()
    )
    module_ids = tuple(module.module_id for module in (*base, *runtime_modules))
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("M7 source set contains duplicate module identities")
    return (*base, *runtime_modules)
