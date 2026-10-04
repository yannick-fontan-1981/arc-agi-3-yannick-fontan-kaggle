"""Exact source-hashed registry for compiled SRC modules and workflows."""

from __future__ import annotations

from pydantic import Field

from agents.yf_arc3_v5.logos.types import FrozenModel, Ref, stable_digest
from agents.yf_arc3_v5.src.compiler import CompiledModule, IrWorkflow


class RegistryConflictError(ValueError):
    """A module identity was already registered with different source."""


class RegistryResolutionError(LookupError):
    """An exact source-hashed module or workflow could not be resolved."""


class RegistryEntry(FrozenModel):
    module_id: Ref
    source_hash: Ref
    semantic_hash: Ref
    transitive_source_hash: Ref
    imports: tuple[Ref, ...]
    workflow_ids: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.src_registry_entry.v0_1"


class WorkflowRegistry:
    """Mutable registry shell over immutable, exact compiled definitions."""

    def __init__(self) -> None:
        self._modules: dict[str, tuple[RegistryEntry, CompiledModule]] = {}
        self._workflows: dict[str, tuple[RegistryEntry, IrWorkflow]] = {}

    @property
    def entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(
            entry
            for entry, _module in sorted(
                self._modules.values(), key=lambda item: item[0].module_id
            )
        )

    def register(
        self,
        module: CompiledModule,
        *,
        imported_entries: tuple[RegistryEntry, ...] | None = None,
    ) -> RegistryEntry:
        import_ids = {item.module_id for item in module.imports}
        if imported_entries is None:
            imported_entries = tuple(
                self._modules[module_id][0]
                for module_id in sorted(import_ids)
                if module_id in self._modules
            )
        if len(imported_entries) != len(
            {entry.module_id for entry in imported_entries}
        ):
            raise RegistryResolutionError("duplicate imported registry entry")
        import_index = {entry.module_id: entry for entry in imported_entries}
        if set(import_index) != import_ids:
            missing = sorted(import_ids - set(import_index))
            extra = sorted(set(import_index) - import_ids)
            raise RegistryResolutionError(
                f"registry import mismatch; missing={missing}, extra={extra}"
            )
        unattached = sorted(
            module_id
            for module_id, entry in sorted(import_index.items())
            if module_id not in self._modules or self._modules[module_id][0] != entry
        )
        if unattached:
            raise RegistryResolutionError(
                f"imports are not exact entries in this registry: {unattached}"
            )
        transitive_source_hash = stable_digest(
            {
                "module_id": module.module_id,
                "source_hash": module.source_hash,
                "imports": tuple(
                    (import_id, import_index[import_id].transitive_source_hash)
                    for import_id in sorted(import_index)
                ),
            }
        )
        entry = RegistryEntry(
            module_id=module.module_id,
            source_hash=module.source_hash,
            semantic_hash=module.semantic_hash,
            transitive_source_hash=transitive_source_hash,
            imports=tuple(item.module_id for item in module.imports),
            workflow_ids=tuple(workflow.id for workflow in module.workflows),
        )

        existing = self._modules.get(module.module_id)
        if existing:
            existing_entry, existing_module = existing
            if existing_entry == entry and existing_module == module:
                return existing_entry
            raise RegistryConflictError(
                f"module {module.module_id} already has a different registered source"
            )

        collisions = sorted(
            workflow.id
            for workflow in module.workflows
            if workflow.id in self._workflows
        )
        if collisions:
            raise RegistryConflictError(f"duplicate workflow identities: {collisions}")

        self._modules[module.module_id] = (entry, module)
        for workflow in module.workflows:
            self._workflows[workflow.id] = (entry, workflow)
        return entry

    def register_graph(
        self,
        modules: tuple[CompiledModule, ...],
    ) -> tuple[RegistryEntry, ...]:
        """Register an unordered, closed import graph or change nothing on failure."""

        if len(modules) != len({module.module_id for module in modules}):
            raise RegistryConflictError(
                "duplicate module identities in registration graph"
            )
        pending = {module.module_id: module for module in modules}
        known = set(self._modules)
        missing = sorted(
            {
                item.module_id
                for module in modules
                for item in module.imports
                if item.module_id not in pending and item.module_id not in known
            }
        )
        if missing:
            raise RegistryResolutionError(f"unregistered graph imports: {missing}")

        ordered: list[CompiledModule] = []
        remaining = dict(pending)
        while remaining:
            ready = sorted(
                module_id
                for module_id, module in sorted(remaining.items())
                if all(
                    item.module_id in known or item.module_id not in remaining
                    for item in module.imports
                )
            )
            if not ready:
                raise RegistryResolutionError(
                    f"cyclic module import graph: {sorted(remaining)}"
                )
            for module_id in ready:
                module = remaining.pop(module_id)
                ordered.append(module)
                known.add(module_id)

        shadow = WorkflowRegistry()
        for _module_id, (_entry, existing_module) in sorted(self._modules.items()):
            shadow.register(existing_module)
        new_entries = tuple(shadow.register(module) for module in ordered)
        self._modules = shadow._modules
        self._workflows = shadow._workflows
        return new_entries

    def module(
        self,
        module_id: Ref,
        *,
        expected_transitive_source_hash: Ref,
    ) -> CompiledModule:
        stored = self._modules.get(module_id)
        if stored is None:
            raise RegistryResolutionError(f"unregistered module: {module_id}")
        entry, module = stored
        if entry.transitive_source_hash != expected_transitive_source_hash:
            raise RegistryResolutionError(f"stale source hash for module: {module_id}")
        return module

    def workflow(
        self,
        workflow_id: Ref,
        *,
        expected_transitive_source_hash: Ref,
    ) -> IrWorkflow:
        stored = self._workflows.get(workflow_id)
        if stored is None:
            raise RegistryResolutionError(f"unregistered workflow: {workflow_id}")
        entry, workflow = stored
        if entry.transitive_source_hash != expected_transitive_source_hash:
            raise RegistryResolutionError(
                f"stale source hash for workflow: {workflow_id}"
            )
        return workflow

    def resolve_workflow(
        self,
        workflow_id: Ref,
        *,
        expected_transitive_source_hash: Ref,
    ) -> tuple[RegistryEntry, CompiledModule, IrWorkflow]:
        """Resolve the exact workflow together with its owning source identity."""

        stored = self._workflows.get(workflow_id)
        if stored is None:
            raise RegistryResolutionError(f"unregistered workflow: {workflow_id}")
        entry, workflow = stored
        if entry.transitive_source_hash != expected_transitive_source_hash:
            raise RegistryResolutionError(
                f"stale source hash for workflow: {workflow_id}"
            )
        module = self._modules[entry.module_id][1]
        return entry, module, workflow
