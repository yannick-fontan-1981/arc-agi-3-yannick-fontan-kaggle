"""Exact source-hashed registry for V5 DRM declarations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from agents.yf_arc3_v5.drm.contracts import (
    DrmDeclaration,
    DrmDeclarationKind,
    DrmFactCondition,
    DrmFactOperator,
    EffectivePrincipleContract,
    LoadedDrmDocument,
)
from agents.yf_arc3_v5.src.symbols import (
    CompilerEnvironment,
    ParameterSpec,
    SymbolDefinition,
    SymbolKind,
    type_spec,
)


class DrmRegistryError(ValueError):
    """DRM declarations conflict or cannot be admitted."""


class DrmResolutionError(LookupError):
    """A requested DRM declaration is absent."""


def _stable_fact_value_order_key(value: object) -> tuple[int, int, str]:
    """Total order for the closed scalar domain of an equals condition."""

    if value is None:
        return (0, 0, "")
    if isinstance(value, bool):
        return (1, int(value), "")
    if isinstance(value, int):
        return (2, value, "")
    if isinstance(value, str):
        return (3, 0, value)
    raise DrmRegistryError("unsupported equality value in projection index")


class DrmRegistry:
    def __init__(self) -> None:
        self._documents: dict[str, LoadedDrmDocument] = {}
        self._declarations: dict[str, tuple[DrmDeclaration, LoadedDrmDocument]] = {}
        self._fact_projection_index: dict[tuple[object, ...], tuple[object, ...]] = {}
        self._fact_projection_stable_index: dict[
            str,
            tuple[
                tuple[int, ...],
                tuple[tuple[str, dict[object, tuple[int, ...]]], ...],
            ],
        ] = {}
        self._projection_condition_plans: dict[
            str,
            tuple[
                dict[
                    int,
                    tuple[
                        tuple[tuple[DrmFactCondition, int], ...],
                        tuple[tuple[DrmFactCondition, int], ...],
                    ],
                ],
                tuple[DrmFactCondition, ...],
            ],
        ] = {}

    def register(self, loaded: LoadedDrmDocument) -> None:
        module_id = loaded.document.module_id
        existing_document = self._documents.get(module_id)
        if existing_document is not None and existing_document != loaded:
            raise DrmRegistryError(f"DRM module already registered: {module_id}")
        conflicts = sorted(
            declaration.id
            for declaration in loaded.document.declarations
            if declaration.id in self._declarations
            and self._declarations[declaration.id][1].document.module_id != module_id
        )
        if conflicts:
            raise DrmRegistryError(f"duplicate DRM declaration identities: {conflicts}")
        self._documents[module_id] = loaded
        self._fact_projection_index.clear()
        self._fact_projection_stable_index.clear()
        self._projection_condition_plans.clear()
        for declaration in loaded.document.declarations:
            self._declarations[declaration.id] = (declaration, loaded)

    @property
    def documents(self) -> tuple[LoadedDrmDocument, ...]:
        return tuple(self._documents[key] for key in sorted(self._documents))

    @property
    def declaration_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._declarations))

    def resolve(self, declaration_id: str) -> tuple[DrmDeclaration, LoadedDrmDocument]:
        entry = self._declarations.get(declaration_id)
        if entry is None:
            raise DrmResolutionError(f"unregistered DRM declaration: {declaration_id}")
        return entry

    def fact_projections_for(
        self,
        declaration_id: str,
        alternative_ref: str,
        stable_facts: Mapping[str, object] | None = None,
    ) -> tuple[object, ...]:
        """Return the immutable declared projection set from a tiny index."""

        declaration, _loaded = self.resolve(declaration_id)
        key = (declaration_id, alternative_ref)
        cached = self._fact_projection_index.get(key)
        if cached is not None and stable_facts is None:
            return cached
        selected = tuple(declaration.fact_projections)
        self._fact_projection_index[key] = selected
        if stable_facts is None or not selected:
            return selected

        stable_index = self._fact_projection_stable_index.get(declaration_id)
        if stable_index is None:
            # A fact authored anywhere in the sequential declaration can be a
            # dependency of a later projection.  Index only equality tests over
            # immutable descriptive input; every semantic dependency remains
            # in the ordinary ordered evaluator.
            authored_fields = {
                str(field)
                for projection in selected
                for mapping in (
                    projection.static_facts,
                    projection.bound_facts,
                    projection.templated_facts,
                )
                for field in mapping
            }
            authored_fields.update(
                composition.output_field
                for projection in selected for composition in projection.sequence_compositions
            )
            fallback: list[int] = []
            mutable_buckets: dict[str, dict[object, list[int]]] = {}
            for index, projection in enumerate(selected):
                stable_condition = next(
                    (
                        condition
                        for condition in projection.all_conditions
                        if condition.operator is DrmFactOperator.EQUALS
                        and str(condition.field) not in authored_fields
                    ),
                    None,
                )
                if stable_condition is None:
                    fallback.append(index)
                    continue
                field_buckets = mutable_buckets.setdefault(
                    str(stable_condition.field), {}
                )
                field_buckets.setdefault(stable_condition.expected, []).append(index)
            stable_index = (
                tuple(fallback),
                tuple(
                    (
                        field,
                        {
                            expected: tuple(expected_buckets[expected])
                            for expected in sorted(
                                expected_buckets,
                                key=_stable_fact_value_order_key,
                            )
                        },
                    )
                    for field, expected_buckets in sorted(mutable_buckets.items())
                ),
            )
            self._fact_projection_stable_index[declaration_id] = stable_index
        fallback, field_indexes = stable_index
        viable_indices = set(fallback)
        for field, expected_buckets in field_indexes:
            viable_indices.update(expected_buckets.get(stable_facts.get(field), ()))
        return tuple(selected[index] for index in sorted(viable_indices))

    def projection_condition_plan(
        self, declaration_id: str
    ) -> tuple[
        dict[
            int,
            tuple[
                tuple[tuple[DrmFactCondition, int], ...],
                tuple[tuple[DrmFactCondition, int], ...],
            ],
        ],
        tuple[DrmFactCondition, ...],
    ]:
        """Intern conditions whose inputs cannot change during projection."""

        cached = self._projection_condition_plans.get(declaration_id)
        if cached is not None:
            return cached
        declaration, _loaded = self.resolve(declaration_id)
        authored_fields = {
            str(field)
            for projection in declaration.fact_projections
            for mapping in (
                projection.static_facts,
                projection.bound_facts,
                projection.templated_facts,
            )
            for field in mapping
        }
        authored_fields.update(
            composition.output_field
            for projection in declaration.fact_projections for composition in projection.sequence_compositions
        )
        unique_slots: dict[tuple[object, ...], int] = {}
        conditions_by_slot: list[DrmFactCondition] = []

        def indexed_conditions(
            conditions: tuple[DrmFactCondition, ...],
        ) -> tuple[tuple[DrmFactCondition, int], ...]:
            result: list[tuple[DrmFactCondition, int]] = []
            for condition in conditions:
                mutable = (
                    condition.field in authored_fields
                    or condition.reference_field in authored_fields
                    or (
                        condition.field == "terminal_access_candidate_ref_present"
                        and "terminal_access_candidate_ref" in authored_fields
                    )
                )
                if mutable:
                    slot = -1
                else:
                    token = (
                        condition.field,
                        condition.operator,
                        type(condition.expected),
                        condition.expected,
                        condition.lower,
                        condition.upper,
                        condition.reference_field,
                        condition.factor,
                    )
                    slot = unique_slots.get(token)
                    if slot is None:
                        slot = len(unique_slots)
                        unique_slots[token] = slot
                        conditions_by_slot.append(condition)
                result.append((condition, slot))
            return tuple(result)

        plans = {
            id(projection): (
                indexed_conditions(projection.all_conditions),
                indexed_conditions(projection.any_conditions),
            )
            for projection in declaration.fact_projections
        }
        compiled = (plans, tuple(conditions_by_slot))
        self._projection_condition_plans[declaration_id] = compiled
        return compiled

    def validate_principle_generalizations(self) -> None:
        for declaration_id in self.declaration_ids:
            declaration, _loaded = self.resolve(declaration_id)
            if declaration.kind is DrmDeclarationKind.PRINCIPLE:
                self._principle_lineage_ids(declaration_id)

    def validate_principle_families(self) -> None:
        required_proof_fields = (
            "positive_test_refs",
            "negative_test_refs",
            "mutation_test_refs",
            "identity_transfer_test_refs",
            "composition_family_refs",
            "composition_test_refs",
            "inter_level_test_refs",
            "activation_evidence_refs",
        )
        for declaration_id in self.declaration_ids:
            declaration, _loaded = self.resolve(declaration_id)
            if declaration.family_ref is None:
                continue
            family, _family_source = self.resolve(declaration.family_ref)
            if family.kind is not DrmDeclarationKind.SCHEMA:
                raise DrmRegistryError(
                    f"principle family must resolve to a schema: {declaration.family_ref}"
                )
            missing = tuple(
                field for field in required_proof_fields if not getattr(family, field)
            )
            if missing:
                raise DrmRegistryError(
                    f"principle family lacks proof metadata: {declaration.family_ref}: "
                    f"{missing}"
                )
            for partner_ref in family.composition_family_refs:
                if partner_ref == family.id:
                    raise DrmRegistryError(
                        f"principle family cannot compose only with itself: {family.id}"
                    )
                partner, _partner_source = self.resolve(partner_ref)
                if partner.kind is not DrmDeclarationKind.SCHEMA:
                    raise DrmRegistryError(
                        f"composition family must resolve to a schema: {partner_ref}"
                    )

    def resolve_effective_principle(
        self,
        declaration_id: str,
    ) -> EffectivePrincipleContract:
        lineage_refs = self._principle_lineage_ids(declaration_id)
        declarations = tuple(self.resolve(ref)[0] for ref in lineage_refs)
        return EffectivePrincipleContract(
            principle_ref=declaration_id,
            lineage_refs=lineage_refs,
            premises=_ordered_unique(
                item for declaration in declarations for item in declaration.premises
            ),
            applicability=_ordered_unique(
                item for declaration in declarations for item in declaration.applicability
            ),
            invariants=_ordered_unique(
                item for declaration in declarations for item in declaration.invariants
            ),
            falsifiers=_ordered_unique(
                item for declaration in declarations for item in declaration.falsifiers
            ),
            reason_refs=_ordered_unique(
                item for declaration in declarations for item in declaration.reason_refs
            ),
            typed_role_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.typed_role_parameters
            ),
            aspect_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.aspect_parameters
            ),
            action_family_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.action_family_parameters
            ),
            relation_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.relation_parameters
            ),
            cardinality_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.cardinality_parameters
            ),
            resource_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.resource_parameters
            ),
            prediction_domain_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.prediction_domain_parameters
            ),
            configuration_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.configuration_parameters
            ),
            temporal_parameters=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.temporal_parameters
            ),
            known_counterexamples=_ordered_unique(
                item
                for declaration in declarations
                for item in declaration.known_counterexamples
            ),
        )

    def _principle_lineage_ids(self, declaration_id: str) -> tuple[str, ...]:
        lineage: list[str] = []
        resolved: set[str] = set()
        active: list[str] = []

        def visit(current_ref: str) -> None:
            if current_ref in active:
                cycle = (*active[active.index(current_ref) :], current_ref)
                raise DrmRegistryError(
                    f"cyclic principle generalization: {' -> '.join(cycle)}"
                )
            if current_ref in resolved:
                return
            try:
                declaration, _loaded = self.resolve(current_ref)
            except DrmResolutionError as error:
                raise DrmRegistryError(
                    f"unregistered general principle: {current_ref}"
                ) from error
            if declaration.kind is not DrmDeclarationKind.PRINCIPLE:
                raise DrmRegistryError(
                    f"general principle reference is not a principle: {current_ref}"
                )
            active.append(current_ref)
            for general_ref in declaration.general_principle_refs:
                visit(general_ref)
            active.pop()
            resolved.add(current_ref)
            lineage.append(current_ref)

        visit(declaration_id)
        return tuple(lineage)

    def symbol_definitions(self) -> tuple[SymbolDefinition, ...]:
        return tuple(
            _symbol_definition(*self._declarations[declaration_id])
            for declaration_id in sorted(self._declarations)
        )

    def extend_environment(
        self, environment: CompilerEnvironment
    ) -> CompilerEnvironment:
        return environment.model_copy(
            update={"symbols": (*environment.symbols, *self.symbol_definitions())}
        )


def build_production_drm_registry(
    documents: tuple[LoadedDrmDocument, ...],
) -> DrmRegistry:
    registry = DrmRegistry()
    for document in documents:
        registry.register(document)
    registry.validate_principle_generalizations()
    registry.validate_principle_families()
    return registry


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _symbol_definition(
    declaration: DrmDeclaration,
    loaded: LoadedDrmDocument,
) -> SymbolDefinition:
    kind = {
        DrmDeclarationKind.SCHEMA: SymbolKind.DRM_DECLARATION,
        DrmDeclarationKind.PRINCIPLE: SymbolKind.DRM_DECLARATION,
        DrmDeclarationKind.MODEL: SymbolKind.MODEL,
        DrmDeclarationKind.PREDICATE: SymbolKind.PREDICATE,
        DrmDeclarationKind.RULE: SymbolKind.RULE,
        DrmDeclarationKind.CRITERION: SymbolKind.CRITERION,
        DrmDeclarationKind.UPDATE_POLICY: SymbolKind.POLICY,
        DrmDeclarationKind.SELECTION_POLICY: SymbolKind.POLICY,
    }[declaration.kind]
    return SymbolDefinition(
        id=declaration.id,
        kind=kind,
        parameters=tuple(
            ParameterSpec(
                name=parameter.name,
                type_spec=type_spec(parameter.type_name),
                required=parameter.required,
            )
            for parameter in declaration.parameters
        ),
        output_type=type_spec(declaration.output_type),
        effect_class=declaration.effect_class,
        source_unit=loaded.source_unit,
        source_hash=loaded.source_hash,
    )
