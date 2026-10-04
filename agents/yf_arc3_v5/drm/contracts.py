"""Typed, immutable contracts for declarative V5 domain knowledge."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    EpistemicStatus,
    FrozenMap,
    FrozenModel,
    OperatorName,
    Ref,
    RelationFamily,
    StaticKnowledgeNodeKind,
    TermKind,
    require_unique,
)


class DrmDeclarationKind(str, Enum):
    SCHEMA = "schema"
    PRINCIPLE = "principle"
    MODEL = "model"
    PREDICATE = "predicate"
    RULE = "rule"
    CRITERION = "criterion"
    UPDATE_POLICY = "update_policy"
    SELECTION_POLICY = "selection_policy"


class DrmEvaluationKind(str, Enum):
    DECLARATIVE_ONLY = "declarative_only"
    DISTINGUISH_REFERENCES = "distinguish_references"
    BASIS_PRESENT = "basis_present"
    PROJECT_REQUESTED_CONCLUSIONS = "project_requested_conclusions"
    EXACT_SET_EQUALITY = "exact_set_equality"
    EXACT_REVISION = "exact_revision"
    SINGLE_ADMISSIBLE = "single_admissible"
    DECLARED_ELIGIBILITY_SINGLE = "declared_eligibility_single"
    SYMBOLIC_PARTIAL_ORDER = "symbolic_partial_order"
    DECLARED_LEXICOGRAPHIC_ORDER = "declared_lexicographic_order"


class DrmFactOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    EQUALS_REFERENCE_FIELD = "equals_reference_field"
    NOT_EQUALS_REFERENCE_FIELD = "not_equals_reference_field"
    GREATER_THAN_REFERENCE_FIELD = "greater_than_reference_field"
    MEMBER_OF_REFERENCE_FIELD = "member_of_reference_field"
    IS_PRESENT = "is_present"
    IS_ABSENT = "is_absent"
    GREATER_THAN = "greater_than"
    BETWEEN_INCLUSIVE = "between_inclusive"
    AT_MOST_MULTIPLE_OF_FIELD = "at_most_multiple_of_field"


class DrmPreferredOrder(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class DrmFactCondition(FrozenModel):
    """One exact condition over descriptive alternative facts."""

    field: Ref
    operator: DrmFactOperator
    expected: str | bool | int | None = None
    lower: int | None = None
    upper: int | None = None
    reference_field: Ref | None = None
    factor: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_arguments(self) -> "DrmFactCondition":
        if self.operator in {
            DrmFactOperator.EQUALS,
            DrmFactOperator.NOT_EQUALS,
        }:
            valid = (
                self.expected is not None
                and self.lower is None
                and self.upper is None
                and self.reference_field is None
                and self.factor is None
            )
        elif self.operator in {
            DrmFactOperator.EQUALS_REFERENCE_FIELD,
            DrmFactOperator.NOT_EQUALS_REFERENCE_FIELD,
            DrmFactOperator.GREATER_THAN_REFERENCE_FIELD,
            DrmFactOperator.MEMBER_OF_REFERENCE_FIELD,
        }:
            valid = (
                self.expected is None
                and self.lower is None
                and self.upper is None
                and self.reference_field is not None
                and self.factor is None
            )
        elif self.operator in {
            DrmFactOperator.IS_PRESENT,
            DrmFactOperator.IS_ABSENT,
        }:
            valid = (
                self.expected is None
                and self.lower is None
                and self.upper is None
                and self.reference_field is None
                and self.factor is None
            )
        elif self.operator is DrmFactOperator.GREATER_THAN:
            valid = (
                isinstance(self.expected, int)
                and not isinstance(self.expected, bool)
                and self.lower is None
                and self.upper is None
                and self.reference_field is None
                and self.factor is None
            )
        elif self.operator is DrmFactOperator.BETWEEN_INCLUSIVE:
            valid = (
                self.expected is None
                and self.lower is not None
                and self.upper is not None
                and self.lower <= self.upper
                and self.reference_field is None
                and self.factor is None
            )
        else:
            valid = (
                self.expected is None
                and self.lower is None
                and self.upper is None
                and self.reference_field is not None
                and self.factor is not None
            )
        if not valid:
            raise ValueError(
                f"invalid arguments for DRM fact operator {self.operator.value}"
            )
        return self


class DrmSelectionTier(FrozenModel):
    """A named tier: all required conditions and at least one optional signal."""

    id: Ref
    all_conditions: tuple[DrmFactCondition, ...] = ()
    any_conditions: tuple[DrmFactCondition, ...] = ()
    reason_ref: Ref


class DrmTieBreaker(FrozenModel):
    field: Ref
    preferred_order: DrmPreferredOrder
    criterion_ref: Ref | None = None
    reason_ref: Ref | None = None


class DrmFactSequenceComposition(FrozenModel):
    """Bounded concatenation of opaque Ref sequences named by DRM source."""

    output_field: Ref
    source_fields: tuple[Ref, ...] = Field(min_length=1, max_length=8)
    max_items: int = Field(ge=1, le=4096)
    unique_items: bool = False


class DrmFactProjection(FrozenModel):
    """Declaratively add meaning-bearing facts to descriptive capability facts."""

    id: Ref
    all_conditions: tuple[DrmFactCondition, ...] = ()
    any_conditions: tuple[DrmFactCondition, ...] = ()
    static_facts: FrozenMap = Field(default_factory=FrozenMap)
    bound_facts: FrozenMap = Field(default_factory=FrozenMap)
    templated_facts: FrozenMap = Field(default_factory=FrozenMap)
    sequence_compositions: tuple[DrmFactSequenceComposition, ...] = ()
    affine_expressions: tuple["DrmFactAffineExpression", ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_affine_order(self) -> "DrmFactProjection":
        outputs = tuple(item.output_field for item in self.affine_expressions)
        require_unique(outputs, "affine output fields")
        for index, expression in enumerate(self.affine_expressions):
            unavailable = set(outputs[index:])
            if any(term.field in unavailable for term in expression.terms):
                raise ValueError("affine expressions cannot reference self or later outputs")
        return self


class DrmFactAffineTerm(FrozenModel):
    """One integer coefficient and a source-declared input field."""

    field: Ref
    coefficient: int = Field(default=1, strict=True, ge=-1024, le=1024)


class DrmFactAffineExpression(FrozenModel):
    """Exact bounded affine arithmetic; the equation belongs to DRM source."""

    output_field: Ref
    terms: tuple[DrmFactAffineTerm, ...] = Field(min_length=1, max_length=8)
    constant: int = Field(default=0, strict=True, ge=-(2**53), le=2**53)
    divisor: int = Field(default=1, strict=True, ge=1, le=1024)
    max_abs_value: int = Field(default=2**53, strict=True, ge=1, le=2**53)


class DrmTemplateIteration(FrozenModel):
    """Bounded mechanical expansion of one template over measured item facts."""

    source_field: Ref
    item_bindings: FrozenMap
    index_binding: Ref | None = None
    max_items: int = Field(ge=1, le=4096)
    all_conditions: tuple[DrmFactCondition, ...] = ()
    any_conditions: tuple[DrmFactCondition, ...] = ()
    any_condition_groups: tuple[tuple[DrmFactCondition, ...], ...] = ()

    @model_validator(mode="after")
    def validate_condition_groups(self) -> "DrmTemplateIteration":
        if any(not group for group in self.any_condition_groups):
            raise ValueError("DRM template iteration condition groups cannot be empty")
        return self

    @model_validator(mode="after")
    def validate_bindings(self) -> "DrmTemplateIteration":
        if not self.item_bindings:
            raise ValueError("DRM template iteration requires item bindings")
        invalid_sources = tuple(
            source
            for source in (self.item_bindings[__yf_order_key] for __yf_order_key in sorted(self.item_bindings))
            if not isinstance(source, str) or not source.strip()
        )
        if invalid_sources:
            raise ValueError("DRM template iteration sources must be non-empty fields")
        local_fields = set(self.item_bindings)
        if self.source_field in local_fields:
            raise ValueError("iteration source cannot be overwritten by an item binding")
        if self.index_binding is not None and self.index_binding in local_fields:
            raise ValueError("iteration index binding conflicts with an item binding")
        return self


class DrmParameter(FrozenModel):
    name: Ref
    type_name: Ref = "Any"
    required: bool = True


class DrmTermTemplate(FrozenModel):
    """Declarative TERM template instantiated from selected exact facts."""

    id_pattern: Ref
    kind: TermKind
    label: Ref
    required_fields: tuple[Ref, ...] = ()
    required_term_patterns: tuple[Ref, ...] = ()
    static_attributes: FrozenMap = Field(default_factory=FrozenMap)
    attribute_bindings: FrozenMap = Field(default_factory=FrozenMap)
    all_conditions: tuple[DrmFactCondition, ...] = ()
    any_conditions: tuple[DrmFactCondition, ...] = ()
    iteration: DrmTemplateIteration | None = None


class DrmClaimTemplate(FrozenModel):
    """Declarative CLAIM template; no semantic predicate lives in Python."""

    id_pattern: Ref
    predicate: Ref
    family: RelationFamily
    argument_patterns: tuple[Ref, ...] = Field(min_length=1)
    epistemic_status: EpistemicStatus = EpistemicStatus.SUPPORTED
    required_fields: tuple[Ref, ...] = ()
    required_term_patterns: tuple[Ref, ...] = ()
    static_attributes: FrozenMap = Field(default_factory=FrozenMap)
    attribute_bindings: FrozenMap = Field(default_factory=FrozenMap)
    all_conditions: tuple[DrmFactCondition, ...] = ()
    any_conditions: tuple[DrmFactCondition, ...] = ()
    iteration: DrmTemplateIteration | None = None
    derivation_premise_refs_field: Ref | None = None


class DrmDeclaration(FrozenModel):
    id: Ref
    kind: DrmDeclarationKind
    evaluation: DrmEvaluationKind = DrmEvaluationKind.DECLARATIVE_ONLY
    parameters: tuple[DrmParameter, ...] = ()
    output_type: Ref = "Any"
    supported_operators: tuple[OperatorName, ...] = ()
    input_contract: Ref | None = None
    output_contract: Ref | None = None
    effect_class: EffectClass = EffectClass.PURE
    relation_family: RelationFamily | None = None
    premises: tuple[Ref, ...] = ()
    applicability: tuple[Ref, ...] = ()
    invariants: tuple[Ref, ...] = ()
    falsifiers: tuple[Ref, ...] = ()
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    general_principle_refs: tuple[Ref, ...] = ()
    family_ref: Ref | None = None
    abstraction_level: Ref | None = None
    positive_test_refs: tuple[Ref, ...] = ()
    negative_test_refs: tuple[Ref, ...] = ()
    mutation_test_refs: tuple[Ref, ...] = ()
    identity_transfer_test_refs: tuple[Ref, ...] = ()
    composition_family_refs: tuple[Ref, ...] = ()
    composition_test_refs: tuple[Ref, ...] = ()
    inter_level_test_refs: tuple[Ref, ...] = ()
    activation_evidence_refs: tuple[Ref, ...] = ()
    typed_role_parameters: tuple[Ref, ...] = ()
    aspect_parameters: tuple[Ref, ...] = ()
    action_family_parameters: tuple[Ref, ...] = ()
    relation_parameters: tuple[Ref, ...] = ()
    cardinality_parameters: tuple[Ref, ...] = ()
    resource_parameters: tuple[Ref, ...] = ()
    prediction_domain_parameters: tuple[Ref, ...] = ()
    configuration_parameters: tuple[Ref, ...] = ()
    temporal_parameters: tuple[Ref, ...] = ()
    known_counterexamples: tuple[Ref, ...] = ()
    preserves_alternatives: bool = True
    selection_eligibility_conditions: tuple[DrmFactCondition, ...] = ()
    selection_required_fact_fields: tuple[Ref, ...] = ()
    selection_tiers: tuple[DrmSelectionTier, ...] = ()
    selection_precedence_criteria: tuple[DrmTieBreaker, ...] = ()
    selection_tie_breakers: tuple[DrmTieBreaker, ...] = ()
    canonical_equivalence_tie_break_debt_ref: Ref | None = None
    declared_alternative_refs: tuple[Ref, ...] = ()
    fact_projections: tuple[DrmFactProjection, ...] = ()
    term_templates: tuple[DrmTermTemplate, ...] = ()
    claim_templates: tuple[DrmClaimTemplate, ...] = ()
    meaning_template_ref_by_alternative: FrozenMap = Field(default_factory=FrozenMap)
    configuration: FrozenMap = Field(default_factory=FrozenMap)
    # Optional explicit membership in the reusable static knowledge tree.
    # The metadata is authored by DRM; Python only projects it mechanically.
    static_knowledge_node_kind: StaticKnowledgeNodeKind | None = None
    static_knowledge_parent_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.drm_declaration.v1"

    @model_validator(mode="after")
    def validate_declaration(self) -> "DrmDeclaration":
        require_unique(tuple(item.name for item in self.parameters), "DRM parameters")
        require_unique(self.supported_operators, "DRM supported operators")
        require_unique(self.premises, "DRM premises")
        require_unique(self.applicability, "DRM applicability")
        require_unique(self.invariants, "DRM invariants")
        require_unique(self.falsifiers, "DRM falsifiers")
        require_unique(self.reason_refs, "DRM reasons")
        require_unique(self.general_principle_refs, "general principle references")
        require_unique(self.positive_test_refs, "positive test references")
        require_unique(self.negative_test_refs, "negative test references")
        require_unique(self.mutation_test_refs, "mutation test references")
        require_unique(
            self.identity_transfer_test_refs, "identity transfer test references"
        )
        require_unique(self.composition_family_refs, "composition family references")
        require_unique(self.composition_test_refs, "composition test references")
        require_unique(self.inter_level_test_refs, "inter-level test references")
        require_unique(self.activation_evidence_refs, "activation evidence references")
        require_unique(self.typed_role_parameters, "typed role parameters")
        require_unique(self.aspect_parameters, "aspect parameters")
        require_unique(self.action_family_parameters, "action family parameters")
        require_unique(self.relation_parameters, "relation parameters")
        require_unique(self.cardinality_parameters, "cardinality parameters")
        require_unique(self.resource_parameters, "resource parameters")
        require_unique(
            self.prediction_domain_parameters, "prediction domain parameters"
        )
        require_unique(self.configuration_parameters, "configuration parameters")
        require_unique(self.temporal_parameters, "temporal parameters")
        require_unique(self.known_counterexamples, "known counterexamples")
        if self.id in self.general_principle_refs:
            raise ValueError("a DRM declaration cannot generalize itself")
        if self.general_principle_refs and self.kind is not DrmDeclarationKind.PRINCIPLE:
            raise ValueError("only principle declarations may name general principles")
        if (self.family_ref is None) != (self.abstraction_level is None):
            raise ValueError(
                "principle family and abstraction level must be declared together"
            )
        if self.family_ref is not None and self.kind is not DrmDeclarationKind.PRINCIPLE:
            raise ValueError("only principle declarations may name a semantic family")
        if self.abstraction_level is not None and self.abstraction_level not in {
            "common",
            "parameterized_core",
            "causal_variant",
        }:
            raise ValueError("unsupported principle abstraction level")
        family_proof_fields = (
            self.positive_test_refs,
            self.negative_test_refs,
            self.mutation_test_refs,
            self.identity_transfer_test_refs,
            self.composition_family_refs,
            self.composition_test_refs,
            self.inter_level_test_refs,
            self.activation_evidence_refs,
        )
        if any(family_proof_fields) and self.kind is not DrmDeclarationKind.SCHEMA:
            raise ValueError("only schema declarations may own family proof metadata")
        require_unique(
            tuple(item.id for item in self.selection_tiers), "selection tiers"
        )
        require_unique(
            tuple(item.id_pattern for item in self.term_templates), "TERM templates"
        )
        require_unique(
            tuple(item.id_pattern for item in self.claim_templates), "CLAIM templates"
        )
        require_unique(
            tuple(item.field for item in self.selection_tie_breakers),
            "selection tie breakers",
        )
        require_unique(
            tuple(item.field for item in self.selection_precedence_criteria),
            "selection precedence criteria",
        )
        if any(not item.criterion_ref or not item.reason_ref for item in self.selection_precedence_criteria):
            raise ValueError("selection precedence criteria require declared criteria and reasons")
        if any(item.field not in self.selection_required_fact_fields for item in self.selection_precedence_criteria):
            raise ValueError("selection precedence criteria require explicit total facts")
        if any(item.field in {"candidate_ref", "alternative_ref"} for item in self.selection_precedence_criteria):
            raise ValueError("canonical references cannot precede semantic selection tiers")
        require_unique(
            self.selection_required_fact_fields,
            "selection required fact fields",
        )
        require_unique(self.declared_alternative_refs, "declared alternatives")
        require_unique(
            tuple(item.id for item in self.fact_projections), "fact projections"
        )
        runtime_kind = self.kind in {
            DrmDeclarationKind.MODEL,
            DrmDeclarationKind.PREDICATE,
            DrmDeclarationKind.RULE,
            DrmDeclarationKind.CRITERION,
            DrmDeclarationKind.UPDATE_POLICY,
            DrmDeclarationKind.SELECTION_POLICY,
        }
        if runtime_kind:
            if (
                not self.supported_operators
                or not self.input_contract
                or not self.output_contract
            ):
                raise ValueError(
                    "runtime DRM declarations require operators and contracts"
                )
            if self.evaluation is DrmEvaluationKind.DECLARATIVE_ONLY:
                raise ValueError("runtime DRM declarations require an evaluation kind")
        elif (
            self.supported_operators
            or self.input_contract is not None
            or self.output_contract is not None
            or self.evaluation is not DrmEvaluationKind.DECLARATIVE_ONLY
        ):
            raise ValueError(
                "schema/principle declarations cannot expose runtime authority"
            )
        if self.kind is DrmDeclarationKind.UPDATE_POLICY:
            if self.effect_class is not EffectClass.COGNITIVE_UPDATE_ADAPTER:
                raise ValueError("update policy requires cognitive update effect")
        elif self.effect_class is not EffectClass.PURE:
            raise ValueError("only update policy may have a non-pure DRM effect")
        declared_selection = self.evaluation in {
            DrmEvaluationKind.DECLARED_ELIGIBILITY_SINGLE,
            DrmEvaluationKind.DECLARED_LEXICOGRAPHIC_ORDER,
        }
        if declared_selection:
            if self.kind is not DrmDeclarationKind.SELECTION_POLICY:
                raise ValueError(
                    "declared selection is valid only for selection policies"
                )
            if (
                self.evaluation is DrmEvaluationKind.DECLARED_LEXICOGRAPHIC_ORDER
                and (not self.selection_tiers or not self.selection_tie_breakers)
            ):
                raise ValueError(
                    "declared ordering requires selection tiers and tie breakers"
                )
            if (
                self.evaluation is DrmEvaluationKind.DECLARED_ELIGIBILITY_SINGLE
                and (
                    self.selection_tiers
                    or self.selection_precedence_criteria
                    or self.selection_tie_breakers
                    or self.canonical_equivalence_tie_break_debt_ref
                )
            ):
                raise ValueError(
                    "declared single eligibility cannot rank eligible alternatives"
                )
            final_tie_field = (
                self.selection_tie_breakers[-1].field
                if self.selection_tie_breakers
                else None
            )
            uses_canonical_equivalence_tie_break = final_tie_field in {
                "candidate_ref",
                "alternative_ref",
            }
            if uses_canonical_equivalence_tie_break != bool(
                self.canonical_equivalence_tie_break_debt_ref
            ):
                raise ValueError(
                    "a canonical equivalence tie-break requires exactly one declared debt ref"
                )
        elif (
            self.selection_eligibility_conditions
            or self.selection_required_fact_fields
            or self.selection_tiers
            or self.selection_precedence_criteria
            or self.selection_tie_breakers
            or self.canonical_equivalence_tie_break_debt_ref
        ):
            raise ValueError(
                "selection conditions are valid only for declared lexicographic ordering"
            )
        if self.declared_alternative_refs and self.kind is not DrmDeclarationKind.SELECTION_POLICY:
            raise ValueError("declared alternatives belong to a selection policy")
        if self.fact_projections and self.kind is not DrmDeclarationKind.SELECTION_POLICY:
            raise ValueError("fact projections belong to a selection policy")
        if (self.term_templates or self.claim_templates) and self.kind is not DrmDeclarationKind.RULE:
            raise ValueError("meaning templates belong to a declarative DRM rule")
        if self.meaning_template_ref_by_alternative and self.kind is not DrmDeclarationKind.SELECTION_POLICY:
            raise ValueError("meaning template routing belongs to a selection policy")
        if self.static_knowledge_node_kind is not None and self.kind not in {
            DrmDeclarationKind.PRINCIPLE,
            DrmDeclarationKind.SCHEMA,
            DrmDeclarationKind.SELECTION_POLICY,
        }:
            raise ValueError(
                "static knowledge membership belongs to principle/schema/priority declarations"
            )
        if (
            self.static_knowledge_node_kind is StaticKnowledgeNodeKind.PRIORITY
            and self.kind is not DrmDeclarationKind.SELECTION_POLICY
        ):
            raise ValueError("static priority knowledge must be a selection policy")
        if (
            self.kind is DrmDeclarationKind.SELECTION_POLICY
            and self.static_knowledge_node_kind is not None
            and self.static_knowledge_node_kind is not StaticKnowledgeNodeKind.PRIORITY
        ):
            raise ValueError("selection policy static knowledge must be a priority")
        return self


class EffectivePrincipleContract(FrozenModel):
    """Mechanical projection of one principle and all declared general ancestors."""

    principle_ref: Ref
    lineage_refs: tuple[Ref, ...] = Field(min_length=1)
    premises: tuple[Ref, ...] = ()
    applicability: tuple[Ref, ...] = ()
    invariants: tuple[Ref, ...] = ()
    falsifiers: tuple[Ref, ...] = ()
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    typed_role_parameters: tuple[Ref, ...] = ()
    aspect_parameters: tuple[Ref, ...] = ()
    action_family_parameters: tuple[Ref, ...] = ()
    relation_parameters: tuple[Ref, ...] = ()
    cardinality_parameters: tuple[Ref, ...] = ()
    resource_parameters: tuple[Ref, ...] = ()
    prediction_domain_parameters: tuple[Ref, ...] = ()
    configuration_parameters: tuple[Ref, ...] = ()
    temporal_parameters: tuple[Ref, ...] = ()
    known_counterexamples: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.effective_principle_contract.v1"

    @model_validator(mode="after")
    def validate_effective_contract(self) -> "EffectivePrincipleContract":
        require_unique(self.lineage_refs, "principle lineage")
        require_unique(self.premises, "effective principle premises")
        require_unique(self.applicability, "effective principle applicability")
        require_unique(self.invariants, "effective principle invariants")
        require_unique(self.falsifiers, "effective principle falsifiers")
        require_unique(self.reason_refs, "effective principle reasons")
        require_unique(self.typed_role_parameters, "effective typed role parameters")
        require_unique(self.aspect_parameters, "effective aspect parameters")
        require_unique(
            self.action_family_parameters, "effective action family parameters"
        )
        require_unique(self.relation_parameters, "effective relation parameters")
        require_unique(
            self.cardinality_parameters, "effective cardinality parameters"
        )
        require_unique(self.resource_parameters, "effective resource parameters")
        require_unique(
            self.prediction_domain_parameters,
            "effective prediction domain parameters",
        )
        require_unique(
            self.configuration_parameters, "effective configuration parameters"
        )
        require_unique(self.temporal_parameters, "effective temporal parameters")
        require_unique(self.known_counterexamples, "effective known counterexamples")
        if self.lineage_refs[-1] != self.principle_ref:
            raise ValueError("effective principle lineage must end at the requested principle")
        return self


class DrmDocument(FrozenModel):
    module_id: Ref
    declarations: tuple[DrmDeclaration, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.drm_document.v1"

    @model_validator(mode="after")
    def validate_document(self) -> "DrmDocument":
        require_unique(tuple(item.id for item in self.declarations), "DRM declarations")
        return self


class LoadedDrmDocument(FrozenModel):
    document: DrmDocument
    source_unit: Ref
    source_hash: Ref
    semantic_hash: Ref
    schema_version: Ref = "yf_arc3_v5.loaded_drm_document.v1"

    @model_validator(mode="after")
    def validate_hashes(self) -> "LoadedDrmDocument":
        for value, name in (
            (self.source_hash, "source_hash"),
            (self.semantic_hash, "semantic_hash"),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"DRM {name} must be a lowercase SHA-256 digest")
        return self
