"""Shared immutable value types for the V5 cognitive core."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic_core import core_schema

Ref: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeRevision: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
PositiveSequence: TypeAlias = Annotated[int, Field(strict=True, ge=1)]


class TermKind(str, Enum):
    ENTITY = "entity"
    EVENT = "event"
    SCHEMA = "schema"
    PROCEDURE = "procedure"
    GOAL = "goal"
    VALUE = "value"


class RelationFamily(str, Enum):
    STRUCTURAL = "structural"
    DYNAMIC = "dynamic"
    EPISTEMIC = "epistemic"
    CONTROL = "control"


class RelationDimension(str, Enum):
    """Closed generic dimensions whose links must remain independently revisable."""

    IDENTITY_COMPOSITION = "identity_composition"
    MATERIAL_TOPOLOGY = "material_topology"
    AGENCY_CONTROL = "agency_control"
    CORRESPONDENCE_LOCATION = "correspondence_location"
    SUPPORT_OCCUPANCY = "support_occupancy"
    TRANSFORMATION_INVARIANT = "transformation_invariant"
    TEMPORAL_TIMELINE = "temporal_timeline"
    RESOURCE_RECOVERY = "resource_recovery"


class RoleKind(str, Enum):
    """Generic revisable roles; morphology alone never establishes one."""

    SURFACE = "surface"
    SEPARATOR = "separator"
    COMPACT_BODY = "compact_body"
    REPEATED_STRUCTURE = "repeated_structure"
    AXIS_LIKE_BAND = "axis_like_band"
    ENCLOSURE = "enclosure"
    BORDER_DISPLAY = "border_display"
    INTERFACE_CANDIDATE = "interface_candidate"
    CONTROLLED_SOURCE = "controlled_source"
    DEPENDENT_BODY = "dependent_body"
    FIXED_REFERENCE = "fixed_reference"
    TARGET_CANDIDATE = "target_candidate"
    SELECTION_CARRIER = "selection_carrier"
    ACTUATOR = "actuator"
    RESOURCE_INDICATOR = "resource_indicator"
    SUPPORT_SURFACE = "support_surface"
    TEMPORAL_INSTANCE = "temporal_instance"
    WORLD_VIEWPORT_ANCHOR = "world_viewport_anchor"


class GoalKind(str, Enum):
    """Extensible generic goal vocabulary licensed by current claims."""

    ALIGN_POSITION_ROW = "align.position.row"
    ALIGN_POSITION_COLUMN = "align.position.column"
    ALIGN_POSITION_BOTH = "align.position.both"
    ALIGN_CONTACT_EDGE = "align.contact.edge"
    ALIGN_CONTACT_OVERLAP = "align.contact.overlap"
    ALIGN_CONTAINMENT = "align.containment"
    ALIGN_OFFSET_CONSTANT = "align.offset.constant"
    ALIGN_SEPARATION = "align.separation"
    MATCH_SHAPE_EXACT = "match.shape_exact"
    MATCH_SHAPE_TOPOLOGY = "match.shape_topology"
    MATCH_COLOR_AND_SHAPE = "match.color_and_shape"
    MATCH_ORIENTATION = "match.orientation"
    MATCH_SCALE = "match.scale"
    MATCH_NORMALIZED_PATTERN = "match.pattern.normalized"
    MATCH_MARKER_OR_SUBCOMPONENT = "match.marker_or_subcomponent"
    MATCH_BY_EFFECT = "match.by_effect"
    MOVE_DEPENDENT_ONTO_TARGET = "goal.move_dependent_onto_target"
    MOVE_SOURCE_TO_MARKER = "goal.move_source_to_marker"
    QUANTITY_CAPACITY_TRANSFER = "goal.quantity_capacity_transfer"
    DISCRETE_STATE_OR_PERMUTATION_OFFSET = (
        "goal.discrete_state_or_permutation_offset"
    )
    CONNECTIVITY_TRAVERSABILITY_SUPPORT = (
        "goal.connectivity_traversability_support"
    )
    REACH_NEW_INFORMATIVE_ELEMENT = "goal.reach_new_informative_element"
    REACH_SCROLL_TERMINAL_FRONTIER = "goal.reach_scroll_terminal_frontier"
    OPEN_REQUIRED_PASSAGE = "goal.open_required_passage"
    MOVE_TOWARD_OPENED_PASSAGE = "goal.move_toward_opened_passage"
    PREREQUISITE_MECHANISM = "goal.prerequisite_mechanism"
    TEMPORAL_OCCUPANCY_OR_CONCURRENCY = "goal.temporal_occupancy_or_concurrency"
    SAFE_WORLD_REVELATION = "goal.safe_world_revelation"
    LEARNED_TERMINAL_CARDINALITY = "goal.learned_terminal_cardinality"
    CONJUNCTION = "goal.conjunction"
    SUBGOAL_ASSIGNMENT = "goal.subgoal_assignment"


class GoalNodeKind(str, Enum):
    """Generic node families for the declarative exploration graph."""

    OFFICIAL_COMPLETION = "official_completion"
    SAFETY_INVARIANT = "safety_invariant"
    WORLD_ACHIEVEMENT = "world_achievement"
    REQUIRED_PREREQUISITE = "required_prerequisite"
    ACCESS = "access"
    MECHANISM_DISCRIMINATION = "mechanism_discrimination"
    EFFECT_CALIBRATION = "effect_calibration"
    CONFIGURATION_DISCOVERY = "configuration_discovery"
    RECOVERY = "recovery"


class GoalEdgeKind(str, Enum):
    """Declared dependency/revision relations between goal nodes."""

    REQUIRES = "requires"
    ENABLES = "enables"
    REFINES = "refines"
    ALTERNATIVE_TO = "alternative_to"
    CONFLICTS_WITH = "conflicts_with"
    PROTECTS = "protects"
    RESTORES = "restores"
    FALSIFIES = "falsifies"
    ACHIEVED_BY = "achieved_by"


class ObligationKind(str, Enum):
    """Canonical generic meanings carried by the priority dependency graph."""

    TERMINAL = "terminal"
    ENABLING = "enabling"
    PRESERVATION = "preservation"
    INFORMATION = "information"
    REPAIR = "repair"


class ObligationStatus(str, Enum):
    """Lifecycle declared by DRM for one current obligation."""

    PROPOSED = "proposed"
    BLOCKED = "blocked"
    READY = "ready"
    PURSUING = "pursuing"
    AWAITING_EVIDENCE = "awaiting_evidence"
    SATISFIED = "satisfied"
    FALSIFIED = "falsified"
    INVALIDATED = "invalidated"


class ObligationEdgeKind(str, Enum):
    """Typed semantic relations between canonical obligations."""

    REQUIRES = "requires"
    PROVIDES = "provides"
    PRESERVES = "preserves"
    BLOCKS = "blocks"
    CONFLICTS_WITH = "conflicts_with"
    FALSIFIES = "falsifies"
    BENEFITS = "benefits"


class PrioritySafetyGate(str, Enum):
    """Small global safety machine preceding local objective selection."""

    CAUSAL_DEBT = "causal_debt"
    PROTECTION = "protection"
    COMMITMENT = "commitment"
    OBJECTIVE_FRONTIER = "objective_frontier"
    INFORMATION = "information"
    ABSTENTION = "abstention"


class PriorityShadowDisposition(str, Enum):
    """Declared explanation of a shadow comparison with legacy dispatch."""

    EXACT_MATCH = "exact_match"
    LEGACY_OBLIGATION_BLOCKED = "legacy_obligation_blocked"
    DECLARED_ALTERNATIVE = "declared_alternative"


class StaticKnowledgeNodeKind(str, Enum):
    """Generic, scene-independent knowledge-tree node families."""

    PRINCIPLE = "principle"
    QUESTION = "question"
    ACTIVATION_CONDITION = "activation_condition"
    EXPECTED_EFFECT = "expected_effect"
    FALSIFIER = "falsifier"
    DEPENDENCY = "dependency"
    PRIORITY = "priority"


class StaticKnowledgeEdgeKind(str, Enum):
    """Relations inside the reusable knowledge tree."""

    REFINES = "refines"
    REQUIRES = "requires"
    ACTIVATES = "activates"
    PREDICTS = "predicts"
    FALSIFIES = "falsifies"


class EffectSignatureNovelty(str, Enum):
    NONE = "none"
    KNOWN = "known"
    NOVEL = "novel"


class ObservableChangeScope(str, Enum):
    NONE = "none"
    RESOURCE_ONLY = "resource_only"
    KNOWN_CYCLIC_POSE_ONLY = "known_cyclic_pose_only"
    WORLD_SPATIAL_CONFIGURATION = "world_spatial_configuration"
    WORLD_NONSPATIAL_STATE = "world_nonspatial_state"
    GOAL_RELATION = "goal_relation"
    SCORE_OR_LEVEL_BOUNDARY = "score_or_level_boundary"
    MIXED = "mixed"


class DeclaredCausalClass(str, Enum):
    OBSERVABLE_KNOWN = "observable_known"
    EFFECTFUL_KNOWN = "effectful_known"
    EFFECTFUL_NOVEL = "effectful_novel"
    CONTEXTUAL_NO_EFFECT = "contextual_no_effect"


class GoalLifecycleStatus(str, Enum):
    PROPOSED = "proposed"
    ENABLED = "enabled"
    BLOCKED_BY_PREREQUISITE = "blocked_by_prerequisite"
    DISABLED_WITH_REASON = "disabled_with_reason"
    ACCESSIBLE = "accessible"
    PURSUING = "pursuing"
    LOCALLY_SATISFIED = "locally_satisfied"
    AXIS_SATISFIED = "axis_satisfied"
    AXIS_EXHAUSTED_MISSING_SUBGOAL = "axis_exhausted_missing_subgoal"
    FAILED_PARTIAL = "failed_partial"
    FALSIFIED = "falsified"
    ABANDONED_EXACT_VARIANT = "abandoned_exact_variant"
    REOPENED_AFTER_CONTEXT_CHANGE = "reopened_after_context_change"
    LEVEL_ACHIEVED = "level_achieved"


class PeriodicCycleWorkflowStatus(str, Enum):
    """Lifecycle of one observed periodic/cyclic mechanism workflow."""

    SUPPOSED = "supposed"
    IN_PROGRESS = "in_progress"
    CYCLE_UNDERSTOOD = "cycle_understood"


class MechanismGateStatus(str, Enum):
    COMMIT_REVISABLE_MECHANISM = "commit_revisable_mechanism"
    SELECT_DISCRIMINATING_EXPERIMENT = "select_discriminating_experiment"
    EXTEND_SCOPED_ONTOLOGY = "extend_scoped_ontology"
    FAIL_CLOSED_NAMED_GAP = "fail_closed_named_gap"


class DerivedGeometryStateStatus(str, Enum):
    AFFINE_RELATION_FALSIFIED = "affine_relation_falsified"
    AFFINE_AND_CLOSED_ORBIT_SUPPORTED = "affine_and_closed_orbit_supported"
    AFFINE_RELATION_SUPPORTED = "affine_relation_supported"
    CLOSED_ORBIT_SUPPORTED = "closed_orbit_supported"
    TRANSITION_ORDER_ONLY = "transition_order_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ViewportSupportTerrainStatus(str, Enum):
    INCOHERENT_VIEWPORT_OBSERVATIONS = "incoherent_viewport_observations"
    LAYERED_SUPPORT_SUSPENSION = "layered_support_suspension"
    DYNAMIC_TERRAIN_WITH_SUPPORT_CHANGE = "dynamic_terrain_with_support_change"
    DYNAMIC_TRAVERSABILITY = "dynamic_traversability"
    SUPPORT_CHANGE_ONLY = "support_change_only"
    COHERENT_VIEWPORT_RELATION = "coherent_viewport_relation"
    STATIC_CONFIGURATION = "static_configuration"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class InteractionTopologyStatus(str, Enum):
    CONNECTIVITY_CHANGE_WITH_VALID_LANDING = "connectivity_change_with_valid_landing"
    CONNECTIVITY_CHANGE = "connectivity_change"
    ADJACENCY_CHANGE_ONLY = "adjacency_change_only"
    VALID_LANDING_ONLY = "valid_landing_only"
    UNCHANGED_TOPOLOGY = "unchanged_topology"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EliminationViabilityStatus(str, Enum):
    DECLARED_TERMINAL_REACHED = "declared_terminal_reached"
    VIABLE_TO_DECLARED_TERMINAL = "viable_to_declared_terminal"
    EXACT_DEAD_END = "exact_dead_end"
    SEARCH_INCOMPLETE_BOUND = "search_incomplete_bound"


class TemporalConcurrencyStatus(str, Enum):
    ACTION_CONDITIONED_OCCUPANCY_CONFLICT = "action_conditioned_occupancy_conflict"
    ACTION_CONDITIONED_COEVOLUTION = "action_conditioned_coevolution"
    ACTION_CONDITIONED_DUPLICATION = "action_conditioned_duplication"
    ACTION_CONDITIONED_STATE_RETURN = "action_conditioned_state_return"
    ORDERED_TIMELINE_ONLY = "ordered_timeline_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GoalGapDimension(str, Enum):
    """Exact dimensions Python may measure only after DRM declares them."""

    ROW = "row"
    COLUMN = "column"
    POSITION = "position"
    OFFSET = "offset"
    DISTANCE = "distance"
    CONTACT = "contact"
    SEPARATION = "separation"
    OVERLAP = "overlap"
    CONTAINMENT = "containment"
    SHAPE = "shape"
    TOPOLOGY = "topology"
    COLOR = "color"
    SCALE = "scale"
    ROTATION = "rotation"
    NORMALIZED_PATTERN = "normalized_pattern"
    MARKER_CORRESPONDENCE = "marker_correspondence"
    QUANTITY = "quantity"
    CAPACITY = "capacity"
    CONSERVED_TRANSFER = "conserved_transfer"
    DISCRETE_STATE = "discrete_state"
    PERMUTATION_OFFSET = "permutation_offset"
    CONNECTIVITY = "connectivity"
    TRAVERSABILITY = "traversability"
    SUPPORT = "support"
    MECHANISM_STATE = "mechanism_state"
    TEMPORAL_OCCUPANCY = "temporal_occupancy"
    CONCURRENCY = "concurrency"
    WORLD_REVELATION = "world_revelation"
    CARDINALITY = "cardinality"
    CONJUNCTION_ASSIGNMENT = "conjunction_assignment"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EpistemicStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    ESTABLISHED = "established"
    DEFEATED = "defeated"


class Disposition(str, Enum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class Cardinality(str, Enum):
    SINGULAR = "singular"
    SPECIAL = "special"


class OperatorName(str, Enum):
    OBSERVE = "OBSERVE"
    ACT = "ACT"
    DISTINGUISH = "DISTINGUISH"
    RELATE = "RELATE"
    PROPOSE = "PROPOSE"
    DERIVE = "DERIVE"
    COMPARE = "COMPARE"
    UPDATE = "UPDATE"
    SELECT = "SELECT"


class EffectClass(str, Enum):
    PURE = "pure"
    OBSERVATION_ADAPTER = "observation_adapter"
    COGNITIVE_UPDATE_ADAPTER = "cognitive_update_adapter"
    ENVIRONMENT_ADAPTER = "environment_adapter"


@lru_cache(maxsize=1024)
def _canonical_mapping_keys(keys: frozenset[Any]) -> tuple[Any, ...]:
    """Sort a key *shape* once; values and serialized order remain unchanged."""

    return tuple(sorted(keys))


class FrozenMap(Mapping[str, Any]):
    """A recursively immutable JSON-like mapping with deterministic ordering."""

    __slots__ = ("_data", "_items", "_hash", "_digest")

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        if isinstance(value, FrozenMap):
            self._data = value._data
            self._items = value._items
            self._hash = getattr(value, "_hash", None)
            self._digest = getattr(value, "_digest", None)
            return
        raw = {} if value is None else value
        if not isinstance(raw, Mapping):
            raise TypeError("FrozenMap requires a mapping")
        data: dict[str, Any] = {}
        for key in _canonical_mapping_keys(frozenset(raw)):
            if not isinstance(key, str) or not key.strip():
                raise TypeError("FrozenMap keys must be non-empty strings")
            data[key] = freeze_json(raw[key])
        self._data = data
        self._items = tuple(data.items())
        self._hash: int | None = None
        self._digest: str | None = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def canonical_items(self) -> tuple[tuple[str, Any], ...]:
        """Return the immutable, already key-sorted entries without re-sorting."""

        return self._items

    def __eq__(self, other: object) -> bool:
        # Mapping.__eq__ copies both items views into temporary dictionaries.
        # Exact FrozenMap instances already own those immutable indexes. Keep
        # the Mapping protocol for foreign mappings and overriding subclasses.
        if type(self) is FrozenMap and type(other) is FrozenMap:
            return self._data == other._data
        return super().__eq__(other)

    def __hash__(self) -> int:
        # Hash the already frozen structure only when it is used as a key.
        # Canonical JSON remains reserved for a real serialization boundary.
        # Python's salted string hash is not used, so cache layout is stable
        # across processes. Equal numeric values (True/1/1.0) share a hash.
        cached = getattr(self, "_hash", None)
        if cached is None:
            digest = hashlib.sha256()
            _update_frozen_hash(digest, self, root=True)
            raw = digest.digest()
            value = int.from_bytes(raw[:8], byteorder="big", signed=True)
            cached = -2 if value == -1 else value
            self._hash = cached
        return cached


    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        return {key: thaw_json(value) for key, value in self._items}

    def to_shallow_dict(self) -> dict[str, Any]:
        """Copy only the mapping index while preserving immutable child values."""

        return dict(self._items)

    @classmethod
    def from_frozen_items(cls, value: Mapping[str, Any]) -> "FrozenMap":
        """Index values already recursively frozen by a trusted local walker."""

        data: dict[str, Any] = {}
        for key in _canonical_mapping_keys(frozenset(value)):
            if not isinstance(key, str) or not key.strip():
                raise TypeError("FrozenMap keys must be non-empty strings")
            data[key] = value[key]
        frozen = object.__new__(cls)
        frozen._data = data
        frozen._items = tuple(data.items())
        frozen._hash = None
        frozen._digest = None
        return frozen

    @classmethod
    def overlay(
        cls,
        primary: Mapping[str, Any],
        fallback: "FrozenMap",
    ) -> "FrozenMap":
        """Compose immutable facts without recursively refreezing shared context."""

        primary_frozen = primary if isinstance(primary, cls) else cls(primary)
        if primary_frozen is fallback or not primary_frozen:
            return fallback
        if not fallback:
            return primary_frozen
        # Reuse an immutable map only when every overwritten value is the
        # identical object. Equality alone would conflate JSON-distinct values
        # such as 1 and True, or 0.0 and -0.0.
        if all(
            key in fallback and value is fallback[key]
            for key, value in primary_frozen._items
        ):
            return fallback
        data = fallback.to_shallow_dict()
        data.update(primary_frozen.to_shallow_dict())
        combined = object.__new__(cls)
        combined._data = {key: data[key] for key in _canonical_mapping_keys(frozenset(data))}
        combined._items = tuple(combined._data.items())
        combined._hash = None
        combined._digest = None
        return combined

    @classmethod
    def _validate(cls, value: Any) -> "FrozenMap":
        if isinstance(value, cls):
            return value
        return cls(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.to_dict(),
                return_schema=core_schema.dict_schema(),
                when_used="always",
            ),
        )


@lru_cache(maxsize=512)
def _encoded_integer_tuple(value: tuple[int, ...]) -> bytes:
    """Bounded exact hash bytes for short immutable integer sequences."""
    parts = [b"T", len(value).to_bytes(8, "big")]
    for child in value:
        numerator = str(child).encode("ascii")
        parts.extend((b"N", len(numerator).to_bytes(8, "big"), numerator,
                      (1).to_bytes(8, "big"), b"1"))
    return b"".join(parts)


def _update_frozen_hash(digest: Any, value: Any, *, root: bool = False) -> None:
    """Feed one immutable value without encoding or sorting its whole tree."""

    if isinstance(value, FrozenMap):
        if not root:
            digest.update(b"M")
            digest.update(FrozenMap.__hash__(value).to_bytes(8, "big", signed=True))
            return
        digest.update(b"M")
        digest.update(len(value._items).to_bytes(8, "big"))
        for key, child in value._items:
            key_bytes = key.encode("utf-8")
            digest.update(len(key_bytes).to_bytes(8, "big"))
            digest.update(key_bytes)
            _update_frozen_hash(digest, child)
    elif isinstance(value, tuple):
        # Raster rows and coordinate vectors recur inside distinct immutable
        # facts. Reuse their exact existing encoding, without changing hashes.
        # At most 128 signed 64-bit integers per key and 512 cached keys.
        if len(value) <= 128 and all(
            type(child) is int and child.bit_length() <= 63 for child in value
        ):
            digest.update(_encoded_integer_tuple(value))
            return
        digest.update(b"T")
        digest.update(len(value).to_bytes(8, "big"))
        for child in value:
            _update_frozen_hash(digest, child)
    elif value is None:
        digest.update(b"Z")
    elif type(value) in {bool, int, float}:
        digest.update(b"N")
        numerator, denominator = (
            value.as_integer_ratio() if type(value) is float else (int(value), 1)
        )
        numerator_bytes = str(numerator).encode("ascii")
        denominator_bytes = str(denominator).encode("ascii")
        digest.update(len(numerator_bytes).to_bytes(8, "big"))
        digest.update(numerator_bytes)
        digest.update(len(denominator_bytes).to_bytes(8, "big"))
        digest.update(denominator_bytes)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    else:
        raise TypeError(f"unsupported frozen hash value: {type(value).__name__}")


_FROZEN_TUPLE_CACHE_LIMIT = 512
_FROZEN_TUPLE_CACHE: dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}


def freeze_json(value: Any) -> Any:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not canonical JSON values")
        return value
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Enum):
        return freeze_json(value.value)
    if value_type is tuple:
        identity = id(value)
        cached = _FROZEN_TUPLE_CACHE.get(identity)
        if cached is not None and cached[0] is value:
            return cached[1]
        # Exact scalar tuples are already canonical immutable JSON.  Avoid a
        # second walk through every pixel/coordinate when a validated raster
        # or scene model crosses another workflow binding boundary.
        if all(item is None or type(item) in {str, bool, int} for item in value):
            frozen = value
        else:
            frozen = tuple(freeze_json(item) for item in value)
        if len(_FROZEN_TUPLE_CACHE) >= _FROZEN_TUPLE_CACHE_LIMIT:
            _FROZEN_TUPLE_CACHE.pop(next(iter(_FROZEN_TUPLE_CACHE)))
        _FROZEN_TUPLE_CACHE[identity] = (value, frozen)
        return frozen
    if value_type is list:
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        return FrozenMap(value)
    raise TypeError(f"unsupported immutable JSON value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def immutable_json_data(value: Any) -> Any:
    """Project typed values without thawing already immutable payloads.

    This is an in-process representation, not an encoder. Canonical JSON is
    still produced at the archive boundary. Shared FrozenMap subtrees retain
    their exact identity; no traversal, sorting or copying of them is needed.
    """
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, FrozenModel):
        return FrozenMap.from_frozen_items({
            name: immutable_json_data(getattr(value, name))
            for name in value.__class__.model_fields
        })
    if isinstance(value, (tuple, list)):
        converted = tuple(immutable_json_data(item) for item in value)
        if type(value) is tuple and all(a is b for a, b in zip(value, converted)):
            return value
        return converted
    if isinstance(value, Mapping):
        return FrozenMap.from_frozen_items({
            key: immutable_json_data(item) for key, item in sorted(value.items())
        })
    return freeze_json(value)


def canonical_data(value: Any) -> Any:
    # Primitive leaves dominate frame and delta payloads.  Reject them before
    # Pydantic's metaclass-backed isinstance check, which is materially more
    # expensive and cannot match an exact built-in scalar type.
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, FrozenMap):
        return {key: canonical_data(item) for key, item in sorted(value.items())}
    if isinstance(value, BaseModel):
        # Frozen V5 models are already validated immutable values.  Reading
        # their declared fields directly avoids Pydantic materializing a second
        # deep object tree before this canonical walker immediately traverses
        # it again.  Field names and values are unchanged, so hashes remain
        # byte-for-byte compatible with the prior model_dump path.
        return {
            key: canonical_data(getattr(value, key))
            for key in value.__class__.model_fields
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        if all(
            item is None or isinstance(item, (str, bool, int, float))
            for item in value
        ):
            return value
        if value and all(
            isinstance(row, tuple)
            and all(
                item is None or isinstance(item, (str, bool, int, float))
                for item in row
            )
            for row in value
        ):
            return value
        return [canonical_data(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): canonical_data(value[key]) for key in sorted(value)}
    return value


class _CanonicalImmutableJsonEncoder(json.JSONEncoder):
    """Expose validated immutable storage directly to CPython's JSON encoder."""

    def default(self, value: Any) -> Any:
        if isinstance(value, FrozenMap):
            return value._data
        if isinstance(value, BaseModel):
            return {
                key: getattr(value, key)
                for key in value.__class__.model_fields
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {str(key): value[key] for key in sorted(value)}
        return super().default(value)


def canonical_json(value: Any) -> str:
    # Only exact scalar roots bypass the custom encoder.  The encoder handles
    # nested plain tuples/lists/dicts itself; recursively certifying them here
    # walked large immutable scene payloads once *before* every actual encode.
    if value is None or type(value) in {str, bool, int, float}:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return json.dumps(
        value,
        cls=_CanonicalImmutableJsonEncoder,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@lru_cache(maxsize=16384)
def _stable_immutable_digest(value: FrozenModel | FrozenMap) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1024)
def schema_ordered_digest(value: FrozenModel) -> str:
    """Hash one validated immutable contract through Pydantic's fast JSON path.

    This digest is for bounded runtime cache identity, not canonical archives or
    reducer state.  Frozen model field order and FrozenMap key order are stable,
    while Rust-side serialization avoids rebuilding a large Python object tree.
    """

    return hashlib.sha256(value.model_dump_json().encode("utf-8")).hexdigest()


def stable_digest(value: Any) -> str:
    if isinstance(value, FrozenMap):
        cached = getattr(value, "_digest", None)
        if cached is None:
            cached = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
            value._digest = cached
        return cached
    if isinstance(value, FrozenModel):
        return _stable_immutable_digest(value)
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_unique(values: tuple[Ref, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique references")
