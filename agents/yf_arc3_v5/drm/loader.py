"""Strict JSON-backed loader for human-readable `.drm` documents."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from agents.yf_arc3_v5.drm.contracts import DrmDocument, LoadedDrmDocument
from agents.yf_arc3_v5.logos.types import stable_digest


class DrmLoadError(ValueError):
    """A DRM source is unreadable or violates the typed document contract."""


_SOURCE_CACHE_ENTRIES = 32
_MAX_CACHEABLE_SOURCE_BYTES = 4 * 1024 * 1024


def load_drm_document(path: Path) -> LoadedDrmDocument:
    try:
        # Always read current bytes: path, size and mtime cannot attest content.
        source = path.read_bytes()
        if len(source) <= _MAX_CACHEABLE_SOURCE_BYTES:
            return _cached_drm_source(source)
        return _parse_drm_source(source)
    except Exception as error:
        raise DrmLoadError(f"invalid DRM document {path}: {error}") from error


@lru_cache(maxsize=_SOURCE_CACHE_ENTRIES)
def _cached_drm_source(source: bytes) -> LoadedDrmDocument:
    # Models/maps are recursively immutable. No runtime facts or decisions enter
    # this cache; exact byte keys also preserve source-hash distinctions.
    return _parse_drm_source(source)


def _parse_drm_source(source: bytes) -> LoadedDrmDocument:
    raw = json.loads(source.decode("utf-8"))
    document = DrmDocument.model_validate(raw)
    return LoadedDrmDocument(
        document=document,
        source_unit=f"drm.{document.module_id}",
        source_hash=hashlib.sha256(source).hexdigest(),
        semantic_hash=stable_digest(document),
    )


def production_drm_paths() -> tuple[Path, ...]:
    root = Path(__file__).parent
    return tuple(
        root / name
        for name in (
            "arc3.drm",
            "geometry.drm",
            "interaction_families.drm",
            "interaction.drm",
            "inverse_anchor_signature.drm",
            "inverse_anchor_signature_extensions.drm",
            "articulated_channel_kinematics.drm",
            "extent_commands.drm",
            "carried_reference_goals.drm",
            "observed_axis_configuration.drm",
            "typed_recursive_program.drm",
            "recipe_operator_compilation.drm",
            "piercing_rooted_order.drm",
            "ordered_relief_writing.drm",
            "reconfigurable_support.drm",
            "directed_accretive_flow_extensions.drm",
            "local_field_type_production_extensions.drm",
            "mechanism_application_questions.drm",
            "consumptive_type_dependency.drm",
            "local_field_inventory.drm",
            "resources.drm",
            "display_correspondence.drm",
            "announced_motion.drm",
            "alignment.drm",
            "transport.drm",
            "transformation_coupling.drm",
            "oriented_transport.drm",
            "ontology.drm",
            "morphology.drm",
            "perceptual_memory.drm",
            "winning_navigation.drm",
            "rigid_marker_pose.drm",
            "metaplanning.drm",
            "conditional_reasoning_tree.drm",
            "reasoning_tree_v11_todos.drm",
            "dynamic_workflow.drm",
            "priority_graph.drm",
            "functional_intent.drm",
            "abduction.drm",
            "retrodiction.drm",
            "retrospection.drm",
            "temporal_replay.drm",
            "symbolic_editing.drm",
        )
    )


def load_production_drm() -> tuple[LoadedDrmDocument, ...]:
    return tuple(load_drm_document(path) for path in production_drm_paths())
