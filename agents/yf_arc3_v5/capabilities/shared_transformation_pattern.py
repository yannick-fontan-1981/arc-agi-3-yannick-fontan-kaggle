"""Bounded Source-declared projection of effects, without scene identities.

This is an observation pattern, not a causal verdict or a current action permit.
DRM owns its meaning, conditions and epistemic scope.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def effect_descriptor_fields(descriptor, profile):
    fields = tuple(profile["descriptor_fields"]) + tuple(field
        for field in profile.get("optional_descriptor_fields", ()) if field in descriptor)
    if len(fields) > profile["maximum_descriptor_fields"] or len(set(fields)) != len(fields):
        raise ValueError("invalid declared shared transformation projection")
    return fields


def measure_shared_transformation_pattern(*, action_ref, descriptors, profile):
    if len(descriptors) > profile["maximum_effects"]:
        raise ValueError("shared transformation effect bound exceeded; no ties dropped")
    if len(descriptors) != profile["required_effect_count"]:
        return FrozenMap({"shared_pattern_input_complete": False})
    fields = tuple(profile["descriptor_fields"])
    if len(fields) > profile["maximum_descriptor_fields"] or len(set(fields)) != len(fields):
        raise ValueError("invalid declared shared transformation projection")
    values = []
    for descriptor in descriptors:
        fields = effect_descriptor_fields(descriptor, profile)
        if descriptor.get("current_adjacent_cotranslation") is False:
            return FrozenMap({"shared_pattern_input_complete": False})
        if any(field not in descriptor for field in fields):
            return FrozenMap({"shared_pattern_input_complete": False})
        for field in profile["integer_fields"]:
            value = descriptor[field]
            if type(value) is not int or abs(value) > profile["maximum_integer_absolute_value"]:
                raise ValueError("shared transformation integer is invalid or unbounded")
        for field in profile["digest_fields"]:
            value = descriptor[field]
            if not isinstance(value, str) or not value or len(value) > profile["maximum_digest_length"]:
                raise ValueError("shared transformation digest is invalid or unbounded")
        for field in profile.get("palette_fields", ()):
            value = descriptor[field]
            palette = (value,) if type(value) is int else value
            if (not isinstance(palette, tuple) or not 1 <= len(palette) <= profile["maximum_palette_values"]
                    or any(type(item) is not int or abs(item) > profile["maximum_integer_absolute_value"] for item in palette)):
                raise ValueError("shared transformation palette is invalid or unbounded")
        values.append(FrozenMap({field: descriptor[field] for field in fields}))
    # Actor order is not a learned role. Retain multiplicities, no old handles.
    projected = tuple(sorted(values, key=stable_digest))
    return FrozenMap({"shared_pattern_input_complete": True,
        "shared_effect_descriptors": projected,
        "shared_pattern_token": stable_digest((action_ref, projected))})
