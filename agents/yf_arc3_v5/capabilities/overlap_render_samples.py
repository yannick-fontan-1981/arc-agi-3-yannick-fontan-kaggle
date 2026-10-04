"""Bounded local raster samples under supplied, still conditional object masks.

No drawing order, indicator role or action permission is inferred. A sample can
be translated only under exact equality of its local supports and input values.
DRM owns the interpretation and SRC owns any subsequent use of a prediction.
"""

from collections import Counter

from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_overlap_render_samples(*, layers, rendered_rows, maximum_regions,
                                  maximum_support_pixels):
    if not 1 <= maximum_regions <= 256 or not 1 <= maximum_support_pixels <= 8192:
        raise ValueError("invalid overlap raster sample bound")
    if len(layers) > 16:
        raise ValueError("overlap raster layer bound exceeded")
    supports = tuple(frozenset(layer["component_geometry"]["pixels"]) for layer in layers)
    if sum(map(len, supports)) > maximum_support_pixels:
        raise ValueError("overlap raster support bound exceeded")
    height, width = len(rendered_rows), len(rendered_rows[0])
    samples = []
    for first in range(len(layers)):
        for second in range(first + 1, len(layers)):
            pending = set(supports[first] & supports[second])
            while pending:
                seed = min(pending)
                pending.remove(seed)
                region, frontier = {seed}, [seed]
                while frontier:
                    row, col = frontier.pop()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            neighbor = (row + dr, col + dc)
                            if neighbor in pending:
                                pending.remove(neighbor)
                                region.add(neighbor)
                                frontier.append(neighbor)
                if len(samples) >= maximum_regions:
                    raise ValueError("overlap raster region bound exceeded; no sample dropped")
                if any(not (0 <= row < height and 0 <= col < width) for row, col in region):
                    raise ValueError("overlap raster region extends outside observation")
                top = min(row for row, _ in region)
                left = min(col for _, col in region)
                bottom = max(row for row, _ in region)
                right = max(col for _, col in region)
                relative = tuple(sorted((row - top, col - left) for row, col in region))
                local_supports = tuple(tuple(sorted((row - top, col - left)
                    for row, col in support if top <= row <= bottom and left <= col <= right))
                    for support in (supports[first], supports[second]))
                rendered = tuple((row - top, col - left, rendered_rows[row][col])
                    for row, col in sorted(region))
                values = tuple(layers[index]["component_geometry"]["value"] for index in (first, second))
                counts = tuple(sorted(Counter(value for _, _, value in rendered).items()))
                third_refs = tuple(layers[index]["component_geometry"]["component_id"]
                    for index, support in enumerate(supports)
                    if index not in (first, second) and region & support)
                samples.append(FrozenMap({
                    "input_component_refs": tuple(layers[index]["component_geometry"]["component_id"]
                        for index in (first, second)),
                    "input_entity_refs": tuple(layers[index]["prior_entity_ref"] for index in (first, second)),
                    "input_values": values,
                    "origin": (top, left), "extent": (bottom - top + 1, right - left + 1),
                    "relative_overlap_pixels": relative, "local_input_supports": local_supports,
                    "rendered_relative_pixels": rendered, "rendered_value_counts": counts,
                    "render_equals_first_value": len(counts) == 1 and counts[0][0] == values[0],
                    "render_equals_second_value": len(counts) == 1 and counts[0][0] == values[1],
                    "render_is_single_value": len(counts) == 1,
                    "render_uses_only_input_values": all(value in values for value, _ in counts),
                    "other_intersecting_component_refs": third_refs,
                }))
    return tuple(samples)


def project_exact_overlap_sample(*, sample, input_values, local_input_supports,
                                relative_overlap_pixels, origin):
    """Apply a caller-supplied sample, never choose a sample or a mechanism."""
    comparable = (sample["input_values"] == tuple(input_values)
        and sample["local_input_supports"] == tuple(tuple(sorted(support)) for support in local_input_supports)
        and sample["relative_overlap_pixels"] == tuple(sorted(relative_overlap_pixels))
        and not sample["other_intersecting_component_refs"])
    return FrozenMap({"exact_sample_input_equal": comparable,
        "projected_rendered_pixels": tuple((row + origin[0], col + origin[1], value)
            for row, col, value in sample["rendered_relative_pixels"]) if comparable else ()})
