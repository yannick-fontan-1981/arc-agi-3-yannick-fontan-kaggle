"""Exact raster variants between supplied, measured uniform extents."""
from collections import Counter
from fractions import Fraction


def measured_extent_variants(template, before_extent, after_extent, palette_signature, *, max_pixels):
    pixels = {(r, c): v for r, c, v in template}
    if not pixels or len(pixels) != len(template):
        return ()
    height = max(r for r, c in pixels) + 1
    width = max(c for r, c in pixels) + 1
    if min(r for r, c in pixels) != 0 or min(c for r, c in pixels) != 0:
        return ()
    if len(pixels) != height * width:
        return ()
    histogram = Counter(pixels.values())
    proportions = tuple(sorted((v, Fraction(n, len(pixels)).numerator,
        Fraction(n, len(pixels)).denominator) for v, n in histogram.items()))
    if proportions != tuple(tuple(x) for x in palette_signature):
        return ()
    before_extent, after_extent = tuple(before_extent), tuple(after_extent)
    if (height, width) not in (before_extent, after_extent):
        return ()
    target = after_extent if (height, width) == before_extent else before_extent
    th, tw = target
    if min(th, tw) <= 0 or th * width != tw * height or target == (height, width):
        return ()
    if th * tw > max_pixels:
        raise ValueError('observed extent template pixel bound exceeded')
    output = []
    for r in range(th):
        for c in range(tw):
            # Every source pixel touched by one destination pixel must agree.
            values = {pixels[sr, sc]
                for sr in range(r * height // th, ((r + 1) * height + th - 1) // th)
                for sc in range(c * width // tw, ((c + 1) * width + tw - 1) // tw)}
            if len(values) != 1:
                return ()
            output.append((r, c, next(iter(values))))
    return (tuple(output),)
