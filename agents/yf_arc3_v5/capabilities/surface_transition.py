"""Bounded raster comparisons on two explicitly supplied rectangular supports."""
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_surface_transition(*, before, packet, prior_frames, reference_box,
                               output_box, maximum_pixels=4096, maximum_frames=256):
    if len(packet) > maximum_frames or len(prior_frames) > maximum_frames:
        raise ValueError('surface observation packet bound exceeded')
    boxes = (reference_box, output_box)
    if any(len(box) != 4 or any(type(n) is not int for n in box) for box in boxes):
        raise ValueError('surface bounds must be integer rectangles')
    shapes = tuple((b[2]-b[0]+1, b[3]-b[1]+1) for b in boxes)
    if shapes[0] != shapes[1] or any(n <= 0 for n in shapes[0]):
        raise ValueError('surface extents must be equal and nonempty')
    if shapes[0][0]*shapes[0][1] > maximum_pixels:
        raise ValueError('surface pixel bound exceeded')

    def crop(frame, box):
        top, left, bottom, right = box
        if (not frame or top < 0 or left < 0 or bottom >= len(frame)
                or any(right >= len(frame[y]) for y in range(top, bottom+1))):
            raise ValueError('surface lies outside observed frame')
        return tuple(tuple(frame[y][left:right+1]) for y in range(top, bottom+1))

    reference = crop(before, reference_box)
    initial = crop(before, output_box)
    reference_unchanged = all(crop(frame, reference_box) == reference for frame in packet)
    outputs = tuple(crop(frame, output_box) for frame in packet)
    suffix = 0
    for output in reversed(outputs):
        if output != reference:
            break
        suffix += 1
    final = outputs[-1] if outputs else initial
    changed = tuple((y, x) for y, row in enumerate(initial) for x, value in enumerate(row)
                    if final[y][x] != value)
    prior_reference_unchanged = all(crop(frame, reference_box) == reference for frame in prior_frames)
    return FrozenMap({
        'reference_unchanged': reference_unchanged,
        'before_mismatch_count': sum(a != b for ra, rb in zip(initial, reference) for a, b in zip(ra, rb)),
        'terminal_mismatch_count': sum(a != b for ra, rb in zip(final, reference) for a, b in zip(ra, rb)),
        'matching_packet_suffix_count': suffix,
        'changed_pixel_count': len(changed),
        'changed_to_values': tuple(sorted({final[y][x] for y, x in changed})),
        'changed_support_lower_bound': changed,
        'surface_height': shapes[0][0], 'surface_width': shapes[0][1],
        'prior_reference_unchanged': prior_reference_unchanged,
        'prior_matching_observation_count': sum(crop(frame, output_box) == reference for frame in prior_frames),
    })
