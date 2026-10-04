"""Bounded exact geometry of spaced concentric rectangular components.

No palette has a role. These are measurements, not instruction detections.
DRM must decide whether a unique geometry licenses an interaction experiment.
"""

from dataclasses import dataclass
from functools import lru_cache
from collections import Counter

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid

MAX_AREA = 4096
MAX_COMPONENTS = 512
MAX_FRAMES = 32


class ConcentricGeometryBoundExceeded(ValueError):
    """A partial inventory cannot be used to establish singleton geometry."""


@dataclass(frozen=True)
class ConcentricFrame:
    outer_ref: str
    outer_value: int
    top: int
    left: int
    bottom: int
    right: int
    inner_refs: tuple[str, ...]
    maximum_equal_padding: int

    @property
    def center(self):
        return ((self.top + self.bottom) // 2, (self.left + self.right) // 2)

    @property
    def identity(self):
        return (self.top, self.left, self.bottom, self.right, self.outer_value)


@lru_cache(maxsize=8)
def measure_concentric_frames(frame: FrameGrid) -> tuple[ConcentricFrame, ...]:
    if frame.height * frame.width > MAX_AREA:
        raise ConcentricGeometryBoundExceeded("concentric frame raster area exceeds bound")
    components = detect_components(ComponentExtractionInput(frame=frame)).components
    if len(components) > MAX_COMPONENTS:
        raise ConcentricGeometryBoundExceeded("concentric frame component count exceeds bound")
    result = []
    for outer in components:
        box = outer.bbox
        if min(box.height, box.width) < 3:
            continue
        if outer.area != 2 * (box.height + box.width) - 4:
            continue
        if not all(r in (0, box.height - 1) or c in (0, box.width - 1)
                   for r, c in outer.relative_pixels):
            continue
        inners = []
        paddings = []
        for inner in components:
            ib = inner.bbox
            # A singleton point has no two-dimensional extent to frame.
            if min(ib.height, ib.width) < 2:
                continue
            gaps = (ib.top - box.top, ib.left - box.left,
                    box.bottom - ib.bottom, box.right - ib.right)
            # Equal offsets give exact concentricity and one or more raster
            # layers between the outer perimeter and the inner component.
            if min(gaps) > 0 and len(set(gaps)) == 1:
                inners.append(inner.component_id)
                paddings.append(gaps[0])
        if inners:
            result.append(ConcentricFrame(outer.component_id, outer.value, box.top, box.left,
                                          box.bottom, box.right, tuple(inners), max(paddings)))
            if len(result) > MAX_FRAMES:
                # A dense but otherwise valid raster can contain more framed
                # glyphs than the bounded symbolic contract can transport.
                # Discard the partial inventory so it cannot establish a false
                # singleton, while keeping unrelated perception operational.
                return ()
    return tuple(result)


def measure_spaced_concentric_frames(frame: FrameGrid) -> tuple[ConcentricFrame, ...]:
    return tuple(f for f in measure_concentric_frames(frame) if f.maximum_equal_padding > 1)


def advance_concentric_frame_additions(before, after, prior=()):
    before_ids = {f.identity for f in measure_concentric_frames(before)}
    return tuple(f.identity for f in measure_concentric_frames(after)
                 if f.identity not in before_ids or f.identity in prior)


def _nonbackground_runs(row, background_value):
    runs = []
    start = None
    for col, value in enumerate(row):
        if value != background_value and start is None:
            start = col
        elif value == background_value and start is not None:
            runs.append((start, col - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return tuple(runs)


def _run_containing(runs, col):
    return next((run for run in runs if run[0] <= col <= run[1]), None)


def measure_concentric_context(frames, additions, white_value, frame=None):
    added_tight = tuple(
        f
        for f in frames
        if f.maximum_equal_padding == 1 and f.identity in additions
    )
    geometry = {
        "added_tight_rect_wider_lower_span_present": False,
        "added_tight_rect_distinct_horizontal_peer_span_count": 0,
        "added_tight_rect_wider_horizontal_peer_span_count": 0,
        "added_tight_rect_narrower_horizontal_peer_span_count": 0,
    }
    if frame is not None and len(added_tight) == 1:
        tight = added_tight[0]
        background = Counter(value for row in frame.rows for value in row).most_common(1)[0][0]
        width = tight.right - tight.left + 1
        center_row, center_col = tight.center
        center_runs = _nonbackground_runs(frame.rows[center_row], background)
        framed_run = _run_containing(center_runs, center_col)
        peer_runs = tuple(run for run in center_runs if run != framed_run)
        lower_row = tight.bottom + 1
        lower_run = (
            _run_containing(
                _nonbackground_runs(frame.rows[lower_row], background), center_col
            )
            if lower_row < frame.height
            else None
        )
        peer_widths = tuple(right - left + 1 for left, right in peer_runs)
        geometry = {
            "added_tight_rect_wider_lower_span_present": bool(
                lower_run is not None
                and lower_run[0] > 0
                and lower_run[1] < frame.width - 1
                and lower_run[1] - lower_run[0] + 1 > width
            ),
            "added_tight_rect_distinct_horizontal_peer_span_count": len(peer_runs),
            "added_tight_rect_wider_horizontal_peer_span_count": sum(
                peer_width > width for peer_width in peer_widths
            ),
            "added_tight_rect_narrower_horizontal_peer_span_count": sum(
                peer_width < width for peer_width in peer_widths
            ),
        }
    return {
        "spaced_rect_count": sum(f.maximum_equal_padding > 1 for f in frames),
        "spaced_rect_palette_white_count": sum(f.maximum_equal_padding > 1 and f.outer_value == white_value for f in frames),
        "added_spaced_rect_count": sum(f.maximum_equal_padding > 1 and f.identity in additions for f in frames),
        "added_tight_rect_count": sum(
            f.maximum_equal_padding == 1 and f.identity in additions for f in frames
        ),
        **geometry,
    }


def measure_concentric_candidate(
    frames, candidate, additions=(), white_value=0, frame=None
):
    point = candidate.point
    added_tight = tuple(
        f
        for f in frames
        if f.maximum_equal_padding == 1 and f.identity in additions
    )
    local_span_width = 0
    local_span_is_unique_widest_distinct_peer = False
    if frame is not None and len(added_tight) == 1 and point is not None:
        background = Counter(value for row in frame.rows for value in row).most_common(1)[0][0]
        row, col = point
        if 0 <= row < frame.height:
            run = _run_containing(_nonbackground_runs(frame.rows[row], background), col)
            if run is not None:
                local_span_width = run[1] - run[0] + 1
                tight = added_tight[0]
                tight_run = _run_containing(
                    _nonbackground_runs(frame.rows[row], background),
                    tight.center[1],
                )
                peer_runs = tuple(
                    peer
                    for peer in _nonbackground_runs(frame.rows[row], background)
                    if peer != tight_run
                )
                peer_widths = tuple(
                    right - left + 1 for left, right in peer_runs
                )
                local_span_is_unique_widest_distinct_peer = bool(
                    peer_widths
                    and local_span_width == max(peer_widths)
                    and peer_widths.count(local_span_width) == 1
                )
    tight_width = (
        added_tight[0].right - added_tight[0].left + 1
        if len(added_tight) == 1
        else 0
    )
    return {
        "spaced_rect_point_center_count": sum(point == f.center and f.maximum_equal_padding > 1 for f in frames),
        "spaced_rect_point_palette_white_count": sum(point == f.center and f.maximum_equal_padding > 1 and f.outer_value == white_value for f in frames),
        "tight_rect_point_palette_white_count": sum(point == f.center and f.maximum_equal_padding == 1 and f.outer_value == white_value for f in frames),
        "added_spaced_rect_point_center_count": sum(point == f.center and f.maximum_equal_padding > 1 and f.identity in additions for f in frames),
        "added_tight_rect_point_center_count": sum(
            point == f.center for f in added_tight
        ),
        # Pure geometry only: DRM decides whether a unique action-conditioned
        # tight frame is selection, feedback, or decoration.  A destination
        # experiment may then use a distinct point in the same horizontal band.
        "unique_added_tight_rect_distinct_horizontal_peer": (
            len(added_tight) == 1
            and point is not None
            and abs(point[0] - added_tight[0].center[0])
            < (added_tight[0].bottom - added_tight[0].top + 1)
            and abs(point[1] - added_tight[0].center[1])
            > (added_tight[0].right - added_tight[0].left + 1)
        ),
        "candidate_local_horizontal_span_width": local_span_width,
        "candidate_local_horizontal_span_is_narrower_than_added_tight_rect": bool(
            local_span_width > 0 and tight_width > 0 and local_span_width < tight_width
        ),
        "candidate_local_horizontal_span_is_unique_widest_distinct_peer": (
            local_span_is_unique_widest_distinct_peer
        ),
    }
