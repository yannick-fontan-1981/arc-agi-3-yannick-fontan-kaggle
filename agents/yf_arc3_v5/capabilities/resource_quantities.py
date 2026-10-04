"""Exact integer intervals for supplied finite, quantized render models.

No resource role, model choice, budget commitment or action decision is made
here. Model descriptions and their references come from declarative authority.
"""
from __future__ import annotations

from collections.abc import Mapping

from agents.yf_arc3_v5.capabilities.contracts import (
    ResourceGaugeQuantityAnalysis, ResourceGaugeQuantityAnalysisInput,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def refine_rendered_budget_interval(
    *, extent: int, ordinal: int, displayed: int, rounding: str,
    minimum: int = 1, maximum: int | None = None,
) -> tuple[int, int | None] | None:
    """Invert the supplied clipped full-scale model without enumerating budgets.

    Display = rounding(extent * max(B - ordinal, 0) / B), B positive integer.
    Integer inequalities preserve exact tie behavior and unbounded intervals.
    """
    if extent < 1 or ordinal < 0 or not 0 <= displayed <= extent or minimum < 1:
        raise ValueError("invalid rendered-quantity measurement")
    if rounding == "floor":
        factor, lower, upper, lower_strict, upper_strict = 1, displayed, displayed + 1, False, True
    elif rounding == "ceil":
        factor, lower, upper, lower_strict, upper_strict = 1, displayed - 1, displayed, True, False
    elif rounding == "nearest_even":
        factor, lower, upper = 2, 2 * displayed - 1, 2 * displayed + 1
        lower_strict = upper_strict = bool(displayed % 2)
    else:
        raise ValueError("unsupported supplied rounding model")
    numerator = factor * extent * ordinal
    if displayed > 0:
        coefficient = factor * extent - lower
        if coefficient == 0:
            if numerator > 0 or lower_strict:
                return None
        else:
            lower_budget = (numerator // coefficient + 1 if lower_strict
                            else _ceil_div(numerator, coefficient))
            minimum = max(minimum, lower_budget)
    if displayed < extent:
        coefficient = factor * extent - upper
        if coefficient == 0:
            if numerator == 0 and upper_strict:
                return None
        else:
            upper_budget = (_ceil_div(numerator, coefficient) - 1 if upper_strict
                            else numerator // coefficient)
            maximum = upper_budget if maximum is None else min(maximum, upper_budget)
    if maximum is not None and maximum < minimum:
        return None
    return minimum, maximum


def refine_rendered_budget_models(
    *, extent: int, observations: tuple[tuple[int, int], ...],
    profiles: tuple[Mapping[str, object], ...],
    prior_intervals: tuple[Mapping[str, object], ...] | None = None,
    cumulative_cost_samples: Mapping[str, Mapping[int, int]] | None = None,
) -> tuple[dict[str, object], ...]:
    """Refine at most six supplied model intervals; never choose a model.

    None means no earlier calibration. An explicitly empty interval set has no
    surviving profile and never silently restarts a falsified description.
    An extent-multiple budget expresses one displayed quantum per integer
    period. A declared cumulative sum uses supplied measured costs, never an
    inferred primitive count. Its remainder is quantity, not actions.
    """
    if not 1 <= len(profiles) <= 6 or len({str(p['ref']) for p in profiles}) != len(profiles):
        raise ValueError("rendering descriptions must be unique and bounded to six")
    prior = {str(row['profile_ref']):row for row in (prior_intervals or ())}
    records: list[dict[str, object]] = []
    for profile in profiles:
        ref = str(profile['ref'])
        domain = str(profile.get('budget_domain', 'positive_integer'))
        basis = str(profile.get('consumption_basis', 'primitive_count'))
        if domain not in ('positive_integer', 'extent_multiple'):
            raise ValueError('unsupported supplied quantity domain')
        if basis not in ('primitive_count', 'declared_cumulative_sum'):
            raise ValueError('unsupported supplied consumption basis')
        cost_samples = (cumulative_cost_samples or {}).get(ref)
        if basis == 'declared_cumulative_sum' and (
                cost_samples is None or any(n not in cost_samples for n, _ in observations)):
            raise ValueError('cumulative-cost model requires every observed measured cost sum')
        if prior_intervals is not None and ref not in prior:
            continue
        old = prior.get(ref)
        interval: tuple[int, int | None] | None = (
            (int(old['minimum_total_budget']), old['maximum_total_budget']) if old is not None else (1, None)
        )
        for ordinal, displayed in observations:
            assert interval is not None
            consumed = ordinal if basis == 'primitive_count' else int(cost_samples[ordinal])
            interval = refine_rendered_budget_interval(
                extent=extent, ordinal=consumed, displayed=displayed, rounding=str(profile['rounding']),
                minimum=interval[0], maximum=interval[1],
            )
            if interval is not None and domain == 'extent_multiple':
                low = extent * _ceil_div(interval[0], extent)
                high = None if interval[1] is None else extent * (interval[1] // extent)
                interval = None if high is not None and low > high else (low, high)
            if interval is None:
                break
        if interval is not None:
            records.append({'profile_ref':ref, 'rounding':str(profile['rounding']),
                            'consumption_basis':basis, 'budget_domain':domain,
                            'budget_stride':extent if domain == 'extent_multiple' else 1,
                            'minimum_debit_period':interval[0] // extent if domain == 'extent_multiple' else None,
                            'maximum_debit_period':interval[1] // extent if domain == 'extent_multiple' and interval[1] is not None else None,
                            'minimum_total_budget':interval[0], 'maximum_total_budget':interval[1]})
    return tuple(records)


def analyze_resource_gauge_quantities(value: ResourceGaugeQuantityAnalysisInput) -> ResourceGaugeQuantityAnalysis:
    """Measure and refine a source-supplied render hypothesis from real pixels.

    Unchanged raster samples count at their actual primitive ordinal. Only a
    measured full-axis rectangle or its canonical prior geometry supplies scale.
    Interpretation and the attestation of the resource role remain DRM-owned.
    """
    evidence = tuple(item for item in (*value.prior_evidence, *value.current_evidence)
                     if item.evidence_scope_ref == value.evidence_scope_ref)
    current_refs = frozenset(item.indicator_entity_ref for item in value.current_evidence)
    prior_refs = frozenset(str(row.get('indicator_entity_ref') or '') for row in value.prior_render_records)
    refs = current_refs or prior_refs
    ref = next(iter(sorted(refs))) if len(refs) == 1 else ''
    related = tuple(item for item in evidence if item.indicator_entity_ref == ref)
    prior_rows = tuple(row for row in value.prior_render_records
                       if row.get('indicator_entity_ref') == ref and row.get('evidence_scope_ref') == value.evidence_scope_ref)
    prior = prior_rows[0] if len(prior_rows) == 1 else None
    tracked = tuple(item for item in value.before_tracking.entities if item.entity_id == ref)
    full_boxes = tuple(item.render_extent_bbox for item in related if item.render_extent_bbox is not None)
    box_tuple = prior.get('render_extent_bbox') if prior is not None else None
    if box_tuple is None and full_boxes:
        b = full_boxes[0]
        box_tuple = (b.top,b.left,b.bottom,b.right)
    complete = bool(ref and len(tracked) == 1 and box_tuple is not None)
    extent = thickness = displayed = 0
    intervals: tuple[dict[str, object], ...] = ()
    if complete:
        top,left,bottom,right = map(int,box_tuple)
        complete = bool(0 <= top <= bottom < value.after.height and 0 <= left <= right < value.after.width)
        if complete:
            horizontal = right-left >= bottom-top
            extent,thickness = ((right-left+1,bottom-top+1) if horizontal else (bottom-top+1,right-left+1))
            component_box = tracked[0].component.bbox
            complete = (value.before.height == value.after.height and value.before.width == value.after.width
                        and ((left == 0 and right == value.after.width-1 and component_box.top == top and component_box.bottom == bottom)
                             if horizontal else (top == 0 and bottom == value.after.height-1 and component_box.left == left and component_box.right == right)))
            units = []
            pigment = tracked[0].component.value
            for index in range(extent):
                samples = (tuple(value.after.rows[row][left+index] for row in range(top,bottom+1))
                           if horizontal else tuple(value.after.rows[top+index][col] for col in range(left,right+1)))
                count = sum(pixel == pigment for pixel in samples)
                if count not in (0,thickness):
                    complete = False
                    break
                if count:
                    units.append(index)
            displayed = len(units)
            complete = complete and (units == list(range(displayed)) or units == list(range(extent-displayed,extent)))
            if complete:
                previous_intervals = tuple(prior.get('render_fit_intervals') or ()) if prior is not None else None
                observations = ((value.primitive_count_after,displayed),) if prior is not None else tuple(sorted(set(
                    (item.primitive_count_after,item.after_pixel_count // thickness)
                    for item in related if item.primitive_count_after is not None
                    and item.render_pixel_thickness == thickness and item.after_pixel_count % thickness == 0
                )))
                observations = tuple(sorted(set((*observations,(value.primitive_count_after,displayed)))))
                intervals = refine_rendered_budget_models(
                    extent=extent,observations=observations,profiles=value.rendering_profiles,prior_intervals=previous_intervals)
    minimum = min((int(row['minimum_total_budget']) for row in intervals),default=None)
    maximum = (max(int(row['maximum_total_budget']) for row in intervals)
               if intervals and all(row['maximum_total_budget'] is not None for row in intervals) else None)
    facts = FrozenMap({
        'indicator_entity_ref':ref, 'evidence_scope_ref':value.evidence_scope_ref,
        'transition_ref':value.transition_ref, 'role_selection_ref':value.role_selection_ref,
        'prior_role_selection_ref':prior.get('resource_role_selection_ref') if prior is not None else None,
        'render_observation_complete':complete, 'render_extent_bbox':box_tuple,
        'render_extent_units':extent, 'render_pixel_thickness':thickness,
        'primitive_count_after':value.primitive_count_after, 'displayed_extent_units':displayed,
        'render_fit_intervals':intervals, 'fitting_profile_count':len(intervals),
        'minimum_total_budget':minimum, 'maximum_total_budget':maximum,
        'minimum_remaining_action_units':max(0,minimum-value.primitive_count_after) if minimum is not None else None,
        'maximum_remaining_action_units':max(0,maximum-value.primitive_count_after) if maximum is not None else None,
        'unique_total_budget':minimum if minimum is not None and minimum == maximum else None,
        'unique_remaining_action_units':max(0,minimum-value.primitive_count_after) if minimum is not None and minimum == maximum else None,
    })
    alternative_facts = {}
    for alternative_ref in value.alternative_refs:
        transported = dict(facts)
        transported['alternative_ref'] = alternative_ref
        alternative_facts[alternative_ref] = FrozenMap(transported)
    return ResourceGaugeQuantityAnalysis(alternative_refs=value.alternative_refs,
        alternative_facts=FrozenMap(alternative_facts),
        evidence_refs=tuple(dict.fromkeys((*(item.transition_ref for item in related),value.transition_ref))))
