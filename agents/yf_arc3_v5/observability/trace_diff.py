"""Small semantic-prefix comparison for compact V5 SRL traces."""

from __future__ import annotations

import re
from collections.abc import Sequence

from agents.yf_arc3_v5.logos.types import FrozenMap
from agents.yf_arc3_v5.observability.models import (
    CompactSrlLine,
    FirstDivergence,
    TraceDiffResult,
)

_SRL_PATTERN = re.compile(
    r"^(?:\[(?P<ref>#[0-9]+)\]\s+)?(?P<keyword>[A-Z_]+)\s+(?P<statement>.+)$"
)


def diff_compact_traces(
    expected: Sequence[CompactSrlLine | str],
    actual: Sequence[CompactSrlLine | str],
) -> TraceDiffResult:
    expected_rows = tuple(_row(item) for item in expected)
    actual_rows = tuple(_row(item) for item in actual)
    limit = min(len(expected_rows), len(actual_rows))
    matched = 0
    for index in range(limit):
        if _fingerprint(expected_rows[index]) != _fingerprint(actual_rows[index]):
            return TraceDiffResult(
                equivalent=False,
                expected_count=len(expected_rows),
                actual_count=len(actual_rows),
                matched_prefix_count=matched,
                first_divergence=FirstDivergence(
                    index=index,
                    classification=_classify(
                        expected_rows[index],
                        actual_rows[index],
                    ),
                    expected=expected_rows[index],
                    actual=actual_rows[index],
                    previous_match=(expected_rows[index - 1] if index else None),
                ),
            )
        matched += 1
    if len(expected_rows) != len(actual_rows):
        index = limit
        return TraceDiffResult(
            equivalent=False,
            expected_count=len(expected_rows),
            actual_count=len(actual_rows),
            matched_prefix_count=matched,
            first_divergence=FirstDivergence(
                index=index,
                classification=(
                    "missing_actual_event"
                    if len(expected_rows) > len(actual_rows)
                    else "unexpected_actual_event"
                ),
                expected=(expected_rows[index] if index < len(expected_rows) else None),
                actual=(actual_rows[index] if index < len(actual_rows) else None),
                previous_match=(expected_rows[index - 1] if index else None),
            ),
        )
    return TraceDiffResult(
        equivalent=True,
        expected_count=len(expected_rows),
        actual_count=len(actual_rows),
        matched_prefix_count=matched,
    )


def parse_compact_srl(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("# ")
    )


def _row(item: CompactSrlLine | str) -> FrozenMap:
    if isinstance(item, CompactSrlLine):
        return FrozenMap(
            {
                "evidence_ref": item.evidence_ref,
                "keyword": item.keyword,
                "statement": item.statement,
                "record_kind": item.record_kind,
                "workflow_step_id": item.workflow_step_id,
                "frame_id": item.frame_id,
                "source_hash": item.source_hash,
            }
        )
    match = _SRL_PATTERN.match(item.strip())
    if match is None:
        return FrozenMap(
            {
                "keyword": "INVALID",
                "statement": item.strip() or "empty_line",
            }
        )
    return FrozenMap(
        {
            "evidence_ref": match.group("ref"),
            "keyword": match.group("keyword"),
            "statement": match.group("statement"),
        }
    )


def _fingerprint(row: FrozenMap) -> tuple[str, str]:
    return str(row.get("keyword") or ""), str(row.get("statement") or "")


def _classify(expected: FrozenMap, actual: FrozenMap) -> str:
    if expected.get("keyword") != actual.get("keyword"):
        return "control_or_operator_divergence"
    keyword = str(expected.get("keyword") or "")
    if keyword in {"ACTION", "OBSERVE", "EXPECT"}:
        return "action_observation_divergence"
    if keyword in {"MEANING", "UPDATE", "DEDUCTION"}:
        return "meaning_state_divergence"
    return "trace_statement_divergence"
