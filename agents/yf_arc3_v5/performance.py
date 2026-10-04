"""Bounded, request-local performance evidence for the V5 live runtime."""

from __future__ import annotations

import time
from collections import deque
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from typing import Iterator, Mapping


PERFORMANCE_EVENT_LIMIT = 400
PERFORMANCE_EVENT_DURATION_LIMIT = 4000
PERFORMANCE_EVENT_DURATION_RETAIN = 3000
PERFORMANCE_SUFFIX_SUMMARY_LIMIT = 12
PERFORMANCE_PHASE_SUMMARY_LIMIT = 40


@dataclass(frozen=True)
class PerformanceEvent:
    sequence: int
    kind: str
    name: str
    duration_ms: float
    metadata: dict[str, object]


class PerformanceRecorder:
    """Keep a bounded diagnostic trace; never retain runtime payload objects."""

    def __init__(self, *, limit: int = PERFORMANCE_EVENT_LIMIT) -> None:
        self._records: deque[PerformanceEvent] = deque(maxlen=limit)
        self._next_sequence = 1
        self._event_duration_ms: dict[int, float] = {}

    def record(
        self,
        *,
        kind: str,
        name: str,
        duration_ms: float,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        compact_metadata = dict(metadata or {})
        event = PerformanceEvent(
            sequence=self._next_sequence,
            kind=kind,
            name=name,
            duration_ms=round(max(0.0, duration_ms), 3),
            metadata=compact_metadata,
        )
        self._next_sequence += 1
        self._records.append(event)
        event_sequences = compact_metadata.get("event_sequences")
        if isinstance(event_sequences, tuple):
            for sequence in event_sequences:
                if isinstance(sequence, int):
                    self._event_duration_ms[sequence] = event.duration_ms
        if len(self._event_duration_ms) > PERFORMANCE_EVENT_DURATION_LIMIT:
            keep = set(
                sorted(self._event_duration_ms)[-PERFORMANCE_EVENT_DURATION_RETAIN:]
            )
            self._event_duration_ms = {
                key: value
                for key, value in sorted(self._event_duration_ms.items())
                if key in keep
            }

    def duration_for_event(self, event_sequence: int) -> float | None:
        return self._event_duration_ms.get(event_sequence)

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def summary_since(
        self,
        sequence: int,
        *,
        limit: int = PERFORMANCE_SUFFIX_SUMMARY_LIMIT,
    ) -> tuple[dict[str, object], ...]:
        """Aggregate a bounded suffix without retaining any runtime payload."""

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for event in self._records:
            if event.sequence < sequence:
                continue
            key = f"{event.kind}:{event.name}"
            totals[key] = totals.get(key, 0.0) + event.duration_ms
            counts[key] = counts.get(key, 0) + 1
        return tuple(
            {
                "name": key,
                "count": counts[key],
                "total_ms": round(total, 3),
            }
            for key, total in sorted(
                totals.items(), key=lambda item: (-item[1], item[0])
            )[: max(1, limit)]
        )

    def projection(self) -> tuple[dict[str, object], ...]:
        return tuple(asdict(event) for event in self._records)

    def summary(self) -> dict[str, object]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for event in self._records:
            key = f"{event.kind}:{event.name}"
            totals[key] = totals.get(key, 0.0) + event.duration_ms
            counts[key] = counts.get(key, 0) + 1
        rows = tuple(
            {
                "name": key,
                "count": counts[key],
                "total_ms": round(total, 3),
                "mean_ms": round(total / counts[key], 3),
            }
            for key, total in sorted(
                totals.items(), key=lambda item: (-item[1], item[0])
            )
        )
        return {
            "event_count": len(self._records),
            "phase_totals": rows[:PERFORMANCE_PHASE_SUMMARY_LIMIT],
        }


_ACTIVE_RECORDER: ContextVar[PerformanceRecorder | None] = ContextVar(
    "yf_arc3_v5_active_performance_recorder",
    default=None,
)


@contextmanager
def activate_performance_recorder(
    recorder: PerformanceRecorder,
) -> Iterator[None]:
    token: Token[PerformanceRecorder | None] = _ACTIVE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_RECORDER.reset(token)


def begin_profile() -> float | None:
    return time.perf_counter() if _ACTIVE_RECORDER.get() is not None else None


def finish_profile(
    started: float | None,
    *,
    kind: str,
    name: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None or started is None:
        return
    recorder.record(
        kind=kind,
        name=name,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        metadata=metadata,
    )


def profile_span(
    *,
    kind: str,
    name: str,
    metadata: Mapping[str, object] | None = None,
) -> AbstractContextManager[None]:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        return nullcontext()
    return _active_profile_span(
        recorder,
        kind=kind,
        name=name,
        metadata=metadata,
    )


@contextmanager
def _active_profile_span(
    recorder: PerformanceRecorder,
    *,
    kind: str,
    name: str,
    metadata: Mapping[str, object] | None,
) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        recorder.record(
            kind=kind,
            name=name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )
