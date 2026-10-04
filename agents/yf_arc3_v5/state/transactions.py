"""Atomic transaction request/receipt contracts."""

from __future__ import annotations

from pydantic import Field

from agents.yf_arc3_v5.logos.types import FrozenModel, NonNegativeRevision, Ref
from agents.yf_arc3_v5.state.events import CognitiveEvent, StateMutation
from agents.yf_arc3_v5.state.snapshot import CognitiveSnapshot


class StateConflictError(ValueError):
    """The transaction was built from a stale or mismatched before-state."""


class DuplicateTransactionError(ValueError):
    """The transaction identity has already been committed."""


class TransactionRequest(FrozenModel):
    transaction_id: Ref
    run_id: Ref
    timeline_id: Ref
    frame_id: Ref | None = None
    source_hash: Ref
    expected_revision: NonNegativeRevision
    mutations: tuple[StateMutation, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.transaction_request.v1"


class TransactionReceipt(FrozenModel):
    transaction_id: Ref
    committed_events: tuple[CognitiveEvent, ...] = Field(min_length=1)
    snapshot: CognitiveSnapshot
    schema_version: Ref = "yf_arc3_v5.transaction_receipt.v1"
