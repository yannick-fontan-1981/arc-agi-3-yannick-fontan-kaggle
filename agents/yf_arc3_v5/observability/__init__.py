"""Read-only M8 projections over the canonical V5 event journal."""

from agents.yf_arc3_v5.observability.attestation import (
    project_action_lineage,
    query_attestation,
)
from agents.yf_arc3_v5.observability.audit import audit_v5_architecture
from agents.yf_arc3_v5.observability.models import (
    ActiveStateProjection,
    ArchitectureAuditReport,
    AttestationQuery,
    AttestationResult,
    CompactSrlLine,
    DecisionEnvelope,
    FirstDivergence,
    RunReport,
    TraceDiffResult,
)
from agents.yf_arc3_v5.observability.projection import (
    DecisionEnvelopeProjectionError,
    project_active_state,
    project_compact_srl,
    project_decision_envelope,
    render_compact_srl,
)
from agents.yf_arc3_v5.observability.reporting import build_run_report
from agents.yf_arc3_v5.observability.trace_diff import diff_compact_traces

__all__ = [
    "ActiveStateProjection",
    "ArchitectureAuditReport",
    "AttestationQuery",
    "AttestationResult",
    "CompactSrlLine",
    "DecisionEnvelope",
    "DecisionEnvelopeProjectionError",
    "FirstDivergence",
    "RunReport",
    "TraceDiffResult",
    "audit_v5_architecture",
    "project_active_state",
    "project_compact_srl",
    "project_decision_envelope",
    "render_compact_srl",
    "diff_compact_traces",
    "build_run_report",
    "project_action_lineage",
    "query_attestation",
]
