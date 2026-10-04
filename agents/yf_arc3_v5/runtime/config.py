"""Adapter-only configuration for V5 launch surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class IsolatedReasoningCapsule:
    game_id: str
    level: int
    scope_ref: str

    def __post_init__(self) -> None:
        if not self.game_id.strip() or not self.scope_ref.strip():
            raise ValueError("isolated reasoning capsule requires game_id and scope_ref")
        if self.level < 1:
            raise ValueError("isolated reasoning capsule level must be positive")


@dataclass(frozen=True)
class LiveRunConfig:
    game_id: str
    backend: str
    game_service_url: str
    instance_seed: int = 0
    agent_id: str = "yf_arc3_v5"
    run_namespace: str = "live"
    repo_root: Path = DEFAULT_REPO_ROOT
    debug_artifacts: bool = False
    performance_diagnostics: bool = False
    latency_guard_cyclic_gc: bool = False
    isolated_reasoning_capsules: tuple[IsolatedReasoningCapsule, ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in {"standalone", "official"}:
            raise ValueError(f"unsupported V5 backend: {self.backend}")
        if not self.game_id.strip():
            raise ValueError("game_id is required")
        if not 0 <= int(self.instance_seed) <= 0xFFFFFFFF:
            raise ValueError("instance_seed must be between 0 and 2**32-1")
        keys = tuple((item.game_id, item.level) for item in self.isolated_reasoning_capsules)
        scopes = tuple(item.scope_ref for item in self.isolated_reasoning_capsules)
        if len(set(keys)) != len(keys):
            raise ValueError("only one isolated reasoning capsule may own a game/level key")
        if len(set(scopes)) != len(scopes):
            raise ValueError("isolated reasoning capsule scope refs must be unique")

    def isolated_reasoning_scope_for(self, game_id: str, level: int) -> str | None:
        matches = tuple(
            item.scope_ref
            for item in self.isolated_reasoning_capsules
            if item.game_id == game_id and item.level == level
        )
        return matches[0] if matches else None

    @property
    def run_root(self) -> Path:
        return self.repo_root / "runs" / "yf_arc3_v5" / self.run_namespace

    @property
    def saved_dir(self) -> Path:
        return self.run_root / "saved"

    @property
    def debug_dir(self) -> Path:
        return self.run_root / "debug"

    def resolve_save(self, name: str) -> Path:
        safe = "".join(
            character
            for character in str(name)
            if character.isalnum() or character in {"-", "_"}
        ).strip("-_")
        if not safe:
            raise ValueError("save name requires letters, digits, '-' or '_'")
        target = (self.saved_dir / f"{safe}.v5.json").resolve()
        if self.saved_dir.resolve() not in target.parents:
            raise ValueError("save path escapes the V5 run directory")
        return target
