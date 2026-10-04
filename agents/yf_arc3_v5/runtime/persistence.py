"""V5-native save/restore for the canonical journal and bounded world inputs."""

from __future__ import annotations

from pathlib import Path
import hashlib
import gzip
import lzma
from contextlib import contextmanager
from typing import Any

from pydantic import Field, field_validator, model_validator

from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    canonical_json,
    stable_digest,
)
from agents.yf_arc3_v5.runtime.acquisition import WorldInputRepository
from agents.yf_arc3_v5.runtime.contracts import ObservedWorldInput
from agents.yf_arc3_v5.src.registry import RegistryEntry
from agents.yf_arc3_v5.state import EventStore


MAX_NATIVE_ARCHIVE_BYTES = 64 * 1024 * 1024


@contextmanager
def open_run_archive_text(path: str | Path):
    """Read losslessly compressed saves and historical plain JSON saves."""
    target = Path(path)
    with target.open("rb") as probe:
        signature = probe.read(6)
    opener = (lzma.open if signature == b"\xfd7zXZ\x00" else
              gzip.open if signature.startswith(b"\x1f\x8b") else Path.open)
    with opener(target, "rt", encoding="utf-8", newline="") as stream:
        yield stream


class _BoundedArchiveSink:
    """Bound encoded bytes, including codec headers and trailers, before writing."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, data: bytes):
        if self.stream.tell() + len(data) > MAX_NATIVE_ARCHIVE_BYTES:
            raise ValueError("native archive exceeds 64 MiB compressed storage limit")
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()


def write_run_archive(path: str | Path, archive: "V5RunArchive") -> None:
    """Publish an atomic, bounded XZ stream; never stage uncompressed JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    values = {name: getattr(archive, name) for name in type(archive).model_fields}
    try:
        with temporary.open("wb") as raw:
            with lzma.LZMAFile(_BoundedArchiveSink(raw), mode="wb", preset=6) as stream:
                for chunk in _canonical_archive_chunks(values):
                    stream.write(chunk.encode("utf-8"))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_archive_chunks(values: dict[str, object]):
    """Preserve canonical JSON bytes without duplicating a large escaped journal."""
    yield "{"
    for index, key in enumerate(sorted(values)):
        if index:
            yield ","
        yield canonical_json(key) + ":"
        value = values[key]
        if key == "event_jsonl" and isinstance(value, str):
            yield '"'
            for offset in range(0, len(value), 65536):
                yield canonical_json(value[offset:offset + 65536])[1:-1]
            yield '"'
        else:
            yield canonical_json(value)
    yield "}"


def _archive_content_hash(values: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for chunk in _canonical_archive_chunks(values):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


class V5RunArchive(FrozenModel):
    agent_id: str = "yf-arc3-v5"
    agent_version: str = "5"
    run_id: str
    event_jsonl: str
    world_inputs: tuple[ObservedWorldInput, ...] = ()
    registry_entries: tuple[RegistryEntry, ...]
    metadata: FrozenMap = Field(default_factory=FrozenMap)
    content_hash: str
    schema_version: str = "yf_arc3_v5.run_archive.v1"

    @field_validator("event_jsonl", mode="plain")
    @classmethod
    def validate_journal_string(cls, value: object) -> str:
        # Pydantic's string conversion allocates a full UTF-8 copy, even when
        # the canonical exporter already supplied a Python string. Validate
        # every code point in bounded chunks without retaining that copy.
        if not isinstance(value, str):
            raise ValueError("canonical journal must be a string")
        for offset in range(0, len(value), 65536):
            value[offset:offset + 65536].encode("utf-8")
        return value

    @classmethod
    def create(
        cls,
        *,
        event_store: EventStore,
        world_inputs: WorldInputRepository,
        registry_entries: tuple[RegistryEntry, ...],
        metadata: FrozenMap | None = None,
    ) -> "V5RunArchive":
        active_metadata = metadata or FrozenMap()
        values: dict[str, object] = {
            "run_id": event_store.run_id,
            "event_jsonl": event_store.to_jsonl(),
            "world_inputs": world_inputs.values,
            "registry_entries": tuple(
                sorted(registry_entries, key=lambda item: item.module_id)
            ),
            "metadata": active_metadata,
            "agent_id": "yf-arc3-v5",
            "agent_version": "5",
            "schema_version": "yf_arc3_v5.run_archive.v1",
        }
        return cls(
            run_id=event_store.run_id,
            event_jsonl=values["event_jsonl"],
            world_inputs=world_inputs.values,
            registry_entries=tuple(
                sorted(registry_entries, key=lambda item: item.module_id)
            ),
            metadata=active_metadata,
            content_hash=_archive_content_hash(values),
        )

    @model_validator(mode="after")
    def validate_archive(self) -> "V5RunArchive":
        values = {name: getattr(self, name) for name in type(self).model_fields if name != "content_hash"}
        if self.content_hash != _archive_content_hash(values):
            raise ValueError("run archive content hash mismatch")
        if not self.event_jsonl.strip():
            raise ValueError("run archive requires a non-empty canonical journal")
        return self


class RestoredV5State(FrozenModel):
    archive: V5RunArchive
    event_jsonl: str
    schema_version: str = "yf_arc3_v5.restored_state.v1"


class ManualGameAction(FrozenModel):
    action: int
    data: FrozenMap = Field(default_factory=FrozenMap)

    @model_validator(mode="after")
    def validate_action(self) -> "ManualGameAction":
        if not 1 <= self.action <= 7:
            raise ValueError("manual game archive has an invalid action")
        return self


class ManualGameArchive(FrozenModel):
    """Physical replay checkpoint; it contains no V5 journal or cognition."""

    game_id: str
    instance_seed: int
    initial_frame_hash: str
    frame_hash: str
    physical_source_hash: str
    level: int
    score: int
    state: str
    step_index: int
    actions: tuple[ManualGameAction, ...]
    content_hash: str
    schema_version: str = "yf_arc3_v5.manual_game.v1"

    @classmethod
    def create(
        cls,
        *,
        game_id: str,
        instance_seed: int,
        initial_frame_hash: str,
        frame_hash: str,
        physical_source_hash: str,
        level: int,
        score: int,
        state: str,
        step_index: int,
        actions: tuple[ManualGameAction, ...],
    ) -> "ManualGameArchive":
        values: dict[str, Any] = {
            "game_id": game_id,
            "instance_seed": instance_seed,
            "initial_frame_hash": initial_frame_hash,
            "frame_hash": frame_hash,
            "physical_source_hash": physical_source_hash,
            "level": level,
            "score": score,
            "state": state,
            "step_index": step_index,
            "actions": tuple(action.model_dump(mode="python") for action in actions),
            "schema_version": "yf_arc3_v5.manual_game.v1",
        }
        return cls(**values, content_hash=stable_digest(values))

    @model_validator(mode="after")
    def validate_archive(self) -> "ManualGameArchive":
        if self.schema_version != "yf_arc3_v5.manual_game.v1":
            raise ValueError("unsupported manual game archive schema")
        if not all((
            self.game_id, self.initial_frame_hash, self.frame_hash,
            self.physical_source_hash,
        )):
            raise ValueError("manual game archive lacks physical identity")
        if self.level < 1 or self.score < 0 or self.step_index < 0:
            raise ValueError("manual game archive has invalid physical counters")
        if len(self.actions) > 100_000:
            raise ValueError("manual game archive exceeds action bound")
        values = self.model_dump(mode="python", exclude={"content_hash"})
        if self.content_hash != stable_digest(values):
            raise ValueError("manual game archive content hash mismatch")
        return self


def save_manual_game_archive(path: str | Path, archive: ManualGameArchive) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(archive.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(target)


def load_manual_game_archive(path: str | Path) -> ManualGameArchive:
    return ManualGameArchive.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_run_archive(
    path: str | Path,
    *,
    event_store: EventStore,
    world_inputs: WorldInputRepository,
    registry_entries: tuple[RegistryEntry, ...],
    metadata: FrozenMap | None = None,
) -> V5RunArchive:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = V5RunArchive.create(
        event_store=event_store,
        world_inputs=world_inputs,
        registry_entries=registry_entries,
        metadata=metadata,
    )
    write_run_archive(target, archive)
    return archive


def load_run_archive(
    path: str | Path,
    *,
    current_registry_entries: tuple[RegistryEntry, ...],
) -> tuple[V5RunArchive, EventStore, WorldInputRepository]:
    with open_run_archive_text(path) as stream:
        archive = V5RunArchive.model_validate_json(stream.read())
    expected = {
        item.module_id: item.transitive_source_hash for item in current_registry_entries
    }
    saved = {
        item.module_id: item.transitive_source_hash for item in archive.registry_entries
    }
    if saved != expected:
        raise ValueError(
            "saved runtime source registry is stale; "
            f"missing={sorted(set(expected) - set(saved))}, "
            f"extra={sorted(set(saved) - set(expected))}"
        )
    store = EventStore.from_jsonl(archive.event_jsonl)
    if store.run_id != archive.run_id:
        raise ValueError("saved journal run identity mismatch")
    world_inputs = WorldInputRepository()
    for item in archive.world_inputs:
        world_inputs.put(item)
    return archive, store, world_inputs
