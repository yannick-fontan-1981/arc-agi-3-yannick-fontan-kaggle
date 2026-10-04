"""Concrete HTTP transport behind the single V5 environment boundary."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import Field

from agents.yf_arc3_v5.capabilities.contracts import FrameGrid, FrameNormalizationInput
from agents.yf_arc3_v5.capabilities.frame import normalize_frame
from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    FrozenMap,
    FrozenModel,
    stable_digest,
)
from agents.yf_arc3_v5.runtime.contracts import (
    EnvironmentActionRequest,
    EnvironmentActionResponse,
    EnvironmentTransport,
    ObservedWorldInput,
    SessionBootstrapResponse,
)
from agents.yf_arc3_v5.src.symbols import SymbolDefinition, SymbolKind, type_spec

ENVIRONMENT_ADAPTER_ID = "environment.primary"
DISPLAY_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")


class HttpTransportProfile(str, Enum):
    ARC_API = "arc_api"
    OFFICIAL_TOOLKIT = "official_toolkit"


class HttpEnvironmentConfig(FrozenModel):
    profile: HttpTransportProfile
    base_url: str
    game_id: str
    card_id: str | None = None
    guid: str | None = None
    instance_seed: int = 0
    headers: FrozenMap = Field(default_factory=FrozenMap)
    timeout_seconds: float = Field(default=120.0, gt=0)
    schema_version: str = "yf_arc3_v5.http_environment_config.v1"


def environment_symbol_definition() -> SymbolDefinition:
    path = Path(__file__)
    return SymbolDefinition(
        id=ENVIRONMENT_ADAPTER_ID,
        kind=SymbolKind.ENVIRONMENT_ADAPTER,
        output_type=type_spec("ActionEvent"),
        effect_class=EffectClass.ENVIRONMENT_ADAPTER,
        source_unit="agents/yf_arc3_v5/runtime/environment.py",
        source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class HttpEnvironmentTransport(EnvironmentTransport):
    """Transport mechanics only; callers must provide an already valid permit."""

    def __init__(self, config: HttpEnvironmentConfig) -> None:
        self.config = config
        self._card_id = config.card_id
        self._guid = config.guid
        self._checkpoint_action_ref: str | None = None

    def bootstrap(self) -> SessionBootstrapResponse:
        """Explicit reset/select exception; it cannot dispatch a planned action."""

        if self.config.profile is HttpTransportProfile.ARC_API:
            if self._card_id is None:
                opened = self._post_json(
                    "/api/scorecard/open",
                    {"tags": ["yf-arc3-v5-live"]},
                )
                self._card_id = str(opened["card_id"])
            path = "/api/cmd/RESET"
            payload: dict[str, object] = {
                "card_id": self._card_id,
                "game_id": self.config.game_id,
            }
        else:
            path = "/select"
            payload = {
                "game_id": self.config.game_id,
                "instance_seed": self.config.instance_seed,
            }
        response = self._post_json(path, payload)
        self._update_session_ids(response)
        transport_response_digest = stable_digest(response)
        raw_input = world_input_from_response(
            response,
            identity_prefix="bootstrap",
        )
        return SessionBootstrapResponse(
            raw_input=raw_input,
            session_metadata=FrozenMap(
                {
                    "profile": self.config.profile.value,
                    "game_id": self.config.game_id,
                    "card_id": self._card_id,
                    "guid": self._guid,
                    "bootstrap_path": path,
                    "instance_seed": response.get("instance_seed"),
                    "initial_frame_hash": response.get("initial_frame_hash"),
                    "transport_response_digest": transport_response_digest,
                }
            ),
        )

    def recover_level(self, action_ref: str) -> ObservedWorldInput:
        """Issue exactly one explicit official recovery control."""
        if self.config.profile is HttpTransportProfile.ARC_API:
            if action_ref != "RESET":
                raise ValueError("ARC API level recovery requires explicit RESET")
            path = "/api/cmd/RESET"
            payload = {
                "card_id": self._card_id,
                "guid": self._guid,
                "game_id": self.config.game_id,
            }
        else:
            # The local HTTP adapter publishes its checkpoint command under
            # an action alias. Only that observed alias may cross this path;
            # an arbitrary puzzle action must never become a recovery control.
            suffix = action_ref.removeprefix("ACTION")
            if (action_ref != self._checkpoint_action_ref
                    or not action_ref.startswith("ACTION")
                    or suffix not in {str(index) for index in range(1, 8)}):
                raise ValueError("level recovery requires the exposed checkpoint action")
            path = "/action"
            payload = {"action": int(suffix), "data": {}}
        response = self._post_json(path, payload)
        self._update_session_ids(response)
        return world_input_from_response(response, identity_prefix="level-recovery")

    def execute(self, request: EnvironmentActionRequest) -> EnvironmentActionResponse:
        """The only production function that sends an authorized action."""

        action_data = request.intent.action_data or request.world_context.get(
            "action_data", FrozenMap()
        )
        if not isinstance(action_data, FrozenMap):
            action_data = FrozenMap(action_data)
        payload: dict[str, object]
        if self.config.profile is HttpTransportProfile.ARC_API:
            path = f"/api/cmd/{request.intent.action_ref}"
            payload = {
                "card_id": self._card_id,
                "guid": self._guid,
                "game_id": self.config.game_id,
                "reasoning": _reasoning_payload(request),
            }
            if request.intent.action_ref == "ACTION6":
                # ARC REST requires coordinates on the command envelope.
                # The standalone profile below retains its nested data schema.
                payload["x"] = action_data["x"]
                payload["y"] = action_data["y"]
        else:
            action_ref = request.intent.action_ref.upper()
            if not action_ref.startswith("ACTION"):
                raise ValueError(f"unsupported Official action ref: {action_ref}")
            path = "/action"
            payload = {
                "action": int(action_ref.removeprefix("ACTION")),
                "data": action_data.to_dict(),
                "reasoning": _reasoning_payload(request),
            }

        payload_response = self._post_json(path, payload)
        self._update_session_ids(payload_response)
        world_input = world_input_from_response(
            payload_response,
            identity_prefix=request.intent.id,
        )
        reasoning_input_digest = str(
            world_input.metadata["reasoning_input_digest"]
        )
        # All payload-derived fields have already passed the canonical
        # world-input projection above; the remaining envelope fields are
        # local immutable references.  Avoid revalidating the same nested
        # observation on the hot transport path.
        return EnvironmentActionResponse.model_construct(
            dispatch_id=request.dispatch_id,
            environment_action_event_ref=(
                f"environment-action:{request.intent.id}:{reasoning_input_digest[:12]}"
            ),
            action_intent_id=request.intent.id,
            action_release_permit_id=request.permit.id,
            raw_input=world_input,
            response_digest=reasoning_input_digest,
            transport_metadata=FrozenMap(
                {
                    "profile": self.config.profile.value,
                    "path": path,
                    # The canonical observation projection already computed
                    # this bounded response identity.  Reusing it avoids a
                    # second full-payload digest on every action; transport
                    # metadata remains diagnostic and cannot affect cognition.
                    "transport_response_digest": reasoning_input_digest,
                }
            ),
        )

    def execute_human_action(
        self,
        action_ref: str,
        action_data: Mapping[str, object],
        *,
        fallback_frame: tuple[tuple[int, ...], ...] | None = None,
    ) -> ObservedWorldInput:
        """Explicit UI operator override; never callable from an SRC workflow."""

        normalized_ref = action_ref.upper()
        if not normalized_ref.startswith("ACTION"):
            raise ValueError(f"unsupported human action ref: {action_ref}")
        if self.config.profile is HttpTransportProfile.ARC_API:
            path = f"/api/cmd/{normalized_ref}"
            payload: dict[str, object] = {
                "card_id": self._card_id,
                "guid": self._guid,
                "game_id": self.config.game_id,
                "data": dict(action_data),
            }
        else:
            path = "/action"
            payload = {
                "action": int(normalized_ref.removeprefix("ACTION")),
                "data": dict(action_data),
            }
        response = self._post_json(path, payload)
        self._update_session_ids(response)
        return world_input_from_response(
            response,
            identity_prefix=f"human:{normalized_ref}",
            fallback_frame=fallback_frame,
        )

    def inspect_manual_game(self) -> dict[str, object]:
        if self.config.profile is not HttpTransportProfile.OFFICIAL_TOOLKIT:
            raise RuntimeError("physical manual saves require the arcade transport")
        request = Request(
            f"{self.config.base_url.rstrip('/')}/state",
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("arcade state returned non-object JSON")
        return payload

    def replay_manual_game(
        self,
        *,
        game_id: str,
        instance_seed: int,
        actions: list[dict[str, object]],
        expected: dict[str, object],
    ) -> dict[str, object]:
        if self.config.profile is not HttpTransportProfile.OFFICIAL_TOOLKIT:
            raise RuntimeError("physical manual loads require the arcade transport")
        return self._post_json(
            "/replay",
            {
                "game_id": game_id,
                "instance_seed": instance_seed,
                "actions": actions,
                "expected": expected,
            },
        )

    def _post_json(
        self,
        path: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        raw_request = json.dumps(
            dict(payload),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        http_request = Request(
            f"{self.config.base_url.rstrip('/')}{path}",
            data=raw_request,
            headers={
                "content-type": "application/json",
                **{str(key): str(value) for key, value in sorted(self.config.headers.items())},
            },
            method="POST",
        )
        try:
            with urlopen(
                http_request,
                timeout=self.config.timeout_seconds,
            ) as response:
                raw_response = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"environment action failed ({error.code}): {detail[:500]}"
            ) from error
        except URLError as error:
            raise RuntimeError("environment action service is unavailable") from error

        payload_response = json.loads(raw_response.decode("utf-8"))
        if not isinstance(payload_response, dict):
            raise RuntimeError("environment action returned non-object JSON")
        return payload_response

    def _update_session_ids(self, payload: dict[str, object]) -> None:
        checkpoint_ref = payload.get("level_start_checkpoint_action_ref")
        self._checkpoint_action_ref = str(checkpoint_ref) if checkpoint_ref else None
        if payload.get("card_id"):
            self._card_id = str(payload["card_id"])
        if payload.get("guid"):
            self._guid = str(payload["guid"])


def _reasoning_payload(request: EnvironmentActionRequest) -> dict[str, object]:
    return {
        "agent": "yf_arc3_v5",
        "workflow_definition_id": request.workflow_definition_id,
        "workflow_instance_id": request.workflow_instance_id,
        "action_intent_id": request.intent.id,
        "action_release_permit_id": request.permit.id,
        "source_hash": request.intent.source_hash,
        "state_revision": request.intent.state_revision,
    }


def world_input_from_response(
    payload: dict[str, object],
    *,
    identity_prefix: str,
    fallback_frame: tuple[tuple[int, ...], ...] | None = None,
) -> ObservedWorldInput:
    """Project an environment response into the canonical immutable input."""

    raw_frames = payload.get("frames") or payload.get("frame")
    fast_rows = _fast_payload_frame_rows(raw_frames)
    if fast_rows is None:
        state_value = str(payload.get("state") or "NOT_FINISHED").upper()
        terminal_failure = any(
            marker in state_value for marker in ("GAME_OVER", "FAILED", "ERROR")
        )
        empty_terminal_frame = (
            not isinstance(raw_frames, (list, tuple))
            or not raw_frames
            or not isinstance(raw_frames[0], (list, tuple))
            or not raw_frames[0]
            or (
                isinstance(raw_frames[0][0], (list, tuple))
                and not raw_frames[0][0]
            )
        )
        if terminal_failure and empty_terminal_frame and fallback_frame:
            # Some environments signal terminal failure without returning a
            # raster. Keep the last known scene so the UI can show its explicit
            # GAME_OVER overlay instead of failing observation parsing.
            normalized_frame = FrameGrid(rows=fallback_frame)
            transition_frames = ()
            source_frame_count = 1
        else:
            normalized_input = FrameNormalizationInput.from_payload(raw_frames)
            normalized = normalize_frame(normalized_input)
            normalized_frame = normalized.frame
            transition_frames = tuple(item.rows for item in normalized.transition_frames)
            source_frame_count = normalized.source_frame_count
    else:
        # Official bridge payloads are already integer rectangular matrices.
        # Build the same immutable FrameGrid values without the intermediate
        # normalization request/result models on every action response.
        normalized_frame = fast_rows[-1]
        transition_frames = tuple(item.rows for item in fast_rows[:-1])
        source_frame_count = len(fast_rows)
    raw_available_value = (
        payload.get("available_actions") or payload.get("available_action_ids") or ()
    )
    raw_available = (
        raw_available_value if isinstance(raw_available_value, (list, tuple)) else ()
    )
    available = tuple(
        sorted(
            (_action_ref(item) for item in raw_available),
            key=_action_sort_key,
        )
    )
    score = _integer(payload.get("levels_completed", payload.get("score", 0)))
    state = str(payload.get("state") or "NOT_FINISHED")
    # Cognitive identity is derived exclusively from normalized observation
    # facts shared by every transport.  HTTP timing/profile/session fields and
    # JSON key order are intentionally absent: they may remain observable in a
    # transport envelope but can never perturb entity IDs or tie-breaks.
    digest_facts = {
        "available_action_refs": available,
        "frame": normalized_frame.rows,
        "frame_hash": payload.get("frame_hash"),
        "game_id": payload.get("game_id"),
        "initial_frame_hash": payload.get("initial_frame_hash"),
        "instance_seed": payload.get("instance_seed"),
        "last_action": payload.get("last_action"),
        "level": payload.get("level"),
        "level_start_checkpoint_action_ref": payload.get(
            "level_start_checkpoint_action_ref"
        ),
        "levels_completed": payload.get("levels_completed"),
        "score": score,
        "state": state,
        "step_index": payload.get("step_index"),
        "transition_frames": transition_frames,
        "win_levels": payload.get("win_levels"),
    }
    try:
        reasoning_input_digest = hashlib.sha256(
            json.dumps(
                digest_facts,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        # Keep the fully canonical immutable path for an adapter that returns
        # a non-JSON scalar; official bridge/HTTP payloads take the native
        # encoder path above.
        reasoning_input_digest = stable_digest(FrozenMap(digest_facts))
    # The causal identity may itself refer to the workflow started from the
    # previous frame.  Hashing it keeps frame/workflow/action IDs bounded.
    cause_ref = (
        "bootstrap"
        if identity_prefix == "bootstrap"
        else stable_digest(identity_prefix)[:12]
    )
    frame_id = f"frame:{cause_ref}:{reasoning_input_digest[:12]}"
    return ObservedWorldInput(
        raw_input_ref=f"raw:{frame_id}:{reasoning_input_digest[:12]}",
        frame_id=frame_id,
        frame=normalized_frame.rows,
        available_action_refs=available,
        score=score,
        state=state,
        metadata=FrozenMap(
            {
                "game_id": payload.get("game_id"),
                "guid": payload.get("guid"),
                "card_id": payload.get("card_id"),
                "title": payload.get("title"),
                "level": payload.get("level"),
                "levels_completed": payload.get("levels_completed"),
                "win_levels": payload.get("win_levels"),
                "step_index": payload.get("step_index"),
                "default_fps": payload.get("default_fps"),
                "display_palette": _normalize_display_palette(payload.get("palette")),
                "transition_frame_interval_ms": payload.get(
                    "transition_frame_interval_ms"
                ),
                "last_action": payload.get("last_action"),
                "instance_seed": payload.get("instance_seed"),
                "initial_frame_hash": payload.get("initial_frame_hash"),
                "frame_hash": payload.get("frame_hash"),
                "level_start_checkpoint_action_ref": payload.get(
                    "level_start_checkpoint_action_ref"
                ),
                "terminal_message": (
                    str(payload.get("terminal_message") or payload.get("message"))
                    if payload.get("terminal_message") or payload.get("message")
                    else None
                ),
                "transition_frames": transition_frames,
                "source_frame_count": source_frame_count,
                "reasoning_input_digest": reasoning_input_digest,
                "response_digest": reasoning_input_digest,
            }
        ),
    )


def _normalize_display_palette(value: object) -> tuple[str, ...] | None:
    """Validate optional renderer colors without assigning cognitive meaning."""

    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 256:
        return None
    if not all(
        isinstance(color, str) and DISPLAY_COLOR_PATTERN.fullmatch(color)
        for color in value
    ):
        return None
    return tuple(color.upper() for color in value)


def _fast_payload_frame_rows(payload: object) -> tuple[FrameGrid, ...] | None:
    """Normalize trusted bridge matrices without the wrapper model overhead.

    Return ``None`` for non-plain payloads so the general validation path
    remains authoritative for HTTP, tests, and unusual adapters.
    """

    if not isinstance(payload, (list, tuple)) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, (list, tuple)) or not first:
        return None
    raw_stack = payload if isinstance(first[0], (list, tuple)) else (payload,)
    grids: list[FrameGrid] = []
    try:
        for raw_frame in raw_stack:
            rows: list[tuple[int, ...]] = []
            width: int | None = None
            for raw_row in raw_frame:
                if not isinstance(raw_row, (list, tuple)) or not raw_row:
                    return None
                # JSON-decoded bridge rows are already exact Python ints.  Do
                # not call ``int`` once per cell on that trusted path; retain
                # the coercing branch for unusual numeric payloads.
                row = (
                    tuple(raw_row)
                    if all(type(item) is int for item in raw_row)
                    else tuple(int(item) for item in raw_row)
                )
                if width is None:
                    width = len(row)
                if not row or len(row) != width:
                    return None
                rows.append(row)
            if not rows:
                return None
            # The bridge path has already established a rectangular integer
            # matrix above.  Keep the exact FrameGrid contract while avoiding
            # a second recursive Pydantic validation on every primitive
            # action.  Non-bridge/untrusted payloads still use the full
            # normalization path above.
            grids.append(FrameGrid.model_construct(rows=tuple(rows)))
    except (TypeError, ValueError):
        return None
    return tuple(grids) if grids else None


def _action_ref(value: object) -> str:
    if isinstance(value, int):
        return f"ACTION{value}"
    text = str(value)
    if text.upper() == "RESET":
        return "RESET"
    return text if text.upper().startswith("ACTION") else f"ACTION{text}"


def _action_sort_key(value: str) -> tuple[int, int, str]:
    suffix = value.upper().removeprefix("ACTION")
    if suffix.isdigit():
        return (0, int(suffix), value)
    return (1, 0, value)


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
