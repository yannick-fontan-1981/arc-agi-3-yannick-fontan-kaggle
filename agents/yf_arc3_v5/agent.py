"""Standard competition-agent adapter for the V5 source runtime."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from agents.agent import Agent
from agents.structs import ActionInput, FrameData, GameAction, GameState
from agents.yf_arc3_v5.logos.types import FrozenMap
from agents.yf_arc3_v5.runtime.config import LiveRunConfig
from agents.yf_arc3_v5.runtime.contracts import (
    ObservedWorldInput,
    SessionEnvironmentTransport,
)
from agents.yf_arc3_v5.runtime.controller import LiveController
from agents.yf_arc3_v5.runtime.environment import (
    HttpEnvironmentConfig,
    HttpEnvironmentTransport,
    HttpTransportProfile,
)


class YFArc3V5Agent(Agent):
    """Bridge the legacy Swarm lifecycle to V5 without adding an action bypass.

    ``RESET`` is an explicit session-bootstrap exception only.  A level
    recovery must be released by the source runtime without discarding its
    accumulated knowledge.
    Every ordinary action is selected and transported through ``LiveController``
    and its sole source-authorized ``ActionBoundary``.
    """

    def __init__(
        self,
        *args: Any,
        controller: LiveController | None = None,
        transport: SessionEnvironmentTransport | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._bootstrapped = False
        self._controller = controller or LiveController(
            LiveRunConfig(
                game_id=self.game_id,
                backend=_backend_label(self.ROOT_URL),
                game_service_url=self.ROOT_URL,
                run_namespace="standard-agent",
                latency_guard_cyclic_gc=True,
            ),
            transport=transport or self._arc_api_transport(),
        )

    @property
    def name(self) -> str:
        return f"{self.game_id}.yf-arc3-v5"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if not self._bootstrapped:
            return False
        if latest_frame.state is GameState.WIN:
            return True
        if latest_frame.state is GameState.GAME_OVER:
            return False
        return self._controller.control_context()["pending_action"] is None

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        if not self._bootstrapped:
            action = GameAction.RESET
            action.reasoning = {
                "agent": "yf_arc3_v5",
                "authority": "explicit_session_bootstrap_exception",
            }
            return action

        if latest_frame.state is GameState.GAME_OVER:
            raise RuntimeError(
                "terminal recovery awaits source action release and retained knowledge"
            )

        pending = self._controller.control_context()["pending_action"]
        if not isinstance(pending, dict):
            raise RuntimeError("V5 has no source-authorized action to choose")
        action = GameAction.from_name(str(pending["action_ref"]))
        data = dict(pending.get("action_data") or {})
        if action.is_complex() and not {"x", "y"}.issubset(data):
            raise RuntimeError("V5 point action requires its source-authorized x/y payload")
        action.set_data(data)
        action.reasoning = {
            "agent": "yf_arc3_v5",
            "authority": "source_action_release_permit",
            "action_intent_id": pending["action_intent_id"],
            "source_hash": pending["source_hash"],
        }
        return action

    def take_action(self, action: GameAction) -> FrameData:
        """Use only bootstrap or the controller's guarded action boundary."""

        if action is GameAction.RESET:
            if self._bootstrapped:
                raise RuntimeError("RESET is allowed only as V5 session bootstrap")
            self._controller.start()
            self._bootstrapped = True
            return _frame_data(self._controller.current_world_input, action)

        if not self._bootstrapped:
            raise RuntimeError("V5 cannot act before session bootstrap")
        context = self._controller.control_context()
        pending = context.get("pending_action")
        if not isinstance(pending, dict):
            raise RuntimeError("V5 has no source-authorized action to execute")
        if pending.get("action_ref") != action.name:
            raise RuntimeError("legacy action differs from V5 source intent")
        frame = context.get("frame")
        if not isinstance(frame, dict):
            raise RuntimeError("V5 control context has no current frame")
        self._controller.execute_pending_action(
            action_ref=action.name,
            expected_context={
                "run_id": context["run_id"],
                "timeline_id": context["timeline_id"],
                "frame_id": frame["frame_id"],
                "state_revision": context["cognitive_state"]["revision"],
            },
        )
        return _frame_data(self._controller.current_world_input, action)

    def _arc_api_transport(self) -> HttpEnvironmentTransport:
        headers = dict(self.headers)
        cookies = self._session.cookies.get_dict()
        if cookies:
            headers["Cookie"] = "; ".join(
                f"{key}={value}" for key, value in sorted(cookies.items())
            )
        return HttpEnvironmentTransport(
            HttpEnvironmentConfig(
                profile=HttpTransportProfile.ARC_API,
                base_url=self.ROOT_URL,
                game_id=self.game_id,
                card_id=self.card_id,
                headers=FrozenMap(headers),
            )
        )


def _backend_label(root_url: str) -> str:
    host = (urlparse(root_url).hostname or "").lower()
    return "standalone" if host in {"", "localhost", "127.0.0.1"} else "official"


def _frame_data(raw: ObservedWorldInput, action: GameAction) -> FrameData:
    state = GameState(raw.state)
    available = [
        GameAction.from_name(action_ref)
        for action_ref in raw.available_action_refs
    ]
    action_data = action.action_data.model_dump(exclude_none=True)
    return FrameData(
        game_id=str(raw.metadata.get("game_id") or ""),
        frame=[[list(row) for row in raw.frame]],
        state=state,
        score=raw.score,
        action_input=ActionInput(
            id=action,
            data=action_data,
            reasoning=action.reasoning,
        ),
        guid=(str(raw.metadata["guid"]) if raw.metadata.get("guid") else None),
        full_reset=action is GameAction.RESET,
        available_actions=available,
    )


__all__ = ["YFArc3V5Agent"]
