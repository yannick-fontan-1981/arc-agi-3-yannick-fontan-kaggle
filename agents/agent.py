# agents/agent.py

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

if TYPE_CHECKING:
    from requests import Response, Session
    from requests.cookies import RequestsCookieJar

from .recorder import Recorder
from .structs import FrameData, GameAction, GameState, Scorecard
from .tracing import trace_agent_session


class _LazyRequests:
    """Preserve the patchable Session surface without eager HTTP imports."""

    @staticmethod
    def Session():
        from requests import Session

        return Session()


requests = _LazyRequests()

#logger = logging.getLogger()
def _setup_logger(name: str = "agent.base") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname).1s %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        lg.addHandler(handler)
        # Respect env override, default to DEBUG like yf_arc3
        lg.setLevel(os.getenv("ARC_LOG_LEVEL", "DEBUG").upper())
        lg.propagate = False
    # Quiet noisy HTTP logs unless explicitly enabled
    if not os.getenv("ARC_VERBOSE_HTTP"):
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    return lg

logger = _setup_logger()


def get_agent_logger(name: str = "agent.base") -> logging.Logger:
    """Create or retrieve a logger that shares the core agent configuration."""

    return _setup_logger(name)


def _safe_json_preview(data: Any, max_chars: int = 400) -> str:
    """Serialise arbitrary data for logging without flooding the console."""

    try:
        text = json.dumps(data, default=str, ensure_ascii=False)
    except TypeError:
        text = str(data)

    if len(text) > max_chars:
        return f"{text[:max_chars]}… <truncated>"
    return text

class Agent(ABC):
    """Interface for an agent that plays one ARC-AGI-3 game."""

    MAX_ACTIONS: int = 80  # to avoid looping forever if agent doesnt exit
    ROOT_URL: str

    action_counter: int = 0

    timer: float = 0
    agent_name: str
    card_id: str
    game_id: str
    guid: str
    frames: list[FrameData]

    recorder: Recorder
    headers: dict[str, str]
    _session: "Session"

    # AgentOps tracing attributes
    trace: Any = None
    tags: list[str]

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        tags: Optional[list[str]] = None,
        cookies: "RequestsCookieJar | None" = None,
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name
        self.tags = tags or []
        self.frames = [FrameData(score=0)]
        self._cleanup = True
        self.run_reporter: Any = None
        self.segment_step_budget_enabled: bool = False
        self.segment_action_counter: int = 0
        self.MAX_TOTAL_ACTIONS: int | None = None
        self.step_budget_stop_reason: str | None = None
        self.step_budget_resets: list[dict[str, Any]] = []
        if record:
            self.start_recording()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        # Reuse session
        self._session = requests.Session()
        if cookies is not None:
            self._session.cookies = deepcopy(cookies)
        self._session.headers.update(self.headers)

        logger.info(
            "Initialised agent '%s' for game '%s' with scorecard '%s'",
            self.agent_name,
            self.game_id,
            self.card_id,
        )

    def set_run_reporter(self, reporter: Any) -> None:
        """Attach an optional reporter used by external smoke/debug harnesses."""

        self.run_reporter = reporter

    def _emit_run_report(self, hook: str, **payload: Any) -> None:
        reporter = getattr(self, "run_reporter", None)
        if reporter is None:
            return

        callback = getattr(reporter, hook, None)
        if not callable(callback):
            return

        try:
            callback(**payload)
        except Exception:
            logger.exception("Run reporter hook '%s' failed", hook)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _frame_dimensions(self, frame: FrameData) -> tuple[int, int, int]:
        layers = len(frame.frame)
        rows = len(frame.frame[0]) if layers and frame.frame[0] else 0
        cols = len(frame.frame[0][0]) if rows and frame.frame[0][0] else 0
        return layers, rows, cols

    def _format_frame_summary(self, frame: FrameData) -> str:
        layers, rows, cols = self._frame_dimensions(frame)
        available = ", ".join(a.name for a in frame.available_actions) or "<none>"
        return (
            "state=%s score=%s layers=%s grid=%sx%s available_actions=[%s]"
            % (frame.state, frame.score, layers, rows, cols, available)
        )

    def _log_frame_snapshot(self, label: str, frame: FrameData) -> None:
        logger.info("%s | %s", label, self._format_frame_summary(frame))

    def _log_action(self, action: Any) -> None:
        name = getattr(action, "name", getattr(action, "action", "<unknown>"))
        payload_source = getattr(action, "action_data", None)
        if hasattr(payload_source, "model_dump"):
            payload = payload_source.model_dump(exclude_none=True)
        elif isinstance(payload_source, dict):
            payload = payload_source
        else:
            payload = {}
        reasoning = getattr(action, "reasoning", None)
        logger.info(
            "Chose action '%s' with payload=%s reasoning=%s",
            name,
            _safe_json_preview(payload),
            _safe_json_preview(reasoning) if reasoning is not None else "<none>",
        )

    def _segment_budget_exhausted(self) -> bool:
        if not bool(getattr(self, "segment_step_budget_enabled", False)):
            return False
        return int(getattr(self, "segment_action_counter", 0)) > int(self.MAX_ACTIONS)

    def _total_budget_exhausted(self) -> bool:
        limit = getattr(self, "MAX_TOTAL_ACTIONS", None)
        if limit is None:
            return False
        return int(getattr(self, "action_counter", 0)) > int(limit)

    def _should_reset_segment_budget(
        self,
        *,
        previous_frame: FrameData,
        action: Any,
        next_frame: FrameData,
    ) -> str | None:
        if not bool(getattr(self, "segment_step_budget_enabled", False)):
            return None
        try:
            previous_score = int(getattr(previous_frame, "score", 0))
            next_score = int(getattr(next_frame, "score", 0))
        except (TypeError, ValueError):
            previous_score = 0
            next_score = 0
        if next_score != previous_score:
            return "score_changed"
        action_name = str(getattr(action, "name", "") or "").upper()
        if action_name == "RESET" or bool(getattr(next_frame, "full_reset", False)):
            return "reset_action"
        return None

    def _record_segment_budget_reset(
        self,
        *,
        reason: str,
        step_index: int,
        frame: FrameData,
    ) -> None:
        self.segment_action_counter = 0
        self.step_budget_resets.append(
            {
                "reason": str(reason),
                "step": int(step_index),
                "score": int(getattr(frame, "score", 0)),
                "state": getattr(frame, "state", None).value if getattr(frame, "state", None) is not None else None,
            }
        )

    @trace_agent_session
    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        self.segment_action_counter = 0
        self.step_budget_stop_reason = None
        self.step_budget_resets = []
        logger.info(
            "Starting solving loop for game '%s' (max actions=%s)",
            self.game_id,
            self.MAX_ACTIONS,
        )
        if self.frames:
            self._log_frame_snapshot("Initial frame", self.frames[-1])
        self._emit_run_report(
            "on_run_started",
            agent=self,
            initial_frame=self.frames[-1] if self.frames else None,
        )
        while (
            not self.is_done(self.frames, self.frames[-1])
            and (
                self.segment_action_counter <= self.MAX_ACTIONS
                if self.segment_step_budget_enabled
                else self.action_counter <= self.MAX_ACTIONS
            )
            and not self._total_budget_exhausted()
        ):
            latest_frame = self.frames[-1]
            step_index = self.action_counter + 1
            logger.info(
                "Preparing action #%s | elapsed=%ss | fps=%s",
                step_index,
                self.seconds,
                self.fps,
            )
            # self._log_frame_snapshot("Current frame", latest_frame)
            action = self.choose_action(self.frames, self.frames[-1])
            self._log_action(action)
            self._emit_run_report(
                "on_action_selected",
                agent=self,
                step=step_index,
                frame=latest_frame,
                action=action,
            )
            if frame := self.take_action(action):
                self.append_frame(frame)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, score {frame.score}, avg fps {self.fps})"
                )
                self._log_frame_snapshot("Frame after action", frame)
                self._emit_run_report(
                    "on_frame_received",
                    agent=self,
                    step=step_index,
                    action=action,
                    frame=frame,
                )
                reset_reason = self._should_reset_segment_budget(
                    previous_frame=latest_frame,
                    action=action,
                    next_frame=frame,
                )
            else:
                logger.warning(
                    "Action #%s ('%s') did not return a valid frame",
                    step_index,
                    getattr(action, "name", getattr(action, "action", "<unknown>")),
                )
                self._emit_run_report(
                    "on_frame_missing",
                    agent=self,
                    step=step_index,
                    action=action,
                )
                reset_reason = None
            self.action_counter += 1
            if self.segment_step_budget_enabled:
                if isinstance(reset_reason, str) and frame is not None:
                    self._record_segment_budget_reset(
                        reason=reset_reason,
                        step_index=step_index,
                        frame=frame,
                    )
                else:
                    self.segment_action_counter += 1

        if self._total_budget_exhausted():
            self.step_budget_stop_reason = "total_max_steps_reached"
        elif self._segment_budget_exhausted():
            self.step_budget_stop_reason = "segment_max_steps_reached"

        self.cleanup()

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def score(self) -> int:
        return self.frames[-1].score

    @property
    def seconds(self) -> float:
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        elapsed_time = max(self.seconds, 0.1)
        return round(self.action_counter / elapsed_time, 2)

    @property
    def is_playback(self) -> bool:
        return type(self) is Playback

    @property
    def name(self) -> str:
        n = self.__class__.__name__.lower()
        return f"{self.game_id}.{n}"

    def start_recording(self) -> None:
        filename = self.agent_name if self.is_playback else None
        self.recorder = Recorder(prefix=self.name, filename=filename)
        logger.info(
            f"created new recording for {self.name} into {self.recorder.filename}"
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    def do_action_request(self, action: GameAction) -> "Response":
        data = action.action_data.model_dump()
        if action == GameAction.RESET:
            data["card_id"] = self.card_id
        if self.guid:
            data["guid"] = self.guid
        if action.reasoning:
            data["reasoning"] = action.reasoning
        if self.game_id:
            data["game_id"] = self.game_id

        json_str = json.dumps(data)
        logger.info(
            "Submitting '%s' to API with payload=%s",
            action.name,
            _safe_json_preview(data),
        )
        r = self._session.post(
            f"{self.ROOT_URL}/api/cmd/{action.name}",
            json=json.loads(json_str),
            headers=self.headers,
        )
        logger.info(
            "Received response for '%s': status=%s",
            action.name,
            r.status_code,
        )
        if "error" in r.json():
            logger.warning(f"Exception during action request: {r.json()}")
        return r

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Submits the specific action and gets the next frame."""
        frame_data = self.do_action_request(action).json()
        key_preview = (
            list(frame_data.keys())
            if isinstance(frame_data, dict)
            else f"<non-dict:{type(frame_data).__name__}>"
        )
        logger.info(
            "Parsing frame data after '%s': keys=%s",
            getattr(action, "name", getattr(action, "action", "<unknown>")),
            key_preview,
        )
        try:
            frame = FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning(f"Incoming frame data did not validate: {e}")
            return None
        return frame

    def get_scorecard(self) -> Scorecard:
        """Get the scorecard for this agent's game as a Scorecard pydantic object."""
        r = self._session.get(
            f"{self.ROOT_URL}/api/scorecard/{self.card_id}/{self.game_id}",
            timeout=1,
            headers=self.headers,
        )
        response_data = r.json()
        if "error" in response_data:
            logger.warning(f"Exception during scorecard request: {response_data}")
        return Scorecard.model_validate(response_data)

    def cleanup(self, scorecard: Optional[Scorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent
            resolved_scorecard = scorecard
            if hasattr(self, "recorder") and not self.is_playback:
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                else:
                    scorecard_obj = self.get_scorecard()
                    resolved_scorecard = scorecard_obj
                    self.recorder.record(scorecard_obj.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"Exiting: agent reached MAX_ACTIONS of {self.MAX_ACTIONS}, took {self.seconds} seconds ({self.fps} average fps)"
                )
            else:
                logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )
            logger.info(
                "Final frame snapshot for game '%s': %s",
                self.game_id,
                self._format_frame_summary(self.frames[-1]) if self.frames else "<no frames>",
            )
            self._emit_run_report(
                "on_run_finished",
                agent=self,
                scorecard=resolved_scorecard,
            )
            if hasattr(self, "_session"):
                self._session.close()

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        raise NotImplementedError


class Playback(Agent):
    """An agent that plays back from a recorded session from another agent."""

    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 5

    recorded_actions: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = Recorder(
            prefix=Recorder.get_prefix(self.agent_name),
            guid=Recorder.get_guid(self.agent_name),
        )
        self.recorded_actions = []
        if self.agent_name in Recorder.list():
            try:
                self.recorded_actions = self.filter_actions()
                logger.info(
                    f"Loaded {len(self.recorded_actions)} actions from {self.agent_name}"
                )
            except Exception as e:
                logger.error(f"Failed to load recording {self.agent_name}: {e}")
                self.recorded_actions = []
        else:
            logger.warning(
                f"Recording {self.agent_name} not found in available recordings"
            )

    def filter_actions(self) -> list[dict[str, Any]]:
        return [
            a
            for a in self.recorder.get()
            if "data" in a and "action_input" in a["data"]
        ]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(self.action_counter >= len(self.recorded_actions))

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        loop_start_time = time.time()

        if self.action_counter >= len(self.recorded_actions):
            logger.warning(
                f"No more recorded actions available (counter: {self.action_counter}, total: {len(self.recorded_actions)})"
            )
            return GameAction.RESET

        recorded_data = self.recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]

        action = GameAction.from_id(action_input["id"])
        data = action_input["data"].copy()
        data["game_id"] = self.game_id
        action.set_data(data)
        if "reasoning" in action_input and action_input["reasoning"] is not None:
            action.reasoning = action_input["reasoning"]

        logger.debug(
            f"Playback action {self.action_counter}: {action.name} with data {data}"
        )

        target_frame_time = 1.0 / getattr(self, "PLAYBACK_FPS", 5)
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return action

    def append_frame(self, frame: FrameData) -> None:
        # overwrite append_frame to not double record
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
