"""Executed in a separate -I interpreter with only a copied bundle on sys.path."""

import json
import os
from pathlib import Path
import socket
import site
import sys
import time

sys.dont_write_bytecode = True
bundle, repository = map(Path, sys.argv[1:3])
sys.path.insert(0, str(bundle))
dependency_sites = tuple(Path(p).resolve() for p in site.getsitepackages())


def deny_network(*args, **kwargs):
    raise AssertionError("External network access during isolated inference")


socket.socket.connect = deny_network
socket.create_connection = deny_network


def deny_repository(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        installed_dependency = any(path == p or p in path.parents for p in dependency_sites)
        if (path == repository or repository in path.parents) and not installed_dependency:
            raise AssertionError(f"Read outside bundle into development repository: {path}")
    if event in {"subprocess.Popen", "os.system"}:
        raise AssertionError("Subprocess during isolated inference")


sys.addaudithook(deny_repository)
from my_agent import MyAgent
from agents.structs import GameAction, GameState, FrameData
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.runtime.contracts import (
    ObservedWorldInput, SessionBootstrapResponse, EnvironmentActionResponse,
)
from agents.yf_arc3_v5.runtime.environment import (
    HttpEnvironmentTransport, HttpEnvironmentConfig, HttpTransportProfile,
)


class SyntheticTransport:
    stop_reason = None
    def __init__(self):
        self.bootstraps = 0
        self.requests = []

    def bootstrap(self):
        self.bootstraps += 1
        return SessionBootstrapResponse(raw_input=ObservedWorldInput(
            raw_input_ref="raw:synthetic:0", frame_id="frame:synthetic:0",
            frame=((0, 1), (0, 1)), available_action_refs=("ACTION1", "ACTION2"),
            metadata=FrozenMap({"game_id": "synthetic", "level": 1}),
        ))

    def execute(self, request):
        self.requests.append(request)
        assert request.permit.action_intent_id == request.intent.id
        observed = ObservedWorldInput(
            raw_input_ref="raw:synthetic:1", frame_id="frame:synthetic:1",
            frame=((1, 0), (1, 0)), available_action_refs=("ACTION1", "ACTION2"),
            state="WIN", score=1,
            metadata=FrozenMap({"game_id": "synthetic", "level": 1}),
        )
        return EnvironmentActionResponse(
            dispatch_id=request.dispatch_id,
            environment_action_event_ref="event:synthetic:1",
            action_intent_id=request.intent.id, action_release_permit_id=request.permit.id,
            raw_input=observed, response_digest=stable_digest(observed),
        )


kwargs = dict(card_id="synthetic", game_id="synthetic", agent_name="myagent",
              ROOT_URL="http://gateway:8001", record=False)
transport = SyntheticTransport()
agent = MyAgent(**kwargs, transport=transport)
assert not agent.is_done(agent.frames, agent.frames[-1])
reset = agent.choose_action(agent.frames, agent.frames[-1])
assert reset is GameAction.RESET
observed = agent.take_action(reset)  # Real V5 resource compilation + cognition.
assert transport.bootstraps == 1
assert observed.frame == [[[0, 1], [0, 1]]]
action = agent.choose_action([observed], observed)
assert action in (GameAction.ACTION1, GameAction.ACTION2)
assert action.reasoning['authority'] == 'source_action_release_permit'
terminal = agent.take_action(action)  # Real single-authority boundary + reconciliation.
assert len(transport.requests) == 1
assert terminal.state is GameState.WIN
assert agent.is_done([terminal], terminal)

# The competition driver uses full native cycles instead of the legacy 80-iteration loop.
from competition_runner import drive_controller, budgeted_transport, CommandBudgetExceeded
native_transport = SyntheticTransport()
native_agent = MyAgent(**kwargs, transport=native_transport)
assert drive_controller(native_agent._controller, native_transport, time.monotonic() + 60) == 'win'
assert len(native_transport.requests) == 1

# Count physical commands, including bootstrap RESET and every primitive in a series.
original_post = HttpEnvironmentTransport._post_json
sent = []
def synthetic_post(self, path, payload):
    sent.append(path)
    return {'levels_completed': 0, 'state': 'NOT_FINISHED'}
HttpEnvironmentTransport._post_json = synthetic_post
try:
    capped = budgeted_transport(HttpEnvironmentConfig(
        profile=HttpTransportProfile.ARC_API, base_url='http://gateway:8001', game_id='synthetic'),
        maximum=3, deadline=time.monotonic() + 60,
        report_path=bundle.parent / 'physical_budget.json', request_seconds=30)
    for action in ('RESET', 'ACTION1', 'ACTION2'):
        capped._post_json('/api/cmd/' + action, {})
    try:
        capped._post_json('/api/cmd/ACTION2', {})
        raise AssertionError('Physical command cap was exceeded')
    except CommandBudgetExceeded:
        pass
    assert capped.count == capped.completed == 3 and len(sent) == 3
    assert capped.stop_reason == 'action_budget'
    def fail_post(self, path, payload):
        sent.append(path)
        raise TimeoutError('synthetic uncertain delivery')
    HttpEnvironmentTransport._post_json = fail_post
    uncertain = budgeted_transport(HttpEnvironmentConfig(
        profile=HttpTransportProfile.ARC_API, base_url='http://gateway:8001', game_id='synthetic'),
        maximum=1000, deadline=time.monotonic() + 60,
        report_path=bundle.parent / 'uncertain_budget.json', request_seconds=30)
    try:
        uncertain._post_json('/api/cmd/ACTION1', {})
    except TimeoutError:
        pass
    try:
        uncertain._post_json('/api/cmd/ACTION1', {})
    except CommandBudgetExceeded:
        pass
    assert uncertain.count == 1 and uncertain.completed == 0 and len(sent) == 4
    assert uncertain.command_in_flight and uncertain.stop_reason == 'transport_error'
finally:
    HttpEnvironmentTransport._post_json = original_post
try:
    agent.take_action(GameAction.RESET)
    raise AssertionError("Second bootstrap RESET was accepted")
except RuntimeError:
    pass

# ACTION6 and ACTION7 conversion isolate adapter representation from cognition.
# This mock never claims to prove cognitive point/undo selection.
class PendingController:
    def __init__(self):
        self.action_ref = "ACTION6"
        self.data = {"x": 17, "y": 29}

    def control_context(self):
        return {"pending_action": {"action_ref": self.action_ref, "action_data": self.data,
                "action_intent_id": "intent:point", "source_hash": "synthetic-source"}}


pending = PendingController()
point = MyAgent(**kwargs, controller=pending)
point._bootstrapped = True
selected = point.choose_action([], observed)
assert selected is GameAction.ACTION6
assert selected.action_data.x == 17 and selected.action_data.y == 29
assert GameAction.from_id(7) is GameAction.ACTION7 and GameAction.ACTION7.is_simple()
pending.action_ref, pending.data = "ACTION7", {}
assert point.choose_action([], observed) is GameAction.ACTION7
pending.action_ref = "ACTION6"
try:
    point.choose_action([], observed)
    raise AssertionError("Missing point coordinates silently defaulted")
except RuntimeError:
    pass

# Verify the unchanged HTTP adapter carries point data and modern score fields.
http = HttpEnvironmentTransport(HttpEnvironmentConfig(
    profile=HttpTransportProfile.ARC_API, base_url="http://gateway:8001", game_id="synthetic",
))
wire = []


def post(path, payload):
    wire.append((path, payload))
    return {"game_id": "synthetic", "frame": [[[0, 1], [0, 1]]],
            "state": "NOT_FINISHED", "levels_completed": 1, "available_actions": [6, 7]}


http._post_json = post
original = transport.requests[0]
request = original.model_copy(update={"intent": original.intent.model_copy(update={
    "action_ref": "ACTION6", "action_data": FrozenMap({"x": 17, "y": 29}),
})})
response = http.execute(request)
assert wire[0][0] == '/api/cmd/ACTION6'
assert wire[0][1]['x'] == 17 and wire[0][1]['y'] == 29
assert 'data' not in wire[0][1]
assert response.raw_input.score == 1
assert response.raw_input.available_action_refs == ('ACTION6', 'ACTION7')

fresh = MyAgent(**{**kwargs, "game_id": "synthetic-new"}, transport=SyntheticTransport())
assert fresh.choose_action(fresh.frames, fresh.frames[-1]) is GameAction.RESET
assert fresh._controller is not agent._controller
assert not agent.is_done([], FrameData(state=GameState.GAME_OVER))
try:
    agent.choose_action([], FrameData(state=GameState.GAME_OVER))
    raise AssertionError("Terminal recovery bypassed source authority")
except RuntimeError:
    pass
assert not any(n.startswith(('api', 'tools', 'custom_games')) for n in sys.modules)
assert not (bundle / '.env').exists()
try:
    MyAgent(**{**kwargs, "ROOT_URL": "https://external.example"})
    raise AssertionError("External HTTP endpoint accepted")
except ValueError:
    pass
try:
    MyAgent(**kwargs, arc_env=object())
    raise AssertionError("Unverified arc_env interface accepted")
except RuntimeError:
    pass
os.environ['YF_ARC3_V5_MANUAL_PLAY'] = 'true'
try:
    MyAgent(**kwargs)
    raise AssertionError("Manual play accepted")
except RuntimeError:
    pass
finally:
    del os.environ['YF_ARC3_V5_MANUAL_PLAY']
print(json.dumps({"result": "PASS", "real_controller_start": True,
                  "source_authorized_actions": len(transport.requests),
                  "point_payload": {"x": 17, "y": 29}, "terminal": terminal.state.value,
                  "network": "blocked", "repository_reads": "blocked",
                  "python": sys.version.split()[0]}, sort_keys=True))
