"""Process supervision and physical-command budgets; no puzzle decisions."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler


@dataclass(frozen=True)
class RunLimits:
    workers: int = 2
    max_commands: int = 1000
    game_seconds: float = 900
    idle_seconds: float = 300
    request_seconds: float = 30
    total_seconds: float = 30600

    def __post_init__(self):
        if not 1 <= self.workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        if self.max_commands > 1000 or self.game_seconds > 900:
            raise ValueError("User limits: at most 1000 commands and 900 seconds per game")
        if self.max_commands < 1 or any(x <= 0 for x in (
            self.game_seconds, self.idle_seconds, self.request_seconds, self.total_seconds,
        )):
            raise ValueError("all command/time budgets must be positive")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')
    for attempt in range(11):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # A Windows reader/scanner may briefly hold the old report after a child exits.
            if attempt == 10:
                raise
            time.sleep(0.02)


def read_report(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def validate_gateway(url, *, allow_loopback=False):
    parsed = urlparse(url)
    production = (parsed.scheme, parsed.hostname, parsed.port) == ('http', 'gateway', 8001)
    local = allow_loopback and parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
    if not (production or local) or parsed.username or parsed.password or parsed.path not in ('', '/'):
        raise ValueError('Only the official gateway is allowed; loopback is explicit test-only mode')
    if parsed.query or parsed.fragment:
        raise ValueError('Gateway URL cannot contain a query or fragment')
    return url.rstrip('/')


class GatewayClient:
    def __init__(self, base_url, timeout=30, *, allow_loopback=False):
        self.base_url = validate_gateway(base_url, allow_loopback=allow_loopback)
        self.timeout = timeout
        # Never route the internal gateway through an external proxy.
        self.opener = build_opener(ProxyHandler({}))

    def request(self, path, payload=None, *, timeout=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request(self.base_url + path, data=data, headers={
            'Content-Type': 'application/json', 'X-API-Key': os.environ.get('ARC_API_KEY', ''),
        })
        with self.opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
            return json.loads(response.read())

    def ready_games(self, deadline):
        # Retrying discovery is read-only; action commands are never retried here.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Gateway discovery exhausted its budget')
            try:
                games = self.request('/api/games', timeout=min(self.timeout, remaining))
                if not isinstance(games, list) or not games:
                    raise ValueError('Gateway returned no game descriptors')
                ids = [g['game_id'] for g in games]
                if any(not isinstance(g, str) or not g for g in ids) or len(set(ids)) != len(ids):
                    raise ValueError('Gateway returned invalid or duplicate game IDs')
                return ids
            except (URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(1, max(0, deadline - time.monotonic())))


class CommandBudgetExceeded(RuntimeError):
    pass


def budgeted_transport(config, *, maximum, deadline, report_path, request_seconds):
    """Wrap only HTTP serialization; ActionBoundary still owns every release."""
    from agents.yf_arc3_v5.runtime.environment import HttpEnvironmentTransport

    class BudgetedTransport(HttpEnvironmentTransport):
        def __init__(self):
            super().__init__(config)
            self.count = 0
            self.completed = 0
            self.stop_reason = None
            self.last_command = None
            self.last_response = {}
            self.command_in_flight = False
            self.persist('starting')

        def persist(self, status, **extra):
            # No raw frames, cognitive states, action histories or API key.
            value = {
                'game_id': config.game_id, 'status': status,
                'commands_attempted': self.count, 'commands_completed': self.completed,
                'last_command': self.last_command, 'command_in_flight': self.command_in_flight,
                'last_activity': time.time(), 'levels_completed': self.last_response.get(
                    'levels_completed', self.last_response.get('score', 0)),
                'state': self.last_response.get('state'), 'stop_reason': self.stop_reason,
            }
            value.update(extra)
            atomic_json(report_path, value)

        def _post_json(self, path, payload):
            if self.stop_reason:
                raise CommandBudgetExceeded(self.stop_reason)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop_reason = 'time_budget'
                self.persist('time_budget')
                raise CommandBudgetExceeded(self.stop_reason)
            command = path.startswith('/api/cmd/')
            if command and self.count >= maximum:
                self.stop_reason = 'action_budget'
                self.persist('action_budget')
                raise CommandBudgetExceeded(self.stop_reason)
            self.config = self.config.model_copy(update={
                'timeout_seconds': min(request_seconds, remaining),
            })
            if command:
                self.count += 1  # Reserve before sending, including RESET and each series primitive.
                self.last_command = path.rsplit('/', 1)[-1]
                self.command_in_flight = True
                self.persist('running')
            try:
                response = super()._post_json(path, payload)
            except Exception:
                self.stop_reason = 'transport_error'
                self.persist('transport_error', outcome='uncertain' if command else 'failed')
                raise
            if command:
                self.completed += 1
                self.command_in_flight = False
                self.last_response = {k: response[k] for k in ('levels_completed', 'score', 'state') if k in response}
                self.persist('running')
            return response

    return BudgetedTransport()


def drive_controller(controller, transport, deadline):
    """Invoke V5's complete native cycles, including its source-declared series."""
    result = controller.start()
    while True:
        if transport.stop_reason:
            return transport.stop_reason
        if time.monotonic() >= deadline:
            return 'time_budget'
        if controller.current_world_input.state == 'WIN':
            return 'win'
        operation = result.get('operation') or {}
        if operation.get('status') in {'blocked', 'exhausted'}:
            return 'source_' + operation['status']
        # No action/RESET choice or puzzle-specific recovery is introduced here.
        result = controller.next_action()


def run_worker(job):
    from my_agent import MyAgent
    from agents.yf_arc3_v5.logos.types import FrozenMap
    from agents.yf_arc3_v5.runtime.config import LiveRunConfig
    from agents.yf_arc3_v5.runtime.controller import LiveController
    from agents.yf_arc3_v5.runtime.environment import HttpEnvironmentConfig, HttpTransportProfile

    os.environ['NO_PROXY'] = '*'
    deadline = job['deadline_monotonic']  # Includes process launch and imports on this host.
    config = HttpEnvironmentConfig(
        profile=HttpTransportProfile.ARC_API, base_url=job['gateway'], game_id=job['game_id'],
        card_id=job['card_id'], headers=FrozenMap({'X-API-Key': os.environ.get('ARC_API_KEY', '')}),
    )
    transport = budgeted_transport(config, maximum=job['max_commands'], deadline=deadline,
                                  report_path=job['report_path'], request_seconds=job['request_seconds'])
    controller = LiveController(LiveRunConfig(
        game_id=job['game_id'], backend='official', game_service_url=job['gateway'],
        run_namespace='kaggle', latency_guard_cyclic_gc=True,
    ), transport=transport)
    # Retain the official MyAgent identity, without the inherited 80-iteration main loop.
    agent = MyAgent(card_id=job['card_id'], game_id=job['game_id'], agent_name='myagent',
                    ROOT_URL=job['gateway'], record=False, controller=controller)
    try:
        status = drive_controller(agent._controller, transport, deadline)
        transport.persist(status)
    except Exception as error:
        transport.persist(transport.stop_reason or 'error', error_type=type(error).__name__)
        # Error messages may contain server observations or secrets: keep type only.
    return read_report(job['report_path'])


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def supervise(games, card_id, gateway, limits, output, *, worker_script=None, allow_loopback=False):
    gateway = validate_gateway(gateway, allow_loopback=allow_loopback)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if len(set(games)) != len(games):
        raise ValueError('One continuous worker per unique game is required')
    script = Path(worker_script or __file__).resolve()
    waiting, active, results = deque(games), {}, []
    deadline = time.monotonic() + limits.total_seconds
    try:
        while waiting or active:
            remaining = deadline - time.monotonic()
            while waiting and len(active) < limits.workers and remaining > 0:
                game_id = waiting.popleft()
                key = hashlib.sha256(game_id.encode()).hexdigest()[:16]
                report_path = output / (key + '.json')
                job_path = output / (key + '.job.json')
                waves = max(1, math.ceil((len(waiting) + len(active) + 1) / limits.workers))
                seconds = min(limits.game_seconds, remaining / waves)
                started = time.monotonic()
                job = {
                    'game_id': game_id, 'card_id': card_id, 'gateway': gateway,
                    'game_seconds': seconds, 'max_commands': limits.max_commands,
                    'deadline_monotonic': started + seconds,
                    'request_seconds': limits.request_seconds, 'report_path': str(report_path.resolve()),
                }
                atomic_json(job_path, job)
                atomic_json(report_path, {'game_id': game_id, 'status': 'starting',
                                          'commands_attempted': 0, 'commands_completed': 0,
                                          'last_activity': time.time()})
                env = dict(os.environ)
                for name in list(env):
                    if name.startswith(('PYTHONPATH', 'YF_ARC3_V5_', 'AGENTOPS_')):
                        env.pop(name)
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                env['ARC_LOG_LEVEL'] = 'WARNING'
                env['NO_PROXY'] = '*'
                # Direct child per game; no grandchild LLM/server processes.
                process = subprocess.Popen([sys.executable, '-I', '-B', str(script), '--worker', str(job_path.resolve())],
                                           cwd=script.parent, env=env, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                active[game_id] = (process, report_path, started, seconds)
                remaining = deadline - time.monotonic()
            now = time.monotonic()
            for game_id, (process, report_path, started, seconds) in list(active.items()):
                report = read_report(report_path)
                reason = None
                if now >= deadline:
                    reason = 'global_time_budget'
                elif now - started >= seconds:
                    reason = 'time_budget'
                elif time.time() - report.get('last_activity', time.time() - (now - started)) >= limits.idle_seconds:
                    reason = 'stalled'
                if process.poll() is None and not reason:
                    continue
                if reason:
                    stop_process(process)
                else:
                    process.wait(timeout=2)
                # Read the last durable counter after stopping; never overwrite it with zero.
                report = read_report(report_path)
                terminal = {'win', 'action_budget', 'time_budget', 'source_blocked',
                            'source_exhausted', 'error', 'transport_error'}
                if not reason and (process.returncode != 0 or report.get('status') not in terminal):
                    reason = 'crash'
                report.update({'game_id': game_id, 'status': reason or report['status'],
                               'process_exit_code': process.returncode,
                               'elapsed_seconds': round(time.monotonic() - started, 3)})
                atomic_json(report_path, report)
                results.append(report)
                del active[game_id]
                atomic_json(output / 'run_summary.json', {'limits': asdict(limits), 'games': results,
                                                         'complete': not waiting and not active})
            if time.monotonic() >= deadline:
                while waiting:
                    results.append({'game_id': waiting.popleft(), 'status': 'not_started_global_time_budget'})
            if active:
                time.sleep(0.05)
    finally:
        for process, _path, _started, _seconds in active.values():
            stop_process(process)
    return results


def run_competition(gateway, limits, output, *, allow_loopback=False, worker_script=None):
    start = time.monotonic()
    client = GatewayClient(gateway, limits.request_seconds, allow_loopback=allow_loopback)
    summary = {'limits': asdict(limits), 'games': [], 'complete': False, 'scorecard_closed': False}
    atomic_json(Path(output) / 'run_summary.json', summary)
    card_id = None
    try:
        games = client.ready_games(min(start + 600, start + limits.total_seconds))
        card_id = client.request('/api/scorecard/open', {'competition_mode': True})['card_id']
        remaining = limits.total_seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError('Preparation exhausted the global budget')
        run_limits = RunLimits(**{**asdict(limits), 'total_seconds': remaining})
        summary['games'] = supervise(games, card_id, gateway, run_limits, output,
                                    worker_script=worker_script, allow_loopback=allow_loopback)
        summary['complete'] = len(summary['games']) == len(games)
    except Exception as error:
        summary['error_type'] = type(error).__name__
        summary['games'] = read_report(Path(output) / 'run_summary.json').get('games', [])
    finally:
        if card_id:
            try:
                client.request('/api/scorecard/close', {'card_id': card_id})
                summary['scorecard_closed'] = True
            except Exception as error:
                summary['close_error_type'] = type(error).__name__
        summary['elapsed_seconds'] = round(time.monotonic() - start, 3)
        atomic_json(Path(output) / 'run_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--gateway', default='http://gateway:8001')
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/v5_reports'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-commands', type=int, default=1000)
    parser.add_argument('--game-seconds', type=float, default=900)
    parser.add_argument('--idle-seconds', type=float, default=300)
    parser.add_argument('--request-seconds', type=float, default=30)
    parser.add_argument('--total-seconds', type=float, default=30600)
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        run_worker(json.loads(args.worker.read_text(encoding='utf-8')))
        return 0
    limits = RunLimits(args.workers, args.max_commands, args.game_seconds,
                       args.idle_seconds, args.request_seconds, args.total_seconds)
    summary = run_competition(args.gateway, limits, args.output)
    print(json.dumps({'complete': summary['complete'], 'scorecard_closed': summary['scorecard_closed'],
                      'games': len(summary['games']), 'error_type': summary.get('error_type')}))
    return 0 if summary['complete'] and summary['scorecard_closed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
