"""Bounded packaging integration, with synthetic workers and a loopback gateway."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

AREA = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('competition_runner', AREA / 'competition_runner.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def worker(tmp_path):
    script = tmp_path / 'synthetic_worker.py'
    script.write_text('''import json, pathlib, sys, time
job = json.loads(pathlib.Path(sys.argv[sys.argv.index('--worker') + 1]).read_text())
path = pathlib.Path(job['report_path'])
start = time.time()
if job['game_id'] == 'crash':
    raise SystemExit(3)
if job['game_id'] == 'hang':
    time.sleep(30)
time.sleep(0.15)
report = {'status': 'win', 'commands_attempted': 1, 'commands_completed': 1,
          'last_activity': time.time(), 'synthetic_start': start, 'synthetic_end': time.time()}
temporary = path.with_suffix('.part')
temporary.write_text(json.dumps(report))
temporary.replace(path)
''', encoding='utf-8')
    return script


def test_user_limits_are_enforced():
    limits = runner.RunLimits()
    assert (limits.max_commands, limits.game_seconds, limits.workers) == (1000, 900, 2)
    for change in ({'max_commands': 1001}, {'game_seconds': 901}, {'max_commands': 0}):
        with pytest.raises(ValueError):
            runner.RunLimits(**change)


def test_parallelism_and_crash_isolation(tmp_path, worker):
    results = runner.supervise(['a', 'crash', 'b', 'c'], 'card', 'http://gateway:8001',
                              runner.RunLimits(game_seconds=5, total_seconds=15), tmp_path / 'reports',
                              worker_script=worker)
    by_game = {r['game_id']: r for r in results}
    assert by_game['crash']['status'] == 'crash'
    assert all(by_game[g]['status'] == 'win' for g in ('a', 'b', 'c'))
    events = [(r['synthetic_start'], 1) for r in results if 'synthetic_start' in r]
    events += [(r['synthetic_end'], -1) for r in results if 'synthetic_end' in r]
    active = maximum = 0
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    assert maximum <= 2
    assert all(json.loads(p.read_text())['max_commands'] == 1000
               for p in (tmp_path / 'reports').glob('*.job.json'))


@pytest.mark.parametrize('idle, seconds, expected', [(0.6, 5, 'stalled'), (10, 0.7, 'time_budget')])
def test_hung_worker_stops_and_queue_continues(tmp_path, worker, idle, seconds, expected):
    results = runner.supervise(['hang', 'later'], 'card', 'http://gateway:8001',
                              runner.RunLimits(workers=1, game_seconds=seconds, idle_seconds=idle,
                                               total_seconds=10), tmp_path, worker_script=worker)
    assert {r['game_id']: r['status'] for r in results} == {'hang': expected, 'later': 'win'}
    assert next(r for r in results if r['game_id'] == 'hang')['process_exit_code'] is not None


def test_global_budget_accounts_for_unstarted_games(tmp_path, worker):
    results = runner.supervise(['hang', 'later', 'last'], 'card', 'http://gateway:8001',
                              runner.RunLimits(workers=1, total_seconds=0.6), tmp_path,
                              worker_script=worker)
    assert {r['game_id'] for r in results} == {'hang', 'later', 'last'}
    assert any(r['status'] in {'global_time_budget', 'not_started_global_time_budget'} for r in results)


def test_gateway_lifecycle_closes_after_game_crash(tmp_path, worker):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append((self.path, None))
            self.respond([{'game_id': 'a'}, {'game_id': 'crash'}])

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, payload))
            self.respond({'card_id': 'synthetic-card'})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        summary = runner.run_competition(f'http://127.0.0.1:{server.server_port}',
                                        runner.RunLimits(game_seconds=5, total_seconds=15), tmp_path,
                                        allow_loopback=True, worker_script=worker)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert summary['complete'] and summary['scorecard_closed']
    assert [c[0] for c in calls] == ['/api/games', '/api/scorecard/open', '/api/scorecard/close']
    assert calls[1][1] == {'competition_mode': True}
    assert calls[2][1] == {'card_id': 'synthetic-card'}
    assert {r['status'] for r in summary['games']} == {'win', 'crash'}


def test_external_gateway_rejected():
    for url in ('https://example.com', 'http://gateway:8001/x', 'http://127.0.0.1:8001'):
        with pytest.raises(ValueError):
            runner.validate_gateway(url)
