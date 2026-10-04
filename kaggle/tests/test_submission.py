"""Packaging checks only: no game engines or broad ARC regression campaign."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / 'kaggle'
spec = importlib.util.spec_from_file_location('kaggle_builder', AREA / 'build_submission.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope='session')
def built():
    return builder.build()


def test_manifest_and_authoritative_bytes(built):
    output = AREA / 'submission'
    files = [p for p in output.rglob('*') if p.is_file()]
    assert len(files) == built['total_file_count']
    assert sum(p.stat().st_size for p in files) == built['total_byte_size']
    for entry in built['files']:
        data = (output / entry['path']).read_bytes()
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        if entry['path'] not in {
            'agents/__init__.py', 'my_agent.py', 'competition_runner.py',
            'LICENSE', 'THIRD_PARTY_NOTICES.md',
        }:
            assert data == (ROOT / entry['path']).read_bytes()
        elif entry['path'] == 'competition_runner.py':
            assert data == (AREA / 'competition_runner.py').read_bytes()
        elif entry['path'] in {'LICENSE', 'THIRD_PARTY_NOTICES.md'}:
            assert data == (AREA / entry['path']).read_bytes()
    notices = (output / 'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
    assert 'Copyright (c) 2025 ARC Prize' in notices
    assert 'Pydantic' in notices and 'PyArrow' in notices
    assert 'Copyright (c) 2026 Yannick Fontan' in (output / 'LICENSE').read_text(encoding='utf-8')


def test_notebook_archive_and_extraction(built, tmp_path, monkeypatch):
    import sys
    import zipfile
    sys.path.insert(0, str(AREA))
    try:
        import build_notebook as notebook
    finally:
        sys.path.remove(str(AREA))
    archive = tmp_path / 'input' / 'v5_runtime.zip'
    archive.parent.mkdir()
    monkeypatch.setattr(notebook, 'ARCHIVE', archive)
    digest = notebook.make_archive()
    original = archive.read_bytes()
    assert notebook.make_archive() == digest and archive.read_bytes() == original
    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == {e['path'] for e in built['files']} | {'MANIFEST.json'}
    document = notebook.make_notebook(digest)
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec')
    configuration = ''.join(document['cells'][1]['source'])
    assert 'MAX_COMMANDS_PER_GAME = 1000' in configuration
    assert 'MAX_SECONDS_PER_GAME = 900' in configuration
    extraction = ''.join(document['cells'][3]['source'])
    extraction = extraction.replace('/kaggle/input', (tmp_path / 'input').as_posix())
    extraction = extraction.replace('/tmp/arc3_v5_', (tmp_path / 'extracted_').as_posix())
    namespace = {'Path': Path}
    exec(extraction, namespace)
    assert (namespace['BUNDLE_ROOT'] / 'competition_runner.py').is_file()
    (namespace['BUNDLE_ROOT'] / 'unexpected.py').write_text('raise RuntimeError()')
    with pytest.raises(RuntimeError, match='Unexpected or missing'):
        exec(extraction, namespace)
    (namespace['BUNDLE_ROOT'] / 'unexpected.py').unlink()
    with zipfile.ZipFile(archive) as package:
        package.extractall(archive.parent / 'expanded')
    archive.unlink()  # Dataset upload may expose the extracted directory instead.
    exec(extraction, namespace)
    assert (namespace['BUNDLE_ROOT'] / 'my_agent.py').is_file()


def test_isolated_offline_controller(built, tmp_path):
    bundle = tmp_path / 'bundle'
    shutil.copytree(AREA / 'submission', bundle)
    script = tmp_path / 'smoke.py'
    shutil.copyfile(AREA / 'tests/isolated_smoke.py', script)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'ARC_', 'YF_ARC3', 'AGENTOPS'))}
    result = subprocess.run([sys.executable, '-I', str(script), str(bundle), str(ROOT)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-3500:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['result'] == 'PASS'
    assert report['source_authorized_actions'] == 1
    builder.write_json(AREA / 'isolation_report.json', report)


@pytest.mark.parametrize('rel', ['docs/answer.json', 'agents/leaked_manual_oracle.py',
                                'agents/new_unreviewed_module.py', 'local/LS20_engine.py'])
def test_forbidden_and_unreviewed_files_rejected(tmp_path, rel):
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError):
        builder.validate_files(tmp_path, set())


def test_new_leakage_fails_review(tmp_path):
    (tmp_path / 'puzzle.py').write_text('GAME = "ls20"\nSOLUTION = ["ACTION1", "ACTION2", "ACTION3"]\n')
    findings = builder.leakage_findings(tmp_path)
    assert {'game_identifier', 'literal_action_witness'} <= {f['kind'] for f in findings}
    reviewed = json.loads((AREA / 'leakage_review.json').read_text())
    assert {f['id'] for f in findings} - set(reviewed)


def test_resource_closure_includes_discovered_declarations(built):
    resources = set(json.loads((AREA / 'dependency_inventory.json').read_text())['resources'])
    assert 'agents/yf_arc3_v5/drm/dynamic_workflow_components.json' in resources
    assert 'agents/yf_arc3_v5/src/production_workflow_contracts.json' in resources
    assert any(p.endswith('.src') for p in resources)
    assert any(p.endswith('.drm') for p in resources)


def test_rebuild_is_deterministic(built):
    first = (AREA / 'submission/MANIFEST.json').read_bytes()
    rebuilt = builder.build()
    assert rebuilt == built
    assert first == (AREA / 'submission/MANIFEST.json').read_bytes()
