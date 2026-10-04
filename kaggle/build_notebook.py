"""Generate a portable notebook and deterministic agent-input ZIP; never upload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from build_submission import build, AREA, OUTPUT

ARTIFACTS = AREA / 'artifacts'
NOTEBOOK = ARTIFACTS / 'arc3_v5_submission.ipynb'
ARCHIVE = ARTIFACTS / 'v5_runtime.zip'


def make_archive():
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUTPUT.rglob('*')):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(OUTPUT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()


def cell(source, kind='code'):
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def make_notebook(archive_hash):
    manifest_hash = hashlib.sha256((OUTPUT / 'MANIFEST.json').read_bytes()).hexdigest()
    cells = [
        cell('# V5 ARC-AGI-3 submission candidate\n\n'
             'Attach `v5_runtime.zip` as a private notebook input. This notebook does not upload or submit itself.\n'
             'Save/run-all checks the package and emits the official starter placeholder. '
             'Competition rerun uses only the provided gateway.\n', 'markdown'),
        cell('''import os, sys, time
from pathlib import Path
NOTEBOOK_STARTED = time.monotonic()
if sys.version_info < (3, 12):
    raise RuntimeError("Kaggle target requires Python >=3.12")
IS_RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in {"1", "true"}
# Operational limits, independent of puzzle rules. All game IDs come from the gateway.
WORKERS = 2
MAX_COMMANDS_PER_GAME = 1000  # Includes RESET and every primitive of a series.
MAX_SECONDS_PER_GAME = 900  # User limit: 15 minutes.
IDLE_SECONDS = 300
REQUEST_SECONDS = 30
NOTEBOOK_SECONDS = 9 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
WHEEL_DIRS = [Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")]
# Add an attached offline wheel dataset here only if the competition wheelhouse lacks a dependency.
'''),
        cell('''import subprocess
install = [sys.executable, "-m", "pip", "install", "--no-index", "--disable-pip-version-check"]
for directory in WHEEL_DIRS:
    if directory.is_dir():
        install += ["--find-links", str(directory)]
install += ["pydantic>=2.11.7,<3", "requests>=2.32.4,<3", "pandas", "pyarrow"]
subprocess.run(install, check=True)
from importlib.metadata import version
print({name: version(name) for name in ["pydantic", "requests", "pandas", "pyarrow"]})
'''),
        cell(f'''import hashlib, json, shutil, zipfile
EXPECTED_ZIP_SHA256 = {archive_hash!r}
EXPECTED_MANIFEST_SHA256 = {manifest_hash!r}
archives = list(Path("/kaggle/input").rglob("v5_runtime.zip"))
BUNDLE_ROOT = Path("/tmp/arc3_v5_" + EXPECTED_ZIP_SHA256[:16])
BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
if len(archives) == 1:
    if hashlib.sha256(archives[0].read_bytes()).hexdigest() != EXPECTED_ZIP_SHA256:
        raise RuntimeError("Notebook and runtime ZIP are from different builds")
    with zipfile.ZipFile(archives[0]) as archive:
        for name in archive.namelist():
            target = (BUNDLE_ROOT / name).resolve()
            if BUNDLE_ROOT.resolve() not in target.parents:
                raise RuntimeError("Archive path escapes bundle")
        archive.extractall(BUNDLE_ROOT)
elif not archives:
    # Also accept a dataset whose upload workflow has already extracted the ZIP.
    candidates = [path for path in Path("/kaggle/input").rglob("MANIFEST.json")
                  if hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256]
    if len(candidates) != 1:
        raise RuntimeError("Attach exactly one runtime dataset matching this notebook")
    source_root = candidates[0].parent.resolve()
    source_manifest = json.loads(candidates[0].read_text())
    for name in [entry["path"] for entry in source_manifest["files"]] + ["MANIFEST.json"]:
        source, target = (source_root / name).resolve(), (BUNDLE_ROOT / name).resolve()
        if source_root not in source.parents or BUNDLE_ROOT.resolve() not in target.parents:
            raise RuntimeError("Dataset path escapes bundle")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
else:
    raise RuntimeError("Attach exactly one runtime dataset matching this notebook")
if hashlib.sha256((BUNDLE_ROOT / "MANIFEST.json").read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("Runtime manifest differs from this notebook build")
manifest = json.loads((BUNDLE_ROOT / "MANIFEST.json").read_text())
expected_files = {{entry["path"] for entry in manifest["files"]}} | {{"MANIFEST.json"}}
actual_files = {{str(path.relative_to(BUNDLE_ROOT)).replace('\\\\', '/') for path in BUNDLE_ROOT.rglob('*') if path.is_file()}}
if actual_files != expected_files:
    raise RuntimeError("Unexpected or missing files in the extracted bundle")
for entry in manifest["files"]:
    data = (BUNDLE_ROOT / entry["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError("Runtime file failed its manifest hash")
print({{"source": manifest["source_git_commit"], "files": manifest["total_file_count"]}})
'''),
        cell('''# Validate package import outside any mounted framework package.
probe = "import sys; sys.path.insert(0, sys.argv[1]); from my_agent import MyAgent; import competition_runner; print('bundle imports: PASS')"
subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(BUNDLE_ROOT)], check=True)
'''),
        cell('''if IS_RERUN:
    os.environ.setdefault("ARC_API_KEY", "test-key-123")  # Official starter gateway credential.
    remaining = NOTEBOOK_SECONDS - (time.monotonic() - NOTEBOOK_STARTED) - FINAL_RESERVE_SECONDS
    if remaining <= 0:
        raise RuntimeError("No competition budget remains after setup")
    subprocess.run([
        sys.executable, "-I", "-B", str(BUNDLE_ROOT / "competition_runner.py"),
        "--gateway", "http://gateway:8001", "--output", "/kaggle/working/v5_reports",
        "--workers", str(WORKERS), "--max-commands", str(MAX_COMMANDS_PER_GAME),
        "--game-seconds", str(MAX_SECONDS_PER_GAME), "--idle-seconds", str(IDLE_SECONDS),
        "--request-seconds", str(REQUEST_SECONDS), "--total-seconds", str(remaining),
    ], check=True)
    # The gateway, not V5, must produce the actual scoring file.
    submission = Path("/kaggle/working/submission.parquet")
    wait_until = time.monotonic() + 60
    while not submission.is_file() and time.monotonic() < wait_until:
        time.sleep(1)
    if not submission.is_file() or submission.stat().st_size == 0:
        raise RuntimeError("Gateway submission.parquet absent; no fabricated competition result")
else:
    # Official starter placeholder for Save & Run All; not a measured score.
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=[
        "row_id", "game_id", "end_of_game", "score",
    ]).to_parquet("/kaggle/working/submission.parquet", index=False)
    print("Save/run-all package check only; actual gateway execution is still unverified")
'''),
    ]
    return {'nbformat': 4, 'nbformat_minor': 4, 'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'kaggle': {'accelerator': 'none', 'isInternetEnabled': False, 'isGpuEnabled': False,
                   'language': 'python', 'sourceType': 'notebook'},
    }}


def main():
    manifest = build()
    ARTIFACTS.mkdir(exist_ok=True)
    archive_hash = make_archive()
    NOTEBOOK.write_text(json.dumps(make_notebook(archive_hash), indent=1) + '\n', encoding='utf-8')
    print(f'Notebook: {NOTEBOOK}; ZIP: {ARCHIVE.stat().st_size:,} bytes; SHA256: {archive_hash}')
    return manifest


if __name__ == '__main__':
    main()
