"""Build a source-qualified, audited runtime closure. No network or downloads."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "kaggle"
OUTPUT = AREA / "submission"
V5 = ROOT / "agents/yf_arc3_v5"
VERSION = "3"
REQUIRED = ["pydantic>=2.11.7,<3", "requests>=2.32.4,<3"]
GENERATED_INIT = b'"""Isolated V5 package; no template discovery or dotenv loading."""\n'
FORBIDDEN = {
    "api", "standalone_game_api", "custom_games", "tests", "runs", "docs",
    ".git", ".github", ".codex", ".agents", "environment_files", "screenshots",
    "logs", "node_modules", "__pycache__",
}
STATIC_SUFFIXES = {".src", ".drm", ".logos", ".json", ".yaml", ".yml", ".txt"}
SPECIAL_CALLS = {
    "open", "read_text", "read_bytes", "glob", "rglob", "import_module",
    "__import__", "getenv", "urlopen", "Popen", "run", "system",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_submission_glue(files):
    imports = []
    for name in ('adapter.py', 'competition_runner.py'):
        tree = ast.parse((AREA / name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                if module == 'my_agent':
                    continue
                if module.startswith('agents.'):
                    if resolve_module(module) not in files:
                        raise ValueError(f'Glue import outside audited closure: {module}')
                elif module.split('.')[0] not in sys.stdlib_module_names:
                    raise ValueError(f'Unreviewed submission glue dependency: {module}')
                imports.append({'file': name, 'line': node.lineno, 'module': module})
    return imports


def resolve_module(name):
    parts = name.split(".")
    for path in (ROOT.joinpath(*parts).with_suffix(".py"), ROOT.joinpath(*parts) / "__init__.py"):
        if path.is_file():
            return path.relative_to(ROOT).as_posix()
    return None


def dependency_closure():
    todo = ["agents/yf_arc3_v5/agent.py"]
    seen, edges, external, calls, literals, exclusions = set(), [], set(), [], set(), []
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = ROOT / rel
        # Package initializers execute too, except the explicitly generated root.
        for parent in path.parents:
            if parent == ROOT:
                break
            init = parent / "__init__.py"
            if init.is_file() and init != ROOT / "agents/__init__.py":
                todo.append(init.relative_to(ROOT).as_posix())
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        owners = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    owners[id(child)] = node.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.split(".")[-1] in SPECIAL_CALLS or "environ" in name:
                    calls.append({"file": rel, "line": node.lineno, "call": ast.unparse(node)[:240]})
                    if name.endswith("import_module"):
                        raise ValueError(f"Unreviewed dynamic import: {rel}:{node.lineno}")
                    if name == "__import__" and not (
                        node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "os"
                    ):
                        raise ValueError(f"Unreviewed dynamic import: {rel}:{node.lineno}")
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    package = path.parent.relative_to(ROOT).parts
                    prefix = ".".join((*package[:len(package) - node.level + 1], prefix)).rstrip(".")
                names = [prefix, *(prefix + "." + item.name for item in node.names)]
            for name in names:
                if name == "tools.manual_oracle_recorder":
                    if rel != "agents/yf_arc3_v5/runtime/controller.py" or owners.get(id(node)) != "manual_action":
                        raise ValueError("Human recorder import escaped its reviewed human-only method")
                    exclusions.append({"file": rel, "line": node.lineno, "module": name,
                                       "reason": "human-only recording; no inference dependency"})
                    continue
                top = name.split(".")[0]
                if top in {"agents", "tools", "api", "custom_games"}:
                    target = resolve_module(name)
                    if target:
                        if not target.startswith("agents/"):
                            raise ValueError(f"Unreviewed local dependency {name} from {rel}")
                        if target == "agents/__init__.py":
                            continue
                        edges.append({"from": rel, "line": node.lineno, "to": target})
                        todo.append(target)
                elif top:
                    external.add(top)
    allowed = set(sys.stdlib_module_names) | {"__future__", "pydantic", "pydantic_core", "requests", "agentops"}
    unknown = external - allowed
    if unknown:
        raise ValueError(f"Unreviewed external dependencies: {sorted(unknown)}")
    # Resolve literals against runtime resources, retaining their original paths.
    # This includes source-unit/hash references, not merely loader call arguments.
    resources = set()
    candidates = [p for p in V5.rglob("*") if p.is_file() and p.suffix in STATIC_SUFFIXES]
    resource_literals = {s for s in literals if Path(s).suffix in STATIC_SUFFIXES and "\n" not in s}
    for path in candidates:
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.endswith(s.replace("\\", "/")) for s in resource_literals):
            resources.add(rel)
    for literal in resource_literals:
        if literal.startswith("agents/") and literal not in resources and (ROOT / literal).is_file():
            raise ValueError(f"Resource outside reviewed V5 closure: {literal}")
    inventory = {
        "entrypoint": "agents/yf_arc3_v5/agent.py", "python_files": sorted(seen),
        "resources": sorted(resources), "import_edges": sorted(edges, key=lambda x: (x['from'], x['line'], x['to'])),
        "external_import_roots": sorted(external), "execution_sensitive_calls": sorted(calls, key=lambda x: (x['file'], x['line'])),
        "reviewed_exclusions": exclusions,
        "resource_discovery": "AST resource literals and source-hash references; isolated startup validates loading",
    }
    return seen | resources, inventory


def validate_files(directory, expected):
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink forbidden: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(directory).as_posix()
        actual.add(rel)
        lower = rel.lower()
        if (FORBIDDEN.intersection(path.relative_to(directory).parts)
                or "manual_oracle" in lower or "_engine" in lower
                or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".log", ".tmp", ".pyc"}
                or path.name == ".env"):
            raise ValueError(f"Forbidden submission file: {rel}")
        if rel not in expected:
            raise ValueError(f"File outside audited dependency closure: {rel}")
    if actual != set(expected):
        raise ValueError(f"Missing submission files: {sorted(set(expected) - actual)}")


def leakage_findings(directory):
    # Public release scope: official ARC-AGI-3 game identifiers only.
    ids = set("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 "
              "m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 "
              "tu93 vc33 wa30".split())
    patterns = {
        "game_identifier": re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, ids))) + r")\b", re.I),
        "oracle_or_environment": re.compile(r"manual_oracle|docs[/\\]oracles|standalone_game_api|custom_games|environment_files|(?:ls20|lp85|ft09)_engine", re.I),
        "solution_or_level_reference": re.compile(r"\b(?:known.?solution|hard.?coded|oracle|level[_ -]?id|action[_ -]?sequence|level[_ -]?layout)\b", re.I),
        "numbered_level": re.compile(r"\b(?:LVL|L|level[_ -]?)[0-9]+\b", re.I),
        "literal_action_witness": re.compile(r"[\[(]\s*['\"](?:ACTION[1-7]|RESET)['\"]\s*,\s*['\"](?:ACTION[1-7]|RESET)['\"]\s*,\s*['\"](?:ACTION[1-7]|RESET)['\"]"),
    }
    findings = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            for kind, pattern in patterns.items():
                if pattern.search(line):
                    rel = path.relative_to(directory).as_posix()
                    key = digest((rel + "\n" + kind + "\n" + line.strip()).encode())
                    findings.append({"id": key, "file": rel, "line": line_no, "kind": kind,
                                     "text": line.strip()[:350]})
    return findings


def isolated_import(directory):
    # -I removes cwd/PYTHONPATH/user-site; explicitly insert only the copied bundle.
    with tempfile.TemporaryDirectory(prefix="arc3-kaggle-import-") as temp:
        bundle = Path(temp) / "bundle"
        shutil.copytree(directory, bundle)
        code = (
            "import sys,socket; sys.dont_write_bytecode=True; "
            f"sys.path.insert(0,{str(bundle)!r}); "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled')); "
            "from my_agent import MyAgent; "
            "a=MyAgent(card_id='synthetic',game_id='synthetic',agent_name='myagent',"
            "ROOT_URL='http://gateway:8001',record=False); "
            "from agents.structs import GameAction; "
            "assert a.choose_action(a.frames,a.frames[-1]) is GameAction.RESET; "
            "assert not any(n.startswith(('tools','api','custom_games')) for n in sys.modules); "
            "print('isolated import, instantiate and bootstrap selection: PASS')"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "ARC_", "YF_ARC3", "AGENTOPS"))}
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=temp, env=env,
                                capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError("Isolated import failed:\n" + result.stderr[-2500:])
        return result.stdout.strip()


def build():
    # Never delete a computed output outside this precise generated directory.
    if OUTPUT.resolve() != ROOT.resolve() / "kaggle" / "submission" or OUTPUT.is_symlink():
        raise ValueError("Unsafe generated output path")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    try:
        files, inventory = dependency_closure()
        inventory['submission_glue_imports'] = validate_submission_glue(files)
        write_json(AREA / "dependency_inventory.json", inventory)
        for rel in sorted(files):
            source = ROOT / rel
            if source.is_symlink():
                raise ValueError(f"Source symlink forbidden: {rel}")
            target = OUTPUT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (OUTPUT / "agents/__init__.py").write_bytes(GENERATED_INIT)
        shutil.copyfile(AREA / "adapter.py", OUTPUT / "my_agent.py")
        shutil.copyfile(AREA / "competition_runner.py", OUTPUT / "competition_runner.py")
        shutil.copyfile(AREA / "LICENSE", OUTPUT / "LICENSE")
        shutil.copyfile(AREA / "THIRD_PARTY_NOTICES.md", OUTPUT / "THIRD_PARTY_NOTICES.md")
        expected = files | {
            "agents/__init__.py", "my_agent.py", "competition_runner.py",
            "LICENSE", "THIRD_PARTY_NOTICES.md",
        }
        validate_files(OUTPUT, expected)
        findings = leakage_findings(OUTPUT)
        write_json(AREA / "leakage_findings.json", findings)
        review_path = AREA / "leakage_review.json"
        review = json.loads(review_path.read_text()) if review_path.exists() else {}
        unknown = {x['id'] for x in findings} - set(review)
        if unknown or any(not review[x['id']].get('reason') for x in findings if x['id'] in review):
            raise ValueError(f"{len(unknown)} unreviewed leakage findings; inspect kaggle/leakage_findings.json")
        import_result = isolated_import(OUTPUT)
        entries = [{"path": rel, "sha256": digest((OUTPUT / rel).read_bytes()),
                    "bytes": (OUTPUT / rel).stat().st_size} for rel in sorted(expected)]
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        changed = subprocess.run(['git', 'diff', '--name-only', 'HEAD', '--', *sorted(files)],
                                 cwd=ROOT, capture_output=True, text=True, check=True)
        manifest = {
            "builder_version": VERSION, "source_git_commit": commit.stdout.strip() or None,
            "source_git_diff_clean": not changed.stdout.strip(),
            "python_requires": ">=3.12", "external_python_dependencies": REQUIRED,
            "files": entries, "total_file_count": len(entries) + 1,
            "payload_byte_size": sum(x['bytes'] for x in entries),
            "total_byte_size": 0, "manifest_self_hash": "excluded to avoid recursive self-hash",
            "leakage_findings": len(findings), "isolated_import": import_result,
        }
        # Total includes the manifest; solve its small decimal-length fixed point.
        while True:
            write_json(OUTPUT / "MANIFEST.json", manifest)
            size = manifest['payload_byte_size'] + (OUTPUT / "MANIFEST.json").stat().st_size
            if size == manifest['total_byte_size']:
                break
            manifest['total_byte_size'] = size
        validate_files(OUTPUT, expected | {"MANIFEST.json"})
        print(f"Bundle: {manifest['total_file_count']} files, {size:,} bytes; {import_result}")
        print("Notebook runner included; actual Kaggle image/gateway validation remains pending.")
        return manifest
    except Exception:
        # A rejected candidate must never remain as an apparently usable submission.
        shutil.rmtree(OUTPUT)
        raise


if __name__ == "__main__":
    build()
