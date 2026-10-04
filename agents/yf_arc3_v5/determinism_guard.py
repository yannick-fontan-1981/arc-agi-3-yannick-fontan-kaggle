"""Static fail-closed determinism contract for production V5 sources.

The runtime may measure elapsed time for observability, but no production
decision source may obtain entropy, depend on process-randomized hashing, or
consume an explicitly unordered iterator.  The repository CLI additionally
applies the strict ordering checks to every line added on the active branch.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping


V5_ROOT = Path(__file__).resolve().parent

_SOURCE_SUFFIXES = frozenset({".py", ".src", ".drm", ".logos", ".json"})
_ENTROPY_MODULES = frozenset({"random", "secrets", "uuid"})
_ENTROPY_CALLS = frozenset(
    {
        "hash",
        "os.urandom",
        "random.random",
        "random.randrange",
        "random.randint",
        "random.choice",
        "random.choices",
        "random.sample",
        "random.shuffle",
        "secrets.choice",
        "secrets.randbelow",
        "secrets.token_bytes",
        "secrets.token_hex",
        "secrets.token_urlsafe",
        "uuid.uuid1",
        "uuid.uuid4",
    }
)
_RACE_CALL_SUFFIXES = frozenset(
    {
        "as_completed",
        "ThreadPoolExecutor",
        "ProcessPoolExecutor",
    }
)
_ORDERLESS_MAPPING_METHODS = frozenset({"items", "keys", "values"})
_ORDERLESS_CONSTRUCTORS = frozenset({"set", "frozenset"})
_OPAQUE_REPRESENTATION_CALLS = frozenset({"repr"})
_DECLARATIVE_ENTROPY_TOKENS = frozenset(
    {"random", "shuffle", "uuid", "wall_time", "timestamp"}
)


@dataclass(frozen=True)
class DeterminismFinding:
    rule: str
    path: str
    line: int
    detail: str


def _qualified_name(value: ast.AST) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        prefix = _qualified_name(value.value)
        return f"{prefix}.{value.attr}" if prefix else value.attr
    return ""


def _is_sorted_iteration(value: ast.AST) -> bool:
    if not isinstance(value, ast.Call):
        return False
    name = _qualified_name(value.func)
    if name == "sorted":
        return True
    if name in {"enumerate", "reversed"} and value.args:
        return _is_sorted_iteration(value.args[0])
    return False


def _orderless_iteration_kind(value: ast.AST) -> str | None:
    if isinstance(value, (ast.Set, ast.SetComp)):
        return "set"
    if not isinstance(value, ast.Call):
        return None
    name = _qualified_name(value.func)
    suffix = name.rsplit(".", 1)[-1]
    if suffix in _ORDERLESS_MAPPING_METHODS:
        return f"mapping.{suffix}"
    if name in _ORDERLESS_CONSTRUCTORS:
        return name
    return None


class _PythonDeterminismVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        path: Path,
        display_path: str,
        strict_lines: frozenset[int],
    ) -> None:
        self.path = path
        self.display_path = display_path
        self.strict_lines = strict_lines
        self.findings: list[DeterminismFinding] = []
        self.parents: dict[ast.AST, ast.AST] = {}

    def bind_parents(self, tree: ast.AST) -> None:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent

    def _record(self, rule: str, node: ast.AST, detail: str) -> None:
        self.findings.append(
            DeterminismFinding(
                rule=rule,
                path=self.display_path,
                line=int(getattr(node, "lineno", 0) or 0),
                detail=detail,
            )
        )

    def _strict(self, node: ast.AST) -> bool:
        return int(getattr(node, "lineno", 0) or 0) in self.strict_lines

    def _inside_sorted_materialization(self, node: ast.AST) -> bool:
        current = node
        while current in self.parents:
            current = self.parents[current]
            if isinstance(current, ast.Call) and _qualified_name(current.func) == "sorted":
                return True
            if isinstance(current, (ast.For, ast.AsyncFor)):
                return False
        return False

    def _check_iteration(self, node: ast.AST, iterator: ast.AST) -> None:
        # ``ast.comprehension`` itself has no line metadata; its iterator does.
        if not self._strict(iterator) or _is_sorted_iteration(iterator):
            return
        kind = _orderless_iteration_kind(iterator)
        if kind is None or self._inside_sorted_materialization(node):
            return
        self._record(
            "orderless_iteration_added",
            node,
            f"new production iteration over {kind} must be wrapped in sorted(...)",
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.partition(".")[0] in _ENTROPY_MODULES:
                self._record(
                    "entropy_module_forbidden",
                    node,
                    f"production V5 may not import {alias.name}",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = str(node.module or "").partition(".")[0]
        if module in _ENTROPY_MODULES:
            self._record(
                "entropy_module_forbidden",
                node,
                f"production V5 may not import from {node.module}",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _qualified_name(node.func)
        suffix = name.rsplit(".", 1)[-1]
        if name in _ENTROPY_CALLS:
            self._record(
                "entropy_or_process_hash_forbidden",
                node,
                f"production V5 may not call {name}",
            )
        if name in _OPAQUE_REPRESENTATION_CALLS:
            self._record(
                "opaque_representation_forbidden",
                node,
                "production V5 must use an exact typed descriptor, not repr()",
            )
        if suffix in _RACE_CALL_SUFFIXES:
            self._record(
                "race_order_primitive_forbidden",
                node,
                f"production V5 may not call {name}",
            )
        if suffix == "popitem":
            self._record(
                "orderless_removal_forbidden",
                node,
                "popitem() makes the removed member depend on container order",
            )
        if self._strict(node) and name == "json.dumps":
            sort_keyword = next(
                (keyword for keyword in node.keywords if keyword.arg == "sort_keys"),
                None,
            )
            if not (
                sort_keyword is not None
                and isinstance(sort_keyword.value, ast.Constant)
                and sort_keyword.value.value is True
            ):
                self._record(
                    "noncanonical_json_added",
                    node,
                    "new JSON serialization must set sort_keys=True",
                )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_iteration(node, node.iter)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._check_iteration(node, node.iter)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._check_iteration(node, node.iter)
        self.generic_visit(node)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root.parent.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _python_findings(
    path: Path,
    *,
    root: Path,
    strict_lines: frozenset[int],
) -> tuple[DeterminismFinding, ...]:
    display_path = _display_path(path, root)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        return (
            DeterminismFinding(
                rule="determinism_source_unreadable",
                path=display_path,
                line=int(getattr(error, "lineno", 0) or 0),
                detail=str(error),
            ),
        )
    visitor = _PythonDeterminismVisitor(
        path=path,
        display_path=display_path,
        strict_lines=strict_lines,
    )
    visitor.bind_parents(tree)
    visitor.visit(tree)
    if path.name != "performance.py":
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _qualified_name(node.func) in {
                "time.time",
                "time.monotonic",
                "time.perf_counter",
                "datetime.now",
            }:
                visitor._record(
                    "decision_time_source_forbidden",
                    node,
                    "wall/monotonic time is restricted to performance.py observability",
                )
    return tuple(visitor.findings)


def _declarative_findings(path: Path, *, root: Path) -> tuple[DeterminismFinding, ...]:
    display_path = _display_path(path, root)
    text = path.read_text(encoding="utf-8")
    findings: list[DeterminismFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = {
            token.strip(".,:;()[]{}\"'").lower()
            for token in line.split()
        }
        forbidden_tokens = tokens & _DECLARATIVE_ENTROPY_TOKENS
        if forbidden_tokens:
            forbidden = tuple(sorted(forbidden_tokens))
            findings.append(
                DeterminismFinding(
                    rule="declarative_entropy_token_forbidden",
                    path=display_path,
                    line=line_number,
                    detail=f"declarative source contains entropy token(s): {', '.join(forbidden)}",
                )
            )
    return tuple(findings)


def audit_v5_determinism(
    *,
    root: Path = V5_ROOT,
    strict_changed_lines: Mapping[Path, Iterable[int]] | None = None,
) -> tuple[DeterminismFinding, ...]:
    """Audit every V5 source, including explicit order and canonical JSON."""

    resolved_root = root.resolve()
    strict_index = {
        path.resolve(): frozenset(int(line) for line in lines)
        for path, lines in sorted(
            (strict_changed_lines or {}).items(),
            key=lambda item: item[0].as_posix(),
        )
    }
    findings: list[DeterminismFinding] = []
    paths = tuple(
        sorted(
            (
                path
                for path in resolved_root.rglob("*")
                if path.is_file()
                and path.suffix in _SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ),
            key=lambda path: path.as_posix(),
        )
    )
    for path in paths:
        if path.suffix == ".py":
            findings.extend(
                _python_findings(
                    path,
                    root=resolved_root,
                    strict_lines=strict_index.get(
                        path.resolve(),
                        frozenset(
                            range(
                                1,
                                len(path.read_text(encoding="utf-8").splitlines())
                                + 1,
                            )
                        ),
                    ),
                )
            )
        else:
            findings.extend(_declarative_findings(path, root=resolved_root))
    return tuple(
        sorted(
            findings,
            key=lambda item: (item.path, item.line, item.rule, item.detail),
        )
    )


def _source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix in _SOURCE_SUFFIXES
            and "__pycache__" not in candidate.parts
        ),
        key=lambda candidate: candidate.as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def v5_source_tree_digest(*, root: Path = V5_ROOT) -> str:
    """Return the exact cache key for all guarded V5 decision sources."""

    return _source_tree_digest(root.resolve())


@lru_cache(maxsize=8)
def _cached_severe_audit(
    root_text: str,
    source_tree_digest: str,
) -> tuple[DeterminismFinding, ...]:
    # The digest is deliberately part of the cache key: an edit in the same
    # Python process necessarily triggers a fresh AST audit before compilation.
    del source_tree_digest
    return audit_v5_determinism(root=Path(root_text))


def assert_v5_determinism(*, root: Path = V5_ROOT) -> None:
    """Fail before production SRC compilation when severe rules are violated."""

    resolved_root = root.resolve()
    findings = _cached_severe_audit(
        str(resolved_root),
        _source_tree_digest(resolved_root),
    )
    if not findings:
        return
    rendered = "\n".join(
        f"{item.rule}: {item.path}:{item.line}: {item.detail}"
        for item in findings
    )
    raise RuntimeError(f"V5 determinism guard rejected production sources:\n{rendered}")


__all__ = [
    "DeterminismFinding",
    "V5_ROOT",
    "assert_v5_determinism",
    "v5_source_tree_digest",
    "audit_v5_determinism",
]
