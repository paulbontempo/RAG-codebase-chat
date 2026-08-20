from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ImportRecord:
    local_name: str
    imported_module: str  # e.g. "models" or "os.path"
    imported_symbol: (
        str | None
    )  # e.g. "User" for `from models import User`; None for `import models`
    lineno: int

    @property
    def resolved(self) -> str:
        return (
            f"{self.imported_module}.{self.imported_symbol}"
            if self.imported_symbol
            else self.imported_module
        )


@dataclass
class DefRecord:
    qualname: str
    kind: str  # "function" | "method" | "class"
    lineno: int
    end_lineno: int
    class_name: str | None = None


@dataclass
class CallRecord:
    caller_qualname: str
    callee: str
    lineno: int
    resolved: bool


@dataclass
class FileAnalysis:
    module: str
    imports: list[ImportRecord] = field(default_factory=list)
    defs: list[DefRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


class _RawCall:
    __slots__ = ("caller_qualname", "raw", "lineno")

    def __init__(self, caller_qualname: str, raw: str | None, lineno: int) -> None:
        self.caller_qualname = caller_qualname
        self.raw = raw
        self.lineno = lineno


def _relative_import_base(module: str, is_package_init: bool, level: int) -> str:
    """Resolves the package a relative import (`from . import x`, `from .. import y`)
    is anchored to. `level` is ast's count of leading dots. A package's __init__.py
    IS the package for level=1, whereas an ordinary submodule's level=1 refers to
    its *parent* package."""
    parts = module.split(".") if module else []
    up = level - 1 if is_package_init else level
    if up <= 0:
        return module
    if up >= len(parts):
        return ""
    return ".".join(parts[: len(parts) - up])


class _Analyzer(ast.NodeVisitor):
    def __init__(self, module: str, is_package_init: bool = False) -> None:
        self.module = module
        self.is_package_init = is_package_init
        self.imports: list[ImportRecord] = []
        self.defs: list[DefRecord] = []
        self._raw_calls: list[_RawCall] = []
        self._scope_stack: list[str] = []
        self._class_stack: list[str] = []

    @property
    def _current_scope(self) -> str:
        return self._scope_stack[-1] if self._scope_stack else self.module

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.imports.append(
                ImportRecord(
                    local_name=local,
                    imported_module=alias.name,
                    imported_symbol=None,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0:
            base = _relative_import_base(self.module, self.is_package_init, node.level)
            module = f"{base}.{node.module}" if node.module else base
        else:
            module = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            self.imports.append(
                ImportRecord(
                    local_name=local,
                    imported_module=module,
                    imported_symbol=alias.name,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = f"{self.module}.{node.name}"
        self.defs.append(
            DefRecord(
                qualname=qualname,
                kind="class",
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
            )
        )
        self._class_stack.append(node.name)
        self._scope_stack.append(qualname)
        for stmt in node.body:
            self.visit(stmt)
        self._scope_stack.pop()
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        in_class = (
            bool(self._class_stack)
            and self._scope_stack
            and self._scope_stack[-1] == (f"{self.module}.{self._class_stack[-1]}")
        )
        if in_class:
            qualname = f"{self._scope_stack[-1]}.{node.name}"
            kind = "method"
            class_name = self._class_stack[-1]
        else:
            qualname = (
                f"{self._current_scope}.{node.name}"
                if self._scope_stack
                else f"{self.module}.{node.name}"
            )
            kind = "function"
            class_name = None

        self.defs.append(
            DefRecord(
                qualname=qualname,
                kind=kind,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                class_name=class_name,
            )
        )
        self._scope_stack.append(qualname)
        for stmt in node.body:
            self.visit(stmt)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        self._raw_calls.append(_RawCall(self._current_scope, _dotted_name(node.func), node.lineno))
        self.generic_visit(node)

    def resolve(self) -> list[CallRecord]:
        imports_by_local = {imp.local_name: imp.resolved for imp in self.imports}
        local_top_names = {
            d.qualname.rsplit(".", 1)[-1]
            for d in self.defs
            if d.qualname.count(".") == self.module.count(".") + 1
        }
        resolved_calls: list[CallRecord] = []
        for rc in self._raw_calls:
            resolved_calls.append(self._resolve_one(rc, imports_by_local, local_top_names))
        return resolved_calls

    def _resolve_one(
        self, rc: _RawCall, imports_by_local: dict[str, str], local_top_names: set[str]
    ) -> CallRecord:
        if rc.raw is None:
            return CallRecord(rc.caller_qualname, "<dynamic>", rc.lineno, resolved=False)

        parts = rc.raw.split(".")
        head, rest = parts[0], parts[1:]

        if head in ("self", "cls") and len(rest) == 1:
            caller_parts = rc.caller_qualname.split(".")
            if len(caller_parts) >= 3:
                callee = ".".join([*caller_parts[:-1], rest[0]])
                return CallRecord(rc.caller_qualname, callee, rc.lineno, resolved=True)
            return CallRecord(rc.caller_qualname, rc.raw, rc.lineno, resolved=False)

        if head in imports_by_local:
            resolved_module = imports_by_local[head]
            callee = ".".join([resolved_module, *rest]) if rest else resolved_module
            return CallRecord(rc.caller_qualname, callee, rc.lineno, resolved=True)

        if head in local_top_names:
            callee = f"{self.module}.{rc.raw}"
            return CallRecord(rc.caller_qualname, callee, rc.lineno, resolved=True)

        return CallRecord(rc.caller_qualname, rc.raw, rc.lineno, resolved=False)


def analyze_file(module: str, source: str, is_package_init: bool = False) -> FileAnalysis:
    tree = ast.parse(source)
    analyzer = _Analyzer(module, is_package_init=is_package_init)
    for stmt in tree.body:
        analyzer.visit(stmt)
    return FileAnalysis(
        module=module,
        imports=analyzer.imports,
        defs=analyzer.defs,
        calls=analyzer.resolve(),
    )
