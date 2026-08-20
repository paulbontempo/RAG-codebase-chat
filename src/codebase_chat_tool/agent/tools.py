from typing import Any

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.llm.base import ToolSpec


def search_code(ctx: RepoContext, query: str, top_k: int = 5) -> list[dict]:
    results = ctx.retriever.search(query, top_k=top_k)
    return [
        {
            "file": r.chunk.file_path,
            "qualname": r.chunk.qualname,
            "kind": r.chunk.kind,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "text": r.chunk.text,
        }
        for r in results
    ]


def get_definition(ctx: RepoContext, qualname: str) -> dict | None:
    chunk = ctx.chunks_by_qualname.get(qualname)
    if chunk is None:
        return None
    return {
        "file": chunk.file_path,
        "qualname": chunk.qualname,
        "kind": chunk.kind,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
    }


def get_callers(ctx: RepoContext, qualname: str, transitive: bool = False) -> list[str]:
    infos = (
        ctx.resolver.transitive_callers(qualname)
        if transitive
        else ctx.resolver.direct_callers(qualname)
    )
    return sorted({c.qualname for c in infos})


def get_callees(ctx: RepoContext, qualname: str) -> list[str]:
    return sorted({c.qualname for c in ctx.resolver.direct_callees(qualname)})


def get_importers(ctx: RepoContext, module: str) -> list[str]:
    return sorted(ctx.resolver.importers(module))


def find_tests_for(ctx: RepoContext, qualname: str) -> list[dict]:
    """Heuristic: chunks in test files (under a 'tests' dir, or named test_*.py) whose
    text references the symbol's short name. No AST-level call resolution into test
    files is attempted here; this is intentionally a fast, best-effort heuristic."""
    short_name = qualname.rsplit(".", 1)[-1]
    results = []
    for chunk in ctx.chunks_by_id.values():
        path_parts = chunk.file_path.split("/")
        is_test_file = "tests" in path_parts[:-1] or path_parts[-1].startswith("test_")
        if is_test_file and short_name in chunk.text:
            results.append(
                {
                    "file": chunk.file_path,
                    "qualname": chunk.qualname,
                    "start_line": chunk.start_line,
                }
            )
    return results


TOOL_FUNCS = {
    "search_code": search_code,
    "get_definition": get_definition,
    "get_callers": get_callers,
    "get_callees": get_callees,
    "get_importers": get_importers,
    "find_tests_for": find_tests_for,
}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_code",
        description=(
            "Hybrid semantic + keyword search over the indexed codebase. "
            "Use this first for open-ended questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language or keyword query."},
                "top_k": {"type": "integer", "description": "Number of results (default 5)."},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_definition",
        description=(
            "Fetch the exact source of a known fully-qualified symbol, "
            "e.g. 'service.UserService.get_user'."
        ),
        parameters={
            "type": "object",
            "properties": {"qualname": {"type": "string"}},
            "required": ["qualname"],
        },
    ),
    ToolSpec(
        name="get_callers",
        description="List callers of a fully-qualified symbol, from the static call graph.",
        parameters={
            "type": "object",
            "properties": {
                "qualname": {"type": "string"},
                "transitive": {
                    "type": "boolean",
                    "description": "If true, include indirect (transitive) callers.",
                },
            },
            "required": ["qualname"],
        },
    ),
    ToolSpec(
        name="get_callees",
        description=(
            "List symbols that a given fully-qualified symbol calls, from the static call graph."
        ),
        parameters={
            "type": "object",
            "properties": {"qualname": {"type": "string"}},
            "required": ["qualname"],
        },
    ),
    ToolSpec(
        name="get_importers",
        description="List modules that import a given module.",
        parameters={
            "type": "object",
            "properties": {"module": {"type": "string"}},
            "required": ["module"],
        },
    ),
    ToolSpec(
        name="find_tests_for",
        description="Find test functions that likely cover a given fully-qualified symbol.",
        parameters={
            "type": "object",
            "properties": {"qualname": {"type": "string"}},
            "required": ["qualname"],
        },
    ),
]


def call_tool(ctx: RepoContext, name: str, arguments: dict[str, Any]) -> Any:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    return func(ctx, **arguments)
