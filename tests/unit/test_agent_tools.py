from pathlib import Path

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.agent.tools import (
    find_tests_for,
    get_callees,
    get_callers,
    get_definition,
    get_importers,
)
from codebase_chat_tool.graph.ast_visitor import analyze_file
from codebase_chat_tool.graph.call_graph import build_graph
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.ingestion.chunker import chunk_file
from codebase_chat_tool.ingestion.metadata import Chunk


def _build_ctx() -> RepoContext:
    sources = {
        "utils": "def helper():\n    pass\n",
        "service": "from utils import helper\n\ndef run():\n    helper()\n",
    }
    analyses = [analyze_file(module, src) for module, src in sources.items()]
    resolver = GraphResolver(build_graph(analyses))

    chunks: list[Chunk] = []
    for module, src in sources.items():
        chunks.extend(chunk_file(module, f"{module}.py", src))
    test_chunks = chunk_file(
        "tests.test_service",
        "tests/test_service.py",
        "from service import run\n\ndef test_run():\n    run()\n",
    )
    chunks.extend(test_chunks)

    return RepoContext(
        repo_root=Path("/fake"),
        settings=None,
        retriever=None,
        resolver=resolver,
        chunks_by_id={c.chunk_id: c for c in chunks},
        chunks_by_qualname={c.qualname: c for c in chunks},
    )


def test_get_definition_returns_chunk_details():
    ctx = _build_ctx()
    result = get_definition(ctx, "utils.helper")
    assert result["file"] == "utils.py"
    assert result["kind"] == "function"
    assert "def helper" in result["text"]


def test_get_definition_returns_none_for_unknown_symbol():
    ctx = _build_ctx()
    assert get_definition(ctx, "utils.nonexistent") is None


def test_get_callers_and_callees():
    ctx = _build_ctx()
    assert get_callers(ctx, "utils.helper") == ["service.run"]
    assert get_callees(ctx, "service.run") == ["utils.helper"]


def test_get_importers():
    ctx = _build_ctx()
    assert get_importers(ctx, "utils") == ["service"]


def test_find_tests_for_matches_test_file_referencing_symbol():
    ctx = _build_ctx()
    results = find_tests_for(ctx, "service.run")
    qualnames = {r["qualname"] for r in results}
    assert "tests.test_service.test_run" in qualnames


def test_find_tests_for_returns_empty_when_no_test_references_symbol():
    ctx = _build_ctx()
    assert find_tests_for(ctx, "utils.helper") == []
