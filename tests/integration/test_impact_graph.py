import sys
from pathlib import Path

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.agent.impact_graph import run_impact_analysis
from codebase_chat_tool.graph.ast_visitor import analyze_file
from codebase_chat_tool.graph.call_graph import build_graph
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.ingestion.chunker import chunk_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fakes import FakeLLMProvider, text_response  # noqa: E402


def _build_ctx() -> RepoContext:
    sources = {
        "utils": "def helper():\n    pass\n",
        "service": "from utils import helper\n\ndef run():\n    helper()\n",
        "main": "from service import run\n\ndef main():\n    run()\n",
    }
    analyses = [analyze_file(module, src) for module, src in sources.items()]
    resolver = GraphResolver(build_graph(analyses))

    chunks = []
    for module, src in sources.items():
        chunks.extend(chunk_file(module, f"{module}.py", src))
    chunks.extend(
        chunk_file(
            "tests.test_utils",
            "tests/test_utils.py",
            "from utils import helper\n\ndef test_helper():\n    helper()\n",
        )
    )

    return RepoContext(
        repo_root=Path("/fake"),
        settings=None,
        retriever=None,
        resolver=resolver,
        chunks_by_id={c.chunk_id: c for c in chunks},
        chunks_by_qualname={c.qualname: c for c in chunks},
    )


def test_impact_analysis_finds_direct_and_transitive_callers():
    ctx = _build_ctx()
    provider = FakeLLMProvider([text_response("Low risk.")])

    result = run_impact_analysis(ctx, provider, "utils.helper")

    assert result["error"] is None
    assert result["resolved_target"] == "utils.helper"
    assert result["direct_callers"] == ["service.run"]
    assert result["transitive_callers"] == ["main.main"]
    assert result["risk_summary"] == "Low risk."
    assert len(provider.calls) == 1


def test_impact_analysis_finds_dependent_modules_and_tests():
    ctx = _build_ctx()
    provider = FakeLLMProvider([text_response("Low risk.")])

    result = run_impact_analysis(ctx, provider, "utils.helper")

    assert result["target_module"] == "utils"
    assert result["dependent_modules"] == ["service"]
    test_qualnames = {t["qualname"] for t in result["related_tests"]}
    assert "tests.test_utils.test_helper" in test_qualnames


def test_impact_analysis_reports_error_for_unknown_symbol_without_calling_llm():
    ctx = _build_ctx()
    provider = FakeLLMProvider([])  # should never be called

    result = run_impact_analysis(ctx, provider, "utils.nonexistent")

    assert result["error"] is not None
    assert "not found" in result["error"]
    assert result.get("risk_summary") is None
    assert provider.calls == []


def test_impact_analysis_resolves_short_name_when_unambiguous():
    ctx = _build_ctx()
    provider = FakeLLMProvider([text_response("Low risk.")])

    result = run_impact_analysis(ctx, provider, "helper")

    assert result["error"] is None
    assert result["resolved_target"] == "utils.helper"


def test_impact_analysis_reports_ambiguous_short_name_without_calling_llm():
    ctx = _build_ctx()
    ctx.chunks_by_qualname["other.run"] = ctx.chunks_by_qualname["service.run"]  # any Chunk works
    provider = FakeLLMProvider([])  # should never be called -- must ask for clarification instead

    result = run_impact_analysis(ctx, provider, "run")

    assert result["error"] is not None
    assert "Ambiguous" in result["error"]
    assert "service.run" in result["error"]
    assert "other.run" in result["error"]
    assert result.get("risk_summary") is None
    assert provider.calls == []
