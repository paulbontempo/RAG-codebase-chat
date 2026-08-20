from pathlib import Path

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.eval.impact_eval import evaluate_impact_question
from codebase_chat_tool.graph.ast_visitor import analyze_file
from codebase_chat_tool.graph.call_graph import build_graph
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.ingestion.chunker import chunk_file


def _build_ctx() -> RepoContext:
    sources = {
        "utils": "def helper():\n    pass\n",
        "service": "from utils import helper\n\ndef run():\n    helper()\n",
    }
    analyses = [analyze_file(module, src) for module, src in sources.items()]
    resolver = GraphResolver(build_graph(analyses))
    chunks = []
    for module, src in sources.items():
        chunks.extend(chunk_file(module, f"{module}.py", src))
    return RepoContext(
        repo_root=Path("/fake"),
        settings=None,
        retriever=None,
        resolver=resolver,
        chunks_by_id={c.chunk_id: c for c in chunks},
        chunks_by_qualname={c.qualname: c for c in chunks},
    )


def test_evaluate_impact_question_perfect_match():
    ctx = _build_ctx()
    result = evaluate_impact_question(ctx, "utils.helper", ["service.run"], [])
    assert result.resolved is True
    assert result.direct_callers.precision == 1.0
    assert result.direct_callers.recall == 1.0
    assert result.direct_callers.f1 == 1.0


def test_evaluate_impact_question_partial_match_scores_between_zero_and_one():
    ctx = _build_ctx()
    result = evaluate_impact_question(
        ctx, "utils.helper", ["service.run", "nonexistent.caller"], []
    )
    assert result.direct_callers.precision == 1.0  # everything predicted is correct
    assert result.direct_callers.recall == 0.5  # missed one expected caller
    assert 0.0 < result.direct_callers.f1 < 1.0


def test_evaluate_impact_question_unresolvable_target_marks_unresolved():
    ctx = _build_ctx()
    result = evaluate_impact_question(ctx, "utils.does_not_exist", [], [])
    assert result.resolved is False
