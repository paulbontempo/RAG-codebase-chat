from codebase_chat_tool.eval.correctness import score_keyword_coverage
from codebase_chat_tool.eval.groundedness import score_groundedness
from codebase_chat_tool.eval.retrieval_metrics import score_retrieval
from codebase_chat_tool.llm.base import Message


def test_retrieval_score_perfect_match_at_rank_one():
    score = score_retrieval(["pkg.mod.foo"], ["pkg.mod.foo"])
    assert score.precision_at_k == 1.0
    assert score.recall_at_k == 1.0
    assert score.reciprocal_rank == 1.0


def test_retrieval_score_partial_match_ranked_second():
    score = score_retrieval(["pkg.mod.other", "pkg.mod.foo"], ["pkg.mod.foo"])
    assert score.precision_at_k == 0.5
    assert score.recall_at_k == 1.0
    assert score.reciprocal_rank == 0.5


def test_retrieval_score_handles_class_vs_method_granularity():
    # retrieving the containing class chunk should count as a hit for a method-level question
    score = score_retrieval(["pkg.mod.MyClass"], ["pkg.mod.MyClass.my_method"])
    assert score.recall_at_k == 1.0


def test_retrieval_score_no_hits():
    score = score_retrieval(["pkg.mod.unrelated"], ["pkg.mod.foo"])
    assert score.precision_at_k == 0.0
    assert score.recall_at_k == 0.0
    assert score.reciprocal_rank == 0.0


def test_keyword_coverage_all_present():
    score = score_keyword_coverage("The Timeout exception is raised.", ["Timeout", "raised"])
    assert score.matched == 2
    assert score.rate == 1.0


def test_keyword_coverage_case_insensitive():
    score = score_keyword_coverage("the timeout exception", ["Timeout"])
    assert score.matched == 1


def test_keyword_coverage_partial():
    score = score_keyword_coverage("mentions only one thing", ["thing", "nonexistent"])
    assert score.matched == 1
    assert score.rate == 0.5


def test_groundedness_citation_matches_seen_tool_result():
    answer = "See `sessions.py:651` for the implementation."
    messages = [
        Message(
            role="tool",
            content='{"file": "requests/sessions.py", "start_line": 640, "end_line": 660}',
        )
    ]
    score = score_groundedness(answer, messages)
    assert score.has_citations
    assert score.grounded_count == 1
    assert score.rate == 1.0


def test_groundedness_citation_outside_seen_range_is_ungrounded():
    answer = "See `sessions.py:999` for the implementation."
    messages = [
        Message(
            role="tool",
            content='{"file": "requests/sessions.py", "start_line": 640, "end_line": 660}',
        )
    ]
    score = score_groundedness(answer, messages)
    assert score.has_citations
    assert score.grounded_count == 0
    assert score.rate == 0.0


def test_groundedness_no_citations_reports_has_citations_false():
    score = score_groundedness("No file references here.", [])
    assert score.has_citations is False
    assert score.citation_count == 0
