from codebase_chat_tool.eval.regression import check_regression


def _result(repo: str, recall: float, grounded: float, keyword: float, caller_f1: float) -> dict:
    return {
        "repo": repo,
        "retrieval": {"recall_at_k": recall},
        "groundedness": {"grounded_rate": grounded},
        "correctness": {"keyword_coverage": keyword},
        "impact": {"caller_f1": caller_f1},
    }


def test_no_failures_when_metrics_match_baseline():
    baseline = [_result("requests", 0.8, 0.9, 1.0, 1.0)]
    current = [_result("requests", 0.8, 0.9, 1.0, 1.0)]
    assert check_regression(current, baseline) == []


def test_no_failures_within_tolerance():
    baseline = [_result("requests", 0.8, 0.9, 1.0, 1.0)]
    current = [_result("requests", 0.75, 0.85, 0.95, 0.95)]  # small drops, within default tolerance
    assert check_regression(current, baseline, tolerance=0.10) == []


def test_failure_reported_when_metric_drops_beyond_tolerance():
    baseline = [_result("requests", 0.8, 0.9, 1.0, 1.0)]
    current = [_result("requests", 0.3, 0.9, 1.0, 1.0)]  # recall cratered
    failures = check_regression(current, baseline, tolerance=0.10)
    assert len(failures) == 1
    assert failures[0].repo == "requests"
    assert failures[0].metric == "retrieval.recall_at_k"


def test_improvement_never_flagged_as_failure():
    baseline = [_result("requests", 0.5, 0.5, 0.5, 0.5)]
    current = [_result("requests", 0.9, 0.9, 0.9, 0.9)]
    assert check_regression(current, baseline) == []


def test_repo_missing_from_baseline_is_skipped_not_failed():
    baseline = [_result("requests", 0.8, 0.9, 1.0, 1.0)]
    current = [_result("new_repo", 0.0, 0.0, 0.0, 0.0)]
    assert check_regression(current, baseline) == []


def test_multiple_metrics_can_fail_at_once():
    baseline = [_result("requests", 0.9, 0.9, 0.9, 0.9)]
    current = [_result("requests", 0.1, 0.1, 0.9, 0.9)]
    failures = check_regression(current, baseline, tolerance=0.10)
    metrics = {f.metric for f in failures}
    assert metrics == {"retrieval.recall_at_k", "groundedness.grounded_rate"}
