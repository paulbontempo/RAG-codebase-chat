from dataclasses import dataclass

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.agent.tools import find_tests_for


@dataclass
class SetScore:
    precision: float
    recall: float
    f1: float


def _prf1(predicted: set[str], expected: set[str]) -> SetScore:
    if not predicted and not expected:
        return SetScore(1.0, 1.0, 1.0)
    if not predicted or not expected:
        return SetScore(0.0, 0.0, 0.0)
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted)
    recall = true_positives / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return SetScore(precision, recall, f1)


@dataclass
class ImpactQuestionResult:
    target: str
    resolved: bool
    direct_callers: SetScore
    tests: SetScore


def evaluate_impact_question(
    ctx: RepoContext, target: str, expected_direct_callers: list[str], expected_tests: list[str]
) -> ImpactQuestionResult:
    """Purely deterministic: walks the static call graph directly (no LLM call)
    and compares against hand-verified gold caller/test sets via set precision/
    recall/F1."""
    if target not in ctx.chunks_by_qualname and target not in ctx.resolver.graph.nodes:
        return ImpactQuestionResult(
            target=target, resolved=False, direct_callers=SetScore(0, 0, 0), tests=SetScore(0, 0, 0)
        )

    predicted_callers = {c.qualname for c in ctx.resolver.direct_callers(target)}
    predicted_tests = {t["qualname"] for t in find_tests_for(ctx, target)}

    return ImpactQuestionResult(
        target=target,
        resolved=True,
        direct_callers=_prf1(predicted_callers, set(expected_direct_callers)),
        tests=_prf1(predicted_tests, set(expected_tests)),
    )


def aggregate_impact_scores(results: list[ImpactQuestionResult]) -> dict[str, float]:
    resolved = [r for r in results if r.resolved]
    if not resolved:
        return {
            "caller_precision": 0.0,
            "caller_recall": 0.0,
            "caller_f1": 0.0,
            "resolution_rate": 0.0,
        }
    n = len(resolved)
    return {
        "caller_precision": sum(r.direct_callers.precision for r in resolved) / n,
        "caller_recall": sum(r.direct_callers.recall for r in resolved) / n,
        "caller_f1": sum(r.direct_callers.f1 for r in resolved) / n,
        "test_f1": sum(r.tests.f1 for r in resolved) / n,
        "resolution_rate": len(resolved) / len(results),
    }
