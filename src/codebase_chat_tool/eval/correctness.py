from dataclasses import dataclass


@dataclass
class CorrectnessScore:
    matched: int
    total: int

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 1.0


def score_keyword_coverage(answer_text: str, expected_keywords: list[str]) -> CorrectnessScore:
    """Deterministic fact-coverage check: does the answer contain each required
    keyword/fact (case-insensitive substring match)? No LLM judgment involved --
    every question's gold keywords are hand-picked so that their presence is a
    reasonable proxy for "the answer contains the actually-correct information".
    """
    if not expected_keywords:
        return CorrectnessScore(matched=0, total=0)
    lowered = answer_text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in lowered)
    return CorrectnessScore(matched=matched, total=len(expected_keywords))


def aggregate_correctness_scores(scores: list[CorrectnessScore]) -> dict[str, float]:
    scored = [s for s in scores if s.total > 0]
    if not scored:
        return {"keyword_coverage": 0.0}
    return {"keyword_coverage": sum(s.rate for s in scored) / len(scored)}
