from dataclasses import dataclass


@dataclass
class RetrievalScore:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def score_retrieval(
    retrieved_qualnames: list[str], expected_qualnames: list[str]
) -> RetrievalScore:
    """Standard IR metrics: precision@k, recall@k, and reciprocal rank (for MRR
    aggregation), comparing a ranked list of retrieved qualnames against a gold
    set of expected qualnames. A retrieved item counts as a hit if it equals an
    expected qualname or is a member of it (handles method vs. class granularity
    mismatches, e.g. retrieving the containing class chunk for a method question).
    """
    if not retrieved_qualnames:
        return RetrievalScore(0.0, 0.0, 0.0)

    expected = set(expected_qualnames)

    def matches(retrieved: str, gold: str) -> bool:
        return (
            retrieved == gold
            or retrieved.startswith(gold + ".")
            or gold.startswith(retrieved + ".")
        )

    hits = [any(matches(q, e) for e in expected) for q in retrieved_qualnames]
    precision_at_k = sum(hits) / len(retrieved_qualnames)

    covered_gold = {e for e in expected if any(matches(q, e) for q in retrieved_qualnames)}
    recall_at_k = len(covered_gold) / len(expected) if expected else 0.0

    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalScore(precision_at_k, recall_at_k, reciprocal_rank)


def aggregate_retrieval_scores(scores: list[RetrievalScore]) -> dict[str, float]:
    if not scores:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    n = len(scores)
    return {
        "precision_at_k": sum(s.precision_at_k for s in scores) / n,
        "recall_at_k": sum(s.recall_at_k for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
    }
