RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuses multiple ranked lists of chunk_ids into one, via score(d) = sum(1 / (k + rank_i(d))).

    Each input list is assumed sorted best-first; ranks are 1-indexed within each list.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
