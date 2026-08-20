from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = _load_cross_encoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """candidates: list of (chunk_id, chunk_text). Returns (chunk_id, score) sorted desc."""
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        results = list(zip((cid for cid, _ in candidates), scores, strict=True))
        return sorted(results, key=lambda x: x[1], reverse=True)
