from functools import lru_cache

from sentence_transformers import SentenceTransformer

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = _BGE_QUERY_INSTRUCTION + text if "bge" in self.model_name.lower() else text
        vector = self._model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[
            0
        ]
        return vector.tolist()
