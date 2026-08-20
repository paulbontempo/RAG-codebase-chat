import hashlib
from dataclasses import dataclass
from pathlib import Path

from codebase_chat_tool.config import Settings
from codebase_chat_tool.ingestion.metadata import Chunk, load_chunks
from codebase_chat_tool.retrieval.bm25_index import BM25Index, load_bm25_index
from codebase_chat_tool.retrieval.embeddings import Embedder
from codebase_chat_tool.retrieval.hybrid import reciprocal_rank_fusion
from codebase_chat_tool.retrieval.reranker import Reranker
from codebase_chat_tool.retrieval.vector_store import QdrantVectorStore


def collection_name(settings: Settings, repo_root: Path) -> str:
    repo_hash = hashlib.sha1(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{settings.qdrant_collection}_{repo_hash}"


@dataclass
class ScoredChunk:
    chunk: Chunk
    dense_rank: int | None
    bm25_rank: int | None
    fused_score: float
    rerank_score: float | None


class Retriever:
    def __init__(self, repo_root: Path, settings: Settings) -> None:
        self.repo_root = repo_root
        self.settings = settings
        index_dir = settings.index_path(repo_root)

        self.chunks: list[Chunk] = load_chunks(index_dir / "chunks.json")
        self._chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

        self.bm25: BM25Index = load_bm25_index(index_dir / "bm25.pkl")
        self.embedder = Embedder(settings.embedding_model)
        self.vector_store = QdrantVectorStore(
            settings.qdrant_url, collection_name(settings, repo_root)
        )
        self.reranker = Reranker()

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        candidate_k = self.settings.retrieval_candidate_k

        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(query_vector, top_k=candidate_k)
        dense_ranked = [h.chunk_id for h in dense_hits]

        bm25_ranked = [cid for cid, _score in self.bm25.search(query, top_k=candidate_k)]

        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])
        fused_top = fused[:candidate_k]

        dense_rank_of = {cid: i + 1 for i, cid in enumerate(dense_ranked)}
        bm25_rank_of = {cid: i + 1 for i, cid in enumerate(bm25_ranked)}

        candidates = [
            (cid, self._chunks_by_id[cid].text)
            for cid, _score in fused_top
            if cid in self._chunks_by_id
        ]
        reranked = self.reranker.rerank(query, candidates)[:top_k]

        results: list[ScoredChunk] = []
        fused_score_of = dict(fused_top)
        for chunk_id, rerank_score in reranked:
            results.append(
                ScoredChunk(
                    chunk=self._chunks_by_id[chunk_id],
                    dense_rank=dense_rank_of.get(chunk_id),
                    bm25_rank=bm25_rank_of.get(chunk_id),
                    fused_score=fused_score_of.get(chunk_id, 0.0),
                    rerank_score=float(rerank_score),
                )
            )
        return results
