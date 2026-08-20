import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from codebase_chat_tool.ingestion.metadata import Chunk

_ID_NAMESPACE = uuid.UUID("6f6f6f6f-1111-2222-3333-444444444444")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


@dataclass
class VectorHit:
    chunk_id: str
    score: float


class QdrantVectorStore:
    def __init__(self, url: str, collection: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=dimension, distance=qmodels.Distance.COSINE
                ),
            )

    def recreate_collection(self, dimension: int) -> None:
        self.client.delete_collection(self.collection)
        self.ensure_collection(dimension)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "file_path": chunk.file_path,
                    "qualname": chunk.qualname,
                    "kind": chunk.kind,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self, query_vector: list[float], top_k: int, kind_filter: str | None = None
    ) -> list[VectorHit]:
        query_filter = None
        if kind_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="kind", match=qmodels.MatchValue(value=kind_filter))
                ]
            )
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [VectorHit(chunk_id=r.payload["chunk_id"], score=r.score) for r in results]
