from dataclasses import dataclass
from pathlib import Path

from codebase_chat_tool.config import Settings
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.graph.store import load_graph
from codebase_chat_tool.ingestion.metadata import Chunk, load_chunks
from codebase_chat_tool.retrieval.retriever import Retriever


@dataclass
class RepoContext:
    repo_root: Path
    settings: Settings
    retriever: Retriever
    resolver: GraphResolver
    chunks_by_id: dict[str, Chunk]
    chunks_by_qualname: dict[str, Chunk]


def load_repo_context(repo_root: Path, settings: Settings) -> RepoContext:
    index_dir = settings.index_path(repo_root)
    if not (index_dir / "chunks.json").exists():
        raise FileNotFoundError(
            f"No index found at {index_dir}. Run `codebase-chat-tool index {repo_root}` first."
        )

    chunks = load_chunks(index_dir / "chunks.json")
    graph = load_graph(index_dir / "graph.pkl")

    return RepoContext(
        repo_root=repo_root,
        settings=settings,
        retriever=Retriever(repo_root, settings),
        resolver=GraphResolver(graph),
        chunks_by_id={c.chunk_id: c for c in chunks},
        chunks_by_qualname={c.qualname: c for c in chunks},
    )
