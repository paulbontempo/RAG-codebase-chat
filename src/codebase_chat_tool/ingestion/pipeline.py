from pathlib import Path

from rich.console import Console

from codebase_chat_tool.config import get_settings
from codebase_chat_tool.graph.ast_visitor import analyze_file
from codebase_chat_tool.graph.call_graph import build_graph
from codebase_chat_tool.graph.store import save_graph
from codebase_chat_tool.ingestion.chunker import chunk_file
from codebase_chat_tool.ingestion.discover import discover_python_files, path_to_module
from codebase_chat_tool.ingestion.metadata import Chunk, save_chunks
from codebase_chat_tool.retrieval.bm25_index import build_bm25_index, save_bm25_index
from codebase_chat_tool.retrieval.embeddings import Embedder
from codebase_chat_tool.retrieval.retriever import collection_name
from codebase_chat_tool.retrieval.vector_store import QdrantVectorStore

console = Console()


def run_indexing(path: str) -> None:
    settings = get_settings()
    repo_root = Path(path).resolve()
    if not repo_root.is_dir():
        console.print(f"[red]Not a directory: {repo_root}[/red]")
        raise SystemExit(1)

    files = discover_python_files(repo_root, index_dir_name=settings.index_dir)
    if not files:
        console.print(f"[yellow]No Python files found under {repo_root}[/yellow]")
        return

    all_chunks: list[Chunk] = []
    analyses = []
    parse_errors = 0

    for file_path in files:
        rel_path = file_path.relative_to(repo_root).as_posix()
        module = path_to_module(repo_root, file_path)
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            console.print(f"[yellow]Skipping {rel_path}: {exc}[/yellow]")
            parse_errors += 1
            continue

        try:
            is_package_init = file_path.name == "__init__.py"
            # Compute both before mutating shared state, so a syntax error caught mid-way
            # (e.g. tree-sitter tolerates it but `ast.parse` doesn't) never leaves a file
            # partially indexed -- either both succeed and get added, or neither does.
            file_chunks = chunk_file(module, rel_path, source)
            file_analysis = analyze_file(module, source, is_package_init=is_package_init)
        except SyntaxError as exc:
            console.print(f"[yellow]Skipping {rel_path}: syntax error ({exc})[/yellow]")
            parse_errors += 1
            continue
        all_chunks.extend(file_chunks)
        analyses.append(file_analysis)

    graph = build_graph(analyses)
    index_dir = settings.index_path(repo_root)
    save_chunks(all_chunks, index_dir / "chunks.json")
    save_graph(graph, index_dir / "graph.pkl")

    console.print("[cyan]Building BM25 index...[/cyan]")
    bm25 = build_bm25_index([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])
    save_bm25_index(bm25, index_dir / "bm25.pkl")

    console.print("[cyan]Embedding chunks and upserting to Qdrant...[/cyan]")
    embedder = Embedder(settings.embedding_model)
    vector_store = QdrantVectorStore(settings.qdrant_url, collection_name(settings, repo_root))
    vector_store.recreate_collection(embedder.dimension)
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        vector_store.upsert(batch, vectors)

    n_functions = sum(
        1 for d in graph.nodes(data=True) if d[1].get("kind") in ("function", "method")
    )
    n_classes = sum(1 for d in graph.nodes(data=True) if d[1].get("kind") == "class")
    unresolved_calls = sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("type") == "call" and not data.get("resolved")
    )
    resolved_calls = sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("type") == "call" and data.get("resolved")
    )

    console.print(f"[green]Indexed {len(files)} files ({parse_errors} skipped)[/green]")
    console.print(f"  chunks:            {len(all_chunks)}")
    console.print(f"  functions/methods: {n_functions}")
    console.print(f"  classes:           {n_classes}")
    console.print(f"  resolved calls:    {resolved_calls}")
    console.print(f"  unresolved calls:  {unresolved_calls}")
    console.print(f"  index dir:         {index_dir}")
