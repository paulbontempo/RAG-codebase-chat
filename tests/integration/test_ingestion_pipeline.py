import shutil
from pathlib import Path

import pytest

from codebase_chat_tool.config import get_settings
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.graph.store import load_graph
from codebase_chat_tool.ingestion.metadata import load_chunks
from codebase_chat_tool.ingestion.pipeline import run_indexing

FIXTURE_REPO = Path(__file__).resolve().parent.parent / "fixtures" / "sample_repo"


@pytest.fixture
def indexed_repo(tmp_path, monkeypatch):
    repo_copy = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, repo_copy)
    monkeypatch.delenv("INDEX_DIR", raising=False)
    run_indexing(str(repo_copy))
    return repo_copy


def test_pipeline_produces_expected_chunk_count(indexed_repo):
    settings = get_settings()
    chunks = load_chunks(settings.index_path(indexed_repo) / "chunks.json")
    assert len(chunks) == 22


def test_pipeline_produces_expected_def_kinds(indexed_repo):
    settings = get_settings()
    chunks = load_chunks(settings.index_path(indexed_repo) / "chunks.json")
    kinds = [c.kind for c in chunks]
    assert kinds.count("class") == 3  # UserService, BaseEntity, User
    # UserService has 4 methods; BaseEntity and User have 2 each (__init__, describe).
    assert kinds.count("method") == 8


def test_pipeline_call_graph_resolves_known_edges(indexed_repo):
    settings = get_settings()
    graph = load_graph(settings.index_path(indexed_repo) / "graph.pkl")
    resolver = GraphResolver(graph)

    describe_user_callers = {
        c.qualname for c in resolver.direct_callers("service.UserService.get_user")
    }
    assert describe_user_callers == {"service.UserService.describe_user"}

    log_call_callers = {c.qualname for c in resolver.direct_callers("utils.log_call")}
    assert log_call_callers == {"notifier.notify", "service.UserService.create_user"}

    assert set(resolver.importers("utils")) == {"notifier", "service"}


def test_pipeline_handles_circular_import_without_crashing(indexed_repo):
    # service.py and notifier.py import each other (one deferred); indexing must not raise.
    settings = get_settings()
    graph = load_graph(settings.index_path(indexed_repo) / "graph.pkl")
    resolver = GraphResolver(graph)
    assert "notifier" in resolver.importers("service") or "service" in resolver.importers(
        "notifier"
    )


def test_pipeline_skips_files_with_syntax_errors_and_indexes_the_rest(tmp_path):
    repo_copy = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, repo_copy)
    (repo_copy / "broken.py").write_text("def broken(:\n    this is not valid python\n")

    # Must not raise, despite one file being unparseable.
    run_indexing(str(repo_copy))

    settings = get_settings()
    chunks = load_chunks(settings.index_path(repo_copy) / "chunks.json")
    # The other 7 well-formed files still get indexed (22 chunks, as in the clean-repo test).
    assert len(chunks) == 22
    assert all("broken" not in c.file_path for c in chunks)
