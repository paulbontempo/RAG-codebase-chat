import shutil
import sys
from pathlib import Path

from typer.testing import CliRunner

from codebase_chat_tool.cli.main import app

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fakes import FakeLLMProvider, text_response  # noqa: E402

FIXTURE_REPO = Path(__file__).resolve().parent.parent / "fixtures" / "sample_repo"
runner = CliRunner()


def _indexed_repo(tmp_path) -> Path:
    repo_copy = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, repo_copy)
    result = runner.invoke(app, ["index", str(repo_copy)])
    assert result.exit_code == 0, result.output
    return repo_copy


def test_cli_index_smoke(tmp_path):
    repo = _indexed_repo(tmp_path)
    settings_index_dir = repo / ".codebase_chat_tool"
    assert (settings_index_dir / "chunks.json").exists()
    assert (settings_index_dir / "graph.pkl").exists()
    assert (settings_index_dir / "bm25.pkl").exists()


def test_cli_search_smoke(tmp_path):
    repo = _indexed_repo(tmp_path)
    result = runner.invoke(app, ["search", "retry logic", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "Search results" in result.output


def test_cli_graph_callers_smoke(tmp_path):
    repo = _indexed_repo(tmp_path)
    result = runner.invoke(
        app, ["graph", "callers", "service.UserService.get_user", "--repo", str(repo)]
    )
    assert result.exit_code == 0, result.output
    assert "describe_user" in result.output


def test_cli_ask_smoke(tmp_path, monkeypatch):
    repo = _indexed_repo(tmp_path)
    monkeypatch.setattr(
        "codebase_chat_tool.llm.factory.get_provider",
        lambda settings: FakeLLMProvider([text_response("This is a fake answer.")]),
    )
    result = runner.invoke(app, ["ask", "What does this repo do?", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "This is a fake answer." in result.output


def test_cli_impact_smoke(tmp_path, monkeypatch):
    repo = _indexed_repo(tmp_path)
    monkeypatch.setattr(
        "codebase_chat_tool.llm.factory.get_provider",
        lambda settings: FakeLLMProvider([text_response("Low risk.")]),
    )
    result = runner.invoke(app, ["impact", "service.UserService.get_user", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "describe_user" in result.output
    assert "Low risk." in result.output


def test_cli_impact_reports_error_for_unknown_symbol(tmp_path, monkeypatch):
    repo = _indexed_repo(tmp_path)
    monkeypatch.setattr(
        "codebase_chat_tool.llm.factory.get_provider",
        lambda settings: FakeLLMProvider([]),
    )
    result = runner.invoke(app, ["impact", "does.not.exist", "--repo", str(repo)])
    assert result.exit_code == 1
    assert "not found" in result.output
