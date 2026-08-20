import json
import sys
from pathlib import Path

from codebase_chat_tool.agent.chat_graph import run_chat_turn
from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.graph.ast_visitor import analyze_file
from codebase_chat_tool.graph.call_graph import build_graph
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.ingestion.chunker import chunk_file
from codebase_chat_tool.llm.base import LLMResponse, ToolCall, Usage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fakes import FakeLLMProvider, text_response  # noqa: E402


def _build_ctx(source: str = "def helper():\n    pass\n") -> RepoContext:
    sources = {"utils": source}
    analyses = [analyze_file(module, src) for module, src in sources.items()]
    resolver = GraphResolver(build_graph(analyses))
    chunks = chunk_file("utils", "utils.py", sources["utils"])
    return RepoContext(
        repo_root=Path("/fake"),
        settings=None,
        retriever=None,
        resolver=resolver,
        chunks_by_id={c.chunk_id: c for c in chunks},
        chunks_by_qualname={c.qualname: c for c in chunks},
    )


def test_chat_graph_answers_directly_without_tool_calls():
    ctx = _build_ctx()
    provider = FakeLLMProvider([text_response("The answer is 42.")])

    history = run_chat_turn(ctx, provider, [], "What is the answer?")

    assert history[-1].role == "assistant"
    assert history[-1].content == "The answer is 42."
    assert len(provider.calls) == 1


def test_chat_graph_executes_tool_call_then_produces_final_answer():
    ctx = _build_ctx()
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="call_1", name="get_definition", arguments={"qualname": "utils.helper"})
        ],
        stop_reason="tool_use",
        usage=Usage(),
    )
    final_response = text_response("utils.helper is a no-op function.")
    provider = FakeLLMProvider([tool_call_response, final_response])

    history = run_chat_turn(ctx, provider, [], "What does utils.helper do?")

    assert len(provider.calls) == 2
    tool_messages = [m for m in history if m.role == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0].content)
    assert payload["qualname"] == "utils.helper"
    assert history[-1].content == "utils.helper is a no-op function."


def test_chat_graph_stops_after_max_iterations_to_avoid_infinite_tool_loop():
    ctx = _build_ctx()
    looping_call = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="call_x", name="get_definition", arguments={"qualname": "utils.helper"})
        ],
        stop_reason="tool_use",
        usage=Usage(),
    )
    provider = FakeLLMProvider([looping_call] * 10)

    history = run_chat_turn(ctx, provider, [], "loop forever")

    # Should terminate rather than hang, even though every response requests a tool call.
    assert history[-1].role == "assistant"


def test_tool_message_content_stays_valid_json_even_for_large_definitions():
    # Regression test: truncating the *serialized JSON string* (rather than the
    # "text" field inside it) can cut mid-object and produce invalid JSON that
    # silently breaks any downstream consumer (e.g. the groundedness eval check).
    big_source = "def helper():\n    " + "\n    ".join(f"x{i} = {i}" for i in range(2000)) + "\n"
    ctx = _build_ctx(big_source)
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="call_1", name="get_definition", arguments={"qualname": "utils.helper"})
        ],
        stop_reason="tool_use",
        usage=Usage(),
    )
    provider = FakeLLMProvider([tool_call_response, text_response("done")])

    history = run_chat_turn(ctx, provider, [], "describe helper")

    tool_message = next(m for m in history if m.role == "tool")
    payload = json.loads(tool_message.content)  # must not raise
    assert payload["qualname"] == "utils.helper"
    assert "[truncated]" in payload["text"]
