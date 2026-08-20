import json
from functools import partial

from langgraph.graph import END, StateGraph

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.agent.prompts import CHAT_SYSTEM_PROMPT
from codebase_chat_tool.agent.state import ChatState
from codebase_chat_tool.agent.tools import TOOL_SPECS, call_tool
from codebase_chat_tool.llm.base import LLMProvider, Message

MAX_ITERATIONS = 6
MAX_TEXT_FIELD_CHARS = 2000


def _truncate_text_fields(value: object) -> object:
    """Truncates long "text" string values within a tool result so the overall
    payload stays a manageable size, without ever slicing the serialized JSON
    itself (which would produce invalid JSON)."""
    if isinstance(value, dict):
        return {
            k: (
                v[:MAX_TEXT_FIELD_CHARS] + "... [truncated]"
                if k == "text" and isinstance(v, str) and len(v) > MAX_TEXT_FIELD_CHARS
                else _truncate_text_fields(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_truncate_text_fields(v) for v in value]
    return value


def _agent_node(state: ChatState, ctx: RepoContext, provider: LLMProvider) -> ChatState:
    response = provider.generate(state["messages"], tools=TOOL_SPECS, max_tokens=1024)
    assistant_message = Message(
        role="assistant",
        content=response.content,
        tool_calls=response.tool_calls or None,
    )
    return {
        "messages": [*state["messages"], assistant_message],
        "iterations": state["iterations"] + 1,
    }


def _tools_node(state: ChatState, ctx: RepoContext) -> ChatState:
    last = state["messages"][-1]
    tool_messages: list[Message] = []
    for tool_call in last.tool_calls or []:
        try:
            result = call_tool(ctx, tool_call.name, tool_call.arguments)
            content = json.dumps(_truncate_text_fields(result))
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the LLM, not a crash
            content = json.dumps({"error": str(exc)})
        tool_messages.append(Message(role="tool", content=content, tool_call_id=tool_call.id))
    return {"messages": [*state["messages"], *tool_messages]}


def _should_continue(state: ChatState) -> str:
    last = state["messages"][-1]
    if last.tool_calls and state["iterations"] < MAX_ITERATIONS:
        return "tools"
    return "end"


def build_chat_graph(ctx: RepoContext, provider: LLMProvider):
    graph = StateGraph(ChatState)
    graph.add_node("agent", partial(_agent_node, ctx=ctx, provider=provider))
    graph.add_node("tools", partial(_tools_node, ctx=ctx))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_chat_turn(
    ctx: RepoContext, provider: LLMProvider, history: list[Message], user_input: str
) -> list[Message]:
    """Runs one user turn through the chat graph, returning the updated message history."""
    if not history:
        history = [Message(role="system", content=CHAT_SYSTEM_PROMPT)]
    messages = [*history, Message(role="user", content=user_input)]
    compiled = build_chat_graph(ctx, provider)
    result = compiled.invoke({"messages": messages, "iterations": 0})
    return result["messages"]
