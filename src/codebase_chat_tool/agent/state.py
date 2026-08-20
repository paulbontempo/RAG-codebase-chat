from typing import TypedDict

from codebase_chat_tool.llm.base import Message


class ChatState(TypedDict):
    messages: list[Message]
    iterations: int


class ImpactState(TypedDict, total=False):
    target: str
    resolved_target: str
    error: str | None
    target_module: str
    direct_callers: list[str]
    transitive_callers: list[str]
    dependent_modules: list[str]
    related_tests: list[dict]
    risk_summary: str | None
