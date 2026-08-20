import json
from collections.abc import Iterator
from typing import Any

from codebase_chat_tool.llm.base import (
    LLMChunk,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        import openai

        self._client = openai.OpenAI(api_key=api_key)

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def _to_openai_tools(self, tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(messages),
            tools=self._to_openai_tools(tools),
            **kwargs,
        )
        choice = response.choices[0]
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (choice.message.tool_calls or [])
        ]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            usage=Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[LLMChunk]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(messages),
            tools=self._to_openai_tools(tools),
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield LLMChunk(delta=delta)
        yield LLMChunk(delta="", done=True)
