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


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def _to_anthropic_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system: str | None = None
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n" + m.content) if system else m.content
            elif m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return system, out

    def _to_anthropic_tools(self, tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system, anthropic_messages = self._to_anthropic_messages(messages)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 2048),
            system=system,
            messages=anthropic_messages,
            tools=self._to_anthropic_tools(tools) or [],
            **kwargs,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[LLMChunk]:
        system, anthropic_messages = self._to_anthropic_messages(messages)
        with self._client.messages.stream(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 2048),
            system=system,
            messages=anthropic_messages,
            tools=self._to_anthropic_tools(tools) or [],
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield LLMChunk(delta=text)
            yield LLMChunk(delta="", done=True)
