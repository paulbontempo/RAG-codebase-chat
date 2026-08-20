from collections.abc import Iterator

from codebase_chat_tool.llm.base import LLMChunk, LLMProvider, LLMResponse, Message, ToolSpec, Usage


class FakeLLMProvider(LLMProvider):
    """Deterministic LLMProvider for tests: returns pre-scripted responses in order,
    one per call to generate(). Never touches the network."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLMProvider ran out of scripted responses")
        return self._responses.pop(0)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> Iterator[LLMChunk]:
        response = self.generate(messages, tools, **kwargs)
        yield LLMChunk(delta=response.content, done=True)


def text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], stop_reason="end_turn", usage=Usage())
