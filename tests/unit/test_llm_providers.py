from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codebase_chat_tool.llm.anthropic_provider import AnthropicProvider
from codebase_chat_tool.llm.base import Message


def _anthropic_provider_with_mocked_client() -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "claude-test"
    provider._client = MagicMock()
    return provider


def _openai_provider_with_mocked_client():
    from codebase_chat_tool.llm.openai_provider import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-test"
    provider._client = MagicMock()
    return provider


def _anthropic_text_response(text: str, input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _anthropic_tool_use_response(tool_id: str, name: str, tool_input: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _openai_text_response(text: str, prompt_tokens=10, completion_tokens=5):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _openai_tool_call_response(call_id: str, name: str, arguments_json: str):
    tool_call = SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments_json)
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.parametrize("provider_kind", ["anthropic", "openai"])
def test_generate_returns_normalized_text_response(provider_kind):
    if provider_kind == "anthropic":
        provider = _anthropic_provider_with_mocked_client()
        provider._client.messages.create.return_value = _anthropic_text_response("hello world")
    else:
        provider = _openai_provider_with_mocked_client()
        provider._client.chat.completions.create.return_value = _openai_text_response("hello world")

    response = provider.generate([Message(role="user", content="hi")])

    assert response.content == "hello world"
    assert response.tool_calls == []
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


@pytest.mark.parametrize("provider_kind", ["anthropic", "openai"])
def test_generate_returns_normalized_tool_call(provider_kind):
    if provider_kind == "anthropic":
        provider = _anthropic_provider_with_mocked_client()
        provider._client.messages.create.return_value = _anthropic_tool_use_response(
            "call_1", "search_code", {"query": "retry logic"}
        )
    else:
        provider = _openai_provider_with_mocked_client()
        provider._client.chat.completions.create.return_value = _openai_tool_call_response(
            "call_1", "search_code", '{"query": "retry logic"}'
        )

    response = provider.generate([Message(role="user", content="find retry logic")])

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "search_code"
    assert call.arguments == {"query": "retry logic"}


def test_anthropic_system_message_is_extracted_separately():
    provider = _anthropic_provider_with_mocked_client()
    provider._client.messages.create.return_value = _anthropic_text_response("ok")

    provider.generate(
        [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hi"),
        ]
    )

    _, kwargs = provider._client.messages.create.call_args
    assert kwargs["system"] == "You are helpful."
    assert all(m["role"] != "system" for m in kwargs["messages"])


def test_openai_system_message_stays_in_message_list():
    provider = _openai_provider_with_mocked_client()
    provider._client.chat.completions.create.return_value = _openai_text_response("ok")

    provider.generate(
        [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hi"),
        ]
    )

    _, kwargs = provider._client.chat.completions.create.call_args
    roles = [m["role"] for m in kwargs["messages"]]
    assert "system" in roles
