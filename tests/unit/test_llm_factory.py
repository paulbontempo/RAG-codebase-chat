import pytest

from codebase_chat_tool.config import Settings
from codebase_chat_tool.llm.factory import get_provider


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_missing_anthropic_key_raises_clear_error():
    settings = _settings(llm_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider(settings)


def test_missing_openai_key_raises_clear_error():
    settings = _settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_provider(settings)


def test_unknown_provider_raises_clear_error():
    settings = _settings(llm_provider="not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_provider(settings)


def test_anthropic_provider_constructed_when_key_present():
    settings = _settings(llm_provider="anthropic", anthropic_api_key="sk-ant-fake")
    provider = get_provider(settings)
    assert provider.__class__.__name__ == "AnthropicProvider"


def test_openai_provider_constructed_when_key_present():
    settings = _settings(llm_provider="openai", openai_api_key="sk-fake")
    provider = get_provider(settings)
    assert provider.__class__.__name__ == "OpenAIProvider"
