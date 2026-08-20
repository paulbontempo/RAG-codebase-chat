from codebase_chat_tool.config import Settings
from codebase_chat_tool.llm.base import LLMProvider


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        from codebase_chat_tool.llm.anthropic_provider import AnthropicProvider

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_model)

    if settings.llm_provider == "openai":
        from codebase_chat_tool.llm.openai_provider import OpenAIProvider

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)

    raise ValueError(
        f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (expected 'anthropic' or 'openai')"
    )
