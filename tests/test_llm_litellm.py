"""Tests for litellm-based LLM client."""

from unittest.mock import patch, MagicMock
from arag.core.llm import LLMClient


def test_llm_client_init_with_vertex():
    """LLMClient accepts Vertex AI config without requiring API key."""
    client = LLMClient(
        model="vertex_ai/claude-sonnet-4-5-20250929",
        vertex_project="hyprotwin",
        vertex_location="europe-west4",
    )
    assert client.model == "vertex_ai/claude-sonnet-4-5-20250929"


def test_llm_client_init_backward_compat():
    """LLMClient still works with api_key for non-Vertex models."""
    client = LLMClient(model="gpt-4o-mini", api_key="sk-test")
    assert client.model == "gpt-4o-mini"


@patch("arag.core.llm.litellm")
def test_chat_calls_litellm(mock_litellm):
    """chat() delegates to litellm.completion."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": None,
    }
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_litellm.completion.return_value = mock_response

    client = LLMClient(
        model="vertex_ai/claude-sonnet-4-5-20250929",
        vertex_project="hyprotwin",
        vertex_location="europe-west4",
    )
    result = client.chat(messages=[{"role": "user", "content": "Hi"}])

    mock_litellm.completion.assert_called_once()
    call_kwargs = mock_litellm.completion.call_args
    assert call_kwargs.kwargs["model"] == "vertex_ai/claude-sonnet-4-5-20250929"
    assert result["message"]["content"] == "Hello"


@patch("arag.core.llm.litellm")
def test_chat_with_tools(mock_litellm):
    """chat() passes tools to litellm."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "tc_1", "function": {"name": "test", "arguments": "{}"}}],
    }
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_litellm.completion.return_value = mock_response

    client = LLMClient(
        model="vertex_ai/claude-sonnet-4-5-20250929",
        vertex_project="hyprotwin",
        vertex_location="europe-west4",
    )
    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    result = client.chat(
        messages=[{"role": "user", "content": "Hi"}],
        tools=tools,
    )

    call_kwargs = mock_litellm.completion.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert result["message"]["tool_calls"] is not None


def test_count_tokens():
    """Token counting works without API credentials."""
    client = LLMClient.__new__(LLMClient)
    import tiktoken
    client.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    assert client.count_tokens("Hello world") > 0
