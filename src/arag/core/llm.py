"""LLM client for ARAG — unified interface via litellm."""

import os
from typing import Any, Dict, List, Optional

import litellm
import tiktoken


class LLMClient:
    """Unified LLM client using litellm for multi-provider support."""

    PRICING = {
        "claude-sonnet-4-5": (3.0, 0.3, 15.0),
        "claude-opus-4-6": (5.0, 0.5, 25.0),
        "claude-haiku-4-5": (1.0, 0.1, 5.0),
        "gemini-2.5-flash": (0.3, 0.075, 2.5),
        "gemini-2.5-pro": (1.25, 0.125, 10.0),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "gpt-4o": (2.5, 1.25, 10.0),
        "default": (1.0, 0.1, 5.0),
    }

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: str = None,
        vertex_project: str = None,
        vertex_location: str = None,
        vertex_credentials: str = None,
    ):
        self.model = model or os.getenv("ARAG_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.total_cost = 0.0

        self._litellm_kwargs = {}
        if vertex_project or os.getenv("VERTEX_PROJECT"):
            self._litellm_kwargs["vertex_project"] = vertex_project or os.getenv("VERTEX_PROJECT")
        if vertex_location or os.getenv("VERTEX_LOCATION"):
            self._litellm_kwargs["vertex_location"] = vertex_location or os.getenv("VERTEX_LOCATION")
        if vertex_credentials or os.getenv("VERTEX_CREDENTIALS"):
            self._litellm_kwargs["vertex_credentials"] = vertex_credentials or os.getenv("VERTEX_CREDENTIALS")

        if api_key or os.getenv("ARAG_API_KEY"):
            self._litellm_kwargs["api_key"] = api_key or os.getenv("ARAG_API_KEY")
        if base_url or os.getenv("ARAG_BASE_URL"):
            self._litellm_kwargs["api_base"] = (base_url or os.getenv("ARAG_BASE_URL")).rstrip("/")

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total

    def calculate_cost(self, usage: dict) -> float:
        model_lower = self.model.lower()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        input_tokens = max(prompt_tokens - cached_tokens, 0)

        for key in self.PRICING:
            if key in model_lower:
                input_price, cached_price, output_price = self.PRICING[key]
                break
        else:
            input_price, cached_price, output_price = self.PRICING["default"]

        usd_cost = (
            (input_tokens / 1_000_000) * input_price
            + (cached_tokens / 1_000_000) * cached_price
            + (completion_tokens / 1_000_000) * output_price
        )
        return round(usd_cost, 6)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **self._litellm_kwargs,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = litellm.completion(**kwargs)

        message = response.choices[0].message.model_dump()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        cost = self.calculate_cost(usage)
        self.total_cost += cost

        return {
            "message": message,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "cost": cost,
        }

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        **kwargs,
    ) -> tuple:
        if system:
            messages = [{"role": "system", "content": system}] + messages
        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        cost = result["cost"]
        return content, cost
