"""LLM Client — abstraction for calling local (Ollama) or cloud (OpenAI) LLMs.

The LLM is used exclusively for augmenting the deterministic analysis,
never for parsing or extracting CI/CD structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""

    provider: str = "ollama"  # "ollama" or "openai"
    base_url: str = "http://localhost:11434"  # Ollama default
    model: str = "mistral"
    api_key: str = ""  # Required for OpenAI
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout: int = 120


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    model: str
    tokens_used: int = 0
    provider: str = ""


class LLMClient:
    """Unified LLM client supporting Ollama and OpenAI-compatible APIs."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = httpx.Client(timeout=self.config.timeout)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Send a prompt to the LLM and return the response.

        Args:
            system_prompt: System-level instructions for the LLM.
            user_prompt: The actual analysis query.

        Returns:
            LLMResponse with the generated content.
        """
        if self.config.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        elif self.config.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the Ollama /api/chat endpoint."""
        url = f"{self.config.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.config.model,
                tokens_used=data.get("eval_count", 0),
                provider="ollama",
            )
        except httpx.HTTPError as e:
            logger.error("Ollama error: %s", e)
            raise RuntimeError(f"Ollama request failed: {e}") from e

    def _call_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call an OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self.config.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        try:
            resp = self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            return LLMResponse(
                content=choice.get("message", {}).get("content", ""),
                model=self.config.model,
                tokens_used=usage.get("total_tokens", 0),
                provider="openai",
            )
        except httpx.HTTPError as e:
            logger.error("OpenAI error: %s", e)
            raise RuntimeError(f"OpenAI request failed: {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
