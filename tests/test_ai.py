"""Unit tests for atlas-ai — prompt builder, LLM client, advisor."""

from unittest.mock import MagicMock, patch

import pytest

from atlas_ai.llm_client import LLMClient, LLMConfig, LLMResponse
from atlas_ai.prompts import SYSTEM_PROMPT, build_analysis_prompt, build_executive_summary_prompt
from atlas_ai.advisor import ModernizationAdvisor, ModernizationResult

# --- Sample report data ---
SAMPLE_REPORT = {
    "meta": {"name": "CI Workflow", "platform": "github_actions", "generated_at": "2026-02-23T08:00:00Z"},
    "scores": {"complexity_score": 58.0, "fragility_score": 32.0},
    "structure": {
        "total_nodes": 10,
        "total_edges": 11,
        "nodes_by_type": {"pipeline": 1, "job": 3, "step": 4, "environment": 1, "secret_ref": 1},
    },
    "findings": [
        {"rule_id": "no-timeout", "severity": "medium", "message": "Job 'Build' has no timeout"},
        {"rule_id": "unpinned-images", "severity": "high", "message": "Step uses node:latest"},
    ],
}


class TestPrompts:

    def test_system_prompt_exists(self):
        assert "PipelineAtlas AI" in SYSTEM_PROMPT
        assert "NEVER invent" in SYSTEM_PROMPT

    def test_build_analysis_prompt(self):
        prompt = build_analysis_prompt(SAMPLE_REPORT)
        assert "CI Workflow" in prompt
        assert "Complexity: 58.0/100" in prompt
        assert "Fragility: 32.0/100" in prompt
        assert "no-timeout" in prompt
        assert "unpinned-images" in prompt
        assert "Nodes: 10" in prompt
        assert "Modernization Roadmap" in prompt

    def test_build_executive_summary_prompt(self):
        prompt = build_executive_summary_prompt(SAMPLE_REPORT)
        assert "CI Workflow" in prompt
        assert "Findings: 2" in prompt


class TestLLMClient:

    @patch("atlas_ai.llm_client.httpx.Client")
    def test_ollama_generate(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Here is the roadmap..."},
            "eval_count": 150,
        }
        mock_client.post.return_value = mock_resp

        config = LLMConfig(provider="ollama", model="mistral")
        client = LLMClient(config)
        response = client.generate("system", "user prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Here is the roadmap..."
        assert response.tokens_used == 150
        assert response.provider == "ollama"

    @patch("atlas_ai.llm_client.httpx.Client")
    def test_openai_generate(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Summary here"}}],
            "usage": {"total_tokens": 200},
        }
        mock_client.post.return_value = mock_resp

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        client = LLMClient(config)
        response = client.generate("system", "user prompt")

        assert response.content == "Summary here"
        assert response.tokens_used == 200
        assert response.provider == "openai"


class TestAdvisor:

    @patch("atlas_ai.advisor.LLMClient")
    def test_analyze(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.generate.side_effect = [
            LLMResponse(content="Roadmap content", model="mistral", tokens_used=100, provider="ollama"),
            LLMResponse(content="Executive summary", model="mistral", tokens_used=50, provider="ollama"),
        ]

        advisor = ModernizationAdvisor()
        result = advisor.analyze(SAMPLE_REPORT)

        assert isinstance(result, ModernizationResult)
        assert result.roadmap == "Roadmap content"
        assert result.executive_summary == "Executive summary"
        assert result.tokens_used == 150
        assert result.model == "mistral"

    @patch("atlas_ai.advisor.LLMClient")
    def test_generate_roadmap_only(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.generate.return_value = LLMResponse(
            content="Just the roadmap",
            model="mistral",
            tokens_used=80,
            provider="ollama",
        )

        advisor = ModernizationAdvisor()
        roadmap = advisor.generate_roadmap(SAMPLE_REPORT)
        assert roadmap == "Just the roadmap"
