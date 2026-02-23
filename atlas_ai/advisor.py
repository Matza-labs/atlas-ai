"""Modernization Advisor — high-level orchestrator for AI-powered insights.

Wires the deterministic JSON report into the LLM client and generates
modernization roadmaps, executive summaries, and improvement plans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from atlas_ai.llm_client import LLMClient, LLMConfig, LLMResponse
from atlas_ai.prompts import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_executive_summary_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class ModernizationResult:
    """Result of a modernization analysis."""

    roadmap: str
    executive_summary: str
    tokens_used: int = 0
    model: str = ""


class ModernizationAdvisor:
    """Orchestrates AI-powered CI/CD modernization insights.

    Usage:
        advisor = ModernizationAdvisor()
        result = advisor.analyze(report_json)
        print(result.roadmap)
        print(result.executive_summary)
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._client = LLMClient(config)

    def analyze(self, report_json: dict[str, Any]) -> ModernizationResult:
        """Run a full modernization analysis from a JSON report.

        Args:
            report_json: The JSON report dict from atlas-report's JSON renderer.

        Returns:
            ModernizationResult with roadmap and executive summary.
        """
        total_tokens = 0

        # Generate full analysis
        analysis_prompt = build_analysis_prompt(report_json)
        roadmap_response = self._client.generate(SYSTEM_PROMPT, analysis_prompt)
        total_tokens += roadmap_response.tokens_used

        # Generate executive summary
        summary_prompt = build_executive_summary_prompt(report_json)
        summary_response = self._client.generate(SYSTEM_PROMPT, summary_prompt)
        total_tokens += summary_response.tokens_used

        logger.info(
            "Modernization analysis complete: %d tokens used (model: %s)",
            total_tokens,
            roadmap_response.model,
        )

        return ModernizationResult(
            roadmap=roadmap_response.content,
            executive_summary=summary_response.content,
            tokens_used=total_tokens,
            model=roadmap_response.model,
        )

    def generate_roadmap(self, report_json: dict[str, Any]) -> str:
        """Generate only the modernization roadmap."""
        prompt = build_analysis_prompt(report_json)
        response = self._client.generate(SYSTEM_PROMPT, prompt)
        return response.content

    def generate_summary(self, report_json: dict[str, Any]) -> str:
        """Generate only the executive summary."""
        prompt = build_executive_summary_prompt(report_json)
        response = self._client.generate(SYSTEM_PROMPT, prompt)
        return response.content

    def close(self) -> None:
        """Close the LLM client."""
        self._client.close()
