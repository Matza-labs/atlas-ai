"""Prompt builder — constructs LLM prompts from deterministic analysis data.

Crucially, the LLM never parses CI/CD structure. It only consumes
data that has already been deterministically extracted, and generates
human-readable insights, summaries, and modernization recommendations.
"""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are PipelineAtlas AI — an expert CI/CD architecture advisor.

You will receive a JSON summary of a deterministic CI/CD analysis including:
- Pipeline structure (nodes and edges)
- Complexity and fragility scores
- Rule engine findings with severity levels

Your job is to:
1. Provide an executive summary of the pipeline health
2. Generate a modernization roadmap with specific, actionable steps
3. Rank improvements by impact (highest first)
4. Reference specific node names and findings from the data

Rules:
- NEVER invent CI/CD structure that isn't in the data
- ALWAYS reference specific node names and rule IDs from the data
- Keep recommendations practical and specific
- Use markdown formatting for readability
"""


def build_analysis_prompt(report_json: dict[str, Any]) -> str:
    """Build a user prompt from a JSON report dict.

    Args:
        report_json: Output from atlas-report's JSON renderer.

    Returns:
        A formatted user prompt string for the LLM.
    """
    sections = []

    # Meta
    meta = report_json.get("meta", {})
    sections.append(f"## Pipeline: {meta.get('name', 'Unknown')}")
    sections.append(f"Platform: {meta.get('platform', 'unknown')}")
    sections.append(f"Generated: {meta.get('generated_at', 'unknown')}")

    # Scores
    scores = report_json.get("scores", {})
    sections.append(f"\n## Scores")
    sections.append(f"- Complexity: {scores.get('complexity_score', 'N/A')}/100")
    sections.append(f"- Fragility: {scores.get('fragility_score', 'N/A')}/100")

    # Structure
    structure = report_json.get("structure", {})
    sections.append(f"\n## Structure")
    sections.append(f"- Nodes: {structure.get('total_nodes', 0)}")
    sections.append(f"- Edges: {structure.get('total_edges', 0)}")

    nodes = structure.get("nodes_by_type", {})
    if nodes:
        for nt, count in nodes.items():
            sections.append(f"  - {nt}: {count}")

    # Findings
    findings = report_json.get("findings", [])
    if findings:
        sections.append(f"\n## Findings ({len(findings)} total)")
        for f in findings:
            sections.append(
                f"- [{f.get('severity', 'info').upper()}] {f.get('rule_id', '?')}: {f.get('message', '')}"
            )

    sections.append("\n---")
    sections.append("Based on the analysis above, provide:")
    sections.append("1. Executive Summary (2-3 sentences)")
    sections.append("2. Modernization Roadmap (3-5 prioritized improvements)")
    sections.append("3. Risk Assessment (what could break if left unaddressed)")

    return "\n".join(sections)


def build_executive_summary_prompt(report_json: dict[str, Any]) -> str:
    """Build a prompt specifically for generating a concise executive summary."""
    name = report_json.get("meta", {}).get("name", "Unknown Pipeline")
    scores = report_json.get("scores", {})
    findings_count = len(report_json.get("findings", []))

    return (
        f"Generate a concise 2-3 sentence executive summary for '{name}'.\n"
        f"Complexity: {scores.get('complexity_score', 'N/A')}/100, "
        f"Fragility: {scores.get('fragility_score', 'N/A')}/100, "
        f"Findings: {findings_count}.\n"
        f"Focus on the overall health and the single most impactful improvement."
    )
