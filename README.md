# atlas-ai

AI Strategy Layer for **PipelineAtlas** — LLM-powered modernization insights.

## Purpose

Uses LLMs (local or cloud) to generate modernization roadmaps, executive summaries, and structural explanations. AI is **never used for structure extraction** — only for augmenting the deterministic analysis.

## Status: 🟡 Phase 2

This service is planned for Phase 2. The directory structure is scaffolded and ready.

## Planned Features

- Local LLM support (Llama / Mistral via Ollama)
- Cloud LLM integration (OpenAI / Anthropic)
- Modernization roadmap generation
- Executive summaries
- Cross-repo architecture insights
- Token budget control and caching

## Constraints

- Must **reference graph nodes and evidence IDs** — never invent structure
- Token budget enforcement
- Optional in enterprise local mode (can be disabled)

## Dependencies

- `atlas-sdk` (shared models)
- `redis` (Redis Streams)
- `httpx` (query graph service)

## Related Services

Queries ← `atlas-graph`
Publishes to → `atlas-report`
