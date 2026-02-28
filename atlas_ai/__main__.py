"""Entry point for running atlas-ai as a module.

Usage:
    python -m atlas_ai                     # stdin mode (default)
    python -m atlas_ai --mode stream       # Redis stream consumer
    python -m atlas_ai --info              # Print configuration and exit

Environment variables:
    LLM_PROVIDER      - "ollama" or "openai" (default: ollama)
    LLM_BASE_URL      - Base URL for LLM (default: http://localhost:11434)
    LLM_MODEL         - Model name (default: mistral)
    LLM_API_KEY       - API key for OpenAI (default: "")
    LLM_MAX_TOKENS    - Max tokens per response (default: 2048)
    LLM_TIMEOUT       - Request timeout in seconds (default: 120)
    ATLAS_AI_MODE     - "stdin" or "stream" (default: stdin)
    ATLAS_REDIS_URL   - Redis URL for stream mode (default: redis://localhost:6379)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from atlas_ai.advisor import ModernizationAdvisor
from atlas_ai.llm_client import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _config_from_env() -> LLMConfig:
    """Build LLMConfig from environment variables."""
    return LLMConfig(
        provider=os.environ.get("LLM_PROVIDER", "ollama"),
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("LLM_MODEL", "mistral"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
    )


def run_stdin(advisor: ModernizationAdvisor) -> None:
    """Read a JSON report from stdin, analyze it, and print the result to stdout."""
    logger.info("Reading report from stdin...")
    try:
        raw = sys.stdin.read()
        report_json = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON from stdin: %s", exc)
        sys.exit(1)

    result = advisor.analyze(report_json)
    output = {
        "model": result.model,
        "tokens_used": result.tokens_used,
        "roadmap": result.roadmap,
        "executive_summary": result.executive_summary,
    }
    print(json.dumps(output, indent=2))


def run_stream(advisor: ModernizationAdvisor, redis_url: str) -> None:
    """Consume the atlas.reports.ready Redis stream and analyze each report."""
    import redis as _redis

    logger.info("Connecting to Redis at %s ...", redis_url)
    client = _redis.from_url(redis_url, decode_responses=True)

    stream_name = "atlas.reports.ready"
    group_name = "atlas-ai"
    consumer_name = "atlas-ai-1"

    # Create consumer group if it doesn't exist yet
    try:
        client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
    except _redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise

    logger.info("Listening on stream '%s' (group=%s)...", stream_name, group_name)
    while True:
        try:
            messages = client.xreadgroup(
                group_name, consumer_name, {stream_name: ">"}, count=1, block=5000
            )
            if not messages:
                continue
            for _stream, entries in messages:
                for msg_id, fields in entries:
                    try:
                        report_json = json.loads(fields.get("payload", "{}"))
                        logger.info("Analyzing report from message %s", msg_id)
                        result = advisor.analyze(report_json)
                        logger.info(
                            "Analysis complete: %d tokens, model=%s",
                            result.tokens_used,
                            result.model,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to analyze message %s: %s", msg_id, exc)
                    finally:
                        client.xack(stream_name, group_name, msg_id)
        except KeyboardInterrupt:
            logger.info("Shutting down stream consumer.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="PipelineAtlas AI Modernization Advisor")
    parser.add_argument(
        "--mode",
        default=os.environ.get("ATLAS_AI_MODE", "stdin"),
        choices=["stdin", "stream"],
        help="Operation mode: stdin (default) or stream",
    )
    parser.add_argument("--info", action="store_true", help="Print configuration and exit")
    args = parser.parse_args()

    cfg = _config_from_env()
    redis_url = os.environ.get("ATLAS_REDIS_URL", "redis://localhost:6379")

    if args.info:
        print("PipelineAtlas AI Modernization Advisor")
        print(f"  Provider:  {cfg.provider}")
        print(f"  Model:     {cfg.model}")
        print(f"  Base URL:  {cfg.base_url}")
        print(f"  Mode:      {args.mode}")
        return

    advisor = ModernizationAdvisor(config=cfg)
    try:
        if args.mode == "stream":
            run_stream(advisor, redis_url)
        else:
            run_stdin(advisor)
    finally:
        advisor.close()


if __name__ == "__main__":
    main()
