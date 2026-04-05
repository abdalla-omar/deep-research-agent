#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from dotenv import load_dotenv

from research_agent.models import ResearchRequest
from research_agent.orchestrator import DeepResearchEngine, EngineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deep research engine locally")
    parser.add_argument("--request", type=pathlib.Path, help="Path to request JSON", default=None)
    parser.add_argument("--query", type=str, help="Quick query string", default=None)
    parser.add_argument("--memory-mode", type=str, default="hybrid", choices=["none", "vector-only", "summary-only", "hybrid"])
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.request:
        payload = json.loads(args.request.read_text())
    else:
        payload = {
            "query": args.query
            or "Design a budget-aware deep research agent architecture for enterprise intelligence workflows.",
            "constraints": {
                "max_context_tokens_per_call": 2000,
                "max_session_cost_usd": 0.05,
                "max_runtime_seconds": 480,
            },
            "options": {"depth": "medium", "citations_required": True},
        }

    request = ResearchRequest.model_validate(payload)
    engine = DeepResearchEngine(config=EngineConfig(memory_mode=args.memory_mode))
    response = engine.run(request)
    out = response.model_dump(mode="json")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
