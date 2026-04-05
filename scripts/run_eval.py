#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import time
from statistics import mean

from dotenv import load_dotenv
from tabulate import tabulate

from research_agent.models import ResearchRequest
from research_agent.orchestrator import DeepResearchEngine, EngineConfig
from research_agent.verifier import verify_claim_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation evaluation for deep research agent")
    parser.add_argument(
        "--queries",
        type=pathlib.Path,
        default=pathlib.Path("scripts/query_suite.json"),
        help="JSON file with evaluation queries",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/eval/results.json"),
        help="Path for JSON result output",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
    )
    return parser.parse_args()


def keyword_relevancy(query: str, answer_markdown: str) -> float:
    words = [w.strip(".,:;!?()[]{}\"'").lower() for w in query.split()]
    keywords = [w for w in words if len(w) >= 5]
    if not keywords:
        return 1.0
    text = answer_markdown.lower()
    hit = sum(1 for k in set(keywords) if k in text)
    return hit / max(1, len(set(keywords)))


def run_variant(queries: list[str], memory_mode: str, depth: str) -> dict:
    engine = DeepResearchEngine(config=EngineConfig(memory_mode=memory_mode))

    costs = []
    tokens = []
    latencies = []
    compliance = []
    faithfulness = []
    relevancy = []
    retrieval_precision_proxy = []

    for q in queries:
        request = ResearchRequest(
            query=q,
            options={"depth": depth, "citations_required": True},
            constraints={
                "max_context_tokens_per_call": 2000,
                "max_session_cost_usd": 0.05,
                "max_runtime_seconds": 480,
            },
        )
        start = time.perf_counter()
        response = engine.run(request)
        elapsed = time.perf_counter() - start

        costs.append(response.budget_report.total_cost_usd)
        tokens.append(response.budget_report.total_tokens)
        latencies.append(elapsed)

        compliant = (
            response.budget_report.total_cost_usd <= 0.05
            and response.budget_report.max_context_tokens_seen <= 2000
        )
        compliance.append(1.0 if compliant else 0.0)

        ver = verify_claim_coverage(
            response.answer_markdown,
            [
                # Rebuild minimal evidence objects from citations for verifier compatibility.
                # The claim IDs are what matter for this metric.
                type("TmpCard", (), {"id": c.claim_id})() for c in response.citations
            ],
        )
        faithfulness.append(ver.score)
        relevancy.append(keyword_relevancy(q, response.answer_markdown))
        retrieval_precision_proxy.append(1.0 if response.citations else 0.0)

    return {
        "memory_mode": memory_mode,
        "constraint_compliance": round(mean(compliance), 3),
        "avg_cost_usd": round(mean(costs), 5),
        "avg_total_tokens": round(mean(tokens), 2),
        "avg_latency_s": round(mean(latencies), 3),
        "faithfulness": round(mean(faithfulness), 3),
        "answer_relevancy": round(mean(relevancy), 3),
        "retrieval_precision_proxy": round(mean(retrieval_precision_proxy), 3),
    }


def main() -> int:
    load_dotenv()
    args = parse_args()

    queries = json.loads(args.queries.read_text())

    variants = ["none", "vector-only", "summary-only", "hybrid"]
    results = [run_variant(queries, v, args.depth) for v in variants]

    headers = [
        "memory_mode",
        "constraint_compliance",
        "faithfulness",
        "answer_relevancy",
        "retrieval_precision_proxy",
        "avg_cost_usd",
        "avg_total_tokens",
        "avg_latency_s",
    ]

    table = [
        [
            r["memory_mode"],
            r["constraint_compliance"],
            r["faithfulness"],
            r["answer_relevancy"],
            r["retrieval_precision_proxy"],
            r["avg_cost_usd"],
            r["avg_total_tokens"],
            r["avg_latency_s"],
        ]
        for r in results
    ]

    print(tabulate(table, headers=headers, tablefmt="github"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"queries": queries, "results": results}, indent=2))

    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(
        "# Evaluation Results\n\n"
        + tabulate(table, headers=headers, tablefmt="github")
        + "\n"
    )
    print(f"\nSaved JSON: {args.output}")
    print(f"Saved Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
