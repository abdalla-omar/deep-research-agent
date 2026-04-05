# Constrained Deep Research Agent (n8n)

A **memory-constrained deep research agent** that decomposes complex queries, retrieves evidence, and synthesizes citation-grounded answers under strict runtime/cost/token limits.

It breaks down multi-part research questions, retrieves evidence, compresses memory, synthesizes a citation-grounded answer, and enforces hard limits:

- `max_context_tokens_per_call <= 2000`
- `max_session_cost_usd <= 0.05`
- `max_runtime_seconds <= 480`

## Design goals

This implementation focuses on:

- **Reliability under constraints**: deterministic budget governance and context packing.
- **Reproducibility**: local Docker stack, exported n8n workflow, and runnable tests/evals.
- **Traceability**: claim-to-source mapping with explicit budget/trace metadata in responses.
- **Iteration speed**: clear architecture and scripts for fast experimentation.

## Repository layout

- `docker-compose.yml`: local stack (`n8n`, `postgres`, `redis`, `qdrant`)
- `workflows/g3_deep_research.workflow.json`: n8n workflow export
- `src/research_agent/`: core deterministic engine (budget, memory, retrieval, synthesis, verification)
- `schemas/`: request/response JSON schemas
- `tests/`: unit tests for constraints, memory packing, orchestration
- `scripts/run_eval.py`: ablation runner for A/B/C/D memory variants
- `evaluation.md`: architecture trade-offs and experimental results

## Quickstart

### 1) Prerequisites

- Docker + Docker Compose
- Python 3.11+
- `jq` (for demo script output formatting)

### 2) Configure environment

```bash
cp .env.example .env
# Fill OPENAI_API_KEY and TAVILY_API_KEY if available.
```

The system runs in offline-safe fallback mode if API keys are empty.

### 3) Install Python dependencies

```bash
uv sync --extra dev
```

### 4) Start infrastructure

```bash
make up
```

### 5) Import workflow into n8n

```bash
make import-workflow
```

Open n8n UI at `http://localhost:5678`, confirm/imported workflow, and activate it.

### 6) Run demo request

```bash
make demo
```

Default endpoint:

- `POST http://localhost:5678/webhook/research`

## API contract

### Endpoint

`POST /webhook/research`

### Request schema

See `schemas/request.schema.json`.

Example:

```json
{
  "query": "Design a deep research agent architecture under a 2K context and $0.05/session budget.",
  "constraints": {
    "max_context_tokens_per_call": 2000,
    "max_session_cost_usd": 0.05,
    "max_runtime_seconds": 480
  },
  "options": {
    "depth": "medium",
    "citations_required": true
  }
}
```

### Response schema

See `schemas/response.schema.json`.

Top-level response fields:

- `answer_markdown`
- `citations[]` (claim-id-linked)
- `query_breakdown[]`
- `budget_report` (tokens, cost, violations)
- `trace` (processed subquestions/evidence cards, degrade-mode)

## Architecture

Logical flow:

1. Trigger
2. Session initialization with ledger
3. Query decomposition
4. Retrieval loop (multi-query variants)
5. Evidence card normalization
6. Memory compaction and utility knapsack context pack
7. Synthesis under budget
8. Verification and final response formatting

Mermaid source: `docs/architecture.mmd`
Reference set: `docs/references.md`

## Constraint enforcement model

- Central ledger tracks:
  - total token usage
  - projected/actual cost
  - max context tokens seen
  - runtime and violation flags
- Degrade mode triggers when projected work exceeds budget
- Hard limits are fail-closed and reflected in `budget_report.constraint_violations`

## Memory design

Hybrid memory layers:

- **Working memory**: bounded recent subquestion state
- **Summary memory**: rolling compacted synthesis fragments
- **Episodic evidence cards**: scored source-grounded facts

Context pack uses utility scoring:

`utility = 0.45*relevance + 0.25*novelty + 0.20*source_quality + 0.10*recency`

Token budget allocation:

- planner/state: 250
- summary: 450
- evidence cards: 1050
- instructions/citation schema: 250

## Local engine (non-n8n) for testing and evals

Run one request directly through Python engine:

```bash
uv run python scripts/run_engine.py --query "Compare vector-only vs hybrid memory for deep research"
```

Run memory ablations and emit metrics:

```bash
uv run python scripts/run_eval.py --output artifacts/eval/results.json
```

## Tests

```bash
uv run pytest
```

Coverage includes:

- budget guard behavior
- context packing under hard token budget
- offline orchestration path and response structure

## Notes and limitations

- Pricing values in the governor are conservative defaults and should be tuned to your selected model pricing.
- `qdrant` is provisioned in stack for episodic memory expansion, while v1 evidence-card store is in workflow memory + Postgres tables.
- n8n workflow currently prioritizes reproducibility and deterministic behavior over maximal node-level complexity.
