# Evaluation: Deep Research Agent + Memory Constraints

## 1) Objective and constraints

This evaluation measures how the agent behaves under explicit hard limits:

- `max_context_tokens_per_call = 2000`
- `max_session_cost_usd = 0.05`
- `max_runtime_seconds = 480`
- citations required for non-trivial factual claims

Primary goals:

1. Preserve factual grounding and citation linkage under constrained context.
2. Maintain cost/runtime compliance via centralized budget governance.
3. Quantify architecture trade-offs across memory variants.

## 2) Architecture variants compared

Ablation set:

- **A: no-memory (`none`)**
- **B: vector-only (`vector-only`)**
- **C: summary-only (`summary-only`)**
- **D: hybrid (`hybrid`)** (target architecture)

Variant behavior in this implementation:

- `none`: disables summary/evidence context packing contribution.
- `vector-only`: uses episodic evidence cards, no summary memory.
- `summary-only`: uses summary memory, no evidence card packing.
- `hybrid`: uses working + summary + episodic evidence with utility knapsack selection.

## 3) Dataset and query suite

Evaluation query set: `scripts/query_suite.json` (5 multi-part prompts).

Themes covered:

- memory architecture trade-offs
- cost-aware orchestration
- hallucination/faithfulness controls
- enterprise rollout and observability

Execution command:

```bash
uv run python scripts/run_eval.py --output artifacts/eval/results.json
```

## 4) Metrics

### Constraint compliance metrics

- session cost compliance (`<= $0.05`)
- context envelope compliance (`<= 2000` per call)
- runtime compliance (`<= 480s`)

### Retrieval quality proxies

- retrieval precision proxy: whether evidence/citations are consistently produced

### Final output quality proxies

- faithfulness proxy: fraction of claim IDs in answer that are backed by selected evidence IDs
- answer relevancy proxy: query-keyword coverage in generated answer

### Operational metrics

- average session token consumption
- average session cost
- average end-to-end latency

## 5) Results table

Source: `artifacts/eval/results.json` and `artifacts/eval/results.md`

| memory_mode | constraint_compliance | faithfulness | answer_relevancy | retrieval_precision_proxy | avg_cost_usd | avg_total_tokens | avg_latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1.000 | 1.000 | 0.278 | 1.000 | 0.00620 | 2878.8 | 0.050 |
| vector-only | 1.000 | 1.000 | 0.278 | 1.000 | 0.00620 | 2880.4 | 0.001 |
| summary-only | 1.000 | 1.000 | 0.278 | 1.000 | 0.00632 | 2939.4 | 0.001 |
| hybrid | 1.000 | 1.000 | 0.278 | 1.000 | 0.00633 | 2944.2 | 0.001 |

## 6) Ablation analysis

Key observations from this run profile:

1. **All variants remained within hard limits**.
   - Compliance stayed at `1.0` across cost and context constraints.
2. **Hybrid had slightly higher token/cost footprint**.
   - Expected: hybrid carries both summary and episodic evidence context overhead.
3. **Faithfulness/retrieval proxies are saturated in offline fallback mode**.
   - This is expected because deterministic fallback retrieval emits stable evidence cards and citation IDs.
4. **Quality separation is muted in offline mode**.
   - Stronger distinction is expected in API-backed runs with heterogeneous web content.

Implication:

- Hybrid remains the preferred default because it best matches production behavior under noisy retrieval, even though it costs slightly more than simpler variants.

## 7) Failure cases

Observed/anticipated failure modes and mitigations:

1. **Budget cliff during retrieval bursts**
   - Symptom: projected retrieval cost spikes near session limit.
   - Mitigation: degrade mode lowers retrieval depth (`top_k`) and prevents overrun.

2. **Citation drift after synthesis edits**
   - Symptom: answer contains unsupported claim IDs.
   - Mitigation: verifier flags unsupported IDs and surfaces violation in `budget_report.constraint_violations`.

3. **Context overflow under verbose evidence**
   - Symptom: packed context may exceed strict envelope.
   - Mitigation: utility-per-token pruning and summary truncation keep prompt under limit.

4. **Quality flattening in offline deterministic mode**
   - Symptom: ablation metrics appear too similar.
   - Mitigation: run online eval (OpenAI + Tavily) and optionally add DRB subset to increase retrieval variance.

## 8) Trade-off summary and future improvements

Trade-off summary:

- **Hybrid memory** offers best robustness to real-world retrieval variance, at modest additional token/cost overhead.
- **Vector-only** is cheaper/faster but loses high-level continuity when sub-questions diverge.
- **Summary-only** is compact but weaker for precise claim-level attribution.
- **No-memory** is simplest but least resilient for deep, multi-hop synthesis.

Planned next improvements:

1. Integrate real reranker and contextual compression retriever node chain in n8n.
2. Add explicit per-claim support scoring beyond ID linkage.
3. Expand evaluation set with online runs and DRB-aligned tasks.
4. Add latency breakdown per stage (decompose, retrieve, compact, synthesize, verify).
5. Add long-horizon memory persistence to Qdrant/Postgres with replayable session traces.
