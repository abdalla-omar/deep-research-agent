#!/usr/bin/env bash
set -euo pipefail

N8N_URL=${N8N_URL:-http://localhost:5678}
WEBHOOK_PATH=${WEBHOOK_PATH:-webhook/research}

read -r -d '' PAYLOAD <<'JSON' || true
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
JSON

echo "POST ${N8N_URL}/${WEBHOOK_PATH}"
curl -sS -X POST "${N8N_URL}/${WEBHOOK_PATH}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" | jq .
