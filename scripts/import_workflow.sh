#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_PATH="/workflows/g3_deep_research.workflow.json"

echo "Importing workflow from ${WORKFLOW_PATH} ..."
docker compose exec n8n n8n import:workflow --input="${WORKFLOW_PATH}" || {
  echo "Workflow import failed. Ensure containers are running: docker compose up -d"
  exit 1
}

echo "Imported. Activate the workflow in n8n UI if not already active."
