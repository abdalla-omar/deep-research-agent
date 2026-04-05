.PHONY: up down logs import-workflow test lint eval demo

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f n8n

import-workflow:
	bash scripts/import_workflow.sh

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

eval:
	uv run python scripts/run_eval.py --output artifacts/eval/results.json

demo:
	bash scripts/run_demo.sh
