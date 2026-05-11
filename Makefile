.PHONY: help up down logs ingest api eval demo test fmt clean

# Default target
help:
	@echo "FamilyHQ RAG — common commands"
	@echo ""
	@echo "  make up         Start postgres + pgvector (docker compose)"
	@echo "  make down       Stop postgres"
	@echo "  make logs       Tail postgres logs"
	@echo "  make ingest     Parse trend SQLite + product PDFs → embed → pgvector"
	@echo "  make api        Start FastAPI on http://localhost:8000"
	@echo "  make demo       Start Streamlit demo"
	@echo "  make eval       Run full eval suite (retrieval + latency + cost + llm_judge)"
	@echo "  make test       Run pytest"
	@echo "  make fmt        Run ruff format + lint"
	@echo "  make clean      Remove pycache, eval runs (keeps baseline.json)"

up:
	docker compose up -d
	@echo "Waiting for postgres healthcheck..."
	@until docker compose exec -T postgres pg_isready -U rag -d familyhq_rag > /dev/null 2>&1; do sleep 1; done
	@echo "✓ Postgres ready on localhost:5432"

down:
	docker compose down

logs:
	docker compose logs -f postgres

ingest:
	python -m src.ingest

api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

demo:
	streamlit run streamlit_app.py

eval:
	python -m src.eval.run_all

test:
	pytest -v

fmt:
	ruff format src tests
	ruff check --fix src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find eval_results/runs -type f -delete 2>/dev/null || true
