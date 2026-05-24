.PHONY: help up down logs ingest api eval eval-langchain eval-langgraph eval-langgraph-smoke demo embeddings test fmt clean

# Default target
help:
	@echo "FamilyHQ RAG — common commands"
	@echo ""
	@echo "  make up         Start postgres + pgvector (docker compose)"
	@echo "  make down       Stop postgres"
	@echo "  make logs       Tail postgres logs"
	@echo "  make ingest     Parse trend SQLite + product PDFs → embed → pgvector"
	@echo "  make api        Start FastAPI on http://localhost:8000"
	@echo "  make demo       Start Streamlit demo (uses in-memory embeddings; no Postgres needed)"
	@echo "  make embeddings One-shot: regenerate data/embeddings.npy for the demo"
	@echo "  make eval       Run full eval suite (retrieval + latency + cost + llm_judge)"
	@echo "  make eval-langchain  Same eval suite via the LangChain implementation (writes to eval_results/langchain/)"
	@echo "  make eval-langgraph  Same eval suite via the LangGraph 4-agent pipeline (writes to eval_results/langgraph/)"
	@echo "  make eval-langgraph-smoke  Single-query sanity check on the multi-agent runner (no files written)"
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
	streamlit run app/streamlit_app.py

embeddings:
	python -m scripts.precompute_embeddings

eval:
	python -m src.eval.run_all

eval-langchain:
	python -m src.eval.run_langchain

eval-langgraph:
	python -m src.eval.run_langgraph

eval-langgraph-smoke:
	python -m src.eval.run_langgraph --smoke

test:
	pytest -v

fmt:
	ruff format src tests
	ruff check --fix src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find eval_results/runs -type f -delete 2>/dev/null || true
