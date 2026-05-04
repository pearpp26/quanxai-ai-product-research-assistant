# AI Product Research Assistant

A FastAPI service that answers product questions by
routing through a LangGraph agent with three tools: a Qdrant-backed RAG
search over an internal product catalog, a Tavily-backed web search, and
deterministic profit-margin analysis over a SQLite mirror of the catalog.

## Features

- **RAG over the internal catalog** — Cohere `embed-english-v3.0` embeddings
  in a Qdrant vector collection, queried by semantic similarity.
- **Web search** — Tavily for current market prices, competitor data,
  and reviews.
- **Profit-margin analysis** — deterministic Python math over SQLite
  (no LLM-computed arithmetic).
- **Two-node LangGraph agent** — a planner LLM emits a plain-text rationale,
  then a tool-bound LLM issues structured tool calls. The rationale is
  persisted with every query for transparency.
- **Query history & feedback** — every `/query` is logged to SQLite;
  `/queries` returns recent history; `/feedback` records up/down ratings.

## Architecture

See [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) for the
full design, scaling strategy, trade-offs, and load test results.

```
User → POST /query (FastAPI)
        → Planner LLM (OpenRouter / Llama-3.3-70b)
        → Agent LLM with bound tools
            ├── search_product_catalog → Qdrant
            ├── search_web             → Tavily
            └── analyze_profit_margins → SQLite math
        → Response logged to SQLite
        → JSON returned to user
```

## Prerequisites

- Python 3.11+ (project tested with 3.12)
- Docker & Docker Compose
- API keys for: OpenRouter, Cohere, Tavily

## Setup

1. Clone the repo and enter the project directory:
   ```bash
   git clone <repo-url> product-research-assistant
   cd product-research-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` from the example and fill in your keys:
   ```bash
   cp .env.example .env
   # then edit .env and set:
   #   OPEN_ROUTER_API_KEY='...'
   #   COHERE_API='...'
   #   TAVILY_API='...'
   ```

## Data Ingestion

Ingest the 105-product catalog into Qdrant. This is **idempotent** — re-running
overwrites updated rows by integer point ID and inserts new ones.

```bash
# 1. Start Qdrant first (so the script has somewhere to write)
docker-compose up qdrant -d

# 2. Run the ingest script (defaults QDRANT_URL to http://localhost:6333)
python src/ingest.py
```

The product table in SQLite (`data/assistant.db`) is auto-populated on
first FastAPI startup from `data/products_catalog.csv`.

## Run the Application

```bash
docker-compose up --build
```

This starts both `qdrant` (port 6333) and `web` (port 8000). The web
service initializes the SQLite schema and seeds the product table on first
startup.

## Test the API

All `POST` endpoints **require** `Content-Type: application/json` —
FastAPI returns 422 without it.

### Health check
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Internal catalog (RAG)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What wireless headphones do we have in stock?"}'
```

### Web search
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Current market price for noise-cancelling headphones?"}'
```

### Profit-margin analysis
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which products have the lowest profit margins?"}'
```

### Multi-tool (catalog + web + pricing)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Should we adjust AudioMax headphones pricing vs competitors?"}'
```

### Query history
```bash
curl http://localhost:8000/queries
```

### Submit feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_log_id": 1, "rating": "up"}'
```

## Tests

```bash
python -m pytest tests/ -v
```

Covers `/health`, the pricing tool against an in-memory SQLite, and the
RAG tool with a mocked Qdrant + Cohere client.

## Load Testing

```bash
locust -f load_tests/locustfile.py \
       --host=http://localhost:8000 \
       --headless --users 3 --spawn-rate 1 --run-time 2m
```

Each request hits the live LLM provider — keep `--users` and `--run-time`
modest to avoid burning API credits. Results and analysis are recorded in
[architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md).

## Project Layout

```
src/
  main.py              # FastAPI app: /health, /query, /queries, /feedback
  ingest.py            # CSV → Cohere embeddings → Qdrant upsert
  agent/graph.py       # LangGraph: planner → agent → tools
  tools/
    rag_tool.py        # Qdrant semantic search
    search_tool.py     # Tavily web search
    pricing_tool.py    # Deterministic margin math over SQLite
  database/
    models.py          # Product, QueryLog, Feedback
    session.py         # SQLAlchemy engine / session
    init_db.py         # Schema create + product seeding
data/
  products_catalog.csv # Source of truth for the catalog
  assistant.db         # Runtime SQLite (gitignored)
tests/                 # pytest suite
load_tests/            # Locust scenarios
architecture/          # Diagrams and ARCHITECTURE.md
```

## Limitations & Future Improvements

### What I didn't finish and why

- **Hard-deletes in the vector DB.** If a product is removed from the
  CSV, its point remains in Qdrant. The ingest script only upserts; it
  does not diff IDs against the existing collection. Acceptable for an
  MVP with a stable catalog, but documented as a known gap.
- **Test coverage is intentionally minimal.** Smoke tests for `/health`,
  unit tests for the deterministic pricing tool, and tests for the RAG
  tool with a mocked Qdrant + Cohere client. The LangGraph agent itself
  is not unit-tested because covering a non-deterministic two-LLM graph
  cleanly requires more harness than fit in the time budget.
- **No response caching.** Repeated identical queries re-run the full
  agent. A Redis layer keyed on a normalized-query hash would cut both
  latency and OpenRouter spend, but it added complexity that wasn't
  necessary to demonstrate the core agent + tool pattern.

### What I would improve with more time

- **Streaming responses.** OpenRouter round-trips dominate latency
  (3–8 s per query). An SSE `/query/stream` endpoint would let the
  client render progressively instead of waiting on the full
  request/response cycle.
- **Postgres swap.** SQLite is fine for a single container but bottlenecks
  on concurrent writes. The SQLAlchemy interface is already in place, so
  the swap is essentially a `DATABASE_URL` change plus a migration tool
  (Alembic) for schema versioning.
- **Resilience.** Add bounded retries with exponential backoff and a
  circuit breaker around the OpenRouter and Tavily clients. Currently
  Tavily fails open (returns a "search unavailable" string), but
  OpenRouter failures bubble as 500s.
- **Agent integration tests.** With a mocked LLM that returns scripted
  tool-call sequences, we could pin the planner → agent → tool flow and
  catch regressions when prompts or tool signatures change.
- **Observability.** Per-tool latency metrics and structured logs to a
  real backend, plus a `/metrics` Prometheus endpoint.

### What I learned during this assignment

- Wiring Qdrant + Cohere into a clean, idempotent ingestion path —
  particularly the value of a stable integer point ID derived from a
  human-readable `product_id`.
- The two-node planner-then-agent pattern in LangGraph as a way to
  capture an explicit, persistable "reasoning" string without parsing
  ReAct intermediate steps.
- Tool-binding through `langchain-openai` against an OpenRouter-hosted
  open-source model — the open-source Llama needs careful prompt
  engineering ("you MUST actually invoke the tool via the structured
  channel — never write JSON in content") to avoid stringly-typed tool
  calls.

### What challenges I faced

- Getting Llama-3.3-70b on OpenRouter to emit reliable structured tool
  calls instead of inlined JSON in the message content. Solved with
  explicit format rules and worked examples in the system prompt.
- Capturing the planner's rationale and the final assistant answer
  separately from a single `MessagesState` history — required walking
  the message list and identifying the trailing AI message without
  tool calls as the "answer".
- Docker networking between `web` and `qdrant`: the `web` container
  needs `QDRANT_URL=http://qdrant:6333` (service DNS), while local
  scripts use `http://localhost:6333` (published port). Resolved by
  keeping the env var override in `docker-compose.yml` and defaulting
  to localhost in code.
