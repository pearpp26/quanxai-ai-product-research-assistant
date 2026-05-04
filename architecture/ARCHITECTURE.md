# Architecture

This document describes the design, scaling strategy, trade-offs, and observed
performance of the AI Product Research Assistant.

The accompanying diagrams live alongside this file:

- `data_pipeline_diagram.png` — CSV → embedding → Qdrant.
- `system_architecture_diagram.png` — request-time flow through FastAPI,
  the LangGraph agent, the three tools, and SQLite logging.

---

## Monthly Catalog Updates

The catalog is refreshed by re-running the ingestion script:

```bash
python src/ingest.py
```

Key properties:

- **Chunked descriptions.** Each row's `description` is split by
  `chunk_text()` (sentence/word-boundary aware, `chunk_size=400`,
  `overlap=50`). Short descriptions return a single chunk; longer ones
  yield multiple. Every chunk is embedded with the product header
  (`name | category | brand | chunk`) so each vector retains product
  context. The matching `chunk_text` is stored on the payload for
  traceability.
- **Idempotent upsert with stable per-chunk IDs.** Each chunk's Qdrant
  point ID is `product_int_id * 1000 + chunk_index` (e.g. `PROD-042`
  chunk 0 → `42000`, chunk 1 → `42001`). Re-running the script overwrites
  existing chunks in place — no full re-index required.
- **Chunk dedup at query time.** `search_product_catalog` over-fetches
  `limit * 3` candidates and collapses them to one entry per `product_id`,
  keeping the highest-scoring chunk. This prevents multi-chunk products
  from crowding the top-k.
- **Embedding cost is proportional to chunk count.** Each run re-embeds
  every chunk (the script does not diff against Qdrant). For ~100 products
  this is fine; at scale we would hash each chunk and skip unchanged ones.
- **Hard-deletes and orphan chunks are not propagated.** Rows removed
  from the CSV remain in Qdrant, and if a description shrinks to fewer
  chunks than the previous run the trailing chunks become orphans.
  Documented limitations; in production we would diff CSV-derived chunk
  IDs against the collection and delete the difference.

---

## Scaling Strategy

| Layer | Today | Production direction |
|-------|-------|----------------------|
| FastAPI (`web` service) | Single Docker container | Stateless — replicate horizontally behind a load balancer (ECS / Kubernetes). |
| SQLite (`data/assistant.db`) | Single-file, single-writer | Replace with PostgreSQL via the existing SQLAlchemy interface; only `DATABASE_URL` and the engine config change. |
| Qdrant | Single-node container | Qdrant Cloud or a self-hosted sharded cluster. The client URL is environment-driven (`QDRANT_URL`), so the swap is config-only. |
| LLM (OpenRouter / Llama-3.3-70b) | Synchronous request/response | Add `/query/stream` (SSE) or async job queue (Celery + Redis) so the HTTP request returns immediately. |

The dominant bottleneck is the LLM round-trip (see Load Test Results), so the
highest-leverage scaling work is on the LLM call path, not on FastAPI or
Qdrant.

---

## Production Considerations

**Latency.** A single `/query` triggers a planner LLM call plus an agent
LLM call (and one more agent call per tool round-trip). End-to-end latency
is 3–10 seconds, dominated by OpenRouter. For production UX:

- Stream tokens via SSE so the client renders progressively.
- Or accept the request, return a job ID, and poll `/queries/{id}`.

**Cost.** Each request is billed by OpenRouter (LLM tokens) and optionally
Tavily (web search). For repeated or near-duplicate queries, add a Redis
cache keyed by a hash of the normalized query → response. Cache TTL should
be short (minutes) for web-search queries and longer (hours) for catalog
queries that don't depend on live data.

**Security.**
- API keys are loaded from `.env` via `python-dotenv` and never committed.
- `/query` is unauthenticated in this MVP. Production should require an
  API token or session, and CORS should be restricted to known origins.
- The agent has no destructive tools — `analyze_profit_margins` is read-only
  math, the RAG tool is read-only, and `search_web` only reads.

**Observability.** Every query is persisted to `query_logs` with
`tools_used`, `reasoning`, and `response_text`. `GET /queries` exposes the
log for inspection; `POST /feedback` records up/down ratings tied to a
`query_log_id`. In production we would also emit structured logs and per-tool
latency metrics to a real backend (Datadog / OTEL).

---

## Trade-offs

- **SQLite vs PostgreSQL.** SQLite is zero-config, ships in the
  container, and is sufficient for single-node development. Its lack of
  concurrent-write support is a hard ceiling at scale, hence the planned
  swap to PostgreSQL.
- **Local Qdrant vs Qdrant Cloud.** Running Qdrant in `docker-compose`
  avoids egress cost and vendor lock-in but means we own backup, upgrade,
  and HA. Qdrant Cloud trades that for SLA-backed availability.
- **OpenRouter / Llama-3.3-70b vs hosted Claude/GPT-4.** OpenRouter gives
  us model flexibility and a lower per-token cost, at the price of an extra
  routing hop and less predictable latency. The bigger frontier models
  would simplify tool-call reliability but cost ~10× more per query.
- **Two-node graph (planner → agent) vs single-node ReAct.** The separate
  planner produces a plain-text rationale we can persist as `reasoning` in
  `QueryLog`, which is much easier to debug than parsing intermediate
  ReAct traces. The cost is one extra LLM call per query.
- **In-process margin math vs LLM-computed math.** `analyze_profit_margins`
  is deterministic Python that reads from SQLite. We deliberately keep the
  LLM out of the math path because LLM arithmetic is unreliable.

---

## Load Test Results

Tool: [Locust](https://locust.io/) 2.43.x
Endpoint: `POST /query`
Target: `http://localhost:8000` (FastAPI in Docker, Qdrant in Docker,
Llama-3.3-70b via OpenRouter)
Users: 3 concurrent (spawn rate: 1/s)
Duration: 2 minutes
Wait time: `between(2, 5)` seconds per virtual user
Query mix: 12 queries spanning all three tools and multi-tool prompts
(see `load_tests/locustfile.py`).

```
Total requests:  18
Failures:        0 (0.00%)
RPS:             0.15

Latency (ms):
  Min:           5,740
  Avg:          15,061
  p50 (median): 13,000
  p90:          22,000
  p95:          38,000
  p99:          38,000
  Max:          38,493
```

**Observed bottleneck.** OpenRouter LLM round-trips account for the vast
majority of response time. With a median of 13 s and a p95 of 38 s for
3 concurrent users, the LLM is clearly the constraint — Qdrant
`query_points` returns in <50 ms for this collection size, and FastAPI
request-handling overhead is <5 ms. The wide tail (p50 → p95 spread of
~25 s) is the planner+agent multi-call pattern: queries that trigger
multiple tools chain additional LLM round-trips. This is the data point
that justifies the streaming / async-queue direction in **Scaling
Strategy** — at production load the synchronous request/response shape
will time clients out long before FastAPI or Qdrant feel any pressure.

> **Reproduce locally:**
> ```bash
> docker-compose up --build -d
> python src/ingest.py    # only needed once
> locust -f load_tests/locustfile.py \
>        --host=http://localhost:8000 \
>        --headless --users 3 --spawn-rate 1 --run-time 2m
> ```
