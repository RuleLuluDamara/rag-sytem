# Production architecture

How this prototype would be scaled and deployed for real use: thousands of scanned
regulations and court documents, many concurrent analysts, auditable answers.

![Production architecture](architecture.svg)

## Prototype → production mapping

| Prototype (this repo) | Production | Why |
|---|---|---|
| Streamlit app | React web app + **FastAPI query service** behind an API gateway | stateless, horizontally scalable, auth + rate limits at the edge |
| `ingest_pdf()` loop, one process | **Upload API → Pub/Sub → page workers → caption workers → index workers** | pages are processed in parallel; slow OCR never blocks queries |
| `data/store/<doc_id>/*.jpg, *.png` | **Object storage (GCS)** | durable, cheap, lifecycle rules |
| `pages.json`, `chunks.json`, `embeddings.npy` | **PostgreSQL + pgvector** (HNSW) with `tsvector` for BM25 | one transactional store for metadata, vectors and keyword search; row-level security per tenant |
| In-memory BM25 + numpy cosine | pgvector ANN + Postgres full-text, same RRF fusion and Pasal/BAB boosts in the query service | scales past RAM; read replicas for the query path |
| Gemini API key in `.env` | **Vertex AI Gemini** in `asia-southeast2` (Jakarta), key/identity in Secret Manager | data residency, SLAs, provisioned throughput, no training on customer data |
| `pytest` + `scripts/eval_retrieval.py` | CI gate: unit tests + golden Q&A evaluation on every release | catches retrieval/answer regressions before deploy |

## Ingestion path (asynchronous)

1. **Upload API** issues a signed URL; the PDF lands in object storage. The SHA-1 of the file is the
   document id, so re-uploads are deduplicated.
2. A `document.uploaded` event fans out into **one task per page**.
3. **Page workers** (CPU pool, autoscaled on queue lag) render, deskew, OCR and run the OpenCV
   detectors. Each page result is written idempotently, so a crashed or retried job resumes
   where it stopped. Failures go to a dead-letter queue for inspection.
4. **Caption workers** send element crops to Gemini behind a token-bucket limiter shared by all
   workers (Redis), with exponential backoff on 429/503.
5. **Index workers** run once all pages are done: legal-structure chunking (BAB → Pasal → ayat),
   embeddings, and a single transactional write to Postgres. The document then becomes searchable.

Throughput scales linearly with page workers: at ~8 s/page on one vCPU, 50 workers ingest
roughly 20,000 pages per hour.

## Query path (synchronous)

1. Gateway authenticates (OAuth/IAP) and rate-limits per user.
2. The **query service** checks the Redis answer cache, then runs hybrid retrieval in Postgres
   (vector ANN + full-text), fuses with RRF and applies the Pasal/BAB and visual-intent boosts.
3. Crops of retrieved stamps, tables, signatures and figures are fetched from object storage and
   sent with the question to Gemini; the answer streams back to the browser with citations that
   link to the highlighted region of the page image.

Target: p95 under 5 s to first token. The service is stateless, so it autoscales on request
concurrency.

## Operations

- **CI/CD:** GitHub Actions runs `pytest` and the retrieval evaluation, builds container images,
  pushes to Artifact Registry, and deploys blue/green.
- **Observability:** structured logs, traces per question (retrieval → prompt → model), metrics for
  latency, OCR confidence, Gemini tokens and cost per query, alerts on queue lag and 429 rates.
- **Quality loop:** a golden set of real questions per document type, scored for retrieval hit@k
  and citation correctness on every release; analyst feedback feeds new test cases.
- **Security & compliance:** documents stay in the Jakarta region, encryption at rest and in
  transit, per-tenant isolation, audit log of who asked what, aligned with UU 27/2022 (PDP).
