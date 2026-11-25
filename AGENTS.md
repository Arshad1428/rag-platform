# Context for AI coding assistants

Read this and README.md before changing code.

## What this is
A RAG platform: answers only from company documents, with citations; refuses off-topic questions;
says "not found" instead of guessing. One platform for any domain ("lateral"), separated by
tenant / lateral / ACL tags. Agreed high-level design (Preeti): Client → RAG API → Pipeline A
(ingest) and Pipeline B (ask, three gates) → shared data; inference layer = cloud (A/B) or on-prem
GPU (C). Profiles are configuration only – never fork the ingest/ask logic per client.

## Non-negotiable rules
1. Every chunk carries `tenant`, `lateral`, `acl`. Every search filters by the user's tenant and
   groups (+ lateral/doc filters when given). Never filter after retrieval instead.
2. Pipeline B order is fixed: Gate 1 Scope → Retrieve + ACL filter → Gate 2 Match → Generate →
   Gate 3 Fact check (number check + model check). Outcomes: answered | refuse | not found.
3. Logs contain only allow-listed fields (`rag/logs.py`): never document text, questions, answers,
   or exception messages (they can contain document text) – log the exception type only.
4. Profile-specific code lives only in `rag/providers/*`. Everything else is provider-agnostic.
5. All match scores are 0-1 (Azure's 0-4 semantic score is divided by 4).
6. Keep `python -m pytest -q` green; add a test for every behaviour change.
7. No company documents, part numbers or names in the repo (tests use invented sample data).

## Decisions already made
LangGraph (not CrewAI/Strands); own ingestion code (not the Azure built-in indexer); figures are
described as text (no image embeddings in v1); Qdrant (not Chroma) when self-hosting the index;
catalog-driven Gate 1 (no hard-coded topics); Docling for on-prem reading; current Azure models
(gpt-5.1 / gpt-5-mini / text-embedding-3-large at 1024 dims) – gpt-4o and gpt-4.1 are retired.

## Open items
- Chunking method (to be decided by the team).
- Join tables split across pages (Docling: table-stitcher; Azure: merge on Document Intelligence output).
- "Fetch whole table" for list questions; two-phase ingestion (text first, figures later).
- Answer cache (Redis: Docker locally, Azure Managed Redis in Azure), streaming answers.
- Prompt Shields (A/B), owner/admin roles for delete, queue-based worker for A/B.
