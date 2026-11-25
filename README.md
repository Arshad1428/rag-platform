# RAG platform – prototype

Answers questions **only** from your documents, with sources. Off-topic questions are refused, and
if the answer isn't in the documents it says "not found" instead of guessing. One platform for
any domain ("lateral"), with tenant / lateral / access separation, and three deployment profiles
chosen by configuration.

```
Client (user/app + identity) → RAG API (upload + /ask)
  Pipeline A – ingest once:   Land + tag → Convert → Read (OCR/layout) → Describe figures → Chunk
                              → Hybrid index + Doc catalog + Original images
  Pipeline B – ask each time: Gate 1 Scope → Retrieve + ACL filter → Gate 2 Match
                              → Generate cited answer → Gate 3 Fact check → Response
  Inference layer:            cloud models (profiles A/B) or on-prem GPU servers (profile C)
```

| Profile | Models | Reader | Index | Storage | Sign-in |
| --- | --- | --- | --- | --- | --- |
| **C** on-prem / air-gap (prototype) | Ollama (qwen3-vl, bge-m3) on GPU | Docling | Qdrant | local disk | demo users, or OIDC (Keycloak/ADFS) |
| **A** connected cloud | Azure OpenAI | Document Intelligence | Azure AI Search (or Qdrant) | Blob Storage | Entra ID |
| **B** restricted network | same as A, via Private Endpoints only | ← | ← | ← | ← |

Same code for all three. Switch with `PROFILE=A|B|C` (see `deploy/profiles/`).

---

## Run the prototype (profile C, on a GPU machine)

### Option 1 – Docker (Linux, or Windows with WSL2 + NVIDIA Container Toolkit)

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen3-vl:8b      # text + images (bigger GPU: qwen3-vl:30b)
docker compose exec ollama ollama pull bge-m3            # vectors
```
Open **http://localhost:8501**. API docs: http://localhost:8000/docs.

### Option 2 – without Docker (Windows or Linux)

1. Install **Python 3.11/3.12**, **LibreOffice**, **Ollama** (with an NVIDIA GPU, Ollama uses it automatically).
2. `ollama pull qwen3-vl:8b` and `ollama pull bge-m3`
3. In this folder:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate        # Windows Git Bash  (Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   # GPU for Docling + reranker: install the CUDA build of PyTorch first, see pytorch.org
   cp deploy/profiles/profile-c.env .env
   ```
   Windows: add `SOFFICE_PATH=C:/Program Files/LibreOffice/program/soffice.exe` to `.env`.
4. Two terminals:
   ```bash
   uvicorn rag.api:app --port 8000        # the API (processing runs in the background here)
   streamlit run ui/app.py                # the web page → http://localhost:8501
   ```

The first document downloads Docling's and the reranker's models (~2 GB) once. After that it
works without internet; set `HF_HUB_OFFLINE=1` in `.env` to make sure nothing tries to download.

**Fully air-gapped site:** do the steps above once on a connected machine, then copy the
Python packages (`pip download -r requirements.txt`), the Ollama models (`~/.ollama`) and the
Hugging Face cache (`~/.cache/huggingface`) to the offline server on approved media.

---

## Demo script (≈10 minutes)

Demo users (`deploy/demo_users.json`): **Design Engineer · Company A**, **Manufacturing Engineer · Company A**, **Design Engineer · Company B**.

1. Signed in as **Design Engineer · Company A**: upload the cleaned fastener catalogue, lateral
   `design`, readers `design`. Processing runs in the background; the page stays usable.
2. Ask a table question, a rule question and an image question → cited answers with page numbers
   and figures.
3. Ask *"What's the weather tomorrow?"* → refused (Gate 1).
   Ask about something not in the document → "not found" (Gate 2/3).
4. Switch to **Manufacturing Engineer · Company A** → same question gets no answer: the document is for the design group only.
5. Switch to **Design Engineer · Company B** → nothing visible: different tenant.
6. Re-upload the same file → skipped instantly (unchanged, no cost).

Tip: on a laptop CPU, use a 15–20 page extract; on a GPU, the full catalogue.

---

## Quality: measure, don't guess

```bash
python -m eval.run_eval eval/questions.jsonl design.a
```
Put 30–100 real questions in `eval/questions.jsonl` (with ≥20% that must be refused / not found).
Set `MIN_RELEVANCE` between the scores it prints for "should answer" and "should not answer".
`eval/results.csv` contains answers — keep it private.

## Tests

```bash
python -m pytest -q
```
19 tests with stand-in models (runs anywhere in seconds): the three gates, tenant / group / lateral
separation, unchanged-file skipping, cache reuse, retries, the failed state, resume after a crash,
deletion, image uploads, and that **logs never contain document text, questions or answers**.

---

## Reliability built in

- **Background processing** – uploads return immediately; a worker processes the file.
- **Resume** – job state is saved; unfinished jobs restart automatically, and cached Read results
  and figure descriptions mean finished work is never repeated.
- **Retries** – 3 tries with growing waits, then status `failed` for someone to check.
- **No wasted work** – unchanged files are skipped; identical images described once; tiny images
  skipped; images shrunk before the vision model.
- **Safe updates** – a new version replaces the old one; deleting removes pieces, images and the
  catalog entry.
- **Content-free logs** – operators see events, file names, counts, timings and scores only.

## Code map

```
rag/config.py            all settings; PROFILE picks the defaults
rag/security.py          identity: demo | entra_easyauth | oidc | none
rag/logs.py              content-free event logging
rag/api.py               RAG API: /documents, /jobs, /ask, /figures, /health
rag/providers/llm.py     inference layer: Ollama | Azure OpenAI
rag/providers/reader.py  Read: Docling | Azure Document Intelligence
rag/providers/index.py   Hybrid index + ACL filter: Qdrant | Azure AI Search
rag/providers/rerank.py  local reranker (with Qdrant)
rag/providers/storage.py files, figures, doc catalog: local | Blob Storage
rag/ingest/              Pipeline A (convert, figures, chunk, summary, pipeline, worker)
rag/ask/graph.py         Pipeline B (LangGraph, three gates)
ui/app.py                prototype web page (talks only to the API)
deploy/                  profile settings, demo users
```

## Known limits of the prototype

- One worker thread inside the API. Production (A/B): the same `run_job` runs in a Container Apps
  job fed by a queue, with several copies.
- The doc catalog is a JSON file in storage – fine for hundreds of documents; move to a table later.
- Anyone who can see a document can remove it; production needs an owner/admin role.
- Chunking method is still to be decided; the current chunker keeps sections and tables together.
- Tables split across pages are not yet joined.
- Profiles A/B are written against the Azure SDKs and checked offline, but not yet run on Azure.
