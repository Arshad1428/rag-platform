"""RAG API - one front door, two flows (shared sign-in, logging and provider config).
  POST /documents  upload  -> Pipeline A (background job)
  POST /ask        question -> Pipeline B (LangGraph, three gates)
The client never talks to models or the index directly.

Run:  uvicorn rag.api:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from rag.ask.graph import ask
from rag.config import settings
from rag.ingest import worker
from rag.ingest.convert import SUPPORTED
from rag.ingest.pipeline import delete_document
from rag.logs import event
from rag.providers.storage import catalog_all, catalog_visible, storage
from rag.security import User, current_user, demo_users

@asynccontextmanager
async def lifespan(_app):
    worker.start()   # background ingestion + resume of unfinished jobs
    event("api.started", profile=settings.profile)
    yield


app = FastAPI(title="RAG platform", lifespan=lifespan)


def _visible_doc(doc_id: str, user: User) -> dict:
    e = catalog_all().get(doc_id)
    if not e or e["tenant"] != user.tenant or not set(e["acl"]) & set(user.acl_groups):
        raise HTTPException(404, "Not found")
    return e


@app.get("/health")
def health():
    return {"ok": True, "profile": settings.profile, "llm": settings.llm_provider,
            "reader": settings.reader, "index": settings.index, "storage": settings.storage,
            "auth": settings.auth_mode}


@app.get("/demo-users")
def list_demo_users():
    if settings.auth_mode != "demo":
        raise HTTPException(404, "Not available")
    return {k: {"name": v.get("name", k), "tenant": v["tenant"], "groups": v.get("groups", [])}
            for k, v in demo_users().items()}


@app.get("/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "name": user.name, "tenant": user.tenant, "groups": user.acl_groups}


# ---------------- Pipeline A ----------------
@app.post("/documents")
async def upload(file: UploadFile = File(...), lateral: str = Form("general"), acl: str = Form(""),
                 title: str = Form(""), user: User = Depends(current_user)):
    name = (file.filename or "").replace("/", "_").replace("\\", "_")
    if not name or "." + name.rsplit(".", 1)[-1].lower() not in SUPPORTED:
        raise HTTPException(400, "Unsupported file type")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")
    groups = [g.strip() for g in acl.split(",") if g.strip()] or [settings.everyone_group]
    # documents always land in the uploader's own tenant
    job = worker.submit(data, name, user.tenant, lateral.strip().lower() or "general", groups,
                        title.strip() or None, user.id)
    return {"job_id": job["id"], "doc_id": job["doc_id"], "status": job["status"]}


@app.get("/jobs")
def jobs(user: User = Depends(current_user)):
    return [{k: j.get(k) for k in ("id", "doc_id", "file_name", "lateral", "status", "step",
                                    "attempts", "error", "chunks", "created", "updated")}
            for j in worker.list_jobs(user.tenant)[:50]]


@app.get("/documents")
def documents(lateral: str | None = None, user: User = Depends(current_user)):
    docs = catalog_visible(user.tenant, user.acl_groups, [lateral] if lateral else None)
    return [{k: d.get(k) for k in ("doc_id", "doc_title", "file_name", "lateral", "acl", "summary",
                                    "chunks", "figures", "pages", "updated")} for d in docs]


@app.delete("/documents/{doc_id}")
def remove(doc_id: str, user: User = Depends(current_user)):
    # Prototype: anyone who can see a document can remove it. Production: owner/admin role only.
    _visible_doc(doc_id, user)
    delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/file")
def original(doc_id: str, user: User = Depends(current_user)):
    e = _visible_doc(doc_id, user)
    data = storage().get(f"files/{e['tenant']}/{doc_id}/{e['file_name']}")
    if data is None:
        raise HTTPException(404, "Not found")
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{e["file_name"]}"'})


@app.get("/figures/{doc_id}/{name}")
def figure(doc_id: str, name: str, user: User = Depends(current_user)):
    _visible_doc(doc_id, user)
    data = storage().get(f"figures/{doc_id}/{name.replace('/', '')}")
    if data is None:
        raise HTTPException(404, "Not found")
    return Response(data, media_type="image/png")


# ---------------- Pipeline B ----------------
class AskRequest(BaseModel):
    question: str
    laterals: list[str] = []
    doc_ids: list[str] = []
    history: list[dict] = []


@app.post("/ask")
def ask_endpoint(req: AskRequest, user: User = Depends(current_user)):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    out = ask(req.question, user, [l.lower() for l in req.laterals], req.doc_ids, req.history[-6:])
    cited = set(out.get("cited", []))
    return {
        "status": out["status"],
        "answer": out["answer"],
        "best_score": round(out.get("best_score", 0.0), 3),
        "citations": [{
            "n": s["n"], "doc_id": s["doc_id"], "doc_title": s["doc_title"], "section": s["section"],
            "pages": s["pages"], "score": s["score"],
            "figures": [f"/figures/{s['doc_id']}/{k.rsplit('/', 1)[-1]}" for k in s["figure_keys"]],
        } for s in out.get("sources", []) if s["n"] in cited],
    }
