"""Background jobs for Pipeline A.

Uploads return immediately with a job id; a background worker processes the file. Nothing in
the web page can cancel it. Job state is saved to disk (data/jobs), so after a restart
unfinished jobs are picked up again, and thanks to the Read and figure caches they continue
almost where they stopped. Failed steps are retried 3 times with growing waits; after that the
job is marked "failed" (the failed list) for someone to check.

Prototype: one worker thread inside the API process. Production (profiles A/B): the same
`run_job` function runs in a Container Apps job fed by a Storage Queue.
"""
import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from rag.config import settings
from rag.ingest.pipeline import doc_id_for, ingest_file
from rag.logs import event

MAX_ATTEMPTS = 3
_q: "queue.Queue[str]" = queue.Queue()
_started = False
_lock = threading.Lock()
_running: set[str] = set()      # a job never runs twice at the same time


def _job_path(job_id: str) -> Path:
    return settings.jobs_dir / f"{job_id}.json"


def _upload_path(job_id: str) -> Path:
    return settings.jobs_dir / f"{job_id}.upload"


def get_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(job: dict):
    job["updated"] = time.time()
    tmp = _job_path(job["id"]).with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(job), encoding="utf-8")
    tmp.replace(_job_path(job["id"]))


def list_jobs(tenant: str) -> list[dict]:
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in settings.jobs_dir.glob("*.json")
            if not p.name.endswith(".tmp")]
    return sorted([j for j in jobs if j["tenant"] == tenant], key=lambda j: -j["created"])


def submit(data: bytes, file_name: str, tenant: str, lateral: str, acl: list[str], title: str | None,
           user: str) -> dict:
    job_id = uuid.uuid4().hex[:12]
    _upload_path(job_id).write_bytes(data)
    job = {"id": job_id, "doc_id": doc_id_for(tenant, file_name), "file_name": file_name,
           "tenant": tenant, "lateral": lateral, "acl": acl, "title": title, "user": user,
           "status": "queued", "step": "waiting", "attempts": 0, "error": None,
           "created": time.time()}
    _save(job)
    event("job.queued", job_id=job_id, file_name=file_name, tenant=tenant, lateral=lateral, user=user)
    _q.put(job_id)
    return job


def run_job(job_id: str):
    with _lock:
        if job_id in _running:
            return
        _running.add(job_id)
    try:
        _run_job(job_id)
    finally:
        with _lock:
            _running.discard(job_id)


def _run_job(job_id: str):
    job = get_job(job_id)
    if not job or job["status"] in ("done", "skipped", "failed"):
        return
    while job["attempts"] < MAX_ATTEMPTS:
        job["attempts"] += 1
        job["status"], job["error"] = "running", None
        _save(job)

        def progress(step, job=job):
            job["step"] = step
            _save(job)
            event("job.step", job_id=job["id"], step=step.split(" (")[0])

        try:
            res = ingest_file(_upload_path(job_id), job["file_name"], job["tenant"], job["lateral"],
                              job["acl"], job["title"], progress=progress)
            job.update(status="done" if res["status"] == "ready" else "skipped",
                       step="finished", chunks=res.get("chunks", 0))
            _upload_path(job_id).unlink(missing_ok=True)
            event("job.finished", job_id=job_id, status=job["status"], chunks=job.get("chunks"))
            _save(job)   # saved last: "done" means everything, including the log line, is finished
            return
        except Exception as e:  # keep the type only: messages can contain document text
            job["error"] = type(e).__name__
            _save(job)
            event("job.error", job_id=job_id, error_type=type(e).__name__, attempt=job["attempts"])
            traceback.print_exc()
            time.sleep(min(60, 5 * 2 ** job["attempts"]))
    job.update(status="failed", step="gave up after 3 tries")
    _save(job)
    event("job.failed", job_id=job_id, file_name=job["file_name"])


def _loop():
    while True:
        job_id = _q.get()
        try:
            run_job(job_id)
        finally:
            _q.task_done()


def start():
    """Start the worker and re-queue jobs that were unfinished when the app stopped (resume)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    for p in sorted(settings.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        job = json.loads(p.read_text(encoding="utf-8"))
        if job["status"] in ("queued", "running"):
            event("job.resumed", job_id=job["id"], file_name=job["file_name"])
            _q.put(job["id"])
    threading.Thread(target=_loop, daemon=True, name="ingest-worker").start()
