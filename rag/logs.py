"""Content-free logging.

Rule (from the design): operators see process logs and file names, never document text,
questions or answers. So we only ever log events with ids, file names, counts, timings and
scores. Use `event()` everywhere; never log free text from documents or users.
"""
import json
import logging
import time

from rag.config import settings

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("rag")

ALLOWED_FIELDS = {
    "job_id", "doc_id", "file_name", "tenant", "lateral", "status", "step", "seconds", "pages",
    "chunks", "figures", "described", "reused", "skipped", "error_type", "outcome", "best_score",
    "sources", "user", "profile", "attempt", "count",
}


def event(name: str, **fields) -> None:
    """Log one structured event. Fields not on the allow-list are dropped, so document text,
    questions and answers can never end up in the logs by accident."""
    safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS}
    _log.info("%s %s", name, json.dumps(safe, default=str))


class timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t0, 2)
