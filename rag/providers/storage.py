"""Object storage for original files, figure images and the document catalog.
  local       data/files on disk (profile C)
  azure_blob  Azure Blob Storage container (profiles A/B)
Keys look like paths: "files/<tenant>/<doc_id>/<name>", "figures/<doc_id>/<id>.png", "catalog.json".
"""
import json
import threading
from functools import lru_cache
from pathlib import Path

from rag.config import settings


class LocalStorage:
    def __init__(self):
        self.root = settings.data_dir / "store"
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("bad key")
        return p

    def put(self, key: str, data: bytes):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)   # atomic: never a half-written file

    def get(self, key: str) -> bytes | None:
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def delete_prefix(self, prefix: str):
        import shutil
        p = self._p(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


class BlobStorage:
    def __init__(self):
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
        svc = BlobServiceClient(settings.blob_account_url, DefaultAzureCredential())
        self.c = svc.get_container_client(settings.blob_container)
        if not self.c.exists():
            self.c.create_container()

    def put(self, key: str, data: bytes):
        self.c.upload_blob(key, data, overwrite=True)

    def get(self, key: str) -> bytes | None:
        b = self.c.get_blob_client(key)
        return b.download_blob().readall() if b.exists() else None

    def delete_prefix(self, prefix: str):
        for b in self.c.list_blobs(name_starts_with=prefix):
            self.c.delete_blob(b.name)


@lru_cache
def storage():
    if settings.storage == "local":
        return LocalStorage()
    if settings.storage == "azure_blob":
        return BlobStorage()
    raise ValueError(f"Unknown STORAGE {settings.storage}")


# ---------------- document catalog (one entry per document) ----------------
_cat_lock = threading.Lock()


def catalog_all() -> dict[str, dict]:
    raw = storage().get("catalog.json")
    return json.loads(raw) if raw else {}


def catalog_put(doc_id: str, entry: dict | None):
    """Add/replace (entry) or remove (None) one document."""
    with _cat_lock:
        cat = catalog_all()
        if entry is None:
            cat.pop(doc_id, None)
        else:
            cat[doc_id] = entry
        storage().put("catalog.json", json.dumps(cat, ensure_ascii=False, indent=1).encode())


def catalog_visible(tenant: str, groups: list[str], laterals: list[str] | None = None) -> list[dict]:
    out = []
    for e in catalog_all().values():
        if e.get("tenant") != tenant or not set(e.get("acl", [])) & set(groups):
            continue
        if laterals and e.get("lateral") not in laterals:
            continue
        out.append(e)
    return sorted(out, key=lambda e: e["doc_title"].lower())
