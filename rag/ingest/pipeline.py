"""Pipeline A - ingest once per document:
  Land + tag -> Convert -> Read (OCR/layout) -> Describe figures -> Chunk -> Hybrid index
  + Doc catalog + Original images.

Reliability:
  * unchanged file with the same tags -> skipped (no work, no cost)
  * the Read result is cached by file hash -> a retry or resume never reads twice
  * figure descriptions are cached per image -> resumable, no duplicate work
  * the old version of a document is replaced only after the new one is fully ready
"""
import hashlib
import json
import tempfile
import time
from pathlib import Path

from rag.config import settings
from rag.ingest.chunk import chunk_document
from rag.ingest.convert import to_pdf
from rag.ingest.figures import describe_all, too_small
from rag.ingest.summary import summarize
from rag.logs import event, timer
from rag.providers.index import index
from rag.providers.llm import embeddings
from rag.providers.reader import Figure, ParsedDoc, read_pdf
from rag.providers.storage import catalog_all, catalog_put, storage


def doc_id_for(tenant: str, file_name: str) -> str:
    """Same tenant + same file name = same document (a new upload replaces the old version)."""
    return hashlib.sha1(f"{tenant}/{file_name.lower()}".encode()).hexdigest()[:16]


# ---------------- Read cache ----------------
def _cache_dir(file_hash: str) -> Path:
    return settings.cache_dir / "parsed" / f"{file_hash}_{settings.reader}"


def _load_parsed(file_hash: str) -> ParsedDoc | None:
    d = _cache_dir(file_hash)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    figs = [Figure(f["id"], f["page"], f["caption"], (d / f"{f['id']}.png").read_bytes()) for f in meta["figures"]]
    return ParsedDoc((d / "doc.md").read_text(encoding="utf-8"), figs, meta.get("pages", 0))


def _save_parsed(file_hash: str, parsed: ParsedDoc):
    d = _cache_dir(file_hash)
    d.mkdir(parents=True, exist_ok=True)
    (d / "doc.md").write_text(parsed.markdown, encoding="utf-8")
    for f in parsed.figures:
        (d / f"{f.id}.png").write_bytes(f.image_png)
    (d / "meta.json").write_text(json.dumps({"pages": parsed.pages, "figures": [
        {"id": f.id, "page": f.page, "caption": f.caption} for f in parsed.figures]}), encoding="utf-8")


# ---------------- the pipeline ----------------
def ingest_file(path: Path, file_name: str, tenant: str, lateral: str, acl: list[str],
                title: str | None = None, progress=lambda step: None) -> dict:
    data = path.read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()
    doc_id = doc_id_for(tenant, file_name)
    acl = sorted(set(acl or [settings.everyone_group]))
    title = title or Path(file_name).stem
    tags = {"tenant": tenant, "lateral": lateral, "acl": acl}

    # 1. Land + tag (skip if nothing changed)
    old = catalog_all().get(doc_id)
    if old and old.get("file_hash") == file_hash and all(old.get(k) == v for k, v in tags.items()) \
            and old.get("status") == "ready":
        event("ingest.skipped_unchanged", doc_id=doc_id, file_name=file_name, tenant=tenant)
        return {"doc_id": doc_id, "status": "skipped", "chunks": old.get("chunks", 0)}
    progress("land")
    storage().put(f"files/{tenant}/{doc_id}/{file_name}", data)

    # 2-3. Convert + Read (cached)
    parsed = _load_parsed(file_hash)
    if parsed is None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / file_name
            src.write_bytes(data)
            progress("convert")
            pdf = to_pdf(src, Path(tmp))
            progress("read")
            with timer() as t:
                parsed = read_pdf(pdf)
            event("ingest.read", doc_id=doc_id, file_name=file_name, pages=parsed.pages,
                  figures=len(parsed.figures), seconds=t.seconds)
        _save_parsed(file_hash, parsed)

    # 4. Describe figures (skip tiny ones, reuse cache, keep originals for citations)
    md = parsed.markdown
    keep = [f for f in parsed.figures if not too_small(f.image_png)]
    for f in parsed.figures:
        if f not in keep:
            md = md.replace(f"[[FIGURE:{f.id}]]", "")
    texts, described, reused = {}, 0, 0
    if settings.describe_images and keep:
        progress(f"describe figures (0/{len(keep)})")
        items = []
        for f in keep:
            before = md.split(f"[[FIGURE:{f.id}]]")[0]
            heading = next((l for l in reversed(before.splitlines()) if l.startswith("#")), "")
            items.append((f.id, f.image_png, heading.lstrip("# "), f.caption))
        with timer() as t:
            texts, described, reused = describe_all(
                items, progress=lambda d, n: progress(f"describe figures ({d}/{n} new)"))
        event("ingest.figures", doc_id=doc_id, figures=len(keep), described=described,
              reused=reused, skipped=len(parsed.figures) - len(keep), seconds=t.seconds)
    figure_keys = {}
    for f in keep:
        key = f"figures/{doc_id}/{f.id}.png"
        storage().put(key, f.image_png)
        figure_keys[f.id] = key
        body = texts.get(f.id, "")
        md = md.replace(f"[[FIGURE:{f.id}]]",
                        f'<figure id="{f.id}">\nFigure (page {f.page}) {f.caption}\n{body}\n</figure>')

    # 5. Chunk
    progress("chunk")
    chunks = chunk_document(md, title, settings.chunk_target_tokens, settings.chunk_max_tokens,
                            settings.chunk_overlap_tokens)
    if not chunks:
        raise ValueError("No text found in the document")

    # 6. Hybrid index (replaces the old version of this document)
    progress(f"index {len(chunks)} pieces")
    with timer() as t:
        vectors = embeddings().embed_documents([c.text for c in chunks])
        records = [{
            "id": f"{doc_id}-{i:05d}", "doc_id": doc_id, "doc_title": title, "file_name": file_name,
            **tags, "section": c.section, "content": c.text,
            "page_start": c.page_start, "page_end": c.page_end,
            "figure_keys": [figure_keys[x] for x in c.figure_ids if x in figure_keys],
        } for i, c in enumerate(chunks)]
        index().upsert(doc_id, records, vectors)
    event("ingest.indexed", doc_id=doc_id, chunks=len(records), seconds=t.seconds)

    # 7. Doc catalog
    progress("catalog")
    summary = summarize(title, [c.section for c in chunks], "\n\n".join(c.content for c in chunks[:3]))
    catalog_put(doc_id, {"doc_id": doc_id, "doc_title": title, "file_name": file_name, **tags,
                         "summary": summary, "file_hash": file_hash, "status": "ready",
                         "chunks": len(records), "figures": len(keep), "pages": parsed.pages,
                         "updated": time.time()})
    return {"doc_id": doc_id, "status": "ready", "chunks": len(records)}


def delete_document(doc_id: str) -> bool:
    entry = catalog_all().get(doc_id)
    if not entry:
        return False
    index().delete(doc_id)
    storage().delete_prefix(f"files/{entry['tenant']}/{doc_id}")
    storage().delete_prefix(f"figures/{doc_id}")
    catalog_put(doc_id, None)
    event("ingest.deleted", doc_id=doc_id, file_name=entry.get("file_name"), tenant=entry.get("tenant"))
    return True
