"""Describe figures: turn images and flowcharts into searchable text.

Cost and time savers (no quality loss):
  * tiny images (icons, logos) are skipped
  * identical images are described once (cache keyed by the image's SHA-256), also across
    documents and across restarts, so a stopped run resumes without redoing work
  * images are shrunk to VISION_MAX_PX before sending (the full-size image is kept for display)
  * several images are described in parallel (VISION_CONCURRENCY)
"""
import base64
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from rag.config import settings
from rag.providers.llm import vision_model

FIGURE_PROMPT = """You convert one figure from a technical document into text for a search index.
First line: TYPE: flowchart | chart | table-image | technical-drawing | photo | other
Then:
- flowchart: list every step and decision in order, as numbered steps. Show branches as
  "If <condition> -> step N".
- chart: give the data as a Markdown table, then one sentence on what it shows.
- table-image: rewrite it as a Markdown table.
- technical-drawing: name the part, list every labelled dimension letter and what it measures.
- photo/other: describe what is shown in 2-3 sentences.
Use only what is visible. Copy all numbers, codes and labels exactly. Never guess."""

_CACHE_FILE = settings.cache_dir / "figure_descriptions.json"
_lock = threading.Lock()


def _load_cache() -> dict:
    return json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if _CACHE_FILE.exists() else {}


def _save_cache(cache: dict):
    tmp = _CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_CACHE_FILE)


def too_small(png: bytes) -> bool:
    return min(Image.open(io.BytesIO(png)).size) < settings.min_figure_px


def _shrink(png: bytes) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img.thumbnail((settings.vision_max_px, settings.vision_max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _describe_one(png: bytes, section: str, caption: str) -> str:
    url = "data:image/png;base64," + base64.b64encode(_shrink(png)).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": f"Section: {section or 'unknown'}\nCaption: {caption or 'none'}"},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    return vision_model().invoke([SystemMessage(FIGURE_PROMPT), msg]).content.strip()


def describe_all(items: list[tuple[str, bytes, str, str]], progress=lambda done, total: None
                 ) -> tuple[dict[str, str], int, int]:
    """items: (figure_id, png, section, caption). Returns ({figure_id: text}, described, reused)."""
    cache = _load_cache()
    out, todo = {}, {}
    for fid, png, section, caption in items:
        h = hashlib.sha256(png).hexdigest()
        if h in cache:
            out[fid] = cache[h]
        else:
            todo.setdefault(h, []).append((fid, png, section, caption))
    reused = len(out)
    done = 0

    def work(h):
        fid, png, section, caption = todo[h][0]
        return h, _describe_one(png, section, caption)

    with ThreadPoolExecutor(max_workers=max(1, settings.vision_concurrency)) as pool:
        for h, text in pool.map(work, list(todo)):
            with _lock:
                cache[h] = text
                _save_cache(cache)           # saved after each image -> resumable
            for fid, *_ in todo[h]:
                out[fid] = text
            done += 1
            progress(done, len(todo))
    return out, len(todo), reused
