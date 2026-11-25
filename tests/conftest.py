"""Test setup: profile C with stand-in models, so tests run anywhere in seconds, offline.
Only the AI models, the reader and the reranker are replaced; Qdrant, the chunker, the
pipelines, the worker, the API, the gates and the access filters are the real code."""
import hashlib
import io
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rag-test-"))
os.environ.update(PROFILE="C", DATA_DIR=str(_TMP), AUTH_MODE="demo",
                  DEMO_USERS_FILE=str(Path(__file__).resolve().parents[1] / "deploy" / "demo_users.json"))

from PIL import Image  # noqa: E402

SAMPLE_MD = """<!-- PageNumber=1 -->

# Sample Fastener Catalogue

## 1 General Rules

All fasteners M6 and smaller shall be made of stainless steel A2-70.

All fasteners M8 and larger shall be steel grade 8.8 with zinc flake coating.

<!-- PageNumber=2 -->

## 2 Bolts

ISO 4014 - Hexagon head bolts - A2-70

[[FIGURE:1]]

<table><tr><th>Part number</th><th>Thread</th><th>Length</th></tr><tr><td>900001</td><td>M6</td><td>60</td></tr><tr><td>900002</td><td>M6</td><td>70</td></tr></table>

<!-- PageNumber=3 -->

## 3 Washers

ISO 7089 - Plain washers - A2

<table><tr><th>Part number</th><th>Size</th><th>Outside diameter</th></tr><tr><td>900101</td><td>6</td><td>12</td></tr></table>
"""


def _png(size=(400, 300)):
    b = io.BytesIO()
    Image.new("RGB", size, "white").save(b, "PNG")
    return b.getvalue()


class Calls:
    read = 0
    described = 0


def _tok(t):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t.lower())


class FakeEmb:
    def _v(self, t):
        v = np.zeros(64)
        for w in _tok(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return (v + 1e-3).tolist()

    def embed_documents(self, ts):
        return [self._v(t) for t in ts]

    def embed_query(self, t):
        return self._v(t)


def fake_rerank(q, chunks, k):
    qt = set(_tok(q)) - {"which", "are", "for", "in", "the", "what", "is", "of", "listed"}
    for c in chunks:
        c["score"] = len(qt & set(_tok(c["content"]))) / max(1, len(qt))
    return sorted(chunks, key=lambda c: -c["score"])[:k]


class FakeLLM:
    def __init__(self, schema=None):
        self.schema = schema

    def invoke(self, msgs):
        c = msgs[-1].content
        text = c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
        name = self.schema.__name__ if self.schema else None
        if name is None:
            class R:
                content = "Catalogue of fasteners by ISO standard with part numbers and rules."
            return R()
        if name == "ScopeDecision":
            q = text.split("Question: ")[-1]
            return self.schema(in_scope=not re.search(r"weather|python", q, re.I), search_query=q)
        if name == "GroundedAnswer":
            q = text.split("Question:")[-1].lower()
            if "invent" in q:   # used to test Gate 3: answer with a number not in the sources
                return self.schema(found=True, answer="Use part 777777 [1]", cited=[1])
            for block in re.split(r"\n(?=\[\d+\] )", text.split("Question:")[0]):
                m = re.match(r"\[(\d+)\]", block.replace("Sources:\n", "").strip())
                hit = re.search(r"9001\d\d" if "washer" in q else r"9000\d\d", block)
                if m and hit:
                    return self.schema(found=True, answer=f"Part {hit.group(0)} [{m.group(1)}]", cited=[int(m.group(1))])
            return self.schema(found=False, answer="", cited=[])
        if name == "Verdict":
            src, ans = text.split("ANSWER:")
            return self.schema(grounded=all(n in src for n in re.findall(r"\d{6}", ans)))


@pytest.fixture(autouse=True, scope="session")
def fakes():
    import rag.ask.graph as G
    import rag.ingest.figures as FG
    import rag.ingest.pipeline as P
    import rag.ingest.summary as S
    from rag.providers.reader import Figure, ParsedDoc

    def fake_read(pdf):
        Calls.read += 1
        return ParsedDoc(SAMPLE_MD, [Figure("1", 2, "Bolt drawing", _png()), Figure("2", 2, "icon", _png((40, 40)))], 3)

    def fake_describe(png, section, caption):
        Calls.described += 1
        return "TYPE: technical-drawing\nHexagon head bolt; k = head height, s = width across flats."

    emb = FakeEmb()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(P, "read_pdf", fake_read)
    monkeypatch.setattr(P, "embeddings", lambda: emb)
    monkeypatch.setattr(G, "embeddings", lambda: emb)
    monkeypatch.setattr(G, "rerank", fake_rerank)
    monkeypatch.setattr(G, "chat_model", lambda small=False: FakeLLM())
    monkeypatch.setattr(G, "structured", lambda llm, schema: FakeLLM(schema))
    monkeypatch.setattr(S, "chat_model", lambda small=False: FakeLLM())
    monkeypatch.setattr(FG, "_describe_one", fake_describe)
    monkeypatch.setattr("rag.ingest.worker.time.sleep", lambda s: None)
    yield
    monkeypatch.undo()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
