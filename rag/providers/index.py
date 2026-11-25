"""Hybrid index: keyword text + vector + tags (tenant, lateral, ACL) for every chunk.

  qdrant        profile C (also usable in A/B). Dense vectors + BM25-style sparse vectors,
                merged with Reciprocal Rank Fusion. Our own reranker gives the 0-1 match score.
  azure_search  profiles A/B. Hybrid search + Azure semantic ranker (0-4, rescaled to 0-1).

Separation rule: every search is filtered by the user's tenant AND their groups, and optionally
by lateral and document. If a chunk shouldn't be seen, it is never retrieved.
"""
import hashlib
import re
import threading
import uuid
from collections import Counter
from functools import lru_cache

from rag.config import settings

# ---------------- keyword side ----------------
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_TAG = re.compile(r"<[^>]+>")
_STOP = set("a an and are as at be by can do does for from has have how i in is it its may of on or "
            "shall should that the their there these this to was were what when where which who why "
            "will with you your".split())
K1, B, AVG_TOKENS = 1.2, 0.75, 450


def tokenize(text: str) -> list[str]:
    """Keeps codes together ('A2-70', 'M8x40') and also indexes their parts."""
    toks = [t for t in _TOKEN.findall(_TAG.sub(" ", text).lower().replace(":", " ")) if t not in _STOP]
    return toks + [p for t in toks if re.search(r"[-./]", t) for p in re.split(r"[-./]", t) if p]


def _index_tokens(content: str) -> list[str]:
    head = "\n".join(content.split("\n")[:3])   # breadcrumb + title lines count double
    return tokenize(content) + tokenize(head)


def _sparse(tokens: list[str], query: bool = False):
    from qdrant_client import models
    counts: Counter = Counter()
    norm = K1 * (1 - B + B * len(tokens) / AVG_TOKENS)
    for tok, tf in Counter(tokens).items():
        idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        counts[idx] += 1.0 if query else tf * (K1 + 1) / (tf + norm)
    return models.SparseVector(indices=list(counts), values=list(counts.values()))


# ---------------- Qdrant ----------------
class QdrantIndex:
    native_rerank = False
    DENSE, SPARSE = "dense", "bm25"

    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = (QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
                       if settings.qdrant_url else QdrantClient(path=str(settings.data_dir / "qdrant")))
        self.name = settings.index_name
        self._lock = threading.Lock()

    def _ensure(self, dim: int):
        from qdrant_client import models
        if self.client.collection_exists(self.name):
            return
        self.client.create_collection(
            self.name,
            vectors_config={self.DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={self.SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
        if settings.qdrant_url:
            for f in ("doc_id", "tenant", "lateral", "acl"):
                self.client.create_payload_index(self.name, f, models.PayloadSchemaType.KEYWORD)

    def upsert(self, doc_id: str, records: list[dict], vectors: list[list[float]]):
        from qdrant_client import models
        with self._lock:
            self._ensure(len(vectors[0]))
            self.delete(doc_id)
            pts = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["id"])),
                                      vector={self.DENSE: v, self.SPARSE: _sparse(_index_tokens(r["content"]))},
                                      payload=r) for r, v in zip(records, vectors)]
            for i in range(0, len(pts), 256):
                self.client.upsert(self.name, pts[i:i + 256])

    def delete(self, doc_id: str):
        from qdrant_client import models
        if self.client.collection_exists(self.name):
            self.client.delete(self.name, models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))

    def count(self) -> int:
        return self.client.count(self.name).count if self.client.collection_exists(self.name) else 0

    def search(self, query: str, vector: list[float], tenant: str, groups: list[str],
               laterals: list[str] | None = None, doc_ids: list[str] | None = None, k: int = 30) -> list[dict]:
        from qdrant_client import models
        if not self.client.collection_exists(self.name):
            return []
        must = [models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant)),
                models.FieldCondition(key="acl", match=models.MatchAny(any=groups))]
        if laterals:
            must.append(models.FieldCondition(key="lateral", match=models.MatchAny(any=laterals)))
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
        flt = models.Filter(must=must)
        res = self.client.query_points(
            self.name,
            prefetch=[models.Prefetch(query=vector, using=self.DENSE, limit=k * 2, filter=flt),
                      models.Prefetch(query=_sparse(tokenize(query), query=True), using=self.SPARSE,
                                      limit=k * 2, filter=flt)],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True)
        return [dict(p.payload) for p in res.points]


# ---------------- Azure AI Search ----------------
class AzureSearchIndex:
    native_rerank = True   # semantic ranker gives @search.reranker_score (0-4)

    def __init__(self):
        from azure.identity import DefaultAzureCredential
        from azure.search.documents import SearchClient
        self.cred = DefaultAzureCredential()
        self.client = SearchClient(settings.search_endpoint, settings.index_name, self.cred)
        self._ready = False

    def _ensure(self):
        if self._ready:
            return
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType, SearchIndex,
            SemanticConfiguration, SemanticField, SemanticPrioritizedFields, SemanticSearch, SimpleField,
            VectorSearch, VectorSearchProfile)
        S, C = SearchFieldDataType.String, SearchFieldDataType.Collection
        fields = [
            SimpleField(name="id", type=S, key=True),
            SimpleField(name="doc_id", type=S, filterable=True),
            SimpleField(name="tenant", type=S, filterable=True),
            SimpleField(name="lateral", type=S, filterable=True),
            SimpleField(name="acl", type=C(S), filterable=True),
            SearchableField(name="doc_title", type=S),
            SimpleField(name="file_name", type=S),
            SearchableField(name="section", type=S),
            SearchableField(name="content", type=S, analyzer_name="en.microsoft"),
            SimpleField(name="page_start", type=SearchFieldDataType.Int32),
            SimpleField(name="page_end", type=SearchFieldDataType.Int32),
            SimpleField(name="figure_keys", type=C(S)),
            SearchField(name="vector", type=C(SearchFieldDataType.Single), searchable=True,
                        vector_search_dimensions=settings.embed_dimensions, vector_search_profile_name="vec"),
        ]
        SearchIndexClient(settings.search_endpoint, self.cred).create_or_update_index(SearchIndex(
            name=settings.index_name, fields=fields,
            vector_search=VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                                       profiles=[VectorSearchProfile(name="vec", algorithm_configuration_name="hnsw")]),
            semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
                name="default", prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="section"),
                    content_fields=[SemanticField(field_name="content")]))])))
        self._ready = True

    def upsert(self, doc_id: str, records: list[dict], vectors: list[list[float]]):
        self._ensure()
        self.delete(doc_id)
        docs = [{**r, "vector": v} for r, v in zip(records, vectors)]
        for i in range(0, len(docs), 500):
            self.client.upload_documents(docs[i:i + 500])

    def delete(self, doc_id: str):
        self._ensure()
        old = [{"id": r["id"]} for r in self.client.search("*", filter=f"doc_id eq '{_q(doc_id)}'", select=["id"])]
        for i in range(0, len(old), 1000):
            self.client.delete_documents(old[i:i + 1000])

    def count(self) -> int:
        self._ensure()
        return self.client.get_document_count()

    def search(self, query, vector, tenant, groups, laterals=None, doc_ids=None, k=30):
        from azure.search.documents.models import VectorizedQuery
        self._ensure()
        flt = [f"tenant eq '{_q(tenant)}'",
               f"acl/any(g: search.in(g, '{','.join(_q(g) for g in groups)}', ','))"]
        if laterals:
            flt.append(f"search.in(lateral, '{','.join(_q(x) for x in laterals)}', ',')")
        if doc_ids:
            flt.append(f"search.in(doc_id, '{','.join(_q(x) for x in doc_ids)}', ',')")
        results = self.client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k * 2, fields="vector")],
            filter=" and ".join(flt), query_type="semantic", semantic_configuration_name="default",
            top=settings.top_k)
        out = []
        for r in results:
            d = {k2: r.get(k2) for k2 in ("id", "doc_id", "doc_title", "file_name", "tenant", "lateral",
                                           "section", "content", "page_start", "page_end", "figure_keys")}
            d["score"] = (r.get("@search.reranker_score") or 0.0) / 4.0   # rescale 0-4 -> 0-1
            out.append(d)
        return out


def _q(v: str) -> str:
    """Escape a value for an OData filter (and strip separators used by search.in)."""
    return str(v).replace("'", "''").replace(",", "")


@lru_cache
def index():
    if settings.index == "qdrant":
        return QdrantIndex()
    if settings.index == "azure_search":
        return AzureSearchIndex()
    raise ValueError(f"Unknown INDEX {settings.index}")
