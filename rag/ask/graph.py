r"""Pipeline B - ask every time (LangGraph, fixed order):

  Gate 1 Scope -> Retrieve + ACL filter -> Gate 2 Match -> Generate cited answer -> Gate 3 Fact check
       \-> refuse          (tenant, lateral, groups)  \-> not found                 \-> not found

Response is always one of: cited answer | refuse | not found.
"""
import base64
import re
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag.config import settings
from rag.logs import event
from rag.providers.index import index
from rag.providers.llm import chat_model, embeddings, structured
from rag.providers.rerank import rerank
from rag.providers.storage import catalog_visible, storage

REFUSAL = ("I can only answer questions about the documents available to you. "
           "Please ask about their content.")
NOT_FOUND = "I couldn't find this in the documents available to you."


class Source(TypedDict):
    n: int
    doc_id: str
    doc_title: str
    file_name: str
    section: str
    pages: str
    content: str
    score: float
    figure_keys: list[str]


class AskState(TypedDict, total=False):
    question: str
    history: list[dict]
    user: str
    tenant: str
    groups: list[str]
    laterals: list[str]
    doc_ids: list[str]
    search_query: str
    in_scope: bool
    sources: list[Source]
    best_score: float
    answer: str
    cited: list[int]
    status: Literal["answered", "out_of_scope", "not_found", "not_grounded"]


# ---------------- Gate 1: scope ----------------
class ScopeDecision(BaseModel):
    in_scope: bool = Field(description="False only if the question clearly cannot be about the documents")
    search_query: str = Field(description="The question rewritten as a standalone search query, "
                                          "keeping every code, number and name exactly")


SCOPE_PROMPT = """You are the gatekeeper for a company document assistant.
Below is a catalogue of documents the user can search (title: summary).
Mark the question OUT of scope only if it clearly cannot be answered by documents like these:
chit-chat, general knowledge, news, weather, coding help, maths homework, personal advice.
If it could plausibly be about any document's subject, it is IN scope. When unsure, choose IN scope:
a later step checks whether the documents really contain the answer.
If earlier turns are given, resolve words like "it" or "that one" into a standalone query."""


def gate_scope(state: AskState) -> AskState:
    docs = catalog_visible(state["tenant"], state["groups"], state.get("laterals"))
    if state.get("doc_ids"):
        docs = [d for d in docs if d["doc_id"] in state["doc_ids"]]
    if not docs:
        return {"in_scope": False, "search_query": state["question"]}
    catalog = "\n".join(f"- {d['doc_title']}: {d['summary']}" for d in docs[:40])
    history = "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", [])[-6:])
    d: ScopeDecision = structured(chat_model(small=True), ScopeDecision).invoke([
        SystemMessage(SCOPE_PROMPT + "\n\nCatalogue:\n" + catalog),
        HumanMessage(f"Earlier turns:\n{history or '(none)'}\n\nQuestion: {state['question']}"),
    ])
    return {"in_scope": d.in_scope, "search_query": d.search_query or state["question"]}


# ---------------- Retrieve + ACL filter ----------------
def retrieve(state: AskState) -> AskState:
    q = state["search_query"]
    idx = index()
    found = idx.search(q, embeddings().embed_query(q), state["tenant"], state["groups"],
                       state.get("laterals"), state.get("doc_ids"), k=settings.candidates)
    if not idx.native_rerank:
        found = rerank(q, found, settings.top_k)
    found = found[: settings.top_k]
    sources = [Source(
        n=i, doc_id=c["doc_id"], doc_title=c["doc_title"], file_name=c.get("file_name", ""),
        section=c["section"],
        pages=f"{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else ""),
        content=c["content"], score=round(float(c.get("score", 0.0)), 4),
        figure_keys=c.get("figure_keys") or [],
    ) for i, c in enumerate(found, start=1)]
    return {"sources": sources, "best_score": max((s["score"] for s in sources), default=0.0)}


# ---------------- Gate 2: match ----------------
def gate_match(state: AskState) -> str:
    return "generate" if state.get("best_score", 0.0) >= settings.min_relevance else "not_found"


# ---------------- Generate cited answer ----------------
class GroundedAnswer(BaseModel):
    found: bool = Field(description="False if the sources do not contain the answer")
    answer: str = Field(description="The answer, with [n] citations after each fact")
    cited: list[int] = Field(description="Source numbers actually used")


ANSWER_PROMPT = """You answer questions using ONLY the numbered sources below (company documents).
Rules:
1. Use only facts stated in the sources. Never use outside knowledge, even if you know the answer.
2. Put the source number after every fact, like [2]. Cite tables and figures the same way.
3. Copy numbers, part IDs, standards, grades and units exactly as written.
4. If the sources only partly answer, answer that part and say what is missing.
5. If the sources do not answer the question, set found=false and leave answer empty.
6. When a list is asked for, list every matching item from the sources, as a table if there are
   several columns.
7. Be concise. No introductions."""


def _figure_parts(sources: list[Source], limit: int = 2) -> list[dict]:
    parts = []
    for s in sources:
        for key in s["figure_keys"]:
            if len(parts) >= 2 * limit:
                return parts
            png = storage().get(key)
            if png:
                parts += [{"type": "text", "text": f"Image from source [{s['n']}]:"},
                          {"type": "image_url",
                           "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
    return parts


def _source_block(s: Source) -> str:
    return f"[{s['n']}] {s['doc_title']} | {s['section']} | page {s['pages']}\n{s['content']}"


def generate(state: AskState) -> AskState:
    context = "\n\n".join(_source_block(s) for s in state["sources"])
    content = [{"type": "text", "text": f"Sources:\n{context}\n\nQuestion: {state['question']}"}]
    content += _figure_parts(state["sources"])
    a: GroundedAnswer = structured(chat_model(), GroundedAnswer).invoke(
        [SystemMessage(ANSWER_PROMPT), HumanMessage(content=content)])
    if not a.found or not a.answer.strip():
        return {"status": "not_found", "answer": NOT_FOUND, "cited": []}
    cited = [n for n in a.cited if any(s["n"] == n for s in state["sources"])]
    return {"answer": a.answer, "cited": cited or [s["n"] for s in state["sources"]]}


# ---------------- Gate 3: fact check ----------------
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def numbers_supported(answer: str, sources_text: str) -> bool:
    """Instant check: every number in the answer must appear somewhere in the cited sources."""
    have = {n.replace(",", ".") for n in _NUM.findall(sources_text)}
    need = {n.replace(",", ".") for n in _NUM.findall(_CITE.sub(" ", answer))}
    return need <= have


class Verdict(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is supported by the sources")


VERIFY_PROMPT = """You are a strict fact checker. Compare the ANSWER with the SOURCES.
grounded=true only if every fact, number and part ID in the answer appears in the sources.
Ignore wording and formatting differences."""


def gate_fact_check(state: AskState) -> AskState:
    if state.get("status") == "not_found":
        return {}
    used = [s for s in state["sources"] if s["n"] in state.get("cited", [])] or state["sources"]
    text = "\n\n".join(_source_block(s) for s in used)
    if not numbers_supported(state["answer"], text):
        return {"status": "not_grounded", "answer": NOT_FOUND, "cited": []}
    v: Verdict = structured(chat_model(small=True), Verdict).invoke([
        SystemMessage(VERIFY_PROMPT), HumanMessage(f"SOURCES:\n{text}\n\nANSWER:\n{state['answer']}")])
    if not v.grounded:
        return {"status": "not_grounded", "answer": NOT_FOUND, "cited": []}
    return {"status": "answered"}


def refuse(state: AskState) -> AskState:
    return {"status": "out_of_scope", "answer": REFUSAL, "sources": [], "cited": []}


def not_found(state: AskState) -> AskState:
    return {"status": "not_found", "answer": NOT_FOUND, "cited": []}


def build_graph():
    g = StateGraph(AskState)
    for name, fn in [("gate_scope", gate_scope), ("refuse", refuse), ("retrieve", retrieve),
                     ("not_found", not_found), ("generate", generate), ("gate_fact_check", gate_fact_check)]:
        g.add_node(name, fn)
    g.add_edge(START, "gate_scope")
    g.add_conditional_edges("gate_scope", lambda s: "retrieve" if s["in_scope"] else "refuse",
                            ["retrieve", "refuse"])
    g.add_conditional_edges("retrieve", gate_match, ["generate", "not_found"])
    g.add_edge("generate", "gate_fact_check")
    for end in ("gate_fact_check", "refuse", "not_found"):
        g.add_edge(end, END)
    return g.compile()


ask_graph = build_graph()


def ask(question: str, user, laterals=None, doc_ids=None, history=None) -> dict:
    out = ask_graph.invoke({"question": question, "history": history or [], "user": user.id,
                            "tenant": user.tenant, "groups": user.acl_groups,
                            "laterals": laterals or [], "doc_ids": doc_ids or []})
    event("ask.done", user=user.id, tenant=user.tenant, outcome=out["status"],
          best_score=round(out.get("best_score", 0.0), 3), sources=len(out.get("cited", [])))
    return out
