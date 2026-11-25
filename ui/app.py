"""Prototype web page. Talks ONLY to the RAG API (never to models or the index).
Run:  streamlit run ui/app.py      (API_URL defaults to http://localhost:8000)
"""
import os
import time

import httpx
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Document Q&A", layout="wide")

STATUS_NOTE = {"out_of_scope": "Outside the documents available to you",
               "not_found": "Not found in the documents", "not_grounded": "Could not be verified"}


def call(method, path, **kw):
    headers = {"X-Demo-User": st.session_state.get("user", "")}
    r = httpx.request(method, API + path, headers=headers, timeout=900, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    return r


try:
    health = httpx.get(API + "/health", timeout=10).json()
except Exception:
    st.error(f"Cannot reach the RAG API at {API}. Start it with: uvicorn rag.api:app --port 8000")
    st.stop()

# ---------------- sidebar: who am I, upload, library ----------------
with st.sidebar:
    st.caption(f"Profile {health['profile']} · models: {health['llm']} · index: {health['index']}")
    if health["auth"] == "demo":
        users = httpx.get(API + "/demo-users", timeout=10).json()
        pick = st.selectbox("Signed in as (demo)", list(users), format_func=lambda k: users[k]["name"])
        if pick != st.session_state.get("user"):
            st.session_state.user, st.session_state.history = pick, []
        u = users[pick]
        st.caption(f"Tenant **{u['tenant']}** · groups: {', '.join(u['groups']) or '-'}")
        group_options = sorted({g for v in users.values() for g in v["groups"]} | {"all"})
    else:
        group_options = ["all"]

    st.header("Add documents")
    files = st.file_uploader("Files", type=["pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg", "tif", "tiff"],
                             accept_multiple_files=True)
    lateral = st.text_input("Lateral (domain)", "design")
    acl = st.multiselect("Who may read it", group_options, default=["all"])
    if files and st.button(f"Upload {len(files)} file(s)", type="primary"):
        for f in files:
            res = call("POST", "/documents", files={"file": (f.name, f.getvalue())},
                       data={"lateral": lateral, "acl": ",".join(acl)}).json()
            st.toast(f"{f.name}: queued")
        st.info("Processing runs in the background. You can keep using the page.")

    st.header("Processing")
    jobs = call("GET", "/jobs").json()
    for j in jobs[:8]:
        icon = {"done": "✅", "skipped": "⏭️", "failed": "❌", "running": "⏳", "queued": "🕒"}.get(j["status"], "")
        st.write(f"{icon} **{j['file_name']}** · {j['status']} · {j['step']}")
    if any(j["status"] in ("queued", "running") for j in jobs) and st.button("Refresh status"):
        st.rerun()

    st.header("Library")
    docs = call("GET", "/documents").json()
    st.caption(f"{len(docs)} document(s) you can see")
    for d in docs:
        with st.expander(d["doc_title"]):
            st.caption(f"{d['lateral']} · readers: {', '.join(d['acl'])} · {d['chunks']} pieces · {d['figures']} images")
            st.write(d["summary"])
            if st.button("Remove", key=f"rm-{d['doc_id']}"):
                call("DELETE", f"/documents/{d['doc_id']}")
                st.rerun()
    only = st.multiselect("Ask only these documents", [d["doc_title"] for d in docs])
    debug = st.toggle("Show match scores")

# ---------------- main: questions ----------------
st.title("Ask the documents")
st.caption("Answers come only from documents you're allowed to see, with sources. "
           "If the answer isn't there, you'll be told.")
st.session_state.setdefault("history", [])


def show(turn, i):
    if turn["status"] == "answered":
        st.markdown(turn["answer"])
    else:
        st.info(f"**{STATUS_NOTE.get(turn['status'], turn['status'])}.** {turn['answer']}")
    if turn["citations"]:
        with st.expander(f"Sources ({len(turn['citations'])})"):
            for c in turn["citations"]:
                st.markdown(f"**[{c['n']}] {c['doc_title']}** · {c['section']} · page {c['pages']}")
                for fig in c["figures"][:2]:
                    try:
                        st.image(call("GET", fig).content, width=420)
                    except Exception:
                        pass
    if debug:
        st.caption(f"best match score {turn['best_score']} · {turn['seconds']} s")


for i, t in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(t["question"])
    with st.chat_message("assistant"):
        show(t, i)

q = st.chat_input("Ask a question about the documents")
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Checking the documents..."):
            hist = []
            for t in st.session_state.history[-3:]:
                hist += [{"role": "user", "content": t["question"]}, {"role": "assistant", "content": t["answer"]}]
            t0 = time.time()
            ids = [d["doc_id"] for d in docs if d["doc_title"] in only]
            res = call("POST", "/ask", json={"question": q, "doc_ids": ids, "history": hist}).json()
        turn = {"question": q, **res, "seconds": round(time.time() - t0, 1)}
        st.session_state.history.append(turn)
        show(turn, len(st.session_state.history) - 1)
