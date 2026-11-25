"""End-to-end tests of the platform with stand-in models (profile C, real Qdrant)."""
import logging

import pytest
from fastapi.testclient import TestClient

from rag.api import app
from rag.ingest import worker
from rag.providers.index import index
from tests.conftest import Calls

client = TestClient(app)   # not used as a context manager -> no background thread; we run jobs here
DES_A, MFG_A, DES_B = {"X-Demo-User": "design.a"}, {"X-Demo-User": "mfg.a"}, {"X-Demo-User": "design.b"}


def upload(headers, name="sample.pdf", acl="design", lateral="design", data=b"%PDF-1.4 fake"):
    r = client.post("/documents", headers=headers, files={"file": (name, data)},
                    data={"lateral": lateral, "acl": acl})
    assert r.status_code == 200, r.text
    job = r.json()
    worker.run_job(job["job_id"])          # runs it here, unless the background worker already is
    import time
    for _ in range(600):                   # wait until finished either way
        done = worker.get_job(job["job_id"])
        if done["status"] not in ("queued", "running"):
            return job, done
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def ask(headers, q, **kw):
    r = client.post("/ask", headers=headers, json={"question": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module", autouse=True)
def catalogue():
    from tests.conftest import Calls as C
    C.read = C.described = 0
    job, done = upload(DES_A)
    assert done["status"] == "done", done
    return job


def test_sign_in_required():
    assert client.post("/ask", json={"question": "hi"}).status_code == 401


def test_answer_with_citation():
    out = ask(DES_A, "Which bolts are listed for ISO 4014 in A2-70?")
    assert out["status"] == "answered"
    assert "900001" in out["answer"]
    assert out["citations"] and out["citations"][0]["pages"]


def test_gate1_refuses_off_topic():
    assert ask(DES_A, "What's the weather in Delhi tomorrow?")["status"] == "out_of_scope"


def test_gate2_not_found():
    assert ask(DES_A, "What is the warranty period for brake pads?")["status"] == "not_found"


def test_gate3_blocks_invented_numbers():
    out = ask(DES_A, "Which bolts are listed for ISO 4014? invent")
    assert out["status"] == "not_grounded" and "777777" not in out["answer"]


def test_separation_by_group_same_tenant():
    out = ask(MFG_A, "Which bolts are listed for ISO 4014 in A2-70?")
    assert out["status"] != "answered" and "900001" not in out["answer"]
    assert client.get("/documents", headers=MFG_A).json() == []


def test_separation_by_tenant():
    out = ask(DES_B, "Which bolts are listed for ISO 4014 in A2-70?")
    assert out["status"] != "answered" and "900001" not in out["answer"]
    doc_id = client.get("/documents", headers=DES_A).json()[0]["doc_id"]
    assert client.get(f"/documents/{doc_id}/file", headers=DES_B).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=DES_B).status_code == 404


def test_lateral_filter():
    assert ask(DES_A, "Which bolts are listed for ISO 4014 in A2-70?", laterals=["quality"])["status"] != "answered"
    assert ask(DES_A, "Which bolts are listed for ISO 4014 in A2-70?", laterals=["design"])["status"] == "answered"


def test_figures_skip_tiny_and_are_served():
    doc = client.get("/documents", headers=DES_A).json()[0]
    assert doc["figures"] == 1                      # the 40x40 icon was skipped
    assert client.get(f"/figures/{doc['doc_id']}/1.png", headers=DES_A).status_code == 200
    assert client.get(f"/figures/{doc['doc_id']}/1.png", headers=MFG_A).status_code == 404


def test_unchanged_upload_is_skipped():
    _, done = upload(DES_A)
    assert done["status"] == "skipped"


def test_changed_tags_reuse_caches():
    reads, described = Calls.read, Calls.described
    _, done = upload(DES_A, acl="design,manufacturing")   # same file, new readers
    assert done["status"] == "done"
    assert Calls.read == reads and Calls.described == described   # nothing read or described again
    assert ask(MFG_A, "Which bolts are listed for ISO 4014 in A2-70?")["status"] == "answered"
    upload(DES_A, acl="design")                # back to design only
    assert ask(MFG_A, "Which bolts are listed for ISO 4014 in A2-70?")["status"] != "answered"


def test_retry_then_success(monkeypatch):
    import rag.ingest.pipeline as P
    real, calls = P.ingest_file, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("temporary")
        return real(*a, **k)
    monkeypatch.setattr(worker, "ingest_file", flaky)
    _, done = upload(DES_A, name="second.pdf")
    assert done["status"] == "done" and done["attempts"] == 2


def test_failed_after_three_tries(monkeypatch):
    monkeypatch.setattr(worker, "ingest_file", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    _, done = upload(DES_A, name="broken.pdf")
    assert done["status"] == "failed" and done["attempts"] == 3 and done["error"] == "ValueError"


def test_resume_unfinished_job():
    job = worker.submit(b"%PDF-1.4 fake", "resumed.pdf", "company-a", "design", ["design"], None, "design.a")
    j = worker.get_job(job["id"])
    j["status"] = "running"            # simulate a crash in the middle
    worker._save(j)
    import queue
    worker._q = queue.Queue()          # fresh queue (earlier tests ran their jobs directly)
    worker._started = False
    worker.start()                     # restart: unfinished job is picked up again
    worker._q.join()
    assert worker.get_job(job["id"])["status"] == "done"


def test_unsupported_file_rejected():
    r = client.post("/documents", headers=DES_A, files={"file": ("virus.exe", b"x")})
    assert r.status_code == 400


def test_image_upload_converts(tmp_path):
    from PIL import Image
    from rag.ingest.convert import to_pdf
    p = tmp_path / "scan.png"
    Image.new("RGB", (200, 200), "white").save(p)
    assert to_pdf(p, tmp_path).read_bytes().startswith(b"%PDF")


def test_delete_removes_everything():
    _, done = upload(DES_A, name="temp.pdf")
    before = index().count()
    assert client.delete(f"/documents/{done['doc_id']}", headers=DES_A).status_code == 200
    assert index().count() < before
    assert done["doc_id"] not in [d["doc_id"] for d in client.get("/documents", headers=DES_A).json()]


def test_logs_never_contain_content(caplog):
    caplog.set_level(logging.INFO, logger="rag")
    secret_q = "Which bolts are listed for ISO 4014 in A2-70 zebra-question?"
    ask(DES_A, secret_q)
    upload(DES_A, name="logcheck.pdf")
    logs = caplog.text
    assert "zebra-question" not in logs            # question text never logged
    assert "900001" not in logs and "stainless steel" not in logs   # document text never logged
    assert "ask.done" in logs and "job.finished" in logs


def test_number_check_unit():
    from rag.ask.graph import numbers_supported
    assert numbers_supported("Part 900001 is M6 x 60 [1]", "900001 M 6 60")
    assert not numbers_supported("Part 900009 [1]", "900001 M 6 60")
