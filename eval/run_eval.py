"""Score the system with your own test questions:  python -m eval.run_eval [questions.jsonl] [demo-user]

Each line of the questions file:
  {"question": "...", "expected": "...", "should_answer": true}
Include table questions, image/flowchart questions, and at least 20% that must be refused or
"not found". Prints how many were handled correctly and the match scores, so you can set
MIN_RELEVANCE between the "should answer" and "should not answer" groups.
Writes eval/results.csv for review with the engineers (contains answers: keep it private)."""
import csv
import json
import sys
import time

from rag.ask.graph import ask
from rag.config import settings
from rag.security import User, demo_users

path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.jsonl"
uid = sys.argv[2] if len(sys.argv) > 2 else next(iter(demo_users()))
u = demo_users()[uid]
user = User(id=uid, tenant=u["tenant"], groups=tuple(u.get("groups", [])))

rows, ok_count = [], 0
for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    case = json.loads(line)
    t0 = time.time()
    out = ask(case["question"], user)
    answered = out["status"] == "answered"
    ok = answered == case["should_answer"]
    ok_count += ok
    rows.append({"question": case["question"], "should_answer": case["should_answer"], "status": out["status"],
                 "ok": ok, "best_score": round(out.get("best_score", 0.0), 3),
                 "seconds": round(time.time() - t0, 1), "answer": out["answer"], "expected": case.get("expected", ""),
                 "sources": " || ".join(f"{s['doc_title']} | {s['section']} | p.{s['pages']}"
                                        for s in out.get("sources", []) if s["n"] in out.get("cited", []))})
    print(f"{'OK ' if ok else 'BAD'} {out['status']:<13} score={rows[-1]['best_score']:.2f}  {case['question']}")

with open("eval/results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nHandled correctly: {ok_count}/{len(rows)}  (target 95%+)")
print("Best scores, should answer:     ", sorted(r["best_score"] for r in rows if r["should_answer"]))
print("Best scores, should NOT answer: ", sorted(r["best_score"] for r in rows
                                                 if not r["should_answer"] and r["status"] != "out_of_scope"))
print(f"MIN_RELEVANCE is {settings.min_relevance}; set it between the two groups. Details: eval/results.csv")
