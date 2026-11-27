"""Evaluate /qa/ask answers end to end (real Qdrant, real planner, real answer model), planner on vs off.

    uv run python evals/answer_eval.py                  # both modes, compared
    uv run python evals/answer_eval.py --mode planner   # or: plain (planner off)

Questions:
- evals/retrieval_golden.json   answerable ones: found=true? does a cited source match an expected chunk?
                                no-answer ones:  found=false?
- evals/answer_golden.json      list / total / section questions: found=true? which expected items does the answer name?

Metrics per mode:
  answer rate     answerable questions with found=true
  source hit      answerable questions where at least one cited source is an expected golden chunk
  no-answer acc   no-answer questions with found=false
  item recall     list questions: items named in the answer / items expected (average)
Each question costs 1 (plain) or 2 (planner + answer) LLM calls. Saves outputs/evals/<date>__answer__<mode>.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when run from the IDE

from sqlalchemy import select  # noqa: E402

from app.agents.planner_agent import PlannerAgent  # noqa: E402
from app.agents.qa_agent import QAAgent  # noqa: E402
from app.db.database import SessionLocal, engine  # noqa: E402
from app.db.qdrant import qdrant  # noqa: E402
from app.models import User  # noqa: E402
from app.services.qa_service import answer_question  # noqa: E402

CONCURRENCY = 3  # questions in flight at once (keeps provider rate limits happy)


def named(answer: str, item) -> bool:
    """item = "AELMS" or ["6 years 4 months", "6 years and 4 months"]; case-insensitive."""
    spellings = item if isinstance(item, list) else [item]
    return any(s.lower() in answer.lower() for s in spellings)


def source_hit(sources: list[dict], expected: list[dict]) -> bool:
    return any(s["document_hash"] == e["document_hash"] and e["contains"] in s["text"]
               for s in sources for e in expected)


def build_questions(retrieval: dict, extra: dict) -> list[dict]:
    docs = retrieval["documents"]
    out = []
    for q in retrieval["questions"]:
        # pinned to the golden resumes: the eval user is a recruiter, who can read every user's resumes
        out.append({"id": q["id"], "kind": "answerable" if q["expected"] else "no_answer", "type": q["type"],
                    "question": q["question"],
                    "hashes": list(docs.values()),
                    "expected": [{"document_hash": docs[e["document"]], "contains": e["contains"]} for e in q["expected"]]})
    for q in extra["questions"]:
        out.append({"id": q["id"], "kind": "list", "type": "list", "question": q["question"],
                    "hashes": [docs[q["document"]]] if q["document"] else list(docs.values()), "items": q["items"]})
    return out


def load_questions() -> tuple[list[dict], str]:
    """-> (questions, email of the eval user). Also used by ragas_eval.py."""
    retrieval = json.loads((ROOT / "evals" / "retrieval_golden.json").read_text())
    extra = json.loads((ROOT / "evals" / "answer_golden.json").read_text())
    return build_questions(retrieval, extra), retrieval["search_as_user"]


async def eval_user(email: str) -> User:
    async with SessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == email))
    if not user:
        sys.exit(f"User {email} not found.")
    return user


async def ask_one(q: dict, user: User, qa: QAAgent, planner: PlannerAgent | None, top_k: int, sem) -> dict:
    async with sem:
        start = time.perf_counter()
        try:
            r = await answer_question(user, q["question"], qa, top_k, q["hashes"], planner)
            error = None
        except Exception as e:  # counted as a wrong answer, not fatal
            r, error = {"found": False, "answer": "", "sources": [], "plan": None}, type(e).__name__
        row = {"id": q["id"], "kind": q["kind"], "question": q["question"], "found": r["found"],
               "answer": r["answer"], "plan": r.get("plan"), "error": error,
               "n_sources": len(r["sources"]), "seconds": round(time.perf_counter() - start, 1)}
        if q["kind"] == "answerable":
            row["ok"] = r["found"]
            row["source_hit"] = source_hit(r["sources"], q["expected"])
        elif q["kind"] == "no_answer":
            row["ok"] = not r["found"]
        else:
            hits = [named(r["answer"], i) for i in q["items"]]
            row["ok"] = r["found"]
            row["item_recall"] = sum(hits) / len(hits)
            row["missed"] = [i if isinstance(i, str) else i[0] for i, h in zip(q["items"], hits) if not h]
        return row


def summary(rows: list[dict]) -> dict:
    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None
    ans = [r for r in rows if r["kind"] == "answerable"]
    no = [r for r in rows if r["kind"] == "no_answer"]
    lst = [r for r in rows if r["kind"] == "list"]
    return {"answer_rate": avg([r["ok"] for r in ans]), "source_hit": avg([r["source_hit"] for r in ans]),
            "no_answer_acc": avg([r["ok"] for r in no]), "list_found": avg([r["ok"] for r in lst]),
            "item_recall": avg([r["item_recall"] for r in lst]), "seconds_avg": avg([r["seconds"] for r in rows]),
            "errors": sum(bool(r["error"]) for r in rows), "n": {"answerable": len(ans), "no_answer": len(no), "list": len(lst)}}


async def main(args):
    questions, email = load_questions()
    user = await eval_user(email)

    modes = ["plain", "planner"] if args.mode == "both" else [args.mode]
    qa, sem, reports = QAAgent(), asyncio.Semaphore(CONCURRENCY), {}
    for mode in modes:
        planner = PlannerAgent() if mode == "planner" else None
        rows = await asyncio.gather(*(ask_one(q, user, qa, planner, args.top_k, sem) for q in questions))
        reports[mode] = {"date": date.today().isoformat(), "mode": mode, "top_k": args.top_k,
                         "summary": summary(rows), "questions": rows}
        out = ROOT / "outputs" / "evals" / f"{date.today().isoformat()}__answer__{mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports[mode], indent=2, ensure_ascii=False))
        print(f"saved: {out.relative_to(ROOT)}")

    n = next(iter(reports.values()))["summary"]["n"]
    print(f"\nAnswer eval  {date.today()}  top_k {args.top_k}  "
          f"({n['answerable']} answerable, {n['no_answer']} no-answer, {n['list']} list questions)\n")
    keys = ["answer_rate", "source_hit", "no_answer_acc", "list_found", "item_recall", "seconds_avg", "errors"]
    print(f"{'':<15}" + "".join(f"{m:>10}" for m in modes))
    for k in keys:
        print(f"{k:<15}" + "".join(f"{str(reports[m]['summary'][k]):>10}" for m in modes))

    for mode in modes:
        bad = [r for r in reports[mode]["questions"]
               if not r["ok"] or r.get("source_hit") is False or (r.get("item_recall") or 1) < 1]
        print(f"\n{mode}: problems")
        for r in bad:
            what = (f"recall {r['item_recall']:.2f} missed {r['missed']}" if r["kind"] == "list"
                    else "no source hit" if r["kind"] == "answerable" and r["ok"] else f"found={r['found']}")
            plan = f"{r['plan']['route']} {r['plan']['sections']}" if r.get("plan") else "-"
            print(f"  {r['id']}  {r['question'][:50]:<50}  {what}  [{plan}]{'  ERROR ' + r['error'] if r['error'] else ''}")

    await qdrant.close()
    await engine.dispose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["both", "planner", "plain"], default="both")
    p.add_argument("--top-k", type=int, default=5)
    asyncio.run(main(p.parse_args()))
