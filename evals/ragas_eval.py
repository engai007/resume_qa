"""RAGAS eval of /qa/ask answers: is every claim supported by the chunks, and does the answer address the question?
See docs/ragas_eval_plan.md.

    uv run python evals/ragas_eval.py                      # planner on, judge on cerebras
    uv run python evals/ragas_eval.py --mode plain         # planner off
    uv run python evals/ragas_eval.py --provider groq      # judge on the fallback provider
    uv run python evals/ragas_eval.py --only q04 l01       # a few questions

Questions: the answerable golden questions + the list questions (same as answer_eval.py). Skipped: the "not found"
golden questions, and answers that came back found=false (nothing to judge).

Per question:
  faithfulness      supported claims / all claims, judged against EVERY chunk the answer model saw (header included)
  answer_relevancy  how well questions written from the answer match the real one (bge cosine; read it relatively)
Each question: 1-2 app LLM calls + ~5 judge calls. Saves outputs/evals/<date>__ragas__<mode>.json
(questions, answers, scores, unsupported claims; no chunk text).
"""
import argparse
import asyncio
import json
import math
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.ragas_adapters import BgeEmbeddings, judge_model_name, make_judge  # noqa: E402  (sets RAGAS_DO_NOT_TRACK)
from ragas.metrics.collections import AnswerRelevancy, Faithfulness  # noqa: E402

from app.agents.planner_agent import PlannerAgent  # noqa: E402
from app.agents.qa_agent import QAAgent, format_chunk  # noqa: E402
from app.db.database import engine  # noqa: E402
from app.db.qdrant import qdrant  # noqa: E402
from app.services.qa_service import answer_question  # noqa: E402
from evals.answer_eval import eval_user, load_questions  # noqa: E402

CONCURRENCY = 3
LOW_FAITHFULNESS = 0.9  # below this: read the question (plan §6)

MARKERS = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def strip_markers(answer: str) -> str:
    """"Asha used RabbitMQ [1], [2]." -> "Asha used RabbitMQ." The judge splits the answer into claims; the
    [n] markers aren't claims, only noise."""
    return MARKERS.sub("", answer).strip()


def to_sample(q: dict, result: dict) -> tuple[dict | None, str | None]:
    """-> (RAGAS sample, None) or (None, why skipped)."""
    if q["kind"] == "no_answer":
        return None, "no_answer question"
    if not result["found"]:
        return None, "answer not found"
    return {"user_input": q["question"],
            "response": strip_markers(result["answer"]),
            # every chunk the model saw, exactly as it saw it (the candidate name is only in the header)
            "retrieved_contexts": [format_chunk(n, c) for n, c in enumerate(result["chunks"], start=1)]}, None


def number(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)


async def judge(sample: dict, faith: Faithfulness, rel: AnswerRelevancy) -> dict:
    """Faithfulness in its two steps (claims, then verdicts) to keep each claim's verdict, which ascore drops."""
    statements = await faith._create_statements(sample["user_input"], sample["response"])
    claims = []
    if statements:
        verdicts = await faith._create_verdicts(statements, "\n".join(sample["retrieved_contexts"]))
        claims = verdicts.statements
    relevancy = await rel.ascore(user_input=sample["user_input"], response=sample["response"])
    return {"faithfulness": number(sum(c.verdict for c in claims) / len(claims)) if claims else None,
            "answer_relevancy": number(relevancy.value),
            "n_claims": len(claims),
            "unsupported": [{"claim": c.statement, "reason": c.reason} for c in claims if not c.verdict]}


async def run_one(q, user, qa, planner, faith, rel, top_k, sem) -> dict:
    async with sem:
        start = time.perf_counter()
        row = {"id": q["id"], "type": q["type"], "question": q["question"]}
        try:
            result = await answer_question(user, q["question"], qa, top_k, q["hashes"], planner)
        except Exception as e:
            return {**row, "skipped": f"answer error {type(e).__name__}"}
        sample, skipped = to_sample(q, result)
        row.update(answer=result["answer"], n_chunks=len(result["chunks"]),
                   route=(result["plan"] or {}).get("route"))
        if skipped:
            return {**row, "skipped": skipped}
        try:
            row.update(await judge(sample, faith, rel))
        except Exception as e:  # judge down / bad structured output: left out of the averages, not fatal
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        row["seconds"] = round(time.perf_counter() - start, 1)
        return row


def summarise(rows: list[dict]) -> dict:
    """Averages over the judged questions, overall and per type. None scores (no claims) are left out."""
    judged = [r for r in rows if "skipped" not in r and "error" not in r]

    def avg(rs, key):
        xs = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    def block(rs):
        return {"n": len(rs), "faithfulness": avg(rs, "faithfulness"), "answer_relevancy": avg(rs, "answer_relevancy")}

    types = sorted({r["type"] for r in judged})
    return {"overall": block(judged), "by_type": {t: block([r for r in judged if r["type"] == t]) for t in types},
            "skipped": sum("skipped" in r for r in rows), "errors": sum("error" in r for r in rows)}


def print_report(report: dict):
    s = report["summary"]
    print(f"\nRAGAS eval  {report['date']}  judge {report['judge']}  mode {report['mode']}  "
          f"({s['overall']['n']} judged, {s['skipped']} skipped, {s['errors']} errors)\n")
    print(f"{'':<14}{'n':>4}{'faithfulness':>15}{'relevancy':>12}")
    for name, b in [("overall", s["overall"]), *s["by_type"].items()]:
        print(f"{name:<14}{b['n']:>4}{str(b['faithfulness']):>15}{str(b['answer_relevancy']):>12}")

    rows = report["questions"]
    low = sorted((r for r in rows if r.get("faithfulness") is not None and r["faithfulness"] < LOW_FAITHFULNESS),
                 key=lambda r: r["faithfulness"])
    print(f"\nfaithfulness < {LOW_FAITHFULNESS}: {len(low)}")
    for r in low:
        print(f"  {r['id']}  {r['faithfulness']:.2f}  {r['question'][:60]}")
        for u in r["unsupported"]:
            print(f"        unsupported: {u['claim']}\n          judge: {u['reason']}")
    for r in rows:
        if "skipped" in r and r["skipped"] != "no_answer question" or "error" in r:
            print(f"  {r['id']}  {r.get('skipped') or 'ERROR ' + r['error']}")


async def main(args):
    questions, email = load_questions()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    user = await eval_user(email)

    llm = make_judge(args.provider)
    faith, rel = Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=BgeEmbeddings())
    qa = QAAgent()
    planner = PlannerAgent() if args.mode == "planner" else None
    sem = asyncio.Semaphore(CONCURRENCY)

    rows = await asyncio.gather(*(run_one(q, user, qa, planner, faith, rel, args.top_k, sem) for q in questions))
    report = {"date": date.today().isoformat(), "mode": args.mode, "top_k": args.top_k,
              "judge": judge_model_name(args.provider), "summary": summarise(rows), "questions": rows}
    if not args.only:  # a partial run doesn't overwrite the full report
        out = ROOT / "outputs" / "evals" / f"{date.today().isoformat()}__ragas__{args.mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"saved: {out.relative_to(ROOT)}")
    print_report(report)

    await qdrant.close()
    await engine.dispose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["planner", "plain"], default="planner")
    p.add_argument("--provider", default=None, help="judge provider (default: JUDGE_PROVIDER)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--only", nargs="*", help="question ids, e.g. q04 l01")
    asyncio.run(main(p.parse_args()))
