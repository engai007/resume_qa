"""Evaluate retrieval against the golden dataset, using the real Qdrant + embeddings.

    uv run python evals/retrieval_eval.py                          # evals/retrieval_golden.json, top_k 5
    uv run python evals/retrieval_eval.py --top-k 3
    uv run python evals/retrieval_eval.py --golden evals/other.json --threshold 0.5

Prints a report and saves it to outputs/evals/<date>__<model>__v<index_version>__k<top_k>.json
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when run from the IDE

from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.database import SessionLocal, engine  # noqa: E402
from app.db.qdrant import qdrant  # noqa: E402
from app.models import Document, User  # noqa: E402
from app.services.indexing_service import INDEX_VERSION  # noqa: E402
from app.services.retrieval_metrics import average, score_no_answer, score_question  # noqa: E402
from app.services.retriever_service import search_chunks  # noqa: E402


async def find_user(email: str, hashes: list[str]) -> User:
    """The user whose resumes are searched; warns if a golden document isn't indexed for them."""
    async with SessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if not user:
            sys.exit(f"User {email} not found. Upload the golden resumes as this user first.")
        docs = {d.document_hash: d for d in await db.scalars(
            select(Document).where(Document.generated_by == user.id, Document.document_hash.in_(hashes)))}
    for h in hashes:
        if h not in docs:
            print(f"WARNING  document {h[:12]}… not uploaded by {email}")
        elif docs[h].indexed_version != INDEX_VERSION:
            print(f"WARNING  document {h[:12]}… indexed_version={docs[h].indexed_version}, expected {INDEX_VERSION}")
    return user


async def run(golden_path: Path, k: int, threshold: float) -> dict:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    hashes = list(golden["documents"].values())
    user = await find_user(golden["search_as_user"], hashes)

    answered, no_answer = [], []
    for q in golden["questions"]:
        # search only the golden resumes, so other uploads by this user can't change the score
        results = await search_chunks(user.id, q["question"], top_k=k, document_hashes=hashes)
        retrieved = [{"rank": i, "score": r["score"], "document_hash": r["document_hash"][:12],
                      "first_line": r["text"].splitlines()[0]} for i, r in enumerate(results, start=1)]
        if q["expected"]:
            expected = [{"document_hash": golden["documents"][e["document"]], "contains": e["contains"]}
                        for e in q["expected"]]
            s = score_question(results, expected, k)
            s["missed"] = [e["contains"] for e in s["missed"]]
            answered.append({"id": q["id"], "type": q["type"], "question": q["question"], **s, "retrieved": retrieved})
        else:
            no_answer.append({"id": q["id"], "question": q["question"], **score_no_answer(results, threshold),
                              "retrieved": retrieved})

    by_type = defaultdict(list)
    for s in answered:
        by_type[s["type"]].append(s)

    return {
        "date": date.today().isoformat(),
        "golden": golden_path.name,
        "golden_version": golden.get("version"),
        "embedding_model": settings.embedding_model,
        "index_version": INDEX_VERSION,
        "top_k": k,
        "no_answer_threshold": threshold,
        "overall": {"questions": len(answered), **average(answered)},
        "by_type": {t: {"questions": len(v), **average(v)} for t, v in by_type.items()},
        "no_answer": {"questions": len(no_answer), "passed": sum(q["passed"] for q in no_answer)},
        "questions": answered,
        "no_answer_questions": no_answer,
    }


def print_report(r: dict) -> None:
    print(f"\nRetrieval eval  {r['date']}  model: {r['embedding_model']}  "
          f"index_version: {r['index_version']}  top_k: {r['top_k']}")
    k = r["top_k"]
    print(f"\n{'':16}{'n':>4}  {'hit@' + str(k):>7}  {'recall@' + str(k):>9}  {'precision@' + str(k):>12}  {'MRR':>6}")
    rows = [("overall", r["overall"])] + sorted(r["by_type"].items())
    for name, m in rows:
        print(f"{name:16}{m['questions']:>4}  {m['hit']:>7.2f}  {m['recall']:>9.2f}  {m['precision']:>12.2f}  {m['mrr']:>6.2f}")

    na = r["no_answer"]
    print(f"\nno-answer accuracy (top score < {r['no_answer_threshold']}): {na['passed']}/{na['questions']}")
    for q in r["no_answer_questions"]:
        print(f"  {'ok  ' if q['passed'] else 'FAIL'}  {q['id']}  top score {q['top_score']:.3f}  {q['question']}")

    worst = sorted(r["questions"], key=lambda q: (q["recall"], q["mrr"]))[:5]
    print("\nworst questions")
    for q in worst:
        rank = q["first_correct_rank"] or "-"
        print(f"  {q['id']}  recall {q['recall']:.2f}  first correct at #{rank}  {q['question']}")
        for m in q["missed"]:
            print(f"        missed: {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a golden dataset.")
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "retrieval_golden.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.45, help="no-answer questions pass below this score")
    args = parser.parse_args()

    async def main():
        report = await run(args.golden, args.top_k, args.threshold)
        await qdrant.close()
        await engine.dispose()
        return report

    report = asyncio.run(main())
    print_report(report)

    out_dir = ROOT / settings.parser_output_dir / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = settings.embedding_model.split("/")[-1]
    out_file = out_dir / f"{report['date']}__{model}__v{report['index_version']}__k{report['top_k']}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out_file.relative_to(ROOT)}")
