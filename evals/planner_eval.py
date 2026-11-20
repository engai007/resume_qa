"""Evaluate the query planner (route + sections) against evals/planner_golden.json, using the real model.

    uv run python evals/planner_eval.py                                              # model/provider from settings
    uv run python evals/planner_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --provider featherless-ai

Only the planner runs (no search, no answer), so it's cheap: one small-model call per question.
No retry and no fallback provider here: we measure the model on this provider, as it is.
Prints a report and saves it to outputs/evals/<date>__planner__<model>__<provider>.json
"""
import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when run from the IDE

from app.agents import base  # noqa: E402
from app.agents.planner_agent import PlannerAgent  # noqa: E402
from app.schemas.plan import InvalidPlan  # noqa: E402


def score(item: dict, plan) -> tuple[bool, bool]:
    """(route ok, sections ok) for one question. plan=None means the planner's reply was unusable."""
    if plan is None:  # code falls back to plain search: right only for injection questions
        ok = item["type"] == "injection"
        return ok, ok
    routes = item["route"] if isinstance(item["route"], list) else [item["route"]]
    route_ok = plan.route in routes
    if not route_ok or plan.route == "search":
        return route_ok, route_ok  # sections on search only narrow it; not scored
    got, required = set(plan.sections), set(item["sections"])
    return True, required <= got <= required | set(item["also_ok"])


async def main(args):
    golden = json.loads(Path(args.golden).read_text())
    agent = PlannerAgent()
    agent.model_id = args.model or agent.model_id
    agent.provider = args.provider or agent.provider
    agent.fallback_provider = ""
    base.RETRY_WAITS = []  # one try per question

    rows = []
    for item in golden["questions"]:
        start = time.perf_counter()
        plan, error = None, None
        try:
            plan = await agent.plan(item["question"])
        except InvalidPlan as e:
            error = f"invalid plan: {e}"
        except Exception as e:  # provider error: counted, not fatal
            error = f"{type(e).__name__}"
        seconds = round(time.perf_counter() - start, 2)
        route_ok, sections_ok = score(item, plan)
        rows.append({"id": item["id"], "type": item["type"], "question": item["question"],
                     "expected_route": item["route"], "expected_sections": item["sections"],
                     "plan": plan.model_dump() if plan else None, "error": error,
                     "route_ok": route_ok, "sections_ok": sections_ok, "seconds": seconds})

    n = len(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    secs = sorted(r["seconds"] for r in rows if not (r["error"] or "").startswith(("Read", "Connect", "Hf")))
    report = {
        "date": date.today().isoformat(), "model": agent.model_id, "provider": agent.provider, "n": n,
        "valid_plan": sum(r["plan"] is not None for r in rows) / n,
        "route_accuracy": sum(r["route_ok"] for r in rows) / n,
        "sections_accuracy": sum(r["sections_ok"] for r in rows) / n,
        "seconds_avg": round(sum(secs) / len(secs), 2) if secs else None,
        "seconds_p90": secs[int(len(secs) * 0.9) - 1] if secs else None,
        "by_type": {t: {"n": len(rs), "route": sum(r["route_ok"] for r in rs) / len(rs),
                        "sections": sum(r["sections_ok"] for r in rs) / len(rs)} for t, rs in sorted(by_type.items())},
        "questions": rows,
    }

    print(f"\nPlanner eval  {report['date']}  {agent.model_id} @ {agent.provider}  n={n}\n")
    print(f"valid plan {report['valid_plan']:.2f}   route {report['route_accuracy']:.2f}   "
          f"sections {report['sections_accuracy']:.2f}   avg {report['seconds_avg']}s   p90 {report['seconds_p90']}s\n")
    print(f"{'':<12}{'n':>3}{'route':>8}{'sections':>10}")
    for t, s in report["by_type"].items():
        print(f"{t:<12}{s['n']:>3}{s['route']:>8.2f}{s['sections']:>10.2f}")
    wrong = [r for r in rows if not (r["route_ok"] and r["sections_ok"])]
    if wrong:
        print("\nwrong")
    for r in wrong:
        got = r["error"] or f"{r['plan']['route']} {r['plan']['sections']}"
        print(f"  {r['id']}  {r['question'][:55]:<55}  expected {r['expected_route']} {r['expected_sections']}  got {got}")

    out = ROOT / "outputs" / "evals" / f"{report['date']}__planner__{agent.model_id.split('/')[-1]}__{agent.provider}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(ROOT / "evals" / "planner_golden.json"))
    p.add_argument("--model", help="default: settings.planner_model_id")
    p.add_argument("--provider", help="default: settings.planner_provider")
    asyncio.run(main(p.parse_args()))
