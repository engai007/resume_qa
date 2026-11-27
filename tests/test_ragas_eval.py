"""RAGAS eval plumbing, no network: samples, markers, adapter, report math. Made-up data."""
import asyncio

from ragas.metrics.collections import Faithfulness

from evals import ragas_adapters, ragas_eval
from evals.ragas_eval import strip_markers, summarise, to_sample

CHUNKS = [
    {"candidate_name": "Asha Rao", "document_hash": "ab12cd34", "text": "Moved order events to RabbitMQ consumers."},
    {"candidate_name": None, "document_hash": "ef56ab78", "text": "Skills: Python, SQL"},
]
ANSWERABLE = {"id": "q04", "kind": "answerable", "type": "skill", "question": "Who has worked with RabbitMQ?"}


def result(found=True, answer="Asha Rao moved order events to RabbitMQ [1].", chunks=CHUNKS):
    return {"found": found, "answer": answer, "chunks": chunks, "sources": [{"n": 1}], "plan": None}


def test_sample_holds_every_chunk_the_model_saw_with_its_header():
    sample, skipped = to_sample(ANSWERABLE, result())
    assert skipped is None
    # both chunks, although only [1] was cited: a claim can come from an uncited chunk
    assert sample["retrieved_contexts"] == [
        "[1] candidate: Asha Rao | resume: ab12cd\nMoved order events to RabbitMQ consumers.",
        "[2] candidate: unknown | resume: ef56ab\nSkills: Python, SQL",
    ]
    assert sample["response"] == "Asha Rao moved order events to RabbitMQ."
    assert sample["user_input"] == ANSWERABLE["question"]


def test_nothing_to_judge_is_skipped():
    assert to_sample({**ANSWERABLE, "kind": "no_answer"}, result()) == (None, "no_answer question")
    assert to_sample(ANSWERABLE, result(found=False)) == (None, "answer not found")


def test_strip_markers():
    assert strip_markers("Python [1], SQL [2, 3] and Go [4][5].") == "Python, SQL and Go."
    assert strip_markers("No markers here.") == "No markers here."


def test_summary_averages_only_judged_questions():
    rows = [
        {"type": "skill", "faithfulness": 1.0, "answer_relevancy": 0.8},
        {"type": "skill", "faithfulness": 0.5, "answer_relevancy": 0.6},
        {"type": "list", "faithfulness": None, "answer_relevancy": 0.7},  # no claims: not in the faithfulness avg
        {"type": "list", "skipped": "answer not found"},
        {"type": "list", "error": "APITimeoutError"},
    ]
    s = summarise(rows)
    assert s["overall"] == {"n": 3, "faithfulness": 0.75, "answer_relevancy": 0.7}
    assert s["by_type"]["skill"] == {"n": 2, "faithfulness": 0.75, "answer_relevancy": 0.7}
    assert s["skipped"] == 1 and s["errors"] == 1


def test_bge_adapter_uses_our_embeddings_without_the_query_prefix(monkeypatch):
    sent = []

    async def fake_embed(texts):
        sent.append(texts)
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ragas_adapters, "embed_texts", fake_embed)
    emb = ragas_adapters.BgeEmbeddings()
    assert asyncio.run(emb.aembed_text("who?")) == [4.0]
    assert asyncio.run(emb.aembed_texts(["a", "bb"])) == [[1.0], [2.0]]
    assert sent == [["who?"], ["a", "bb"]]  # one request for many texts, no "Represent this sentence..." prefix


def test_judge_name_goes_through_the_router():
    assert ragas_adapters.judge_model_name() == "openai/gpt-oss-120b:cerebras"
    assert ragas_adapters.judge_model_name("groq") == "openai/gpt-oss-120b:groq"


def test_ragas_still_has_the_faithfulness_steps_we_call():
    """The eval calls these private steps to keep per-claim verdicts; an upgrade that renames them fails here."""
    assert all(hasattr(Faithfulness, m) for m in ("_create_statements", "_create_verdicts"))
    assert ragas_eval.LOW_FAITHFULNESS == 0.9
