"""Retrieval metrics: plain counting, no LLM, same input -> same score.

`results`  = search output, best first: [{"document_hash", "text", "score", ...}]
`expected` = golden items:              [{"document_hash", "contains"}]

Metrics (k = how many results we look at, e.g. top 5):

- Hit@k       Did at least one right chunk show up in the top k?  1 = yes, 0 = no.
              "Is search usable at all for this question?"
- Recall@k    Share of ALL right chunks that are in the top k.  2 of 3 found -> 0.67.
              Most important for RAG: the answer can only use what was retrieved.
- Precision@k Share of the top k that are right.  2 right out of 5 -> 0.40.
              Measures noise. If a question has only 1 right chunk, the best possible is 1/k,
              so always read it together with recall.
- MRR         1 / rank of the first right chunk.  #1 -> 1.0, #2 -> 0.5, #4 -> 0.25, none -> 0.
              "Is the best match near the top?"
- No-answer   For questions with no right chunk: passes if the best score is below a threshold.
              "Does search stay unsure when there is nothing to find?"
"""


def is_match(chunk: dict, item: dict) -> bool:
    return chunk["document_hash"] == item["document_hash"] and item["contains"] in chunk["text"]


def score_question(results: list[dict], expected: list[dict], k: int) -> dict:
    """Hit@k, Recall@k, Precision@k and MRR for one question that has an answer."""
    top = results[:k]
    correct = [any(is_match(c, e) for e in expected) for c in top]  # one True/False per returned chunk
    found = [e for e in expected if any(is_match(c, e) for c in top)]
    first = next((rank for rank, ok in enumerate(correct, start=1) if ok), None)
    return {
        "hit": 1.0 if first else 0.0,
        "recall": len(found) / len(expected),
        "precision": sum(correct) / k,
        "mrr": 1 / first if first else 0.0,
        "first_correct_rank": first,
        "missed": [e for e in expected if e not in found],
    }


def score_no_answer(results: list[dict], threshold: float) -> dict:
    """A question with no answer passes when even the best chunk scores below the threshold."""
    top_score = results[0]["score"] if results else 0.0
    return {"passed": top_score < threshold, "top_score": top_score}


METRICS = ("hit", "recall", "precision", "mrr")


def average(scores: list[dict]) -> dict:
    """Mean of each metric over several questions."""
    if not scores:
        return {m: None for m in METRICS}
    return {m: round(sum(s[m] for s in scores) / len(scores), 3) for m in METRICS}
