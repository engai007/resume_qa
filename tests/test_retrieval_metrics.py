from app.services.retrieval_metrics import average, is_match, score_no_answer, score_question


def chunk(doc, text, score=0.5):
    return {"document_hash": doc, "text": text, "score": score}


EXPECTED = [  # like golden q04: 6 expected chunks
    {"document_hash": "A", "contains": "Edge"}, {"document_hash": "B", "contains": "Edge"},
    {"document_hash": "A", "contains": "Domain"}, {"document_hash": "B", "contains": "Domain"},
    {"document_hash": "A", "contains": "Skills"}, {"document_hash": "B", "contains": "Skills"},
]


def test_is_match_needs_same_document_and_snippet():
    item = {"document_hash": "A", "contains": "Kafka"}
    assert is_match(chunk("A", "built Kafka pipelines"), item)
    assert not is_match(chunk("B", "built Kafka pipelines"), item)  # other resume
    assert not is_match(chunk("A", "built kafka pipelines"), item)  # case-sensitive


def test_score_question_q04_example():
    results = [chunk("A", "Experience: Edge"), chunk("B", "Experience: Edge"), chunk("A", "Domain Knowledge"),
               chunk("A", "Experience: Infinite"), chunk("B", "Domain Knowledge")]
    s = score_question(results, EXPECTED, k=5)
    assert s["hit"] == 1.0
    assert s["recall"] == 4 / 6
    assert s["precision"] == 4 / 5
    assert s["mrr"] == 1.0
    assert [m["contains"] for m in s["missed"]] == ["Skills", "Skills"]


def test_score_question_first_correct_lower_down():
    results = [chunk("A", "Profile"), chunk("A", "Awards"), chunk("A", "Education: DTU")]
    s = score_question(results, [{"document_hash": "A", "contains": "DTU"}], k=5)
    assert (s["hit"], s["recall"], s["mrr"], s["first_correct_rank"]) == (1.0, 1.0, 1 / 3, 3)
    assert s["precision"] == 1 / 5  # divided by k, not by how many came back


def test_score_question_only_counts_top_k():
    results = [chunk("A", "Profile"), chunk("A", "Education: DTU")]
    s = score_question(results, [{"document_hash": "A", "contains": "DTU"}], k=1)
    assert (s["hit"], s["recall"], s["mrr"]) == (0.0, 0.0, 0.0)


def test_score_no_answer():
    assert score_no_answer([chunk("A", "x", score=0.30)], threshold=0.45)["passed"] is True
    assert score_no_answer([chunk("A", "x", score=0.60)], threshold=0.45)["passed"] is False
    assert score_no_answer([], threshold=0.45)["passed"] is True


def test_average():
    assert average([{"hit": 1, "recall": 1, "precision": 0.4, "mrr": 1},
                    {"hit": 0, "recall": 0, "precision": 0, "mrr": 0}]) == \
        {"hit": 0.5, "recall": 0.5, "precision": 0.2, "mrr": 0.5}
