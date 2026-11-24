from datetime import date

import pytest

from app.services.match_scoring import (WEIGHTS, candidate_months, combine, experience_part, score_resume,
                                        semantic_part)

TODAY = date(2026, 11, 24)
RESUME = {
    "name": "Jane",
    "skills": {"Languages": ["Python", "SQL"], "DevOps": ["Docker"]},
    "experience": [
        {"company": "Acme", "start": "Jan 2020", "end": "Dec 2022", "projects": [
            {"name": "Payments", "details": ["Built REST APIs with FastAPI"]}]},
        {"company": "Beta", "start": "Jan 2023", "end": "present", "details": ["Ran Kafka pipelines"]},
    ],
    "projects": [{"name": "Bot", "details": ["Personal LLM bot"]}],
}


def test_candidate_months_uses_the_summary_rules():
    assert candidate_months(RESUME, TODAY) == 35 + 46  # Jan 2020-Dec 2022, Jan 2023-Nov 2026
    assert candidate_months({"experience": [{"company": "X"}]}, TODAY) is None  # no readable dates


@pytest.mark.parametrize("years, score", [
    ({"min": 5, "max": None}, 1.0),      # 6.8 years >= 5
    ({"min": 10, "max": None}, 0.68),    # 6.8 / 10
    ({"min": 3, "max": 5}, 1.0),         # over the max is not penalised
])
def test_experience_part(years, score):
    part = experience_part(RESUME, years, TODAY)
    assert part["score"] == score and part["candidate_years"] == 6.8 and part["required_min"] == years["min"]


def test_experience_left_out_without_a_minimum():
    assert experience_part(RESUME, None, TODAY) is None
    assert experience_part(RESUME, {"min": None, "max": 5}, TODAY) is None


def test_experience_unknown_dates_score_zero():
    assert experience_part({"experience": [{"company": "X"}]}, {"min": 3}, TODAY)["score"] == 0.0


def test_combine_rescales_missing_parts():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert combine({"required": 1.0, "preferred": 1.0, "experience": 1.0, "semantic": 1.0}) == 1.0
    # only required + experience given: 0.5*1 + 0.15*0 over 0.65
    assert combine({"required": 1.0, "preferred": None, "experience": 0.0, "semantic": None}) == round(0.5 / 0.65, 4)
    assert combine({"required": None, "preferred": None, "experience": None, "semantic": None}) == 0.0


def test_score_resume_end_to_end():
    jd = {"required_skills": ["Python", "FastAPI", "Rust"], "preferred_skills": ["Kafka"],
          "experience_years": {"min": 5, "max": None}}
    out = score_resume(RESUME, jd, similarity=0.64, best_chunk_index=2, today=TODAY)
    b = out["breakdown"]
    assert b["required"]["missing"] == ["Rust"] and b["required"]["score"] == round(2 / 3, 4)
    assert b["preferred"]["matched"][0]["section"] == "experience"  # "Ran Kafka pipelines", a company bullet
    assert b["experience"]["score"] == 1.0
    assert b["semantic"] == {"score": 0.6, "similarity": 0.64, "best_chunk_index": 2}  # (0.64 - 0.40) / 0.40
    expected = 0.5 * (2 / 3) + 0.15 * 1 + 0.15 * 1 + 0.2 * 0.6
    assert out["score"] == round(expected, 4)


@pytest.mark.parametrize("similarity, score", [
    (0.41, 0.025),   # ICU nurse JD vs a software resume (measured): ~0
    (0.59, 0.475),   # payments JD vs a backend resume
    (0.763, 0.9075), # LLM JD vs a resume with an LLM project
    (0.95, 1.0),     # capped
    (0.20, 0.0),
])
def test_semantic_stretch(similarity, score):
    assert semantic_part(similarity)["score"] == score


def test_semantic_left_out_without_work_chunks():
    assert semantic_part(None) is None


def test_requirements_only_jd_is_not_penalised():
    out = score_resume(RESUME, {"required_skills": ["Python", "SQL"]}, today=TODAY)
    assert out["score"] == 1.0  # every required skill found; the parts the JD doesn't give are left out
    assert out["breakdown"]["preferred"] is None and out["breakdown"]["experience"] is None
