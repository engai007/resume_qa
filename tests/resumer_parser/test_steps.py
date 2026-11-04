"""Unit tests for the pure (no PDF, no LLM) parsing steps."""
from app.services.resumer_parser.builder import build_resume, lines_in, parse_skills
from app.services.resumer_parser.contact import extract_contact, find_phone
from app.services.resumer_parser.llm_input import compact_numbered
from app.services.resumer_parser.pdf_text import merge_wrapped
from app.services.resumer_parser.ranges import fix_ranges
from app.llm.json_output import extract_json as parse_json


def test_merge_wrapped_joins_continuations():
    lines = [{"text": "Built a data"}, {"text": "pipeline in Python,"}, {"text": "SQL"}, {"text": "Skills"}]
    assert merge_wrapped(lines) == ["Built a data pipeline in Python, SQL", "Skills"]


def test_contact():
    texts = ["Jane Doe", "jane@x.com | +91 98765 43210 | linkedin.com/in/jane | github.com/jane"]
    assert extract_contact(texts) == {
        "email": "jane@x.com",
        "phone": "+91 98765 43210",
        "linkedin": "linkedin.com/in/jane",
        "github": "github.com/jane",
    }


def test_find_phone_ignores_date_ranges():
    assert find_phone("Acme 2020 - 2022") is None


def test_compact_numbered():
    texts = [
        "jane@x.com",                                         # only contact -> dropped
        "• Acme Corp Jan 2020 - Present Senior Engineer Bangalore India",  # date line kept whole
        "∗ one two three four five six seven eight nine ten",
    ]
    assert compact_numbered(texts, max_words=8).splitlines() == [
        "1|• Acme Corp Jan 2020 - Present Senior Engineer Bangalore India",
        "2|∗ one two three four five six seven eight…",
    ]


def test_parse_json_strips_think_and_fences():
    raw = '<think>hmm</think>\n```json\n{"name": "Jane"}\n```'
    assert parse_json(raw) == {"name": "Jane"}


def test_parse_skills():
    assert parse_skills(["Languages: Python, SQL", "Docker, Git"]) == {
        "Languages": ["Python", "SQL"],
        "Other": ["Docker", "Git"],
    }


TEXTS = [
    "Jane Doe",                       # 0
    "Experience",                     # 1
    "Acme Corp Jan 2020 - Present",   # 2
    "• Built data pipelines",         # 3
    "• Led a team of five engineers", # 4
    "Skills",                         # 5
    "Languages: Python, SQL",         # 6
]


def test_fix_ranges_trims_headers_and_overlaps():
    parsed = {"experience": [{"company": "Acme", "lines": [2, 6]}], "skills": [5, 6]}
    fix_ranges(parsed, TEXTS)
    assert parsed["experience"][0]["lines"] == [3, 4]
    assert parsed["skills"] == [6, 6]


def test_fix_ranges_clamps_out_of_bounds():
    parsed = {"skills": [6, 99]}
    fix_ranges(parsed, TEXTS)
    assert parsed["skills"] == [6, 6]


def test_lines_in_strips_bullets():
    assert lines_in([3, 4], TEXTS) == ["Built data pipelines", "Led a team of five engineers"]


def test_build_resume():
    parsed = {"name": "Jane Doe", "experience": [{"company": "Acme", "lines": [3, 4]}], "skills": [6, 6]}
    resume = build_resume(parsed, {"email": "jane@x.com"}, TEXTS)
    assert resume["email"] == "jane@x.com"
    assert resume["experience"] == [{"company": "Acme", "details": ["Built data pipelines",
                                                                    "Led a team of five engineers"]}]
    assert resume["skills"] == {"Languages": ["Python", "SQL"]}
