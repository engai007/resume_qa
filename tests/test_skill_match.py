"""Skill matching decides 65% of the match score, so every rule has a test, most of them a real false match."""
import pytest

from app.services.skill_match import alternatives, find_skill, match_skills

CHUNKS = [
    ("profile", "Profile\nName: Jane\nSummary: Backend engineer who likes to go the extra mile."),
    ("skills", "Skills - Languages: Python, SQL, Go, C++"),
    ("skills", "Skills - DevOps: Docker, k8s, GitHub Actions"),
    ("project", "Experience: Acme\nProject: Payments\n- Built REST APIs with FastAPI on Postgres\n"
                "- Worked with Google Cloud and the rest of the team"),
    ("project", "Experience: Beta\nProject: Search\n- Used MongoDB and JavaScript"),
]


def found(skill):
    ev = find_skill(skill, CHUNKS)
    return (ev.found_as, ev.section, ev.chunk_index) if ev else None


def test_found_in_the_skills_list():
    assert found("Python") == ("Python", "skills", 1)
    assert found("python") == ("Python", "skills", 1)  # case doesn't matter


def test_found_only_in_a_project_bullet():
    assert found("FastAPI") == ("FastAPI", "project", 3)  # people often list it only there


def test_skills_list_is_preferred_as_evidence():
    assert found("SQL")[1] == "skills"


@pytest.mark.parametrize("skill, alias", [("PostgreSQL", "Postgres"), ("Kubernetes", "k8s"),
                                          ("MongoDB", "MongoDB"), ("REST APIs", "REST APIs"),
                                          ("GCP", "Google Cloud")])
def test_aliases(skill, alias):
    assert found(skill)[0] == alias


@pytest.mark.parametrize("skill", [
    "Java",      # only "JavaScript" in the resume
    "Rust",      # not there
    "C",         # only "C++": another language
    "AWS",       # "Google Cloud" is not AWS
])
def test_no_false_matches(skill):
    assert found(skill) is None


@pytest.mark.parametrize("skill, word_text, name_text", [
    ("REST", "- Coordinated with the rest of the team", "- Built REST services"),
    ("React", "- Helped react to production incidents", "- Frontend in React and Redux"),
    ("Spring", "- Shipped the spring release on time", "- Java services on Spring Boot"),
    ("Express", "- Wanted to express ideas clearly", "- APIs in Node with Express"),
])
def test_common_words_need_a_capital(skill, word_text, name_text):
    assert find_skill(skill, [("project", word_text)]) is None       # the everyday word
    assert find_skill(skill, [("project", name_text)]) is not None   # the technology


def test_rest_apis_found_as_a_name():
    assert found("REST")[0] == "REST"  # "Built REST APIs"
    assert find_skill("Kafka", [("skills", "Skills: KAFKA, Spark")]).found_as == "KAFKA"


def test_short_names_count_only_in_the_skills_list():
    assert found("Go") == ("Go", "skills", 1)
    only_bullet = [("project", "Project: X\n- Helped the team go live")]
    assert find_skill("Go", only_bullet) is None  # "go live" is not the language
    assert find_skill("Golang", only_bullet) is None  # its short alias "go" too


def test_c_plus_plus_and_dotted_names():
    assert found("C++") == ("C++", "skills", 1)
    assert find_skill("Node.js", [("skills", "Skills: NodeJS, Express")]).found_as == "NodeJS"


@pytest.mark.parametrize("skill, alts", [
    ("Kafka/RabbitMQ", ["Kafka", "RabbitMQ"]),
    ("FastAPI or Django", ["FastAPI", "Django"]),
    ("CI/CD", ["CI/CD"]),          # one term, not a choice
    ("TCP/IP", ["TCP/IP"]),
])
def test_alternatives(skill, alts):
    assert alternatives(skill) == alts


def test_any_alternative_is_enough():
    assert found("Kafka/MongoDB")[0] == "MongoDB"
    assert found("Django or FastAPI")[0] == "FastAPI"


def test_match_skills_score_and_lists():
    part = match_skills(["Python", "FastAPI", "Rust", "Kubernetes"], CHUNKS)
    assert part["score"] == 0.75
    assert [m["skill"] for m in part["matched"]] == ["Python", "FastAPI", "Kubernetes"]
    assert part["missing"] == ["Rust"]
    assert part["matched"][1] == {"skill": "FastAPI", "found_as": "FastAPI", "section": "project", "chunk_index": 3}


def test_no_skills_in_the_jd_means_part_left_out():
    assert match_skills([], CHUNKS) is None
