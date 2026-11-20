"""Section filter (search) and fetch_sections (every chunk of a section), always inside the user's access."""
import asyncio

import pytest

from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.models import User
from app.services.document_service import file_hash
from app.services.retriever_service import fetch_sections
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf
from tests.test_documents import fake_agent, upload
from tests.test_retriever import search

# the test resume gives 4 chunks: 0 profile, 1 skills, 2 experience (Acme, no projects), 3 experience_summary
PDF_A, PDF_B = make_pdf(RESUME_LINES), make_pdf(RESUME_LINES + ["Extra line"])


def me(client, headers) -> User:
    return User(id=client.get("/api/v1/auth/me", headers=headers).json()["id"])


def fetch(user, sections, **kw):
    return asyncio.run(fetch_sections(user, sections, **kw))


@pytest.fixture
def alice_and_bob(client):
    """alice uploads two resumes, bob one."""
    app.dependency_overrides[get_parser_agent] = fake_agent
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    upload(client, alice, ("a1.pdf", PDF_A), ("a2.pdf", PDF_B))
    upload(client, bob, ("b.pdf", PDF_A))
    return alice, bob


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_with_a_section_filter(client, alice_and_bob, mode):
    alice, _ = alice_and_bob
    results = search(client, alice, query="Python", sections=["skills"], mode=mode).json()["results"]
    assert results and {r["section"] for r in results} == {"skills"}
    assert len(results) == 2  # one skills chunk in each of alice's two resumes, none of bob's


def test_unknown_section_is_rejected(client):
    r = search(client, auth_header(client), sections=["jobs"])
    assert r.status_code == 422 and "unknown sections" in r.text


def test_fetch_sections_returns_every_match_in_resume_order(client, alice_and_bob):
    alice, _ = alice_and_bob
    chunks, truncated = fetch(me(client, alice), ["experience_summary", "experience"])

    assert not truncated
    assert [(c["section"], c["chunk_index"]) for c in chunks] == [
        ("experience", 2), ("experience_summary", 3),   # resume 1
        ("experience", 2), ("experience_summary", 3),   # resume 2
    ]
    assert chunks[0]["document_id"] < chunks[2]["document_id"]
    assert all(c["score"] is None for c in chunks)  # not ranked


def test_fetch_sections_for_chosen_resumes_only(client, alice_and_bob):
    alice, _ = alice_and_bob
    chunks, _ = fetch(me(client, alice), ["skills"], document_hashes=[file_hash(PDF_B)])
    assert [c["document_hash"] for c in chunks] == [file_hash(PDF_B)]


def test_limit_keeps_whole_resumes_first(client, alice_and_bob):
    alice, _ = alice_and_bob
    chunks, truncated = fetch(me(client, alice), ["profile", "skills"], limit=3)
    assert truncated
    assert [c["section"] for c in chunks] == ["profile", "skills", "profile"]  # all of resume 1, then resume 2


def test_someone_elses_resume_hash_finds_nothing(client, alice_and_bob):
    """document_hashes can narrow access, never widen it: bob passes alice's resume hash."""
    alice, bob = alice_and_bob
    alice_only = file_hash(PDF_B)  # bob uploaded PDF_A too, but not PDF_B
    assert search(client, bob, document_hashes=[alice_only]).json()["results"] == []
    assert fetch(me(client, bob), ["skills"], document_hashes=[alice_only]) == ([], False)


def test_each_user_fetches_only_their_own(client, alice_and_bob):
    alice, bob = alice_and_bob
    assert len(fetch(me(client, alice), ["profile"])[0]) == 2
    bobs, _ = fetch(me(client, bob), ["profile"])
    assert len(bobs) == 1 and bobs[0]["document_hash"] == file_hash(PDF_A)


def test_fetch_sections_needs_a_section(client):
    with pytest.raises(ValueError):
        fetch(User(id=1), [])
