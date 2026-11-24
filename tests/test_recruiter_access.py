"""Recruiters read every user's resumes (search, Q&A, matching, list); candidates only their own.
Deleting stays owner-only for everyone."""
from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.services.document_service import file_hash
from tests.conftest import auth_header
from tests.test_match import add_jd, add_resume, match
from tests.test_retriever import search


def people(client):
    """two candidates with one resume each, and a recruiter with none."""
    cand_a = auth_header(client, email="a@x.com", user_type="candidate")
    cand_b = auth_header(client, email="b@x.com", user_type="candidate")
    recruiter = auth_header(client, email="r@x.com", user_type="recruiter")
    doc_a, hash_a = add_resume(client, cand_a, "Languages: Python, SQL, Kafka")
    doc_b, hash_b = add_resume(client, cand_b, "Languages: Java")
    return cand_a, cand_b, recruiter, (doc_a, hash_a), (doc_b, hash_b)


def test_recruiter_matches_a_jd_against_every_users_resumes(client):
    cand_a, _, recruiter, (doc_a, _), (doc_b, hash_b) = people(client)
    body = match(client, recruiter, add_jd(client, recruiter)).json()
    assert body["resumes_compared"] == 2
    assert [r["document_id"] for r in body["results"]] == [doc_a, doc_b]  # ranked: a has the skills

    only_b = match(client, recruiter, add_jd(client, recruiter), document_hashes=[hash_b]).json()
    assert [r["document_id"] for r in only_b["results"]] == [doc_b]  # hashes still narrow it


def test_candidate_matches_only_own_resumes(client):
    cand_a, _, _, (doc_a, _), (_, hash_b) = people(client)
    jd_id = add_jd(client, cand_a)
    assert [r["document_id"] for r in match(client, cand_a, jd_id).json()["results"]] == [doc_a]
    assert match(client, cand_a, jd_id, document_hashes=[hash_b]).json()["resumes_compared"] == 0


def test_recruiter_searches_and_lists_every_resume(client):
    cand_a, _, recruiter, (doc_a, _), (doc_b, _) = people(client)
    assert {r["document_id"] for r in search(client, recruiter).json()["results"]} == {doc_a, doc_b}
    assert {r["document_id"] for r in search(client, cand_a).json()["results"]} == {doc_a}

    listed = client.get("/api/v1/documents/", headers=recruiter).json()
    assert {d["document_id"] for d in listed} == {doc_a, doc_b}
    assert not any(d["uploaded_by_me"] for d in listed)
    [own] = client.get("/api/v1/documents/", headers=cand_a).json()
    assert own["document_id"] == doc_a and own["uploaded_by_me"] is True


def test_recruiter_cannot_delete_other_users_resumes(client):
    cand_a, _, recruiter, (doc_a, _), _ = people(client)
    assert client.delete(f"/api/v1/documents/{doc_a}", headers=recruiter).status_code == 404
    assert client.delete(f"/api/v1/documents/{doc_a}", headers=cand_a).status_code == 200


def test_jds_stay_private_even_for_recruiters(client):
    cand_a, _, recruiter, _, _ = people(client)
    jd_id = add_jd(client, cand_a)
    assert match(client, recruiter, jd_id).status_code == 404  # a JD is its owner's only
