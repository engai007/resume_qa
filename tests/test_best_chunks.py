"""best_chunks_per_resume: each resume's closest chunks to a text, one grouped Qdrant call, inside the user's access."""
import asyncio

from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.services.document_service import file_hash
from app.services.match_scoring import jd_focus_text
from app.services.retriever_service import best_chunks_per_resume
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf
from tests.test_documents import fake_agent, upload
from tests.test_fetch_sections import PDF_A, PDF_B, me


def best(user, **kw):
    return asyncio.run(best_chunks_per_resume(user, "Python data pipelines", **kw))


def setup(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    upload(client, alice, ("a1.pdf", PDF_A), ("a2.pdf", PDF_B))
    upload(client, bob, ("b.pdf", make_pdf(RESUME_LINES + ["Bob only"])))
    return me(client, alice), me(client, bob)


def test_one_group_per_resume_best_first(client):
    alice, _ = setup(client)
    groups = best(alice, group_size=2)
    assert len(groups) == 2  # alice's two resumes, not bob's
    for doc_id, hits in groups.items():
        assert len(hits) == 2 and all(h["document_id"] == doc_id for h in hits)
        assert hits[0]["score"] >= hits[1]["score"]


def test_sections_and_resume_filters(client):
    alice, _ = setup(client)
    groups = best(alice, sections=["skills", "experience"], group_size=5)
    assert {h["section"] for hits in groups.values() for h in hits} <= {"skills", "experience"}

    only = best(alice, document_hashes=[file_hash(PDF_B)])
    assert [hits[0]["document_hash"] for hits in only.values()] == [file_hash(PDF_B)]


def test_only_resumes_the_user_can_access(client):
    alice, bob = setup(client)
    assert len(best(bob)) == 1
    assert best(bob, document_hashes=[file_hash(PDF_B)]) == {}  # alice's resume hash: nothing


def test_jd_focus_text_keeps_the_work_parts_only():
    jd = {"title": "Data Engineer", "company": "Acme", "summary": "Build pipelines.", "salary": {"min": 1},
          "responsibilities": ["Build ETL", "Own the warehouse"], "required_skills": ["Spark", "SQL"]}
    assert jd_focus_text(jd) == "Data Engineer. Build pipelines. Build ETL; Own the warehouse. Spark, SQL"
    assert jd_focus_text({}) == ""
    assert len(jd_focus_text({"responsibilities": ["x" * 3000]})) == 1500
