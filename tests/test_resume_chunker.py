from app.services.resume_chunker import make_chunks


def test_make_chunks():
    resume = {
        "name": "Jane Doe", "title": "Engineer", "summary": "Builds things.", "email": "jane@x.com",
        "skills": {"Languages": ["Python", "SQL"]},
        "experience": [{"company": "Acme", "start": "2020", "end": "Present",
                        "details": ["Built pipelines"],
                        "projects": [{"name": "Search", "details": ["Added ranking"]}]}],
    }
    profile, skills, acme = make_chunks(resume)
    assert profile == "Profile\nName: Jane Doe\nTitle: Engineer\nEmail: jane@x.com\nSummary: Builds things."
    assert skills == "Skills\nLanguages: Python, SQL"
    assert "- Built pipelines" in acme and "Project: Search\n- Added ranking" in acme
