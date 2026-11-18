from app.services.resume_chunker import SKILLS_PER_CHUNK, make_chunks


def test_make_chunks():
    resume = {
        "name": "Jane Doe", "title": "Engineer", "summary": "Builds things.", "email": "jane@x.com",
        "skills": {"Languages": ["Python", "SQL"], "DevOps": ["Docker"]},
        "experience": [{"company": "Acme", "start": "2020", "end": "Present",
                        "details": ["Built pipelines"],
                        "projects": [{"name": "Search", "details": ["Added ranking"]},
                                     {"name": "Chat", "details": ["Used WebSockets"]}]}],
    }
    profile, languages, devops, acme, search, chat = make_chunks(resume)
    assert profile == "Profile\nName: Jane Doe\nTitle: Engineer\nEmail: jane@x.com\nSummary: Builds things."
    assert languages == "Skills - Languages: Python, SQL"
    assert devops == "Skills - DevOps: Docker"
    header = "Experience: Acme\nStart: 2020\nEnd: Present"
    assert acme == f"{header}\n- Built pipelines"  # company bullets outside any project
    assert search == f"{header}\nProject: Search\n- Added ranking"
    assert chat == f"{header}\nProject: Chat\n- Used WebSockets"


def test_company_without_projects_is_one_chunk():
    [_, acme] = make_chunks({"name": "J", "experience": [{"company": "Acme", "details": ["Did X"]}]})
    assert acme == "Experience: Acme\nStart: None\nEnd: None\n- Did X"


def test_long_skill_list_is_split_into_groups():
    skills = [f"skill{i}" for i in range(SKILLS_PER_CHUNK + 2)]  # e.g. an uncategorized resume
    chunks = make_chunks({"name": "J", "skills": {"Other": skills}})[1:]
    assert len(chunks) == 2
    assert chunks[0] == "Skills: " + ", ".join(skills[:SKILLS_PER_CHUNK])
    assert chunks[1] == "Skills: " + ", ".join(skills[SKILLS_PER_CHUNK:])
