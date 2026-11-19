from datetime import date

from app.services.resume_chunker import SECTIONS, SKILLS_PER_CHUNK, as_years, experience_summary, make_chunks, total_months

TODAY = date(2026, 11, 19)


def test_make_chunks():
    resume = {
        "name": "Jane Doe", "title": "Engineer", "summary": "Builds things.", "email": "jane@x.com",
        "skills": {"Languages": ["Python", "SQL"], "DevOps": ["Docker"]},
        "experience": [{"company": "Acme", "start": "2020", "end": "Present",
                        "details": ["Built pipelines"],
                        "projects": [{"name": "Search", "details": ["Added ranking"]},
                                     {"name": "Chat", "details": ["Used WebSockets"]}]}],
    }
    chunks = make_chunks(resume, TODAY)
    assert [section for section, _ in chunks] == [
        "profile", "skills", "skills", "experience", "project", "project", "experience_summary"]
    profile, languages, devops, acme, search, chat, summary = [text for _, text in chunks]
    assert profile == "Profile\nName: Jane Doe\nTitle: Engineer\nEmail: jane@x.com\nSummary: Builds things."
    assert languages == "Skills - Languages: Python, SQL"
    assert devops == "Skills - DevOps: Docker"
    header = "Experience: Acme\nStart: 2020\nEnd: Present"
    assert acme == f"{header}\n- Built pipelines"  # company bullets outside any project
    assert search == f"{header}\nProject: Search\n- Added ranking"
    assert chat == f"{header}\nProject: Chat\n- Used WebSockets"
    assert summary.startswith("Experience summary\nTotal work experience: 6 years 10 months")  # Jan 2020 - Nov 2026


def test_company_without_projects_is_one_chunk():
    [_, (_, acme), (_, summary)] = make_chunks({"name": "J", "experience": [{"company": "Acme", "details": ["Did X"]}]})
    assert acme == "Experience: Acme\nStart: None\nEnd: None\n- Did X"
    assert summary == "Experience summary\nTotal work experience: unknown (job dates unclear)"


def test_long_skill_list_is_split_into_groups():
    skills = [f"skill{i}" for i in range(SKILLS_PER_CHUNK + 2)]  # e.g. an uncategorized resume
    chunks = [text for _, text in make_chunks({"name": "J", "skills": {"Other": skills}})[1:]]
    assert len(chunks) == 2
    assert chunks[0] == "Skills: " + ", ".join(skills[:SKILLS_PER_CHUNK])
    assert chunks[1] == "Skills: " + ", ".join(skills[SKILLS_PER_CHUNK:])


def test_experience_summary_adds_up_jobs():
    resume = {  # dates as the parser gives them (real resume shapes), newest first
        "experience": [
            {"company": "Verticurl", "start": "Feb 2026", "end": "present"},
            {"company": "Infinite", "start": "June 2024", "end": "Feb 2026"},
            {"company": "Edge", "start": "June 2023", "end": "Feb 2024"},
            {"company": "SupplyCopia", "start": "December 2022", "end": "Jun 2023"},
            {"company": "Tradeindia", "start": "Mar 2020", "end": "Dec 2022"},
        ],
        "projects": [{"name": "AI Job Search Agent", "start": "Feb 2026", "end": "Aug 2026"}],
    }
    # 9 + 20 + 8 + 6 + 33 = 76 months; the Feb-Jun 2024 gap and the personal project don't count
    assert experience_summary(resume, TODAY) == (
        "Experience summary\n"
        "Total work experience: 6 years 4 months (company jobs only, personal projects not counted; as of Nov 2026)\n"
        "Working since: Mar 2020, at 5 companies")  # no company names: they made it match every company question


def test_overlapping_jobs_count_once():
    assert total_months([(0, 12), (6, 18)]) == 18   # Jan-Jan and Jul-Jul next year: 18, not 24
    assert total_months([(0, 12), (24, 30)]) == 18  # a gap is not counted
    assert total_months([(0, 24), (3, 6)]) == 24    # a side job inside a main job


def test_date_formats():
    jobs = [{"company": "A", "start": "03/2018", "end": "2018-09"},   # 6 months
            {"company": "B", "start": "Sept. 2019", "end": "2019"},   # Sep - Dec = 3
            {"company": "C", "start": "N/A", "end": "2020"}]          # unreadable, not counted
    summary = experience_summary({"experience": jobs}, TODAY)
    assert "Total work experience: 9 months" in summary
    assert summary.endswith("Working since: Mar 2018, at 3 companies; 1 with unclear dates not counted")


def test_as_years():
    assert [as_years(m) for m in (0, 1, 9, 12, 13, 80)] == [
        "0 months", "1 month", "9 months", "1 year", "1 year 1 month", "6 years 8 months"]


def test_personal_project_is_labelled_and_no_summary_without_jobs():
    chunks = make_chunks({"name": "J", "projects": [{"name": "Bot", "start": "2025", "end": "2026", "details": ["x"]}]})
    assert chunks[1] == ("personal_project", "Personal project: Bot\nStart: 2025\nEnd: 2026\n- x")
    assert len(chunks) == 2  # profile + project, no experience summary


def test_every_section_is_known():
    resume = {"name": "J", "skills": {"Other": ["x"]}, "education": [{"degree": "BSc"}],
              "experience": [{"company": "A", "start": "2020", "end": "2021", "details": ["d"],
                              "projects": [{"name": "P"}]}],
              "projects": [{"name": "Bot"}], "certifications": ["AWS"], "publications": ["Paper"],
              "awards": ["Best"], "other_sections": [{"heading": "Languages", "details": ["English"]}]}
    assert {section for section, _ in make_chunks(resume, TODAY)} == set(SECTIONS)
