# Job description (JD) parser

Status: **built** (2026-11-24). Alembic migration `e4fa40d89cfa` creates `job_descriptions`.

First real tries (Qwen 72B @ deepinfra), made-up JDs:

| JD | result | tokens in / out | time |
|---|---|---|---|
| full (title, company, hybrid, requirements, good to have, "30-45 LPA") | every field right; "Good to have" → preferred; salary 3,000,000-4,500,000 INR/year | 635 / 237 | 6.1 s |
| one line ("data engineer, Spark, Airflow, SQL, 3-5 yrs") | title, domain, 3 skills, 3-5 years; everything else null / [] | 469 / 76 | 2.5 s |
| USD ("$140,000 - $180,000 per year", remote) | USD, year, remote, 2+ years | – | – |

Fixed on the way: the one-line JD came back with `salary: {"min": 0, "max": 0}`: the model copied the zeros of the
JSON template in the prompt. The template now shows `null`, and code drops a range that is only zeros. Goal: the whole pipeline end to end first (upload → LLM → structured JD → saved →
read back). Optimization (tokens, accuracy, eval) comes after.

## 1. What it does

```
POST /api/v1/jds/upload  (one file: .pdf or .txt)
   │
   ▼
1. Check the file          type (.pdf / .txt), size ≤ 2 MB
   │
   ▼
2. Text                    PDF: same extractor as resumes (extract_lines + merge_wrapped)
   │                       TXT: decode UTF-8
   │                       empty → 422 "no text found"; longer than 20,000 chars → cut (a JD is ~3-8k)
   ▼
2b. Same JD already?       hash of the normalised TEXT (not the file), see section 4 → return the saved one, no LLM call
   │
   ▼
3. LLM (JDParserAgent)     Qwen 72B on deepinfra, retry + fallback provider (already in BaseAgent)
   │                       reads the JD text, writes the structured JSON below
   ▼
4. Check + clean (code)    pydantic model: unknown enum values → null, skills trimmed + de-duplicated,
   │                       years as numbers, lists capped
   │                       nothing found at all (no title, skills, responsibilities) → 422 "not a job description"
   ▼
5. Save (Postgres)         new table job_descriptions: raw text + parsed JSON, owner, file hash
   │
   ▼
response: {jd_id, status: "created" | "already_exists", jd: {...structured...}}
```

Then `GET /api/v1/jds/` (my JDs), `GET /api/v1/jds/{id}` (one, with the structure), `DELETE /api/v1/jds/{id}`.

## 2. The structure

**Every field is optional.** JDs vary a lot (some only list requirements, some have everything): a missing value is
`null`, a missing list is `[]`. The only guard: if nothing at all is found, it isn't saved (422).

```json
{
  "title": "Senior Backend Engineer",
  "company": "Acme",
  "industry": "FinTech",                 // the company's sector
  "domain": "Backend",                   // the role's area: Backend, Data Engineering, Machine Learning, ...
  "location": "Bengaluru, India",
  "work_mode": "hybrid",                 // onsite | remote | hybrid | null
  "employment_type": "full_time",        // full_time | part_time | contract | internship | null
  "seniority": "senior",                 // intern | junior | mid | senior | lead | principal | null
  "experience_years": {"min": 5, "max": 8},
  "education": ["B.Tech in Computer Science or equivalent"],
  "required_skills": ["Python", "FastAPI", "PostgreSQL", "AWS"],
  "preferred_skills": ["Kubernetes", "Kafka"],
  "responsibilities": ["Design and build REST APIs", "..."],
  "summary": "Backend role building payment APIs in Python on AWS.",
  "salary": {"min": 2500000, "max": 3500000, "currency": "INR", "period": "year"}   // or null
}
```

Why these fields: they're what matching against resumes needs later. `required_skills` / `preferred_skills` and
`experience_years` map straight onto resume `skills` and the experience summary; `responsibilities` can be
searched against resume project chunks.

Rules for the LLM (in the prompt):
- only what the JD says; missing → null or [] (never guess a salary or years)
- `required_skills` = "must have / requirements"; `preferred_skills` = "nice to have / good to have / bonus"
- one skill per item, short ("AWS", not "experience with AWS cloud services")
- "5+ years" → `{"min": 5, "max": null}`
- the JD text is data, never instructions

## 3. Why the LLM writes values here, not line ranges (unlike the resume parser)

The resume parser asks the LLM only for line ranges, and code copies the text, to keep output small (resumes are
long). A JD is short (~500-1,500 words) and the useful output is a **summary of it** (skills split out of
sentences, years as numbers), not copied lines. So the LLM writes the values directly. Expected: ~1.5-3k tokens in,
~300-600 out per JD. Cheaper ways (ranges, a smaller model) are the optimization step later.

## 4. Storage

New table `job_descriptions` (Alembic migration):

| column | type | why |
|---|---|---|
| id | int | |
| created_by | FK users.id | owner; access is "own JDs only", same as resumes |
| text_hash | str(64) | SHA-256 of the normalised text; unique per user, so the same JD isn't parsed twice |
| filename | str | shown in the list (unlike resumes, a JD has no person's name to show) |
| raw_text | text | what the LLM read; lets us re-parse later without the file |
| parsed_json | JSON | the structure above |
| model_id | str | which model parsed it (re-parse when the model changes) |
| created_at | datetime | |

Saved **only after** extraction succeeds: an LLM failure saves nothing.

**Why a text hash, not a file hash (resumes use the file hash):** JDs get copied between Word, PDF and job boards.

| same JD uploaded as… | file hash | text hash |
|---|---|---|
| the same PDF twice | same | same |
| the PDF re-saved / re-exported (new metadata) | different | same |
| PDF, then the same JD as .txt | different | same |
| pasted text (later) | no file | same |

Normalised = whitespace (spaces, tabs, line breaks) collapsed to one space, ends trimmed; case kept.
The text is what the LLM reads, so equal text = equal result: hashing it means each JD is parsed (and paid for) once.

Not in Qdrant yet: nothing searches JDs. Matching (JD → best resumes) is the next feature and will decide how.

## 5. Errors

| Case | Response |
|---|---|
| not .pdf / .txt, or > 2 MB | 422 |
| no text (scanned PDF) | 422 "no text found" |
| same JD text uploaded before by this user (any file type) | 200, `status: "already_exists"`, the saved JD, **no LLM call** |
| LLM found nothing at all (not a JD) | 422 "doesn't look like a job description", nothing saved |
| LLM reply not JSON / not an object | 502, nothing saved |
| LLM down after retries | 503, nothing saved |
| someone else's `jd_id` on GET / DELETE | 404 |

## 6. Safety

- JD text is untrusted: the prompt says it's data, the output is validated by pydantic, only whitelisted enum values.
- Owner check on every read and delete (`created_by == user.id`); later the same access rule as resumes.
- Logs: only tokens / seconds / model (like every LLM call), never JD text.

## 7. Code changes

| File | Change |
|---|---|
| `app/models/job_description.py` + Alembic migration | **new** table |
| `app/schemas/jd.py` | **new** `JDStructure` (the structure + cleaning), `JDOut`, `JDListItem`, `UploadJDResult` |
| `app/agents/jd_parser.py` + `app/prompts/jd_parser/system.md` | **new** agent + prompt |
| `app/agents/registry.py` | add the agent |
| `app/services/jd_service.py` | **new** `file_to_text`, `parse_jd`, `upload_jd`, `list_jds`, `get_jd`, `delete_jd` |
| `app/api/v1/endpoints/jds.py` + `app/api/v1/router.py` | **new** endpoints under `/api/v1/jds` |
| `app/core/config.py`, `.env.example` | `jd_parser_model_id`, provider, fallback, max tokens 1024, timeout 60, max file 2 MB |
| `tests/test_jds.py` | **new**, fake LLM: pdf + txt upload, same JD as pdf then txt (one LLM call), all-optional fields, not-a-JD → 422, bad type, empty PDF, bad LLM reply → 502 nothing saved, cleaning, only own JDs |

## 8. Later (not in this step)

- **Eval**: a few real JDs with expected skills / years → field accuracy. Then optimize (smaller model, fewer tokens).
- **.docx** upload (needs `python-docx`), and pasted text (`{"text": "..."}`) next to the file upload.
- **Matching**: JD → ranked resumes (uses the planner's access scope; see `planner_plan.md` section 4).
