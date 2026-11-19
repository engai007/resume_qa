# Experience summary chunk

Status: **built** (INDEX_VERSION 5, migration `0006`). The same rebuild also adds a `section` field to every chunk (below).

## Problem

"What is the total working experience?" pulled in the personal project chunk:

```
Project: AI Job Search Agent
Start: Feb 2026
End: Aug 2026
- Built a multi-agent job-search system ...
```

Two reasons:

1. **No chunk holds the answer.** The total is a sum over all jobs, and every job is a separate chunk.
   Search finds similar chunks; it can't add them up. So the 5 slots fill with anything that looks like work history.
2. **A personal project looked like a job.** Same shape as a company project chunk (name, start, end, "Built ..."),
   only the `Experience: <company>` line was missing. Nothing in it said "personal".

If the LLM sees that chunk, it may count Feb-Aug 2026 as work experience.

## Fix

Both changes are in `make_chunks()` (`app/services/resume_chunker.py`), so they happen at upload time:

```
PDF -> parser (LLM) -> resume JSON -> make_chunks() -> embed -> Qdrant
                                        ^ here
```

**1. One extra "Experience summary" chunk per resume**, calculated in Python from `resume["experience"]`:

```
Experience summary
Total work experience: 6 years 4 months (company jobs only, personal projects not counted; as of Nov 2026)
Working since: Mar 2020, at 5 companies
```

Only the total, **no per-company lines**. A first version listed every company with its dates; then the chunk
matched every "work at <company>" question and pushed real chunks down (eval: q12 "Work done at Infinite" and
q20 "recognition at work" got worse). Each company's dates are already in its own `Experience:` chunk.

The LLM only reads the number; it never adds dates itself.

**2. Personal projects are labelled** `Personal project: AI Job Search Agent` (was `Project: ...`).
Company projects keep `Project: ...` inside their `Experience: <company>` chunk.

**3. Every chunk gets a `section` payload field**, so Qdrant can filter on it (keyword index):

| section | chunk |
|---|---|
| `profile` | name, title, contacts, summary |
| `skills` | one skill category (or a group of 8) |
| `experience` | company bullets that aren't under a project |
| `project` | one project at a company |
| `personal_project` | a project outside any company |
| `education` | one degree |
| `certifications`, `publications`, `awards`, `other` | those sections |
| `experience_summary` | the calculated total |

`make_chunks()` returns `(section, text)` pairs; `section` is also in search results (`null` for old collections).
Not used by search yet: it's ready for "list every project of resume 7" (filter + scroll instead of top 5),
see the routing idea in the next plan.

## How the total is counted

| Rule | Example |
|---|---|
| A job is `start` to `end`, in months | Mar 2020 - Dec 2022 = 33 months |
| Overlapping jobs count once (merged) | Dec 2022 - Jun 2023 and Jun 2023 - Feb 2024 share no extra month |
| Gaps between jobs are not counted | Feb 2024 - Jun 2024 = gap, 0 |
| `present` / `current` / `now` / `till date` = the indexing date | Feb 2026 - present, indexed Nov 2026 = 9 months |
| Year only: start = January, end = December | 2018 - 2019 = 23 months |
| Dates it can't read (empty, "N/A") | left out of the total, the chunk says "1 with unclear dates not counted" |
| Personal projects (`resume["projects"]`) | never counted |

Date formats read: `Mar 2020`, `March 2020`, `Sept 2020`, `03/2020`, `2020-03`, `2020`.

## Limits

- **"present" is frozen at indexing time.** A resume indexed in Nov 2026 still says "as of Nov 2026" a year later.
  That's why the chunk says "as of". Re-indexing (next INDEX_VERSION or a reindex job, see `todo/reindex-job.md`)
  refreshes it.
- It trusts the parser's dates. If the parser put a personal project under experience, it's counted
  (that's what `move_personal_projects` in the parser is for).
- Only total experience. "Years with Python" is not calculated (skills have no dates).

## Rollout

- `INDEX_VERSION = 5`
- `vector_migrations/0006_experience_summary_chunk.py`: same shape as `0005` (same vectors), builds
  `resume_chunks_v5`, re-indexes all active documents, moves the alias. v4 is kept for rollback.
- Golden set (v5): new question `q34` "What is the total work experience of the candidate?",
  expected = the `Experience summary` chunk of each resume. Golden matching uses `contains`, not chunk numbers,
  so existing questions are unaffected.

## Files

| File | Change |
|---|---|
| `app/services/resume_chunker.py` | `parse_month`, `job_span`, `total_months`, `as_years`, `experience_summary`; `Personal project:` label; `SECTIONS`; `make_chunks(r, today=None)` returns `(section, text)` |
| `app/schemas/chunk.py`, `app/services/chunk_service.py` | `section` in chunk metadata (Qdrant payload) |
| `app/services/retriever_service.py`, `app/schemas/retriever.py` | `section` in search results |
| `app/services/indexing_service.py` | `INDEX_VERSION = 5` |
| `vector_migrations/0006_experience_summary_chunk.py` | new collection v5 (+ `section` index) + re-index + alias |
| `tests/test_resume_chunker.py`, `tests/test_chunk_service.py`, `tests/test_vector_migrations.py` | new and updated tests |
| `evals/retrieval_golden.json` | v5, `q34` |
