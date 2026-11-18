This Repo is to qa the resume by parsing and embedding it for multi purpose
eg - Match the user resume score against the job descipription,
- Find the matching users on basis of Job description
- Find the available jobs for individual resume
- Vector DB use for the RAG is Qdrant
- Right now only Pdf document is acceptable

## 1. Fresh start on a new environment

Postgres and Qdrant must already be running.

```bash
# 1. get the code
git clone <repo> && cd resumer_qa

# 2. install packages
uv sync

# 3. create your settings file, then fill in DATABASE_URL, QDRANT_URL, HF_TOKEN, JWT_SECRET
cp .env.example .env

# 4. create the empty Postgres database (only once)
createdb resumer_qa

# 5. create all Postgres tables
uv run alembic upgrade head

# 6. create the Qdrant collections + alias
uv run python scripts/vector_migrate.py upgrade

# 7. start the app  (API docs: http://localhost:8000/docs)
uv run uvicorn app.main:app
```

## 2. After any change on a running project

Run the same commands every time. Each one only does what is pending, so running them is always safe.

```bash
git pull
uv sync                                            # packages, if changed
uv run alembic upgrade head                        # Postgres changes, if any
uv run python scripts/vector_migrate.py upgrade    # Qdrant changes, if any
# then restart the app
```

Order matters: `alembic` → `vector_migrate` → start app.

| What changed | `uv sync` | `alembic upgrade` | `vector_migrate upgrade` | Restart app |
|---|---|---|---|---|
| Normal code (endpoint, bug fix, prompt) | – | – | – | ✅ |
| New package in `pyproject.toml` | ✅ | – | – | ✅ |
| New / changed Postgres table | – | ✅ | – | ✅ |
| New Qdrant index | – | – | ✅ | ✅ |
| New embedding model or new chunking | – | – | ✅ rebuilds + re-embeds (can take a while) | ✅ |
| Only `.env` values (URL, token, timeout) | – | – | – | ✅ |

If you forget `vector_migrate.py upgrade` after a model change, the app refuses to start and tells you the command to run.

## 3. Useful commands

```bash
uv run pytest                                          # run tests (no real Postgres/Qdrant/HF needed)
uv run python scripts/vector_migrate.py status         # which Qdrant migrations are applied / pending
uv run alembic current                                 # current Postgres migration
uv run python scripts/parse_resumes.py <pdf or folder> # parse resumes to outputs/ (no DB)
uv run python scripts/show_chunks.py [document_id]     # save chunks + metadata of uploaded docs to outputs/
```

## Docs
- Vector DB (Qdrant) migrations: [docs/vector_migrations.md](docs/vector_migrations.md)
- Hybrid search (dense + BM25 + weighted RRF): [docs/hybrid_search_plan.md](docs/hybrid_search_plan.md)