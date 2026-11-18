# Vector DB migrations (Qdrant)

How we change Qdrant (collections, vector size, indexes, embedding model) in a tracked way,
so every environment (laptop, test server, production) ends up exactly the same.
It works like Alembic, but for Qdrant.

---

## 1. The big idea

```
Postgres  (master copy)                         Qdrant  (search copy)
documents + document_meta_data   ──index──►     resume_chunks  (vectors + metadata)
```

- **Postgres is the source of truth.** Every resume's parsed JSON lives in `document_meta_data`.
- **Qdrant is built from Postgres.** We can delete it and rebuild it at any time. Nothing is lost.
- So we never "repair" data inside Qdrant. **Small changes** are done in place. **Big changes** are done by building a new collection from Postgres.

---

## 2. The pieces

| Piece | Where | What it does |
|---|---|---|
| Migration files | `vector_migrations/0001_….py`, `0002_….py` | One file per Qdrant change, run in number order |
| Runner | `app/services/vector_migration_service.py` | Finds pending files and runs them |
| Command | `scripts/vector_migrate.py` | `upgrade` / `status` |
| History table | Postgres `vector_migrations` | Remembers which files already ran (like `alembic_version`) |
| Alias | Qdrant alias `resume_chunks` | The name the app uses. It points to the real collection (`resume_chunks_v3`) |
| Index version | `INDEX_VERSION` in `app/services/indexing_service.py` | Number stamped on every chunk: "made with chunking X + model Y" |
| Index status | `documents.indexed_version`, `index_attempts`, `index_error` | Which documents are in Qdrant, and why one failed |
| Startup check | `check_vector_db()` in `indexing_service.py` | App refuses to start if Qdrant doesn't match the code |

### Current state

| Migration | What it did |
|---|---|
| `0001_create_resume_chunks_v1` | Created `resume_chunks_v1` (384 numbers, all-MiniLM-L6-v2), indexes, alias |
| `0002_rebuild_with_bge_base` | Created `resume_chunks_v2` (768 numbers, `BAAI/bge-base-en-v1.5`), re-embedded every document, moved the alias to v2 |
| `0003_index_document_hash` | Added a payload index on `document_hash` in v2 (search can be limited to given resumes) |
| `0004_rebuild_one_chunk_per_project` | New chunking (one chunk per project, skills per category in groups of 8): built `resume_chunks_v3` (same model, 768), re-indexed, moved the alias to v3 |

```
resume_chunks  ──►  resume_chunks_v3   (768, bge-base-en-v1.5, new chunking)   ← in use
                    resume_chunks_v2   (768, bge-base-en-v1.5, old chunking)   ← kept for rollback
                    resume_chunks_v1   (384, MiniLM)             ← old, kept for rollback
INDEX_VERSION = 3
```

---

## 3. When and how migrations run

**Migrations never run by themselves, and the app never runs them.** You run them once per environment, when you deploy new code:

```bash
uv run alembic upgrade head                        # 1. Postgres tables
uv run python scripts/vector_migrate.py upgrade    # 2. Qdrant
uv run uvicorn app.main:app                        # 3. start the app
```

The order matters. Step 2 needs the Postgres tables from step 1 (`vector_migrations`, `documents`).

Check what's applied:

```bash
uv run python scripts/vector_migrate.py status
```
```
applied  0001_create_resume_chunks_v1
applied  0002_rebuild_with_bge_base
applied  0003_index_document_hash
applied  0004_rebuild_one_chunk_per_project
PENDING  0005_add_index_on_candidate_name      ← will run on next `upgrade`
```

Running `upgrade` twice is safe. The second time it prints `Nothing to apply, already up to date.`

### Why the app doesn't run them on startup

With several app workers starting at once, they'd all try to run the same migration at the same time.
A rebuild can also take minutes, and app startup shouldn't. Alembic follows the same rule.

### What the app does on startup

`check_vector_db()` runs when the app starts:

| Situation | What happens |
|---|---|
| Alias exists, vector size = `EMBEDDING_SIZE` | Starts normally |
| Alias missing | **Refuses to start**: `Run: uv run python scripts/vector_migrate.py upgrade` |
| Size doesn't match (e.g. model changed, migration not run) | **Refuses to start**: `has vector size 384, but EMBEDDING_SIZE is 768` |
| Qdrant is down | Starts with a warning. Uploads still save to Postgres, they just aren't indexed |

### A brand-new environment

Run the same three commands. `upgrade` runs every file from `0001`, so the new environment ends up
just like production. (It briefly creates an empty `resume_chunks_v1` along the way, which is harmless.)

---

## 4. Which change needs what?

The one question to ask: **do the vectors change?**

| Change | Vectors change? | Type | Bump `INDEX_VERSION`? |
|---|---|---|---|
| Add a payload index (fast filter) | no | small, in place | no |
| Add / rename a metadata field on chunks | no | small, in place (re-index into the same collection) | yes |
| Change `make_chunks` (how a resume is split) | **yes** | big, rebuild into a new collection | yes |
| Change the embedding model | **yes** | big, rebuild into a new collection | yes |
| Change vector size or distance | **yes** | big, rebuild into a new collection | yes |
| Delete an old collection | – | cleanup | no |
| Change `QDRANT_URL`, `HF_TOKEN`, timeouts | no | `.env` only, **no migration** | no |
| Change `EMBEDDING_PROVIDER` (same model) | no | `.env` only, **no migration** | no |

> **Warning about `EMBEDDING_MODEL` in `.env`:** changing it is a big change and **must** come with a
> rebuild migration. The startup check only compares vector *size*. If the new model has the same size,
> the app starts anyway and new vectors get mixed with old ones, which quietly makes search worse.

---

## 5. Writing a migration file

Name it `NNNN_short_description.py`, with the next free number. It needs one function:

```python
async def upgrade(client, db) -> None:   # client = Qdrant, db = Postgres session
    ...
```

Rules:

0. **A rebuild must create every payload index the current collection has** (today: `uploaded_by`,
   `document_id`, `document_hash`). Indexes belong to a collection, so they don't move with the alias.
1. **Write values out in the file** (`size=768`, `"resume_chunks_v2"`). Don't read them from `settings`.
   The file must do the same thing forever, even after `.env` changes.
2. **Never edit a migration that has already run** anywhere. Add a new file instead.
3. **Make it safe to run again** (`if not await client.collection_exists(...)`). If it crashed halfway,
   the next `upgrade` runs it again.
4. **If it fails, raise an error.** The runner then doesn't record it, and it runs again next time.
5. There is **no downgrade**. To roll back, point the alias back to the old collection (section 7).

Each file is recorded right after it succeeds. If `0003` fails, `0001` and `0002` stay recorded and only `0003` runs again.

---

## 6. Examples

### Example A: small change, add a payload index

We want fast filtering by candidate name. The vectors don't change, so we update in place.

`vector_migrations/0005_add_index_on_candidate_name.py`
```python
"""Fast filter on candidate_name."""
from qdrant_client import AsyncQdrantClient, models


async def upgrade(client: AsyncQdrantClient, db) -> None:
    await client.create_payload_index("resume_chunks_v3", "candidate_name", models.PayloadSchemaType.KEYWORD)
```
Deploy: `vector_migrate.py upgrade`. That's it. There's no rebuild and `INDEX_VERSION` stays the same.

---

### Example B: small change, add a metadata field

We want `user_type` (recruiter / candidate) on every chunk.

1. Add the field in code: `ChunkMetadata` in `app/schemas/chunk.py` and `build_metadata` in `app/services/chunk_service.py`.
   New uploads get it automatically.
2. Bump `INDEX_VERSION = 4` in `indexing_service.py`.
3. Existing chunks don't have the field yet. Re-index them **into the same collection**, since the vectors are the same size and model:

`vector_migrations/0006_add_user_type_to_chunks.py`
```python
"""Re-index every document so its chunks get the new user_type field."""
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Document, User
from app.services.indexing_service import index_document


async def upgrade(client, db) -> None:
    rows = (await db.execute(
        select(Document, User).join(User, User.id == Document.generated_by)
        .where(Document.is_active).options(selectinload(Document.meta_data))
    )).all()
    failed = [doc.id for doc, user in rows if not await index_document(db, doc, user, collection="resume_chunks_v3")]
    if failed:
        raise RuntimeError(f"could not index documents {failed}, run upgrade again")
```
Re-indexing is safe because point IDs are fixed (`point_id(document_id, chunk_index)`). Old points are overwritten, not duplicated.

---

### Example C: big change, new embedding model (this is what `0002` did)

Say we move from `bge-base-en-v1.5` (768) to a model with 1024 numbers.

**In one commit:**

1. `.env` and `.env.example`: `EMBEDDING_MODEL=<new model>`, `EMBEDDING_SIZE=1024`
2. `indexing_service.py`: `INDEX_VERSION = 4` (and add it to the comment listing the versions)
3. New file `vector_migrations/0007_rebuild_with_new_model.py`, the same shape as `0002`:

```python
"""Rebuild with <new model> (1024 numbers)."""
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Document, User
from app.services.indexing_service import index_document

NEW = "resume_chunks_v4"
ALIAS = "resume_chunks"


async def upgrade(client: AsyncQdrantClient, db) -> None:
    # 1. new empty collection next to the old one
    if not await client.collection_exists(NEW):
        await client.create_collection(NEW, vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    await client.create_payload_index(NEW, "uploaded_by", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(NEW, "document_id", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(NEW, "document_hash", models.PayloadSchemaType.KEYWORD)

    # 2. fill it from Postgres; search keeps using v3 meanwhile
    rows = (await db.execute(
        select(Document, User).join(User, User.id == Document.generated_by)
        .where(Document.is_active).options(selectinload(Document.meta_data))
    )).all()
    failed = [doc.id for doc, user in rows if not await index_document(db, doc, user, collection=NEW)]
    if failed:
        raise RuntimeError(f"could not index documents {failed}. Alias not moved, fix and run upgrade again.")

    # 3. switch the alias in one step
    await client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)),
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=NEW, alias_name=ALIAS)),
    ])
```

**What happens during deploy:**

```
before         resume_chunks ──► v3        search works
upgrade runs   v4 created, filling...      search still on v3, users notice nothing
alias moved    resume_chunks ──► v4        search now uses the new model
               v3 kept                     rollback is possible
```

If Hugging Face is down halfway: the migration raises an error, the alias stays on v3, and `0007` is not recorded.
Run `upgrade` again later. Documents already in v4 are overwritten, not duplicated.

**What if the app starts before the migration?** The startup check sees 768 ≠ 1024 and refuses to start. That's on purpose.

---

### Example D: big change, new chunking

We change `make_chunks` (e.g. one chunk per project instead of one per company).
The chunks are different, so their vectors are different too. This is the same as Example C, with these differences:

- edit `app/services/resume_chunker.py`
- bump `INDEX_VERSION`
- new migration builds `resume_chunks_v4` with the **same** size as now (768), re-indexes and moves the alias
- `.env` doesn't change

Why a new collection, even though the size is the same? Rebuilding in place would mix old and new chunks while
it runs, and old leftover chunks would stay behind. A new collection gives a clean switch and a rollback.

---

### Example E: cleanup, delete the old collection

When the new collection has worked well for a few days:

`vector_migrations/0008_drop_resume_chunks_v1.py`
```python
"""v3 is in use; v1 (MiniLM) is no longer needed."""


async def upgrade(client, db) -> None:
    if await client.collection_exists("resume_chunks_v1"):
        await client.delete_collection("resume_chunks_v1")
```
After this, rolling back to v1 is no longer possible.

---

## 7. Rollback

There are no downgrade functions. To go back, point the alias at the previous collection:

```bash
uv run python -c "
import asyncio
from qdrant_client import models
from app.db.qdrant import qdrant
asyncio.run(qdrant.update_collection_aliases(change_aliases_operations=[
    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name='resume_chunks')),
    models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name='resume_chunks_v1', alias_name='resume_chunks')),
]))"
```

Also put the **code** back to the matching version (`git revert`). The app must embed questions with the
same model that made the vectors in the collection, otherwise the startup check stops it (if sizes differ) or search gets bad results (if they're the same).

---

## 8. How indexing fits in (every upload)

```
POST /api/v1/documents/upload
  1. parse PDF → save in Postgres          documents.indexed_version = NULL
  2. chunk → embed (HF) → upsert into the alias "resume_chunks"
       success → indexed_version = INDEX_VERSION
       failure → indexed_version stays NULL, index_attempts + 1, index_error = message
  3. response: {"status": "created", "indexed": true/false, ...}
```

A failed index **does not fail the upload**. Documents with `indexed_version` NULL, or lower than `INDEX_VERSION`,
are the ones still to index. A Celery retry job for them is planned and doesn't exist yet. Until then,
the next rebuild migration picks them up.

Find them:
```sql
select id, indexed_version, index_attempts, index_error
from documents
where indexed_version is null or indexed_version < 3;   -- 3 = current INDEX_VERSION
```

---

## 9. Testing

Tests never touch the real Qdrant or Hugging Face:

- `AsyncQdrantClient(":memory:")`: Qdrant running inside the test process
- `fake_embed` in `tests/conftest.py`: returns fixed vectors instead of calling HF
- `tests/test_vector_migrations.py` checks that:
  - every migration runs exactly once
  - the final collection size matches `EMBEDDING_SIZE` (this catches a model change without a migration)
  - `0002` re-indexes documents and moves the alias
  - if indexing fails, the alias stays on the old collection and the migration isn't recorded

When you add a migration, run `uv run pytest`. The size test fails if `.env` and the last migration disagree.

---

## 10. Checklist

**Changing the embedding model or the chunking:**
- [ ] code change (`.env` model/size, or `resume_chunker.py`)
- [ ] `INDEX_VERSION` + 1
- [ ] new migration file: new `resume_chunks_vN`, re-index, move the alias
- [ ] `uv run pytest`
- [ ] deploy: `alembic upgrade head` → `vector_migrate.py upgrade` → start app
- [ ] a few days later: cleanup migration that deletes the old collection

**Adding an index:**
- [ ] new migration file with `create_payload_index`
- [ ] deploy: `vector_migrate.py upgrade`

**Changing only URL / token / provider / timeout:**
- [ ] `.env` only, no migration
