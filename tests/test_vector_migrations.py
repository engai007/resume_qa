import asyncio

import pytest
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models import Document, DocumentMetaData, User, UserType
from app.services import indexing_service
from app.services import vector_migration_service as vm
from tests.conftest import create_test_db, use_fake_vectors


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Empty in-memory Qdrant + SQLite DB, fake embeddings."""
    client = AsyncQdrantClient(":memory:")
    use_fake_vectors(monkeypatch, client)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    asyncio.run(create_test_db(engine))
    return client, async_sessionmaker(engine, expire_on_commit=False)


async def add_document(db) -> Document:
    user_type = await db.scalar(select(UserType).where(UserType.name == "recruiter"))
    user = User(email="rec@x.com", hashed_password="h", user_type=user_type)
    db.add(user)
    await db.flush()
    doc = Document(document_hash="abc", generated_by=user.id, meta_data=DocumentMetaData(
        document_plain_data=[], parsed_doc_json={"name": "Jane", "skills": {"Languages": ["Python"]}}))
    db.add(doc)
    await db.commit()
    return doc


def alias_target(client) -> str:
    [alias] = asyncio.run(client.get_aliases()).aliases
    return alias.collection_name


def test_upgrade_runs_each_migration_once(setup):
    client, session = setup

    async def go():
        async with session() as db:
            return await vm.upgrade(db, client), await vm.upgrade(db, client), await vm.pending(db)

    first, second, left = asyncio.run(go())
    assert first == [f.stem for f in vm.migration_files()]
    assert second == [] and left == []
    params = asyncio.run(client.get_collection(settings.qdrant_collection)).config.params  # via the alias
    assert params.vectors["dense"].size == settings.embedding_size  # last migration matches the model
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF


def test_rebuilds_reindex_documents_and_move_alias_to_latest(setup):
    client, session = setup

    async def go():
        async with session() as db:
            doc = await add_document(db)
            await vm.upgrade(db, client)
            await db.refresh(doc)
            return doc

    doc = asyncio.run(go())
    assert alias_target(client) == "resume_chunks_v5"  # 0002 -> v2, 0004 -> v3, 0005 -> v4, 0006 -> v5
    for name in ("resume_chunks_v2", "resume_chunks_v3", "resume_chunks_v4", "resume_chunks_v5"):  # all indexed fine
        assert asyncio.run(client.count(name)).count == 2  # profile + skills chunks
    [point], _ = asyncio.run(client.scroll("resume_chunks_v5", limit=1, with_vectors=True))
    assert set(point.vector) == {"dense", "bm25"}  # both vectors in one point
    assert point.payload["section"] in ("profile", "skills")  # v5 chunks carry their section
    assert doc.indexed_version == indexing_service.INDEX_VERSION


def test_0002_keeps_v1_when_indexing_fails(setup, monkeypatch):
    client, session = setup

    async def hf_down(texts):
        raise ConnectionError("HF down")

    monkeypatch.setattr(indexing_service, "embed_texts", hf_down)

    async def go():
        async with session() as db:
            await add_document(db)
            with pytest.raises(RuntimeError, match="Alias not moved"):
                await vm.upgrade(db, client)
            return await vm.applied_ids(db)

    applied = asyncio.run(go())
    assert applied == {"0001_create_resume_chunks_v1"}  # 0002 not recorded -> runs again next time
    assert alias_target(client) == "resume_chunks_v1"  # search untouched
