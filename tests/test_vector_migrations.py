import asyncio

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models import Document, DocumentMetaData, User, UserType
from app.services import indexing_service
from app.services import vector_migration_service as vm
from tests.conftest import create_test_db, fake_embed


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Empty in-memory Qdrant + SQLite DB, fake embeddings."""
    client = AsyncQdrantClient(":memory:")
    monkeypatch.setattr(indexing_service, "qdrant", client)
    monkeypatch.setattr(indexing_service, "embed_texts", fake_embed)
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
    info = asyncio.run(client.get_collection(settings.qdrant_collection))  # via the alias
    assert info.config.params.vectors.size == settings.embedding_size  # last migration matches the model


def test_0002_reindexes_documents_and_moves_alias(setup):
    client, session = setup

    async def go():
        async with session() as db:
            doc = await add_document(db)
            await vm.upgrade(db, client)
            await db.refresh(doc)
            return doc

    doc = asyncio.run(go())
    assert alias_target(client) == "resume_chunks_v2"
    assert asyncio.run(client.count("resume_chunks_v2")).count == 2  # profile + skills chunks
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
