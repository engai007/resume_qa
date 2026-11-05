import asyncio

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app
from app.models import UserType
from app.services import indexing_service
from app.services import vector_migration_service


@pytest.fixture(autouse=True)
def llm_log(tmp_path, monkeypatch):
    """Tests write LLM logs to a temp file, not logs/llm_calls.jsonl."""
    path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(settings, "llm_log_file", str(path))
    return path


async def fake_embed(texts):
    return [[1.0] + [0.0] * (settings.embedding_size - 1) for _ in texts]


async def run_vector_migrations(client: AsyncQdrantClient, db_path) -> None:
    """Apply every vector migration to `client` (using an empty throwaway DB)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", poolclass=NullPool)
    await create_test_db(engine)
    async with async_sessionmaker(engine)() as db:
        await vector_migration_service.upgrade(db, client)


@pytest.fixture
def vector_db(tmp_path, monkeypatch):
    """Indexing goes to an in-memory Qdrant with fake embeddings (no real Qdrant, no HF calls)."""
    client = AsyncQdrantClient(":memory:")
    monkeypatch.setattr(indexing_service, "qdrant", client)
    monkeypatch.setattr(indexing_service, "embed_texts", fake_embed)
    asyncio.run(run_vector_migrations(client, tmp_path / "migrations.db"))
    return client


async def create_test_db(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine)() as db:
        db.add_all([UserType(name="recruiter"), UserType(name="candidate")])
        await db.commit()


@pytest.fixture
def client(tmp_path, monkeypatch, vector_db):
    """App wired to a throwaway SQLite DB instead of Postgres."""
    monkeypatch.setattr(settings, "jwt_secret", "test-secret-that-is-long-enough-for-hs256")
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db", poolclass=NullPool)
    asyncio.run(create_test_db(engine))
    session = async_sessionmaker(engine, expire_on_commit=False)

    async def test_db():
        async with session() as db:
            yield db

    app.dependency_overrides[get_db] = test_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def auth_header(client, email="jane@x.com", password="secret123", user_type="candidate") -> dict:
    """Sign up + log in, return the Authorization header."""
    client.post("/api/v1/auth/signup", json={"email": email, "password": password, "user_type": user_type})
    token = client.post("/api/v1/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
