import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app
from app.models import UserType


@pytest.fixture(autouse=True)
def llm_log(tmp_path, monkeypatch):
    """Tests write LLM logs to a temp file, not logs/llm_calls.jsonl."""
    path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(settings, "llm_log_file", str(path))
    return path


async def create_test_db(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine)() as db:
        db.add_all([UserType(name="recruiter"), UserType(name="candidate")])
        await db.commit()


@pytest.fixture
def client(tmp_path, monkeypatch):
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
