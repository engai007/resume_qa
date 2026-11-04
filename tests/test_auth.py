import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app
from app.models import UserType


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


def signup(client, email="jane@x.com", password="secret123", user_type="candidate"):
    return client.post("/api/v1/auth/signup", json={"email": email, "password": password, "user_type": user_type})


def login(client, email="jane@x.com", password="secret123"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def test_signup(client):
    r = signup(client)
    assert r.status_code == 201
    assert r.json() == {"id": 1, "email": "jane@x.com", "user_type": "candidate", "is_active": True}


def test_signup_duplicate_email(client):
    signup(client)
    assert signup(client, email="JANE@x.com").status_code == 409


def test_signup_unknown_user_type(client):
    assert signup(client, user_type="admin").status_code == 422


def test_login_and_me(client):
    signup(client, user_type="recruiter")
    token = login(client).json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["email"] == "jane@x.com" and me.json()["user_type"] == "recruiter"


def test_login_wrong_password(client):
    signup(client)
    assert login(client, password="wrong-pass").status_code == 401


def test_me_needs_valid_token(client):
    assert client.get("/api/v1/auth/me").status_code in (401, 403)
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
