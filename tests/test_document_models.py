import asyncio

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.database import Base
from app.models import Document, DocumentMetaData, User, UserType


@pytest.fixture
def session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())
    return async_sessionmaker(engine, expire_on_commit=False)


async def add_user(db) -> User:
    user = User(email="a@x.com", hashed_password="h", user_type=UserType(name="candidate"))
    db.add(user)
    await db.flush()
    return user


def test_document_with_meta_data(session):
    async def go():
        async with session() as db:
            user = await add_user(db)
            doc = Document(document_hash="abc", generated_by=user.id)
            doc.meta_data = DocumentMetaData(document_plain_data=["Jane Doe", "Skills"],
                                             parsed_doc_json={"name": "Jane Doe"})
            db.add(doc)
            await db.commit()
            return doc

    doc = asyncio.run(go())
    assert doc.is_active is True
    assert doc.meta_data.doc_id == doc.id
    assert doc.meta_data.document_plain_data[1] == "Skills"  # index = line number


def test_same_file_twice_for_same_user_is_rejected(session):
    async def go():
        async with session() as db:
            user = await add_user(db)
            db.add_all([Document(document_hash="abc", generated_by=user.id),
                        Document(document_hash="abc", generated_by=user.id)])
            await db.commit()

    with pytest.raises(IntegrityError):
        asyncio.run(go())
