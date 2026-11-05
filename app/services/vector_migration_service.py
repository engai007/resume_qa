"""Runs the Qdrant migration files in vector_migrations/ - our small "Alembic for Qdrant".

Each file has `async def upgrade(client, db)`. Applied files are recorded in the
Postgres table vector_migrations, so each one runs only once per environment.
"""
import importlib.util
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import BASE_DIR
from app.models import VectorMigration

MIGRATIONS_DIR = BASE_DIR / "vector_migrations"


def migration_files(folder: Path = MIGRATIONS_DIR) -> list[Path]:
    """0001_x.py, 0002_y.py, ... in order."""
    return sorted(folder.glob("[0-9]*.py"))


async def applied_ids(db: AsyncSession) -> set[str]:
    return set(await db.scalars(select(VectorMigration.id)))


async def pending(db: AsyncSession, folder: Path = MIGRATIONS_DIR) -> list[Path]:
    done = await applied_ids(db)
    return [f for f in migration_files(folder) if f.stem not in done]


async def upgrade(db: AsyncSession, client: AsyncQdrantClient, folder: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations in order. Returns the ids that were applied."""
    applied = []
    for f in await pending(db, folder):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        await module.upgrade(client, db)
        db.add(VectorMigration(id=f.stem))
        await db.commit()  # record each one right away, so a later failure doesn't redo it
        applied.append(f.stem)
    return applied
