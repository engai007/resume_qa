"""Apply Qdrant migrations from vector_migrations/ (run after `alembic upgrade head`).

    uv run python scripts/vector_migrate.py upgrade    # apply pending migrations
    uv run python scripts/vector_migrate.py status     # show applied / pending
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when run from the IDE

from app.db.database import SessionLocal, engine  # noqa: E402
from app.db.qdrant import qdrant  # noqa: E402
from app.services import vector_migration_service as vm  # noqa: E402


async def main(command: str) -> None:
    async with SessionLocal() as db:
        if command == "upgrade":
            applied = await vm.upgrade(db, qdrant)
            print("\n".join(f"APPLIED  {m}" for m in applied) or "Nothing to apply, already up to date.")
        elif command == "status":
            done = await vm.applied_ids(db)
            for f in vm.migration_files():
                print(f"{'applied ' if f.stem in done else 'PENDING '} {f.stem}")
        else:
            print(__doc__)
    await qdrant.close()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else ""))
