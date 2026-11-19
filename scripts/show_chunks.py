"""Save the chunks + metadata that would go into the vector DB, one text file per document in outputs/.

    uv run python scripts/show_chunks.py          # all documents
    uv run python scripts/show_chunks.py 3 5      # only document ids 3 and 5
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when run from the IDE

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import selectinload  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.database import SessionLocal, engine  # noqa: E402
from app.models import Document, User  # noqa: E402
from app.services.chunk_service import build_chunks  # noqa: E402


async def main(doc_ids: list[int]) -> None:
    query = select(Document, User).join(User, User.id == Document.generated_by) \
        .options(selectinload(Document.meta_data)).order_by(Document.id)
    if doc_ids:
        query = query.where(Document.id.in_(doc_ids))

    async with SessionLocal() as db:
        rows = (await db.execute(query)).all()

    if not rows:
        print("No documents found. Upload some through POST /api/v1/documents/upload first.")
    out_dir = ROOT / settings.parser_output_dir
    out_dir.mkdir(exist_ok=True)
    for doc, user in rows:
        chunks = build_chunks(doc, user)
        text = "\n\n".join(
            f"{'=' * 60}\nCHUNK {c.metadata.chunk_index}\n"
            f"METADATA: {json.dumps(c.metadata.model_dump(mode='json'), indent=2)}\n"
            f"TEXT:\n{c.text}"
            for c in chunks
        )
        out_file = out_dir / f"document_{doc.id}__chunks_meta.txt"
        out_file.write_text(text, encoding="utf-8")
        print(f"OK    document {doc.id} -> {out_file.relative_to(ROOT)} ({len(chunks)} chunks)")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main([int(x) for x in sys.argv[1:]]))
