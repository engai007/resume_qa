"""Parse resume PDFs and save each one as JSON in outputs/.

    uv run python scripts/parse_resumes.py                  # all PDFs in resumes/
    uv run python scripts/parse_resumes.py a.pdf folder/    # specific files / folders
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when run from the IDE

from app.agents.resume_parser import ResumeParserAgent  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.services.resume_chunker import make_chunks  # noqa: E402
from app.services.resumer_parser import parse_resumes  # noqa: E402


def find_pdfs(paths: list[str]) -> list[Path]:
    pdfs = []
    for p in (Path(x).expanduser().resolve() for x in paths):
        if p.is_dir():
            pdfs += sorted(p.glob("*.pdf"))
        elif p.is_file():
            pdfs.append(p)
        else:
            print(f"SKIP  {p} (not found)")
    return list(dict.fromkeys(pdfs))  # drop duplicates, keep order


async def main(paths: list[str]) -> None:
    pdfs = find_pdfs(paths)
    if not pdfs:
        print(f"No PDFs found in {paths}")
        return

    print(f"Parsing {len(pdfs)} resume(s)...")
    agent = ResumeParserAgent()
    results = await parse_resumes({p.name: p.read_bytes() for p in pdfs}, agent)

    out_dir = ROOT / settings.parser_output_dir
    out_dir.mkdir(exist_ok=True)
    model = agent.model_id.split("/")[-1]
    for name, result in results.items():
        if isinstance(result, Exception):
            print(f"FAIL  {name}: {result}")
            continue
        _, resume = result
        out_file = out_dir / f"{Path(name).stem}__{model}.json"
        out_file.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

        chunks_file = out_dir / f"{Path(name).stem}__{model}__chunks.txt"
        chunks = make_chunks(resume)
        text = "\n\n".join(f"{'=' * 60}\nCHUNK {i}\n{c}" for i, c in enumerate(chunks))
        chunks_file.write_text(text, encoding="utf-8")
        print(f"OK    {name} -> {out_file.name}, {chunks_file.name} ({len(chunks)} chunks)")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main(sys.argv[1:] or [str(ROOT / settings.parser_input_dir)]))
