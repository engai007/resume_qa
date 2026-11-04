import asyncio

from app.agents.resume_parser import ResumeParserAgent
from app.core.config import settings
from app.services.resumer_parser.builder import build_resume
from app.services.resumer_parser.contact import extract_contact
from app.services.resumer_parser.llm_input import compact_numbered
from app.services.resumer_parser.pdf_text import extract_lines, merge_wrapped
from app.services.resumer_parser.ranges import fix_ranges


async def parse_resume(pdf_bytes: bytes, agent: ResumeParserAgent) -> dict:
    """PDF bytes -> resume dict."""
    lines = await asyncio.to_thread(extract_lines, pdf_bytes, settings.parser_gap_ratio)
    if not lines:
        raise ValueError("no text found in PDF (scanned image?)")
    texts = merge_wrapped(lines)
    contact = extract_contact(texts)
    llm_input = compact_numbered(texts, settings.parser_max_words)
    parsed = await agent.run(llm_input)
    parsed = fix_ranges(parsed, texts)
    return build_resume(parsed, contact, texts)


async def parse_resumes(files: dict[str, bytes], agent: ResumeParserAgent) -> dict:
    """{filename: pdf bytes} -> {filename: resume dict, or the error if that one failed}."""
    limit = asyncio.Semaphore(settings.parser_concurrency)

    async def one(pdf_bytes):
        async with limit:
            return await parse_resume(pdf_bytes, agent)

    results = await asyncio.gather(*(one(b) for b in files.values()), return_exceptions=True)
    return dict(zip(files, results))
