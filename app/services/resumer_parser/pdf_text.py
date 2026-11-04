"""Step 1: PDF bytes -> layout-aware lines -> merged text lines."""
import re

import pymupdf


def line_text(spans: list[dict], gap_ratio: float = 0.15) -> str:
    """Rebuild line text from characters, inserting spaces where there's a visual gap."""
    out, prev_x1 = [], None
    for s in spans:
        for ch in s["chars"]:
            x0, x1 = ch["bbox"][0], ch["bbox"][2]
            gap = prev_x1 is not None and x0 - prev_x1 > gap_ratio * s["size"]
            if gap and out and out[-1] != " " and ch["c"] != " ":
                out.append(" ")
            out.append(ch["c"])
            prev_x1 = x1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def extract_lines(pdf_bytes: bytes, gap_ratio: float = 0.15) -> list[dict]:
    """One dict per visual line with its text and font/position info. Sync (CPU-bound)."""
    lines = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_no, page in enumerate(doc):
            data = page.get_text("rawdict", sort=True)  # rawdict = per-character positions
            for block in data["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if "".join(c["c"] for c in s["chars"]).strip()]
                    if not spans:
                        continue
                    lines.append({
                        "page": page_no,
                        "text": line_text(spans, gap_ratio),
                        "font": spans[0]["font"],
                        "size": round(max(s["size"] for s in spans) * 2) / 2,
                        "bold": all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                        "color": spans[0]["color"],
                        "x0": round(line["bbox"][0], 1),
                        "y0": round(line["bbox"][1], 1),
                    })
    return lines


def merge_wrapped(lines: list[dict]) -> list[str]:
    """Join lines that are continuations of the previous line (wrapped bullets)."""
    texts: list[str] = []
    for l in lines:
        t = l["text"]
        if texts and (t[0].islower() or texts[-1].endswith((",", "-", "&", " and"))):
            texts[-1] += " " + t
        else:
            texts.append(t)
    return texts
