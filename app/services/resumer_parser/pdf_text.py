"""Step 1: PDF bytes -> layout-aware lines -> merged text lines."""
import logging
import re

import pymupdf

logger = logging.getLogger(__name__)


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


def find_column_split(blocks: list[dict], page_width: float) -> int | None:
    """x of a vertical gap that splits the page into two real text columns, else None."""
    total = sum(b["chars"] for b in blocks)
    for x in range(int(page_width * 0.2), int(page_width * 0.8), 2):
        if any(b["x0"] < x < b["x1"] for b in blocks):
            continue
        left = sum(b["chars"] for b in blocks if b["x1"] <= x)
        if 0.15 * total < left < 0.85 * total:      # both sides have real content
            return x
    return None


def extract_lines(pdf_bytes: bytes, gap_ratio: float = 0.15) -> list[dict]:
    """One dict per visual line with its text and font/position info. Sync (CPU-bound).
    Two-column pages are read left column first, then right column."""
    lines = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_no, page in enumerate(doc):
            blocks = []
            for block in page.get_text("rawdict")["blocks"]:  # rawdict = per-character positions
                if block["type"] != 0:
                    continue
                block_lines = []
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if "".join(c["c"] for c in s["chars"]).strip()]
                    if not spans:
                        continue
                    block_lines.append({
                        "page": page_no,
                        "text": line_text(spans, gap_ratio),
                        "font": spans[0]["font"],
                        "size": round(max(s["size"] for s in spans) * 2) / 2,
                        "bold": all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                        "color": spans[0]["color"],
                        "x0": round(line["bbox"][0], 1),
                        "y0": round(line["bbox"][1], 1),
                    })
                if block_lines:
                    x0, y0, x1, _ = block["bbox"]
                    blocks.append({"x0": x0, "y0": y0, "x1": x1, "lines": block_lines,
                                   "chars": sum(len(l["text"]) for l in block_lines)})

            split = find_column_split(blocks, page.rect.width)
            logger.debug("page %s: column split at x=%s", page_no, split)
            if split is None:
                columns = [blocks]
            else:
                columns = [[b for b in blocks if b["x1"] <= split],
                           [b for b in blocks if b["x1"] > split]]

            for col in columns:
                for b in sorted(col, key=lambda b: (round(b["y0"] / 5), b["x0"])):
                    lines.extend(b["lines"])
    return lines


def merge_wrapped(lines: list[dict]) -> list[str]:
    """Join lines that are continuations of the previous line."""
    texts: list[str] = []
    for l in lines:
        t = l["text"]
        if texts and (t[0].islower() or t[0] == "·"
                      or texts[-1].endswith((",", "-", "–", "—", "&", "·", " and"))):
            texts[-1] += " " + t
        else:
            texts.append(t)
    return texts
