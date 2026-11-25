"""Chunk: cut the Markdown into chunks that follow the document's own structure.

Rules (in plain words):
  * A new heading starts a new chunk. Headings are Markdown '#' lines OR numbered
    lines like "3.3.2 Common Forms of Corrosion" (many Word files type headings as
    normal text, so we detect the numbering too).
  * Small sections are merged with the next one until we reach the target size.
  * A table is never cut in the middle of a row. Big tables are split into row groups
    and every piece gets the table's header row again.
  * A figure description stays in one piece.
  * Every chunk starts with "Document > Section > Subsection" so it makes sense alone.
"""
import re
from dataclasses import dataclass, field

NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+([A-Z][a-z][^.]{1,90})$")
NUMBER_ONLY = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,4}\.?$")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
PAGE_NUMBER = re.compile(r"<!--\s*PageNumber=\"?(\d+)\"?\s*-->")
PAGE_BREAK = "<!-- PageBreak -->"
TOC_LINE = re.compile(r"(\.{3,}|\s{2,}|\t)\d+\s*$")  # "Introduction ..... 2"


def n_tokens(text: str) -> int:
    """Close enough for sizing chunks (about 4 characters per token)."""
    return max(1, len(text) // 4)


@dataclass
class Block:
    kind: str            # "text" | "table" | "figure"
    text: str
    page: int
    path: list[str]      # heading trail at this point
    lead: str = ""       # for tables: the short title line just above the table


@dataclass
class Chunk:
    text: str            # what gets embedded and searched (with breadcrumb)
    content: str         # the raw piece (shown to the model)
    section: str
    page_start: int
    page_end: int
    kinds: set[str] = field(default_factory=set)
    figure_ids: list[str] = field(default_factory=list)


def _heading_level(line: str):
    m = MD_HEADING.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = NUMBERED_HEADING.match(line.strip())
    if m and not TOC_LINE.search(line):
        return m.group(1).count(".") + 1, f"{m.group(1)} {m.group(2).strip()}"
    return None


def _join_split_headings(lines: list[str]) -> list[str]:
    """Some parsers put "3.2.1" and "Anodizing" on separate lines: glue them back."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].strip().lstrip("#").strip()
        if NUMBER_ONLY.match(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and NUMBERED_HEADING.match(f"{line.rstrip('.')} {lines[j].strip().lstrip('#').strip()}"):
                out.append(f"{line.rstrip('.')} {lines[j].strip().lstrip('#').strip()}")
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return out


def _drop_toc(lines: list[str]) -> list[str]:
    """Remove the table of contents: from the "Contents" line up to the first heading
    that appears a second time (that is where the real text starts)."""
    start = next((i for i, l in enumerate(lines[:400])
                  if re.match(r"^\s*(#+\s*)?(table of )?contents\s*$", l, flags=re.I)), None)
    if start is None:
        return lines
    seen = set()
    for i in range(start + 1, len(lines)):
        h = _heading_level(lines[i])
        if not h:
            continue
        title = re.sub(r"^[\d.]+\s+", "", h[1])
        if title in seen:
            return lines[:start] + lines[i:]
        seen.add(title)
    return lines[:start] + lines[start + 1:]


def to_blocks(markdown: str) -> list[Block]:
    blocks, path, page = [], [], 1
    buf: list[str] = []
    in_table = in_figure = False
    last_lead = [None, ""]

    def flush(kind="text"):
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            blocks.append(Block(kind, text, page, list(path)))
        buf = []

    for line in _drop_toc(_join_split_headings(markdown.splitlines())):
        m = PAGE_NUMBER.search(line)
        if m:
            page = int(m.group(1))
            continue
        if PAGE_BREAK in line:
            page += 1
            continue
        if line.strip().startswith("<!--"):
            continue

        # tables (Document Intelligence writes them as HTML)
        if "<table" in line:
            flush(); in_table = True
        if in_table:
            buf.append(line)
            if "</table>" in line:
                flush("table"); in_table = False
                tbl = blocks.pop()
                # Glue a table's title lines and small header tables to it, e.g.
                # "ISO 4014 - Hexagon head bolts - A2-70" + "Standard | ISO 4014" + "Preferred Parts"
                lead = []
                while blocks and len(lead) < 4 and blocks[-1].path == tbl.path and (
                        (blocks[-1].kind == "text" and len(blocks[-1].text) <= 300)
                        or (blocks[-1].kind == "table" and len(blocks[-1].text) <= 1500)):
                    b = blocks.pop()
                    lead.insert(0, (b.lead + "\n" + b.text).strip())
                hit_big_table = bool(blocks) and blocks[-1].kind == "table" and blocks[-1].path == tbl.path
                if hit_big_table and last_lead[0] == tbl.path:
                    # a follow-on table ("Non-Preferred Parts") under the same title
                    lead.insert(0, last_lead[1])
                tbl.lead = "\n".join(lead)
                if tbl.lead and not hit_big_table:
                    last_lead[:] = [tbl.path, tbl.lead]
                blocks.append(tbl)
            continue

        # figures (we replaced them with <figure>...</figure> holding the description)
        if "<figure" in line:
            flush(); in_figure = True
        if in_figure:
            buf.append(line)
            if "</figure>" in line:
                flush("figure"); in_figure = False
            continue

        h = _heading_level(line)
        if h:
            flush()
            level, title = h
            path[:] = path[: level - 1] + [title]
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush()
    return blocks


def _split_table(html: str, max_tokens: int, lead: str = "") -> list[str]:
    lead = (lead + "\n") if lead else ""
    rows = re.findall(r"<tr>.*?</tr>", html, flags=re.S)
    if len(rows) < 2 or n_tokens(html) <= max_tokens:
        return [lead + html]
    caption = re.search(r"<caption>.*?</caption>", html, flags=re.S)
    head = (caption.group(0) if caption else "") + rows[0]
    pieces, cur = [], []
    for row in rows[1:]:
        if cur and n_tokens(head + "".join(cur) + row) > max_tokens:
            pieces.append(lead + "<table>" + head + "".join(cur) + "</table>")
            cur = []
        cur.append(row)
    if cur:
        pieces.append(lead + "<table>" + head + "".join(cur) + "</table>")
    return pieces


def chunk_document(markdown: str, doc_title: str, target=600, max_tokens=900,
                   overlap=80) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Block] = []

    def emit():
        nonlocal cur
        if not cur:
            return
        section = " > ".join(cur[0].path) or "(start of document)"
        content = "\n\n".join(b.text for b in cur)
        figure_ids = re.findall(r'<figure id="([^"]+)"', content)
        chunks.append(Chunk(
            text=f"{doc_title} > {section}\n\n{content}",
            content=content, section=section,
            page_start=cur[0].page, page_end=cur[-1].page,
            kinds={b.kind for b in cur}, figure_ids=figure_ids,
        ))
        # carry a little text over so a sentence split at the border is not lost
        tail = cur[-1]
        cur = [Block("text", tail.text[-overlap * 4:], tail.page, tail.path)] \
            if tail.kind == "text" and overlap else []

    for b in to_blocks(markdown):
        pieces = _split_table(b.text, max_tokens, b.lead) if b.kind == "table" else [b.text]
        for piece in pieces:
            blk = Block(b.kind, piece, b.page, b.path)
            new_section = cur and blk.path[:1] != cur[-1].path[:1]
            size = sum(n_tokens(x.text) for x in cur)
            if cur and (new_section or size + n_tokens(piece) > max_tokens
                        or (blk.kind == "table" and blk.lead and size >= target // 2)
                        or (size >= target and blk.path != cur[-1].path)):
                emit()
                if new_section:
                    cur = []  # no overlap across top-level sections
            cur.append(blk)
    emit()
    return chunks
