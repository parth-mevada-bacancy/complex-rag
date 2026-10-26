"""
PDF -> element chunks for the iMac Product Environmental Report.

The report is designed like a brochure: plain text extraction gets every word but loses
which value belongs to which column, and which caveat belongs to which claim. So:
  1. Tables are rebuilt — every table sits under a dark header bar, so the bars are found
     in the page's drawings and rows are rebuilt from text positions under them.
  2. Footnotes travel with their claim — superscript markers become [^n], and every chunk
     that cites a note gets that endnote's text appended ("[Note 18] ...").
  3. One chunk per element — one per table, prose in small sentence groups.
  4. Visual pages (charts, icons, call-outs) are described by a vision model; the
     description becomes a chunk of its own (see vision_chunks, called from chatbot.py).

Run on its own to inspect the chunks (no vision, no API key needed):
    uv run python ingest.py
"""

import re

import pymupdf

ENDNOTE_PAGES = (11, 12)


def tokens(text):
    """Rough token count (~4 characters per token) — good enough for sizing chunks."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Text pieces with positions; superscript note markers become [^n]
# ---------------------------------------------------------------------------
def page_lines(page, markers=True):
    """Every text line as {x0, y0, x1, y1, spans: [{x0, x1, text, sup}]}, in the PDF's order."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = []
            for s in line["spans"]:
                text = s["text"].strip()
                if not text:
                    continue
                sup = markers and bool(s["flags"] & 1) and text.isdigit()
                spans.append({"x0": s["bbox"][0], "x1": s["bbox"][2], "text": f"[^{text}]" if sup else text,
                              "sup": sup, "spaced": s["text"][:1].isspace() or s["text"][-1:].isspace()})
            if spans:
                x0, y0, x1, y1 = line["bbox"]
                lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "spans": spans})
    return lines


def join(spans):
    """Spans -> text; a note marker, or a span touching the one before it ("CO" + "2" + "e"),
    sticks to the previous text."""
    out, prev = "", None
    for s in spans:
        touching = prev is not None and not (s["spaced"] or prev["spaced"]) and -0.5 < s["x0"] - prev["x1"] < 1
        out += s["text"] if (s["sup"] or touching or not out) else " " + s["text"]
        prev = s
    return out


def is_footer(line, page_height):
    text = join(line["spans"])
    return line["y0"] > page_height - 50 and ("Product Environmental Report" in text or text.isdigit())


# ---------------------------------------------------------------------------
# 1. Tables rebuilt under their dark header bars
# ---------------------------------------------------------------------------
def header_bars(page):
    """Dark filled rectangles, grouped into one header bar per table."""
    rects = sorted((d["rect"] for d in page.get_drawings()
                    if d.get("fill") and sum(d["fill"]) < 1.2
                    and d["rect"].width > 30 and 8 < d["rect"].height < 60), key=lambda r: r.y0)
    bars = []
    for r in rects:
        if bars and r.y0 < bars[-1]["bottom"] - 1:            # overlaps the bar above: same header
            bars[-1]["rects"].append(r)
            bars[-1]["bottom"] = max(bars[-1]["bottom"], r.y1)
        else:
            bars.append({"rects": [r], "top": r.y0, "bottom": r.y1})
    return bars


def spans_in(lines, x0, y0, x1, y1):
    found = []
    for line in lines:
        if y0 - 1 <= (line["y0"] + line["y1"]) / 2 <= y1 + 1:
            found += [s for s in line["spans"] if x0 - 1 <= (s["x0"] + s["x1"]) / 2 <= x1 + 1]
    return found


def rebuild_tables(page, lines):
    """Columns are the header cells that reach the bar's bottom edge; a cell spanning above
    them holds the table's title. Rows are the text lines under the bar, until a line runs
    across a column boundary (a sentence, not a row)."""
    bars, tables = header_bars(page), []
    for i, bar in enumerate(bars):
        bottom = bar["bottom"]
        columns = sorted((r for r in bar["rects"] if abs(r.y1 - bottom) < 1), key=lambda r: r.x0)
        spanning = [r for r in bar["rects"] if abs(r.y1 - bottom) >= 1]
        heads = [join(spans_in(lines, c.x0, c.y0, c.x1, c.y1)) for c in columns]
        if len(columns) < 2 or not all(heads[1:]):
            continue                                           # a dark box or icon, not a table header
        title = join(spans_in(lines, spanning[0].x0, spanning[0].y0, spanning[0].x1, spanning[0].y1)) if spanning else heads[0]
        left, right = columns[0].x0, columns[-1].x1
        boundaries = [c.x0 for c in columns[1:]]
        stop = bars[i + 1]["top"] if i + 1 < len(bars) else page.rect.height

        rows, last_y = [], bottom
        for line in sorted((l for l in lines if bottom - 1 <= l["y0"] < stop
                            and l["x1"] > left and l["x0"] < right), key=lambda l: l["y0"]):
            if any(s["x0"] < b - 2 < b + 2 < s["x1"] for s in line["spans"] for b in boundaries) or line["y0"] - last_y > 40:
                break                                          # a sentence running across columns: table over
            if rows and abs(line["y0"] - rows[-1]["y0"]) < 4:
                rows[-1]["spans"] += line["spans"]             # same visual row, separate PDF line
            else:
                rows.append({"y0": line["y0"], "spans": list(line["spans"])})
            last_y = line["y1"]

        body = []
        for row in rows:
            cells = [join([s for s in row["spans"] if (c.x0 - 2) <= (s["x0"] + s["x1"]) / 2 <= (c.x1 + 2)])
                     for c in columns]
            if len(columns) == 2:
                body.append(f"{cells[0]}: {cells[1]}" if cells[1] else cells[0])
            else:
                values = "; ".join(f"{h} {v}" for h, v in zip(heads[1:], cells[1:]) if v)
                body.append(f"{cells[0]}: {values}" if values else cells[0])
        head = (f"Table: {title}. Column: {heads[1]}" if len(columns) == 2
                else f"Table: {title}. Columns: {' | '.join(heads)}")
        tables.append({"text": "\n".join([head] + body), "top": bar["top"], "bottom": last_y})
    return tables


# ---------------------------------------------------------------------------
# 2. Endnotes, so every [^n] can pull in the note it points to
# ---------------------------------------------------------------------------
def read_endnotes(doc):
    """{n: note text} from the endnote pages, read in order: note n+1 starts at the next
    line that begins with the number n+1, so "1) No use of ..." inside note 7 is not a note."""
    text = ""
    for number in ENDNOTE_PAGES:
        page = doc[number - 1]
        lines = page_lines(page, markers=False)
        tables = [(t["top"] - 6, t["bottom"] + 6) for t in rebuild_tables(page, page_lines(page))]
        legal = min([l["y0"] for l in lines if join(l["spans"]).startswith("©")], default=page.rect.height)
        for line in lines:
            line_text = " ".join(s["text"] for s in line["spans"])     # keeps "17" apart from "Responsible"
            if line_text == "Endnotes" or line["y0"] >= legal or is_footer(line, page.rect.height) \
                    or any(top <= line["y0"] <= bottom for top, bottom in tables):
                continue                                       # title, legal small print, footer, table
            text += line_text + "\n"

    notes, starts, pos, n = {}, [], 0, 1
    while m := re.compile(rf"(?:^|\n)\s*{n}\s+(?=\S)").search(text, pos):
        starts.append((n, m.start(), m.end()))
        pos, n = m.end(), n + 1
    for i, (n, _, begin) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        notes[n] = re.sub(r"\s+", " ", text[begin:end]).strip()
    return notes


# ---------------------------------------------------------------------------
# 3. Element chunks
# ---------------------------------------------------------------------------
def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(•])|(?<=\[\^\d\])\s+(?=[A-Z])", text) if s.strip()]


def prose_with_markers(page, lines, skip):
    """Page text outside the table areas, footer dropped, note markers kept as [^n]."""
    kept = [join(l["spans"]) for l in lines
            if not is_footer(l, page.rect.height) and not any(top <= l["y0"] <= bottom for top, bottom in skip)]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def element_chunks(path, max_tokens=220):
    """One chunk per table; prose in small sentence groups. Any [^n] marker pulls its endnote into the chunk."""
    doc = pymupdf.open(path)
    notes = read_endnotes(doc)
    chunks = []

    def add(text, page, kind):
        refs = sorted({int(n) for n in re.findall(r"\[\^(\d+)\]", text) if int(n) in notes})
        body = re.sub(r"\[\^\d+\]", "", text).strip()
        body += "".join(f"\n[Note {n}] {notes[n][:700]}" for n in refs)
        chunks.append({"id": len(chunks), "text": body, "page_start": page, "page_end": page, "kind": kind,
                       "heading_path": f"page {page}, {kind}"})

    for number, page in enumerate(doc, start=1):
        lines = page_lines(page)
        tables = rebuild_tables(page, lines)
        for t in tables:
            add(t["text"], number, "table")
        if number in ENDNOTE_PAGES:                            # the endnote pages themselves
            continue
        prose = prose_with_markers(page, lines, [(t["top"] - 6, t["bottom"] + 6) for t in tables])
        current, size = [], 0
        for s in split_sentences(prose):
            if current and size + tokens(s) > max_tokens:
                add(" ".join(current), number, "text")
                current, size = [], 0
            current.append(s)
            size += tokens(s)
        if current:
            add(" ".join(current), number, "text")
    return chunks


# ---------------------------------------------------------------------------
# 4. Visual pages -> a vision model's description
# ---------------------------------------------------------------------------
def visual_pages(path):
    """Pages that have images or real vector artwork, rendered as PNG: {page: png bytes}."""
    pages = {}
    for number, page in enumerate(pymupdf.open(path), start=1):
        if not page.get_images() and len(page.get_drawings()) < 5:
            continue                                           # plain text page: nothing to see
        pages[number] = page.get_pixmap(dpi=110).tobytes("png")
    return pages


def vision_chunks(path, describe, start_id):
    """`describe(page_number, png_bytes) -> text` is the vision model call (see chatbot.py)."""
    out = []
    for number, png in visual_pages(path).items():
        out.append({"id": start_id + len(out), "text": describe(number, png), "page_start": number,
                    "page_end": number, "kind": "vision", "heading_path": f"page {number}, vision description"})
    return out


if __name__ == "__main__":
    import os

    pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "iMac_PER_Oct2024.pdf")
    chunks = element_chunks(pdf)
    kinds = {k: sum(c["kind"] == k for c in chunks) for k in ("table", "text")}
    print(f"{len(chunks)} element chunks {kinds}, {len(visual_pages(pdf))} pages for the vision model\n")
    for c in chunks:
        if c["kind"] == "table" or "[Note 18]" in c["text"]:
            print(f"--- chunk {c['id']} ({c['heading_path']})\n{c['text']}\n")
