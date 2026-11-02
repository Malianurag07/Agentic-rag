"""PDF loader. Generic text extractors scramble three things in a document like this one, so each is handled directly:

* running headers and footers  -> detected by repetition across pages and removed
* tables                       -> rebuilt from the thin rule segments the PDF draws (they give exact column edges and row
                                  boundaries) plus the header row that sits above a thick rule; a table that continues on
                                  the next page repeats its header and is merged back into one table
* two-column prose             -> lines with a central gutter are split into a left and a right column, read in that order
* images / charts              -> kept as figure blocks (no text can be read from them without OCR or a vision model)

Paragraphs that run across a page break are stitched back together."""

import html
import logging
import re
import statistics
import subprocess
from collections import Counter, defaultdict

import pdfplumber

from .blocks import Block
from .loaders import clean

logging.getLogger("pdfminer").setLevel(logging.ERROR)

JOIN = "\u0001"                                   # marks a line-end hyphen so it can be resolved once the vocabulary is known
# a line that always starts a new paragraph: section/numbered headings, bullets, figure captions, boxed rule callouts
HEADING_START = re.compile(r"^(Section\s+\d+\s*[:.]|\d+\.\d+(\.\d+)?\s+\S|[\u2022\-]\s|Figure\s+\d+(\.\d+)?\s*:|MANDATORY CROSS-DOMAIN RULE)", re.I)


def _poppler_words(path) -> list[list[dict]]:
    """Word boxes from poppler (`pdftotext -bbox`). It reads fonts that pdfminer cannot (e.g. the digits in currency
    ranges) and does not merge hyphenated line ends."""
    xml = subprocess.run(["pdftotext", "-bbox", str(path), "-"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    pages = []
    for page_xml in xml.split("<page ")[1:]:
        pages.append([_tighten({"text": html.unescape(m.group(5)), "x0": float(m.group(1)), "top": float(m.group(2)),
                                "x1": float(m.group(3)), "bottom": float(m.group(4))})
                      for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', page_xml)])
    return pages


def _tighten(w: dict) -> dict:
    """Poppler boxes include the font's full ascent and descent (about 2.7x the glyph height). The centre is right, so
    shrink the box around it to the size of the glyphs; rules and line grouping then compare like with like."""
    centre, half = (w["top"] + w["bottom"]) / 2, (w["bottom"] - w["top"]) * 0.19
    w["top"], w["bottom"] = centre - half, centre + half
    return w


def _mark_bold(page, words: list[dict]) -> None:
    """Flag each word as bold using the page's glyph fonts (poppler's word boxes carry no font information)."""
    grid: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for ch in page.chars:
        if "bold" in ch["fontname"].lower() and ch["text"].strip():
            cx, cy = (ch["x0"] + ch["x1"]) / 2, (ch["top"] + ch["bottom"]) / 2
            grid[(int(cx // 20), int(cy // 20))].append((cx, cy))
    for w in words:
        cy = (w["top"] + w["bottom"]) / 2
        near = [pt for gx in range(int(w["x0"] // 20) - 1, int(w["x1"] // 20) + 2)
                for gy in (int(cy // 20) - 1, int(cy // 20), int(cy // 20) + 1) for pt in grid.get((gx, gy), [])]
        w["bold"] = any(w["x0"] - 0.5 <= x <= w["x1"] + 0.5 and abs(y - cy) <= 3.5 for x, y in near)


# ---- low-level helpers ----------------------------------------------------------------------------------------------------
def _centre(w: dict) -> float:
    return (w["top"] + w["bottom"]) / 2


def _group_lines(words: list[dict], tol: float = 3.5) -> list[list[dict]]:
    """Group words into visual lines by the vertical CENTRE of their boxes. (Not the top edge: a hyperlink is drawn in a
    shorter box, so its top edge sits several points below the surrounding words even though it is on the same line.)"""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (_centre(w), w["x0"])):
        if lines and abs(_centre(w) - _centre(lines[-1][0])) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]


def _line_text(line: list[dict]) -> str:
    return " ".join(w["text"] for w in line)


def _join_lines(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if not out:
            out = t
        elif out.endswith("-") and t[:1].isalpha():
            out = out[:-1] + "-" + JOIN + t               # "three-" + "pronged": keep the hyphen unless the vocabulary says otherwise
        else:
            out += " " + t
    return out


def _furniture(pages_words: list[list[dict]], height: float) -> set[str]:
    """Normalised text of lines that sit in the top/bottom margin and repeat on many pages (headers, footers, page numbers)."""
    seen: Counter = Counter()
    for words in pages_words:
        margin = [w for w in words if w["top"] < height * 0.10 or w["bottom"] > height * 0.92]
        for line in _group_lines(margin):
            seen[re.sub(r"\d+", "#", _line_text(line))] += 1
    need = max(2, int(0.4 * len(pages_words)))
    return {text for text, n in seen.items() if n >= need}


def _strip_furniture(words: list[dict], height: float, furniture: set[str]) -> list[dict]:
    keep: list[dict] = []
    for line in _group_lines(words):
        in_margin = line[0]["top"] < height * 0.10 or line[-1]["bottom"] > height * 0.92
        if in_margin and re.sub(r"\d+", "#", _line_text(line)) in furniture:
            continue
        keep += line
    return keep


# ---- tables ---------------------------------------------------------------------------------------------------------------
def _rule_rows(page) -> list[dict]:
    """Thin filled rectangles are the table's row rules. Rules made of one long segment are plain dividers, not tables."""
    rows: dict[float, dict] = {}
    for r in page.rects:
        if r["height"] <= 2.2 and 15 < r["width"] < page.width - 20:
            key = round(r["top"] * 2) / 2
            row = rows.setdefault(key, {"top": r["top"], "bottom": r["top"], "segs": set(), "thick": False})
            row["segs"].add((round(r["x0"]), round(r["x1"])))
            row["bottom"] = max(row["bottom"], r["bottom"])
            row["thick"] = row["thick"] or r["height"] >= 1.2
    out = []
    for key in sorted(rows):
        row = rows[key]
        row["segs"] = sorted(row["segs"])
        if len(row["segs"]) >= 2:
            out.append(row)
    return out


def _signature(row: dict) -> tuple:
    return tuple(s[0] for s in row["segs"]) + (row["segs"][-1][1],)


def _assemble_tables(page, words: list[dict]) -> list[dict]:
    tables: list[dict] = []
    current = None
    for rule in _rule_rows(page):
        sig = _signature(rule)
        last_bottom = (current["rules"][-1] if current and current["rules"] else current["head_rule"])["bottom"] if current else 0
        if rule["thick"] or current is None or sig != current["sig"] or rule["top"] - last_bottom > 240:
            if not rule["thick"]:
                current = None
                continue                                   # a body rule with no header above it: not a table we can read
            current = {"sig": sig, "head_rule": rule, "rules": []}
            tables.append(current)
        else:
            current["rules"].append(rule)
    result = []
    for t in tables:
        if not t["rules"]:
            continue
        edges = list(t["sig"])

        def column(w):
            cx = (w["x0"] + w["x1"]) / 2
            idx = 0
            for i, e in enumerate(edges[:-1]):
                if cx >= e - 1:
                    idx = i
            return idx

        def cells(band_words):
            grid = defaultdict(list)
            for w in band_words:
                grid[column(w)].append(w)
            return [clean(_join_lines([_line_text(l) for l in _group_lines(grid[i])])) for i in range(len(edges) - 1)]

        x_lo, x_hi = edges[0] - 2, edges[-1] + 2
        in_x = [w for w in words if w["x0"] >= x_lo and w["x1"] <= x_hi]
        head_rule = t["head_rule"]
        above = [w for w in in_x if w["bottom"] <= head_rule["top"] + 0.5]
        header_lines: list[list[dict]] = []
        for line in reversed(_group_lines(above)):
            if line[0]["top"] < head_rule["top"] - 70:
                break
            if header_lines and header_lines[-1][0]["top"] - line[0]["top"] > 18:
                break                                      # a bigger gap: this line belongs to the paragraph above the table
            if not header_lines and head_rule["top"] - line[-1]["bottom"] > 14:
                break
            header_lines.append(line)
        header_words = [w for l in header_lines for w in l]
        header_top = min((w["top"] for w in header_words), default=head_rule["top"])
        header = cells(header_words)
        rows, prev = [], head_rule["bottom"]
        for rule in t["rules"]:
            band = [w for w in in_x if w["top"] >= prev - 0.5 and w["bottom"] <= rule["top"] + 1]
            row = cells(band)
            if any(row):
                rows.append(row)
            prev = rule["bottom"]
        result.append({"top": header_top, "bottom": t["rules"][-1]["bottom"], "header": header, "rows": rows})
    return result


# ---- prose ----------------------------------------------------------------------------------------------------------------
def _gaps(line: list[dict], width: float) -> list[tuple[float, float]]:
    """Wide horizontal gaps near the middle of the page. One of them may be a column gutter."""
    return [(a["x1"], b["x0"]) for a, b in zip(line, line[1:])
            if b["x0"] - a["x1"] >= 11 and 0.38 * width <= (a["x1"] + b["x0"]) / 2 <= 0.62 * width]


def _overlap(g1: tuple[float, float], g2: tuple[float, float]) -> tuple[float, float] | None:
    lo, hi = max(g1[0], g2[0]), min(g1[1], g2[1])
    return (lo, hi) if hi - lo >= 6 else None


def _find_gutter(lines: list[list[dict]], i: int, width: float) -> tuple[float, float] | None:
    """A real gutter shows up at the same place on at least two consecutive lines; one wide gap in a justified line does not."""
    here = _gaps(lines[i], width)
    for nxt in (lines[i + 1:i + 2]):
        for g1 in here:
            for g2 in _gaps(nxt, width):
                common = _overlap(g1, g2)
                if common:
                    return common
    return None


def _paragraphs(lines: list[list[dict]]) -> list[str]:
    """Group consecutive lines into paragraphs using the vertical gap and numbered-heading starts."""
    if not lines:
        return []
    tops = [l[0]["top"] for l in lines]
    diffs = [b - a for a, b in zip(tops, tops[1:]) if 0 < b - a < 20]
    pitch = statistics.median(diffs) if diffs else 12.0
    paras, buf = [], [_line_text(lines[0])]
    all_bold = lambda l: all(w.get("bold") for w in l)
    for prev, line in zip(lines, lines[1:]):
        text = _line_text(line)
        # an all-bold line is a heading; it ends where a line starts in regular text (a bold lead-in that wraps does not)
        font_break = all_bold(prev) and not line[0].get("bold")
        if line[0]["top"] - prev[0]["top"] > pitch * 1.45 or HEADING_START.match(text) or font_break:
            paras.append(_join_lines(buf))
            buf = [text]
        else:
            buf.append(text)
    paras.append(_join_lines(buf))
    return [clean(p) for p in paras if clean(p)]


def _prose(words: list[dict], width: float) -> list[str]:
    lines = _group_lines(words)
    out: list[str] = []
    i = 0
    while i < len(lines):
        gutter = _find_gutter(lines, i, width)
        if gutter is None:
            j = i + 1
            while j < len(lines) and _find_gutter(lines, j, width) is None:
                j += 1
            out += _paragraphs(lines[i:j])
            i = j
            continue
        split = (gutter[0] + gutter[1]) / 2
        left, right, j = [], [], i
        while j < len(lines):
            line = lines[j]
            has_gutter = any(_overlap(g, gutter) for g in _gaps(line, width))
            if has_gutter:
                pass
            elif line[-1]["x1"] <= gutter[0] + 1 or line[0]["x0"] >= gutter[1] - 1:
                pass                                          # a line that sits entirely inside one column
            else:
                break                                         # a full-width line ends the two-column block
            lo = [w for w in line if (w["x0"] + w["x1"]) / 2 < split]
            hi = [w for w in line if (w["x0"] + w["x1"]) / 2 >= split]
            if lo:
                left.append(lo)
            if hi:
                right.append(hi)
            j += 1
        out += _paragraphs(left) + _paragraphs(right)
        i = j
    return out


# ---- the loader -------------------------------------------------------------------------------------------------------------
def load_pdf(path) -> list[Block]:
    blocks: list[Block] = []
    with pdfplumber.open(str(path)) as pdf:
        all_words = _poppler_words(path)
        furniture = _furniture(all_words, pdf.pages[0].height)
        for number, (page, words) in enumerate(zip(pdf.pages, all_words), start=1):
            height = page.height
            words = _strip_furniture(words, height, furniture)
            _mark_bold(page, words)
            tables = _assemble_tables(page, words)
            figures = [{"top": im["top"], "bottom": im["bottom"], "x0": im["x0"], "x1": im["x1"]} for im in page.images
                       if im["width"] > 100 and im["height"] > 40]
            elements = [("table", t) for t in tables] + [("figure", f) for f in figures]
            claimed = [(e["top"] - 1, e["bottom"] + 1) for _, e in elements]
            free = [w for w in words if not any(a <= (w["top"] + w["bottom"]) / 2 <= b for a, b in claimed)]
            segments: list[tuple[float, list[dict]]] = []
            cuts = sorted(claimed)
            edges = [0.0] + [x for c in cuts for x in c] + [height]
            for lo, hi in zip(edges[0::2], edges[1::2]):
                seg = [w for w in free if lo <= w["top"] < hi]
                if seg:
                    segments.append((lo, seg))
            items = [(lo, "text", seg) for lo, seg in segments] + [(e["top"], kind, e) for kind, e in elements]
            for _, kind, payload in sorted(items, key=lambda x: x[0]):
                if kind == "text":
                    blocks += [Block("para", text=t, page=number) for t in _prose(payload, page.width)]
                elif kind == "table":
                    blocks.append(Block("table", header=payload["header"], rows=payload["rows"], page=number))
                else:
                    w, h = round(payload["x1"] - payload["x0"]), round(payload["bottom"] - payload["top"])
                    blocks.append(Block("figure", page=number, meta={"width": w, "height": h},
                                        text=f"[Figure on page {number}, {w}x{h} pt: an image or chart. No text can be read "
                                             f"from it without OCR or a vision model.]"))
    blocks = _merge_continued_tables(blocks)
    blocks = _stitch_paragraphs(blocks)
    return _resolve_hyphens(blocks)


def _merge_continued_tables(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (b.kind == "table" and prev is not None and prev.kind == "table" and b.page == (prev.meta.get("last_page", prev.page)) + 1
                and [h.lower() for h in b.header] == [h.lower() for h in prev.header]):
            prev.rows += b.rows                           # the same table, continued after a page break
            prev.meta["last_page"] = b.page
            continue
        out.append(b)
    return out


def _stitch_paragraphs(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (b.kind == "para" and prev is not None and prev.kind == "para" and b.page == prev.page + 1
                and not re.search(r"[.!?:;)\]\"”]$", prev.text) and b.text[:1].islower()):
            prev.text = _join_lines([prev.text, b.text]) if prev.text.endswith("-") else prev.text + " " + b.text
            continue
        out.append(b)
    return out


def _resolve_hyphens(blocks: list[Block]) -> list[Block]:
    """A line-end hyphen is a real one ("three-pronged") unless the unhyphenated word exists elsewhere in the document."""
    vocab: Counter = Counter()
    for b in blocks:
        parts = [b.text] if b.kind == "para" else [c for r in [b.header] + b.rows for c in r]
        for p in parts:
            vocab.update(re.findall(r"[A-Za-z]+", p.replace(JOIN, " ")))
    fix = lambda s: re.sub(r"([A-Za-z]+)-" + JOIN + r"([A-Za-z]+)",
                           lambda m: m.group(1) + m.group(2) if vocab[m.group(1) + m.group(2)] > 0 else m.group(1) + "-" + m.group(2), s)
    for b in blocks:
        if b.kind == "para":
            b.text = fix(b.text).replace(JOIN, "")
        elif b.kind == "table":
            b.header = [fix(c).replace(JOIN, "") for c in b.header]
            b.rows = [[fix(c).replace(JOIN, "") for c in r] for r in b.rows]
    return blocks
