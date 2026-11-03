# Part 2: Loading documents (text, Word, PDF)

This is the first stage of the pipeline: **read a file and turn it into Blocks**. Each format needs its own reader, because a text file, a Word file and a PDF store the same information in completely different ways. A text file is just lines. A Word file is a tree of paragraphs and tables. A PDF is only a list of letters placed at coordinates on a page, so a PDF reader has to *reconstruct* paragraphs, tables and columns from positions. `loaders.py` handles the first two and dispatches to `pdf_loader.py` for PDFs.

## `agentic_rag/loaders.py`

The text and Word readers, a small text-cleaning helper, and the `load_document` function that picks the right reader by file extension.

### File description

**Lines 1-8**

```python
"""Format loaders (plan section 5.1: load and clean). Each returns a list of Blocks; see blocks.py.

    .txt / .md   line based; understands the "[TABLE COLUMNS] a | b" / "* x -- y" table notation
    .docx        walks paragraphs and tables in document order; finds embedded images
    .pdf         see pdf_loader.py (rebuilds tables from rule lines, reads two-column prose, strips page furniture)

Legacy binary .doc files are not read directly: convert them to .docx first (for example with LibreOffice:
`soffice --headless --convert-to docx file.doc`)."""
```

The docstring lists the three formats and how each is read, and adds a warning: old binary `.doc` files cannot be read directly and must first be converted to `.docx` (the command shown uses LibreOffice).

### Imports and the table-row pattern

**Line 10**

```python
import re
```

`re` is the regular-expression tool (pattern matching in text).

**Line 11**

```python
from pathlib import Path
```

`Path` is Python's object for file locations. It is safer than plain strings: it handles slashes and extensions for you.

**Line 13**

```python
from .blocks import Block
```

Imports the `Block` record from `blocks.py`. The dot means 'from this same package'.

**Line 15**

```python
TXT_ROW = re.compile(r"^\s*\*\s+(.*)$")
```

`TXT_ROW` is a pattern for one table row in the text-file notation. It matches a line that starts with optional spaces, then `*`, then spaces, then captures the rest of the line (`(.*)`). Rows in the text files look like `* Tier 1  --  $100  --  Manager`.

### clean(): tidy one piece of text

**Line 18**

```python
def clean(text: str) -> str:
```

`clean` takes a string and returns a tidied string.

**Line 19**

```python
    text = text.replace(" ", " ").replace("​", "")
```

Replaces two invisible troublemakers: the non-breaking space (looks like a space but is a different character) becomes a normal space, and the zero-width space (completely invisible) is deleted. Without this, words that look identical would not match.

**Line 20**

```python
    return re.sub(r"[ \t]+", " ", text).strip()
```

Collapses every run of spaces or tabs into a single space, then trims spaces from both ends.

### Reading a text file

**Lines 23-24**

```python
# ---- TXT ----------------------------------------------------------------------------------------------------------------
def load_txt(path: Path) -> list[Block]:
```

`load_txt` reads a `.txt` or `.md` file and returns a list of Blocks.

**Line 25**

```python
    blocks: list[Block] = []
```

Starts with an empty list that will collect the Blocks.

**Line 26**

```python
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
```

Reads the whole file as text (UTF-8, replacing any unreadable bytes instead of crashing) and splits it into a list of lines.

**Line 27**

```python
    i = 0
```

`i` is the position of the line being looked at. A `while` loop is used instead of a `for` loop because the code sometimes needs to jump several lines at once (when it reads a table).

**Line 28**

```python
    while i < len(lines):
```

Keep going until every line has been read.

**Lines 29-30**

```python
        line = lines[i].rstrip()
        stripped = line.strip()
```

Takes the current line, removes trailing spaces (`line`), and also leading spaces (`stripped`).

**Lines 31-33**

```python
        if not stripped or re.fullmatch(r"[=\-_*#~ ]{4,}", stripped):          # blank line or a divider line
            i += 1
            continue
```

If the line is empty, or is only a divider such as `=====` or `-----` (four or more of those characters), skip it: move to the next line and start the loop again.

**Line 34**

```python
        if stripped.startswith("[TABLE COLUMNS]"):
```

If the line starts with `[TABLE COLUMNS]`, a table begins here.

**Line 35**

```python
            header = [clean(c) for c in stripped[len("[TABLE COLUMNS]"):].split("|")]
```

The part after the marker holds the column titles separated by `|`. It is split on `|` and each title is cleaned. This list becomes the table's header.

**Lines 36-37**

```python
            rows = []
            i += 1
```

Prepares an empty list for the rows and moves to the next line.

**Lines 38-39**

```python
            while i < len(lines) and (not lines[i].strip() or re.fullmatch(r"[\-= ]{4,}", lines[i].strip())):
                i += 1                                                           # the dashed rule under the header
```

Skips blank lines and the dashed line printed under the header.

**Lines 40-42**

```python
            while i < len(lines) and TXT_ROW.match(lines[i]):
                rows.append([clean(c) for c in TXT_ROW.match(lines[i]).group(1).split("  --  ")])
                i += 1
```

While the current line looks like a table row (starts with `*`): split its text on the double-dash separator `  --  `, clean each cell, add the row, and move on.

**Line 43**

```python
            blocks.append(Block("table", header=header, rows=rows))
```

Packs the header and rows into one `table` Block.

**Line 44**

```python
            continue
```

Jumps back to the top of the loop (the table has been fully handled).

**Lines 45-46**

```python
        blocks.append(Block("para", text=clean(stripped)))
        i += 1
```

Any other line is an ordinary paragraph: clean it, wrap it in a `para` Block, and move to the next line.

**Line 47**

```python
    return blocks
```

Hands back the list of Blocks.

### Reading a Word file

**Lines 50-51**

```python
# ---- DOCX ---------------------------------------------------------------------------------------------------------------
def load_docx(path: Path) -> list[Block]:
```

`load_docx` reads a `.docx` file.

**Lines 52-55**

```python
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
```

The imports are inside the function on purpose: the Word library is only needed when a Word file is actually loaded, so the rest of the program starts without it. `qn` converts short XML names into the long form the file uses; `Table` and `Paragraph` are the library's types.

**Lines 57-58**

```python
    document = docx.Document(str(path))
    blocks: list[Block] = []
```

Opens the document and prepares the list of Blocks.

**Lines 60-62**

```python
    def image_blocks(paragraph) -> list[Block]:
        out = []
        for blip in paragraph._p.iter(qn("a:blip")):
```

An inner helper that finds images. A paragraph in Word can hold a picture; each picture is an XML element called `blip`. This loops over every `blip` inside the paragraph.

**Lines 63-65**

```python
            rid = blip.get(qn("r:embed"))
            part = document.part.related_parts.get(rid)
            if part is not None:
```

Each picture points to its stored image by an id. The code looks the image part up, and carries on only if it exists.

**Lines 66-67**

```python
                out.append(Block("figure", text="", meta={"image": Path(str(part.partname)).name,
                                                          "bytes": len(part.blob)}))
```

Adds a `figure` Block that records the image's file name and size in bytes. Its text stays empty: the pixels cannot be read as text.

**Line 68**

```python
        return out
```

Returns the figures found (possibly none).

**Lines 70-71**

```python
    def table_to_blocks(table: Table) -> list[Block]:
        grid = []
```

Another inner helper: turn one Word table into Blocks. It begins with an empty grid (a list of rows).

**Lines 72-74**

```python
        for row in table.rows:
            cells, seen = [], set()
            for cell in row.cells:
```

Loops over each row, and inside it over each cell. `cells` is the row being built; `seen` remembers which cells were already used.

**Lines 75-76**

```python
                if id(cell._tc) in seen:                                          # a merged cell repeats itself
                    continue
```

A merged cell appears several times in the row (once per column it spans). Python tells it apart by identity (`id`), so a repeat is skipped.

**Lines 77-79**

```python
                seen.add(id(cell._tc))
                cells.append(clean(cell.text))
            grid.append(cells)
```

Remembers the cell, adds its cleaned text to the row, and adds the finished row to the grid.

**Lines 80-81**

```python
        if not grid:
            return []
```

An empty table produces no Blocks.

**Lines 82-83**

```python
        if len(grid) == 1 and len(grid[0]) == 1:                                  # a one-cell banner or callout box
            return [Block("para", text=grid[0][0])] if grid[0][0] else []
```

A table with exactly one cell is not data: it is a banner or call-out box. It becomes a plain paragraph (or nothing if empty).

**Lines 84-85**

```python
        if all(len(c.split()) > 25 for c in grid[0]) and len(grid) == 1:          # prose set in columns (not a data table)
            return [Block("para", text=c) for c in grid[0] if c]
```

A single row whose cells are all long (over 25 words) is prose laid out in columns, not a data table. Each cell becomes its own paragraph, left to right, which is the correct reading order.

**Line 86**

```python
        return [Block("table", header=grid[0], rows=grid[1:])]
```

Otherwise it is a real table: the first row is the header, the rest are body rows.

**Lines 88-89**

```python
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
```

The main loop. A Word document body is an ordered list of children; this walks them in order so paragraphs and tables stay in their original sequence. A child tagged `w:p` is a paragraph.

**Lines 90-92**

```python
            paragraph = Paragraph(child, document)
            text = clean(paragraph.text)
            if text:
```

Wraps the child as a paragraph object, cleans its text, and carries on only if there is any text.

**Lines 93-94**

```python
                bullet = paragraph.style is not None and "bullet" in paragraph.style.name.lower()
                blocks.append(Block("para", text=("• " + text) if bullet else text))
```

Checks whether the paragraph's style name contains 'bullet'. If so a bullet symbol is put in front of the text. Then the Block is added.

**Line 95**

```python
            blocks.extend(image_blocks(paragraph))
```

Adds any pictures contained in the paragraph.

**Lines 96-97**

```python
        elif child.tag == qn("w:tbl"):
            blocks.extend(table_to_blocks(Table(child, document)))
```

A child tagged `w:tbl` is a table: it is converted with the helper above and added.

**Line 98**

```python
    return blocks
```

Returns the list.

### Choosing the right reader

**Lines 101-103**

```python
def load_document(path: str | Path) -> list[Block]:
    path = Path(path)
    suffix = path.suffix.lower()
```

`load_document` is the single entry point. It turns the path into a `Path` and looks at the file extension in lower case.

**Lines 104-105**

```python
    if suffix in (".txt", ".md"):
        return load_txt(path)
```

`.txt` or `.md` files go to the text reader.

**Lines 106-107**

```python
    if suffix == ".docx":
        return load_docx(path)
```

`.docx` files go to the Word reader.

**Lines 108-110**

```python
    if suffix == ".pdf":
        from .pdf_loader import load_pdf
        return load_pdf(path)
```

`.pdf` files go to the PDF reader. That import is also inside the `if`, so the PDF libraries load only when a PDF is used.

**Lines 111-112**

```python
    if suffix == ".doc":
        raise ValueError(f"{path.name}: legacy .doc is not supported; convert it to .docx first (see loaders.py)")
```

A legacy `.doc` file raises a clear error telling the user to convert it.

**Line 113**

```python
    raise ValueError(f"{path.name}: unsupported file type {suffix!r}")
```

Any other extension raises an 'unsupported file type' error.


---

## `agentic_rag/pdf_loader.py`

The PDF reader. A PDF stores no paragraphs, tables or columns: just letters at coordinates. The file looks like a long list of helper functions, but they form a pipeline, and each one solves one problem that generic PDF tools get wrong: (1) get accurate word positions, (2) remove page headers and footers, (3) rebuild tables from their ruled lines, (4) read two-column text in the right order, (5) rebuild paragraphs, (6) stitch things back together across pages.

### File description

**Lines 1-10**

```python
"""PDF loader. Generic text extractors scramble three things in a document like this one, so each is handled directly:

* running headers and footers  -> detected by repetition across pages and removed
* tables                       -> rebuilt from the thin rule segments the PDF draws (they give exact column edges and row
                                  boundaries) plus the header row that sits above a thick rule; a table that continues on
                                  the next page repeats its header and is merged back into one table
* two-column prose             -> lines with a central gutter are split into a left and a right column, read in that order
* images / charts              -> kept as figure blocks (no text can be read from them without OCR or a vision model)

Paragraphs that run across a page break are stitched back together."""
```

The docstring lists the four layout problems this reader solves on purpose (running headers and footers, tables, two-column prose, images) and notes that paragraphs split across pages are stitched back together.

### Imports and constants

**Lines 12-17**

```python
import html
import logging
import re
import statistics
import subprocess
from collections import Counter, defaultdict
```

Standard-library tools: `html` (decode `&amp;` and friends), `logging` (silence noisy warnings), `re` (patterns), `statistics` (the median), `subprocess` (run an outside program), and two helper containers: `Counter` (counts things) and `defaultdict` (a dictionary that creates missing entries automatically).

**Line 19**

```python
import pdfplumber
```

`pdfplumber` is the third-party library that reads PDF drawing information (lines, rectangles, fonts, images).

**Lines 21-22**

```python
from .blocks import Block
from .loaders import clean
```

Imports the `Block` record and the `clean` helper from the loaders file.

**Line 24**

```python
logging.getLogger("pdfminer").setLevel(logging.ERROR)
```

Turns the PDF library's chatty warnings down to errors only (it complains about harmless font details).

**Line 26**

```python
JOIN = "\u0001"                                   # marks a line-end hyphen so it can be resolved once the vocabulary is known
```

`JOIN` is a special invisible marker character. When a line ends in a hyphen and the next line continues the word, the code leaves this marker so it can decide later whether the hyphen is real ('three-pronged') or just a line-break hyphen.

**Lines 27-28**

```python
# a line that always starts a new paragraph: section/numbered headings, bullets, figure captions, boxed rule callouts
HEADING_START = re.compile(r"^(Section\s+\d+\s*[:.]|\d+\.\d+(\.\d+)?\s+\S|[\u2022\-]\s|Figure\s+\d+(\.\d+)?\s*:|MANDATORY CROSS-DOMAIN RULE)", re.I)
```

`HEADING_START` is a pattern for lines that always begin a *new* paragraph: 'Section 3:' headings, numbered subsections like '3.2 Title', bullets, 'Figure 1.1:' captions, and the boxed 'MANDATORY CROSS-DOMAIN RULE' call-outs. `re.I` makes it ignore capital letters.

### Step 1: get accurate word positions

The first problem is getting each word together with its position on the page. The reader asks an outside program, **poppler**, because it reads fonts the Python library cannot (for example the digits in `$4,999`).

**Lines 31-33**

```python
def _poppler_words(path) -> list[list[dict]]:
    """Word boxes from poppler (`pdftotext -bbox`). It reads fonts that pdfminer cannot (e.g. the digits in currency
    ranges) and does not merge hyphenated line ends."""
```

`_poppler_words` returns, for every page, a list of words. Each word is a small dictionary with its text and its box (left, top, right, bottom). The docstring explains why poppler is used.

**Line 34**

```python
    xml = subprocess.run(["pdftotext", "-bbox", str(path), "-"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
```

Runs the command `pdftotext -bbox file -` and captures its output. `-bbox` makes it print every word with its coordinates as XML text; `-` means 'print to the screen, not a file'. The bytes are decoded to text.

**Lines 35-36**

```python
    pages = []
    for page_xml in xml.split("<page ")[1:]:
```

Prepares a list for the pages. The output is split at each `<page ` tag; `[1:]` throws away the text before the first page.

**Lines 37-39**

```python
        pages.append([_tighten({"text": html.unescape(m.group(5)), "x0": float(m.group(1)), "top": float(m.group(2)),
                                "x1": float(m.group(3)), "bottom": float(m.group(4))})
                      for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', page_xml)])
```

For every word on the page: a pattern pulls out the four numbers (xMin, yMin, xMax, yMax) and the word text; the text is decoded (`&amp;` becomes `&`), the numbers are turned into decimals, and the box is passed through `_tighten`. All the words of the page are added as one list.

**Line 40**

```python
    return pages
```

Returns the list of pages.

**Lines 43-45**

```python
def _tighten(w: dict) -> dict:
    """Poppler boxes include the font's full ascent and descent (about 2.7x the glyph height). The centre is right, so
    shrink the box around it to the size of the glyphs; rules and line grouping then compare like with like."""
```

`_tighten` fixes a poppler quirk: its boxes are about 2.7 times taller than the real letters (they include space for the tallest and lowest possible characters). The centre is correct, so the box is shrunk around the centre.

**Lines 46-47**

```python
    centre, half = (w["top"] + w["bottom"]) / 2, (w["bottom"] - w["top"]) * 0.19
    w["top"], w["bottom"] = centre - half, centre + half
```

Works out the vertical centre and a new half-height (19% of the old height), then sets the top and bottom to centre minus and plus that.

**Line 48**

```python
    return w
```

Returns the adjusted word.

**Lines 51-52**

```python
def _mark_bold(page, words: list[dict]) -> None:
    """Flag each word as bold using the page's glyph fonts (poppler's word boxes carry no font information)."""
```

`_mark_bold` flags which words are bold. Poppler gives no font information, so the information comes from the other library.

**Line 53**

```python
    grid: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
```

Builds a coarse grid (a dictionary from a 20-point square to a list of positions) so that nearby bold letters can be found quickly.

**Lines 54-57**

```python
    for ch in page.chars:
        if "bold" in ch["fontname"].lower() and ch["text"].strip():
            cx, cy = (ch["x0"] + ch["x1"]) / 2, (ch["top"] + ch["bottom"]) / 2
            grid[(int(cx // 20), int(cy // 20))].append((cx, cy))
```

For every letter on the page: if its font name contains 'bold' and it is not a space, its centre point is computed and filed into its grid square.

**Lines 58-59**

```python
    for w in words:
        cy = (w["top"] + w["bottom"]) / 2
```

Now for every word, compute its vertical centre.

**Lines 60-61**

```python
        near = [pt for gx in range(int(w["x0"] // 20) - 1, int(w["x1"] // 20) + 2)
                for gy in (int(cy // 20) - 1, int(cy // 20), int(cy // 20) + 1) for pt in grid.get((gx, gy), [])]
```

Collects the bold-letter points from the grid squares around the word (one square left and right, and the row above, same and below).

**Line 62**

```python
        w["bold"] = any(w["x0"] - 0.5 <= x <= w["x1"] + 0.5 and abs(y - cy) <= 3.5 for x, y in near)
```

Marks the word bold if any bold letter lies inside its horizontal range and within 3.5 points of its vertical centre.

### Step 2: group words into lines and strip headers and footers

**Lines 65-67**

```python
# ---- low-level helpers ----------------------------------------------------------------------------------------------------
def _centre(w: dict) -> float:
    return (w["top"] + w["bottom"]) / 2
```

`_centre` is a tiny helper that returns the vertical centre of a word's box.

**Lines 70-72**

```python
def _group_lines(words: list[dict], tol: float = 3.5) -> list[list[dict]]:
    """Group words into visual lines by the vertical CENTRE of their boxes. (Not the top edge: a hyperlink is drawn in a
    shorter box, so its top edge sits several points below the surrounding words even though it is on the same line.)"""
```

`_group_lines` gathers words into visual lines. The docstring explains an important detail: grouping uses the vertical **centre**, not the top edge, because a hyperlink is drawn in a shorter box, so its top edge sits a few points lower even though it is on the same line (this once caused email addresses to be misplaced).

**Line 73**

```python
    lines: list[list[dict]] = []
```

An empty list of lines.

**Lines 74-76**

```python
    for w in sorted(words, key=lambda w: (_centre(w), w["x0"])):
        if lines and abs(_centre(w) - _centre(lines[-1][0])) <= tol:
            lines[-1].append(w)
```

Walks through the words sorted by vertical centre (then left to right). If there is already a line and this word's centre is within 3.5 points of the line's first word, the word joins that line.

**Lines 77-78**

```python
        else:
            lines.append([w])
```

Otherwise it starts a new line.

**Line 79**

```python
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]
```

Sorts the words of each line left to right and returns the lines.

**Lines 82-83**

```python
def _line_text(line: list[dict]) -> str:
    return " ".join(w["text"] for w in line)
```

`_line_text` joins the words of a line into one string with spaces.

**Lines 86-90**

```python
def _join_lines(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if not out:
            out = t
```

`_join_lines` joins several lines of text into a paragraph. The first piece just starts the paragraph.

**Lines 91-92**

```python
        elif out.endswith("-") and t[:1].isalpha():
            out = out[:-1] + "-" + JOIN + t               # "three-" + "pronged": keep the hyphen unless the vocabulary says otherwise
```

If the text so far ends with a hyphen and the next line starts with a letter, the word was broken across lines. The hyphen is kept and the invisible `JOIN` marker is added, to be resolved later.

**Lines 93-94**

```python
        else:
            out += " " + t
```

Otherwise the next line is just appended after a space.

**Line 95**

```python
    return out
```

Returns the joined text.

**Lines 98-99**

```python
def _furniture(pages_words: list[list[dict]], height: float) -> set[str]:
    """Normalised text of lines that sit in the top/bottom margin and repeat on many pages (headers, footers, page numbers)."""
```

`_furniture` finds the **page furniture**: running headers, footers and page numbers that repeat on every page.

**Lines 100-101**

```python
    seen: Counter = Counter()
    for words in pages_words:
```

A counter, then a loop over each page's words.

**Line 102**

```python
        margin = [w for w in words if w["top"] < height * 0.10 or w["bottom"] > height * 0.92]
```

Takes only the words in the top 10% or bottom 8% of the page.

**Lines 103-104**

```python
        for line in _group_lines(margin):
            seen[re.sub(r"\d+", "#", _line_text(line))] += 1
```

Groups them into lines, replaces every number with `#` (so 'Page 3 of 30' and 'Page 4 of 30' look the same) and counts how often each pattern appears.

**Line 105**

```python
    need = max(2, int(0.4 * len(pages_words)))
```

A line must appear on at least 40% of the pages (and at least 2) to count as furniture.

**Line 106**

```python
    return {text for text, n in seen.items() if n >= need}
```

Returns the set of repeating patterns.

**Lines 109-111**

```python
def _strip_furniture(words: list[dict], height: float, furniture: set[str]) -> list[dict]:
    keep: list[dict] = []
    for line in _group_lines(words):
```

`_strip_furniture` removes them from one page. It groups the words into lines.

**Lines 112-114**

```python
        in_margin = line[0]["top"] < height * 0.10 or line[-1]["bottom"] > height * 0.92
        if in_margin and re.sub(r"\d+", "#", _line_text(line)) in furniture:
            continue
```

A line is dropped if it is in the page margin *and* its normalised text is one of the repeating patterns.

**Lines 115-116**

```python
        keep += line
    return keep
```

All other lines are kept and returned as a flat list of words.

### Step 3: rebuild tables from their ruled lines

This PDF draws each table row as a thin horizontal rectangle (a rule). Where rules start and stop gives the exact edges of the columns, so tables can be rebuilt exactly, with no guessing from text alignment.

**Lines 119-121**

```python
# ---- tables ---------------------------------------------------------------------------------------------------------------
def _rule_rows(page) -> list[dict]:
    """Thin filled rectangles are the table's row rules. Rules made of one long segment are plain dividers, not tables."""
```

`_rule_rows` finds the row rules on a page. The docstring notes that a single very long rule is a plain divider, not a table.

**Lines 122-123**

```python
    rows: dict[float, dict] = {}
    for r in page.rects:
```

A dictionary of rules keyed by their vertical position; loop over every rectangle on the page.

**Lines 124-129**

```python
        if r["height"] <= 2.2 and 15 < r["width"] < page.width - 20:
            key = round(r["top"] * 2) / 2
            row = rows.setdefault(key, {"top": r["top"], "bottom": r["top"], "segs": set(), "thick": False})
            row["segs"].add((round(r["x0"]), round(r["x1"])))
            row["bottom"] = max(row["bottom"], r["bottom"])
            row["thick"] = row["thick"] or r["height"] >= 1.2
```

Keeps only rectangles that are thin (at most 2.2 high) and not nearly page-wide. Rectangles at the same height are grouped (positions rounded to half a point); for each the code records the left-right segments, the lowest edge, and whether any piece is thick (at least 1.2) - the thick rule marks the bottom of a header row.

**Lines 130-135**

```python
    out = []
    for key in sorted(rows):
        row = rows[key]
        row["segs"] = sorted(row["segs"])
        if len(row["segs"]) >= 2:
            out.append(row)
```

Sorts the rules top to bottom, sorts each rule's segments left to right, and keeps only rules made of **two or more** segments (one segment means a plain divider).

**Line 136**

```python
    return out
```

Returns the table rules.

**Lines 139-140**

```python
def _signature(row: dict) -> tuple:
    return tuple(s[0] for s in row["segs"]) + (row["segs"][-1][1],)
```

`_signature` describes a rule by where its column edges are: the left edge of each segment plus the last right edge. Two rules with the same signature belong to the same table.

**Lines 143-145**

```python
def _assemble_tables(page, words: list[dict]) -> list[dict]:
    tables: list[dict] = []
    current = None
```

`_assemble_tables` groups rules into tables. `current` is the table being built.

**Lines 146-148**

```python
    for rule in _rule_rows(page):
        sig = _signature(rule)
        last_bottom = (current["rules"][-1] if current and current["rules"] else current["head_rule"])["bottom"] if current else 0
```

For each rule: compute its signature and the bottom edge of the previous rule.

**Lines 149-152**

```python
        if rule["thick"] or current is None or sig != current["sig"] or rule["top"] - last_bottom > 240:
            if not rule["thick"]:
                current = None
                continue                                   # a body rule with no header above it: not a table we can read
```

A new table starts if the rule is thick, or there is no table yet, or the signature changed, or the gap is over 240 points. If the rule is *not* thick there is no header above it, so it is ignored.

**Lines 153-154**

```python
            current = {"sig": sig, "head_rule": rule, "rules": []}
            tables.append(current)
```

Starts a new table record (signature, header rule, and a list for the body rules) and adds it to the list.

**Lines 155-156**

```python
        else:
            current["rules"].append(rule)
```

Otherwise the rule is another body row of the current table.

**Lines 157-160**

```python
    result = []
    for t in tables:
        if not t["rules"]:
            continue
```

Now it converts each table of rules into real cells. Tables with no body rules are skipped.

**Line 161**

```python
        edges = list(t["sig"])
```

`edges` are the column boundaries.

**Lines 163-169**

```python
        def column(w):
            cx = (w["x0"] + w["x1"]) / 2
            idx = 0
            for i, e in enumerate(edges[:-1]):
                if cx >= e - 1:
                    idx = i
            return idx
```

Inner helper `column`: given a word, find its column. It takes the word's horizontal centre and picks the last column whose left edge is at or before it.

**Lines 171-175**

```python
        def cells(band_words):
            grid = defaultdict(list)
            for w in band_words:
                grid[column(w)].append(w)
            return [clean(_join_lines([_line_text(l) for l in _group_lines(grid[i])])) for i in range(len(edges) - 1)]
```

Inner helper `cells`: given the words of one row band, drop each word into its column, then for each column group its words into lines, join them, and clean the text. The result is the row as a list of cell strings.

**Lines 177-178**

```python
        x_lo, x_hi = edges[0] - 2, edges[-1] + 2
        in_x = [w for w in words if w["x0"] >= x_lo and w["x1"] <= x_hi]
```

Restricts attention to words that lie horizontally within the table's edges.

**Lines 179-180**

```python
        head_rule = t["head_rule"]
        above = [w for w in in_x if w["bottom"] <= head_rule["top"] + 0.5]
```

Takes the thick header rule, and the words lying above it.

**Lines 181-189**

```python
        header_lines: list[list[dict]] = []
        for line in reversed(_group_lines(above)):
            if line[0]["top"] < head_rule["top"] - 70:
                break
            if header_lines and header_lines[-1][0]["top"] - line[0]["top"] > 18:
                break                                      # a bigger gap: this line belongs to the paragraph above the table
            if not header_lines and head_rule["top"] - line[-1]["bottom"] > 14:
                break
            header_lines.append(line)
```

Finds the header row by walking **upwards** from the rule line by line. It stops if a line is more than 70 points above the rule, or more than 18 points above the previous header line (a bigger gap means that line belongs to the paragraph above the table), or the first line is too far above the rule. Otherwise the line is part of the header.

**Lines 190-192**

```python
        header_words = [w for l in header_lines for w in l]
        header_top = min((w["top"] for w in header_words), default=head_rule["top"])
        header = cells(header_words)
```

Collects the header words, finds the header's top edge, and builds the header cells.

**Lines 193-199**

```python
        rows, prev = [], head_rule["bottom"]
        for rule in t["rules"]:
            band = [w for w in in_x if w["top"] >= prev - 0.5 and w["bottom"] <= rule["top"] + 1]
            row = cells(band)
            if any(row):
                rows.append(row)
            prev = rule["bottom"]
```

Rows: for each body rule, the words between the previous rule and this one form one row band. If the row has any text it is kept. The next band starts at this rule's bottom.

**Lines 200-201**

```python
        result.append({"top": header_top, "bottom": t["rules"][-1]["bottom"], "header": header, "rows": rows})
    return result
```

Records the table (top, bottom, header, rows) and returns all tables of the page.

### Step 4: read two-column prose

Two-column text breaks ordinary PDF readers: they read straight across both columns, mixing them. This reader finds the gutter (the empty vertical strip between the columns) and reads the left column, then the right.

**Lines 204-208**

```python
# ---- prose ----------------------------------------------------------------------------------------------------------------
def _gaps(line: list[dict], width: float) -> list[tuple[float, float]]:
    """Wide horizontal gaps near the middle of the page. One of them may be a column gutter."""
    return [(a["x1"], b["x0"]) for a, b in zip(line, line[1:])
            if b["x0"] - a["x1"] >= 11 and 0.38 * width <= (a["x1"] + b["x0"]) / 2 <= 0.62 * width]
```

A comment marking the start of the prose section. `_gaps` returns the wide gaps (at least 11 points) between neighbouring words of a line, but only near the middle of the page (38% to 62% of the width), where a gutter could be.

**Lines 211-213**

```python
def _overlap(g1: tuple[float, float], g2: tuple[float, float]) -> tuple[float, float] | None:
    lo, hi = max(g1[0], g2[0]), min(g1[1], g2[1])
    return (lo, hi) if hi - lo >= 6 else None
```

`_overlap` checks whether two gaps overlap by at least 6 points, and returns the overlapping stretch, or nothing.

**Lines 216-217**

```python
def _find_gutter(lines: list[list[dict]], i: int, width: float) -> tuple[float, float] | None:
    """A real gutter shows up at the same place on at least two consecutive lines; one wide gap in a justified line does not."""
```

`_find_gutter` decides whether line `i` starts a two-column block. The docstring: a real gutter appears at the same place on at least two consecutive lines; one wide gap in a justified line does not count.

**Lines 218-225**

```python
    here = _gaps(lines[i], width)
    for nxt in (lines[i + 1:i + 2]):
        for g1 in here:
            for g2 in _gaps(nxt, width):
                common = _overlap(g1, g2)
                if common:
                    return common
    return None
```

Gets this line's central gaps, then compares each with the next line's gaps. The first overlap found is the gutter. If none, there is no gutter here.

**Lines 228-231**

```python
def _paragraphs(lines: list[list[dict]]) -> list[str]:
    """Group consecutive lines into paragraphs using the vertical gap and numbered-heading starts."""
    if not lines:
        return []
```

`_paragraphs` turns a list of lines into paragraphs. With no lines there is nothing to do.

**Lines 232-234**

```python
    tops = [l[0]["top"] for l in lines]
    diffs = [b - a for a, b in zip(tops, tops[1:]) if 0 < b - a < 20]
    pitch = statistics.median(diffs) if diffs else 12.0
```

Measures the usual distance between lines (the median of line-to-line distances under 20 points, or 12 as a default). This is the 'pitch'.

**Lines 235-236**

```python
    paras, buf = [], [_line_text(lines[0])]
    all_bold = lambda l: all(w.get("bold") for w in l)
```

Starts the first paragraph with the first line. `all_bold` is a tiny test: is every word of a line bold?

**Lines 237-240**

```python
    for prev, line in zip(lines, lines[1:]):
        text = _line_text(line)
        # an all-bold line is a heading; it ends where a line starts in regular text (a bold lead-in that wraps does not)
        font_break = all_bold(prev) and not line[0].get("bold")
```

For each next line: it gets its text, and checks whether the previous line was an all-bold heading and this line starts in regular type (a heading has ended).

**Lines 241-243**

```python
        if line[0]["top"] - prev[0]["top"] > pitch * 1.45 or HEADING_START.match(text) or font_break:
            paras.append(_join_lines(buf))
            buf = [text]
```

A new paragraph starts if the gap is more than 1.45 times the normal pitch, or the line looks like a heading start, or a heading just ended. The finished paragraph is stored.

**Lines 244-245**

```python
        else:
            buf.append(text)
```

Otherwise the line continues the current paragraph.

**Lines 246-247**

```python
    paras.append(_join_lines(buf))
    return [clean(p) for p in paras if clean(p)]
```

Stores the last paragraph, cleans every paragraph and drops empty ones.

**Lines 250-254**

```python
def _prose(words: list[dict], width: float) -> list[str]:
    lines = _group_lines(words)
    out: list[str] = []
    i = 0
    while i < len(lines):
```

`_prose` reads a block of lines in the correct order. It groups the words into lines and walks through them.

**Lines 255-262**

```python
        gutter = _find_gutter(lines, i, width)
        if gutter is None:
            j = i + 1
            while j < len(lines) and _find_gutter(lines, j, width) is None:
                j += 1
            out += _paragraphs(lines[i:j])
            i = j
            continue
```

If no gutter starts here, it keeps taking lines until a gutter is found, and turns that stretch of single-column lines into paragraphs.

**Lines 263-264**

```python
        split = (gutter[0] + gutter[1]) / 2
        left, right, j = [], [], i
```

When a gutter is found: its middle is the split point, and two lists collect left-column and right-column lines.

**Lines 265-267**

```python
        while j < len(lines):
            line = lines[j]
            has_gutter = any(_overlap(g, gutter) for g in _gaps(line, width))
```

Takes lines one at a time and checks whether this line also has the gutter.

**Lines 268-273**

```python
            if has_gutter:
                pass
            elif line[-1]["x1"] <= gutter[0] + 1 or line[0]["x0"] >= gutter[1] - 1:
                pass                                          # a line that sits entirely inside one column
            else:
                break                                         # a full-width line ends the two-column block
```

Accepted if the line has the gutter, or lies entirely inside one column. A full-width line ends the two-column block.

**Lines 274-280**

```python
            lo = [w for w in line if (w["x0"] + w["x1"]) / 2 < split]
            hi = [w for w in line if (w["x0"] + w["x1"]) / 2 >= split]
            if lo:
                left.append(lo)
            if hi:
                right.append(hi)
            j += 1
```

Splits the line's words into left and right by each word's **centre** (so no word can ever be lost) and adds each part to its column.

**Lines 281-283**

```python
        out += _paragraphs(left) + _paragraphs(right)
        i = j
    return out
```

All left-column paragraphs first, then the right-column ones, and continues after the block. Returns the paragraphs.

### Step 5: the main loader

**Lines 286-291**

```python
# ---- the loader -------------------------------------------------------------------------------------------------------------
def load_pdf(path) -> list[Block]:
    blocks: list[Block] = []
    with pdfplumber.open(str(path)) as pdf:
        all_words = _poppler_words(path)
        furniture = _furniture(all_words, pdf.pages[0].height)
```

`load_pdf` is the main function. It opens the PDF with the library, gets all words from poppler, and works out the repeating page furniture from the first page's height.

**Lines 292-295**

```python
        for number, (page, words) in enumerate(zip(pdf.pages, all_words), start=1):
            height = page.height
            words = _strip_furniture(words, height, furniture)
            _mark_bold(page, words)
```

For each page: remove the furniture words, and mark which words are bold.

**Line 296**

```python
            tables = _assemble_tables(page, words)
```

Rebuilds the page's tables.

**Lines 297-298**

```python
            figures = [{"top": im["top"], "bottom": im["bottom"], "x0": im["x0"], "x1": im["x1"]} for im in page.images
                       if im["width"] > 100 and im["height"] > 40]
```

Finds images on the page, ignoring tiny ones (decoration).

**Lines 299-301**

```python
            elements = [("table", t) for t in tables] + [("figure", f) for f in figures]
            claimed = [(e["top"] - 1, e["bottom"] + 1) for _, e in elements]
            free = [w for w in words if not any(a <= (w["top"] + w["bottom"]) / 2 <= b for a, b in claimed)]
```

Tables and figures are 'elements'. Their vertical ranges are 'claimed'; any word inside a claimed range already belongs to a table or figure, so only the other ('free') words are prose.

**Lines 302-308**

```python
            segments: list[tuple[float, list[dict]]] = []
            cuts = sorted(claimed)
            edges = [0.0] + [x for c in cuts for x in c] + [height]
            for lo, hi in zip(edges[0::2], edges[1::2]):
                seg = [w for w in free if lo <= w["top"] < hi]
                if seg:
                    segments.append((lo, seg))
```

The page is cut at the claimed ranges into stretches of free text, so the text above, between and below tables stays separate.

**Lines 309-310**

```python
            items = [(lo, "text", seg) for lo, seg in segments] + [(e["top"], kind, e) for kind, e in elements]
            for _, kind, payload in sorted(items, key=lambda x: x[0]):
```

All text stretches and elements are combined and sorted top to bottom, which restores the page's reading order.

**Lines 311-312**

```python
                if kind == "text":
                    blocks += [Block("para", text=t, page=number) for t in _prose(payload, page.width)]
```

A text stretch becomes paragraphs through `_prose`.

**Lines 313-314**

```python
                elif kind == "table":
                    blocks.append(Block("table", header=payload["header"], rows=payload["rows"], page=number))
```

A table becomes a `table` Block.

**Lines 315-319**

```python
                else:
                    w, h = round(payload["x1"] - payload["x0"]), round(payload["bottom"] - payload["top"])
                    blocks.append(Block("figure", page=number, meta={"width": w, "height": h},
                                        text=f"[Figure on page {number}, {w}x{h} pt: an image or chart. No text can be read "
                                             f"from it without OCR or a vision model.]"))
```

A figure becomes a `figure` Block with its size, and an honest note: no text can be read from an image without OCR.

**Lines 320-322**

```python
    blocks = _merge_continued_tables(blocks)
    blocks = _stitch_paragraphs(blocks)
    return _resolve_hyphens(blocks)
```

After all pages: merge tables that continued across pages, stitch paragraphs split across pages, and resolve hyphens. Returns the result.

### Step 6: repair across page breaks

**Lines 325-328**

```python
def _merge_continued_tables(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
```

`_merge_continued_tables` walks all Blocks, remembering the previous output Block.

**Lines 329-333**

```python
        if (b.kind == "table" and prev is not None and prev.kind == "table" and b.page == (prev.meta.get("last_page", prev.page)) + 1
                and [h.lower() for h in b.header] == [h.lower() for h in prev.header]):
            prev.rows += b.rows                           # the same table, continued after a page break
            prev.meta["last_page"] = b.page
            continue
```

If this block is a table directly following a table from the previous page, and the header cells match, it is the same table continued: add its rows to the earlier one and record the last page.

**Lines 334-335**

```python
        out.append(b)
    return out
```

Otherwise the block is kept as it is.

**Lines 338-341**

```python
def _stitch_paragraphs(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
```

`_stitch_paragraphs` repairs sentences split by a page break.

**Lines 342-345**

```python
        if (b.kind == "para" and prev is not None and prev.kind == "para" and b.page == prev.page + 1
                and not re.search(r"[.!?:;)\]\"”]$", prev.text) and b.text[:1].islower()):
            prev.text = _join_lines([prev.text, b.text]) if prev.text.endswith("-") else prev.text + " " + b.text
            continue
```

If this paragraph begins on the page after the previous paragraph, the previous one does not end with sentence punctuation, and this one starts with a lower-case letter, they are one paragraph. They are joined.

**Lines 346-347**

```python
        out.append(b)
    return out
```

Otherwise the block is kept as it is.

**Lines 350-351**

```python
def _resolve_hyphens(blocks: list[Block]) -> list[Block]:
    """A line-end hyphen is a real one ("three-pronged") unless the unhyphenated word exists elsewhere in the document."""
```

`_resolve_hyphens` decides about the hyphens marked earlier: keep a hyphen ('three-pronged') unless the word without it appears elsewhere in the document.

**Lines 352-356**

```python
    vocab: Counter = Counter()
    for b in blocks:
        parts = [b.text] if b.kind == "para" else [c for r in [b.header] + b.rows for c in r]
        for p in parts:
            vocab.update(re.findall(r"[A-Za-z]+", p.replace(JOIN, " ")))
```

Builds a vocabulary: it counts every word that appears in the document, in paragraphs and in table cells.

**Lines 357-358**

```python
    fix = lambda s: re.sub(r"([A-Za-z]+)-" + JOIN + r"([A-Za-z]+)",
                           lambda m: m.group(1) + m.group(2) if vocab[m.group(1) + m.group(2)] > 0 else m.group(1) + "-" + m.group(2), s)
```

`fix` rewrites each marked hyphen: if the two halves joined together form a word seen elsewhere, the hyphen is dropped; otherwise it stays.

**Lines 359-364**

```python
    for b in blocks:
        if b.kind == "para":
            b.text = fix(b.text).replace(JOIN, "")
        elif b.kind == "table":
            b.header = [fix(c).replace(JOIN, "") for c in b.header]
            b.rows = [[fix(c).replace(JOIN, "") for c in r] for r in b.rows]
```

Applies `fix` and removes the marker in every paragraph, table header and table cell.

**Line 365**

```python
    return blocks
```

Returns the cleaned Blocks.


---
