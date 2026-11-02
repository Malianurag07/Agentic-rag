"""Format loaders (plan section 5.1: load and clean). Each returns a list of Blocks; see blocks.py.

    .txt / .md   line based; understands the "[TABLE COLUMNS] a | b" / "* x -- y" table notation
    .docx        walks paragraphs and tables in document order; finds embedded images
    .pdf         see pdf_loader.py (rebuilds tables from rule lines, reads two-column prose, strips page furniture)

Legacy binary .doc files are not read directly: convert them to .docx first (for example with LibreOffice:
`soffice --headless --convert-to docx file.doc`)."""

import re
from pathlib import Path

from .blocks import Block

TXT_ROW = re.compile(r"^\s*\*\s+(.*)$")


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


# ---- TXT ----------------------------------------------------------------------------------------------------------------
def load_txt(path: Path) -> list[Block]:
    blocks: list[Block] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped or re.fullmatch(r"[=\-_*#~ ]{4,}", stripped):          # blank line or a divider line
            i += 1
            continue
        if stripped.startswith("[TABLE COLUMNS]"):
            header = [clean(c) for c in stripped[len("[TABLE COLUMNS]"):].split("|")]
            rows = []
            i += 1
            while i < len(lines) and (not lines[i].strip() or re.fullmatch(r"[\-= ]{4,}", lines[i].strip())):
                i += 1                                                           # the dashed rule under the header
            while i < len(lines) and TXT_ROW.match(lines[i]):
                rows.append([clean(c) for c in TXT_ROW.match(lines[i]).group(1).split("  --  ")])
                i += 1
            blocks.append(Block("table", header=header, rows=rows))
            continue
        blocks.append(Block("para", text=clean(stripped)))
        i += 1
    return blocks


# ---- DOCX ---------------------------------------------------------------------------------------------------------------
def load_docx(path: Path) -> list[Block]:
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks: list[Block] = []

    def image_blocks(paragraph) -> list[Block]:
        out = []
        for blip in paragraph._p.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            part = document.part.related_parts.get(rid)
            if part is not None:
                out.append(Block("figure", text="", meta={"image": Path(str(part.partname)).name,
                                                          "bytes": len(part.blob)}))
        return out

    def table_to_blocks(table: Table) -> list[Block]:
        grid = []
        for row in table.rows:
            cells, seen = [], set()
            for cell in row.cells:
                if id(cell._tc) in seen:                                          # a merged cell repeats itself
                    continue
                seen.add(id(cell._tc))
                cells.append(clean(cell.text))
            grid.append(cells)
        if not grid:
            return []
        if len(grid) == 1 and len(grid[0]) == 1:                                  # a one-cell banner or callout box
            return [Block("para", text=grid[0][0])] if grid[0][0] else []
        if all(len(c.split()) > 25 for c in grid[0]) and len(grid) == 1:          # prose set in columns (not a data table)
            return [Block("para", text=c) for c in grid[0] if c]
        return [Block("table", header=grid[0], rows=grid[1:])]

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = clean(paragraph.text)
            if text:
                bullet = paragraph.style is not None and "bullet" in paragraph.style.name.lower()
                blocks.append(Block("para", text=("• " + text) if bullet else text))
            blocks.extend(image_blocks(paragraph))
        elif child.tag == qn("w:tbl"):
            blocks.extend(table_to_blocks(Table(child, document)))
    return blocks


def load_document(path: str | Path) -> list[Block]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return load_txt(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".pdf":
        from .pdf_loader import load_pdf
        return load_pdf(path)
    if suffix == ".doc":
        raise ValueError(f"{path.name}: legacy .doc is not supported; convert it to .docx first (see loaders.py)")
    raise ValueError(f"{path.name}: unsupported file type {suffix!r}")
