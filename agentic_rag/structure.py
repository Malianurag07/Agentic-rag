"""Turn a flat list of Blocks into the plan's tree: Section (topic) > numbered subsection (type) > blocks (text).

Recognises, in any of the three formats:
    "SECTION 3: TITLE" / "Section 3: Title"        -> a section
    "3.2 Title (Structured Matrix): body text..."  -> a subsection (title ends at the first colon; body may follow)
Anything before the first section is kept as section 0 (front matter); text between a section heading and its first
numbered subsection becomes an implicit "n.0 Introduction"."""

import re
from dataclasses import dataclass, field

from .blocks import Block

SECTION_RE = re.compile(r"^\s*section\s+(\d{1,2})\s*[:.\-–]\s*(.+)$", re.IGNORECASE | re.DOTALL)
SUB_RE = re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\s+(?=[A-Z\"“(])(.+)$", re.DOTALL)
MAX_TITLE = 190


@dataclass
class Subsection:
    number: str                     # "3.2"
    title: str
    blocks: list = field(default_factory=list)

    def page_range(self) -> tuple[int | None, int | None]:
        pages = [b.page for b in self.blocks if b.page]
        last = [b.meta.get("last_page", b.page) for b in self.blocks if b.page]
        return (min(pages), max(last)) if pages else (None, None)


@dataclass
class Section:
    number: str                     # "3"
    title: str
    subsections: list = field(default_factory=list)


def _split_title(text: str) -> tuple[str, str]:
    """A numbered paragraph is either a bare heading or "Heading: body". Return (title, body).
    A short paragraph with no sentence break is a heading in its own right, colon or not, so a title such as
    "UNSTRUCTURED POLICY PROSE: PERFORMANCE CALIBRATION, PIPS, AND MOONLIGHTING" is kept whole."""
    if len(text) <= MAX_TITLE + 40 and not re.search(r"[.!?]\s", text) and not text.rstrip().endswith("."):
        return text.rstrip(": ").strip(), ""
    cut = text.find(": ")
    if 0 < cut <= MAX_TITLE:
        return text[:cut].strip(), text[cut + 2:].strip()
    if len(text) <= MAX_TITLE + 40:
        return text.rstrip(": ").strip(), ""
    return text[:MAX_TITLE].rsplit(" ", 1)[0] + "...", text            # no clear heading: keep the whole paragraph as body


def build_tree(blocks: list[Block]) -> list[Section]:
    sections: list[Section] = [Section("0", "Front matter")]
    current_sub = Subsection("0.0", "Front matter")
    sections[0].subsections.append(current_sub)

    for block in blocks:
        if block.kind == "para":
            m = SECTION_RE.match(block.text)
            if m and len(block.text) <= MAX_TITLE + 60:
                section = Section(m.group(1), m.group(2).strip().rstrip(":"))
                current_sub = Subsection(f"{section.number}.0", "Introduction")
                section.subsections.append(current_sub)
                sections.append(section)
                continue
            m = SUB_RE.match(block.text)
            if m and m.group(1) == sections[-1].number:                        # n.m only counts inside section n
                title, body = _split_title(f"{m.group(3)}")
                current_sub = Subsection(f"{m.group(1)}.{m.group(2)}", title)
                sections[-1].subsections.append(current_sub)
                if body:
                    current_sub.blocks.append(Block("para", text=body, page=block.page, meta=block.meta))
                continue
        current_sub.blocks.append(block)

    for section in sections:                                                    # drop implicit introductions with no content
        section.subsections = [s for s in section.subsections if s.blocks or not s.number.endswith(".0")]
    return [s for s in sections if s.subsections]
