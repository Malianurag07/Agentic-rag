"""The format-neutral intermediate form. Every loader (TXT, DOCX, PDF) produces a list of Blocks, so structure detection,
chunking and storage are written once and behave the same for all formats."""

from dataclasses import dataclass, field


@dataclass
class Block:
    kind: str                         # "para" | "table" | "figure"
    text: str = ""                    # paragraph text, or a caption/placeholder for a figure
    header: list = field(default_factory=list)    # table column titles
    rows: list = field(default_factory=list)      # table body rows (lists of cell strings)
    page: int | None = None           # PDF page number (1-based); None for formats without pages
    meta: dict = field(default_factory=dict)

    def words(self) -> int:
        if self.kind == "table":
            return sum(len(c.split()) for r in [self.header] + self.rows for c in r)
        return len(self.text.split())
