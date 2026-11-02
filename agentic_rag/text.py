"""Small text helpers shared by every stage (tokenising, light stemming, sentence splitting)."""

import re

_WORD = re.compile(r"[a-z0-9$%]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+")

STOP = set(
    """a an the is are was were be been am i me my we our you your it its of to in on for with at by from and or
    but if then so as that this these those do does did can could should would will shall may might have has had
    not no what which who whom how when where why about into than too very just also please want need get got like
    tell know there here any some more most other such only own same up down out over under again further once
    after before still ever never really actually hasn't haven't didn't don't isn't wasn't won't can't doesn't
    i'm it's what's there's let's
    put give take go come see look find send keep let try ask say make""".split()
)

# Words that appear in many questions but say nothing about the topic. They are used for ranking
# but are not treated as "must-have facts" by the evidence check.
GENERIC = set(
    """long much many window policy rule information info detail process procedure time thing stuff way work use make
    help question possible able mean day week month year hour minute two three four five six seven eight nine ten
    ago today yesterday tomorrow now soon old new first last next one something anything everything someone
    order parcel package item product purchase""".split()
)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """A deliberately tiny stemmer. It only needs to be consistent between documents and questions."""
    if "'" in word:
        word = word.split("'")[0]
    if len(word) > 4:
        if word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 5:
            word = _undouble(word[:-3])
        elif word.endswith("ed"):
            word = _undouble(word[:-2])
        elif word.endswith(("sses", "xes", "zes", "ches", "shes")):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
    if len(word) > 8 and word.endswith("ment"):
        word = word[:-4]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def _undouble(w: str) -> str:
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
        return w[:-1]
    return w


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP]


def content_stems(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


def is_wordlike(token: str) -> bool:
    """True for ordinary words; False for numbers, money and percentages."""
    return token[0].isalpha()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


GENERIC_STEMS = {stem(w) for w in GENERIC}
