"""Small, safe tools the agent may call (plan section 7.2) plus the prompt-injection sanitizer (section 8)."""

import ast
import operator
import re

# ---- calculator ---------------------------------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic. Anything that is not a number or an arithmetic operator is rejected."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](walk(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    return walk(ast.parse(expression.strip(), mode="eval"))


_MONEY = re.compile(r"(?<![\w.])\$?(\d+(?:\.\d+)?)(?!\s*%)(?![\d.]*\s*%)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def try_calculate(question: str) -> tuple[str, float] | None:
    """If the question has an amount, a percentage and a fee/deduction wording, compute the net amount.
    Returns (expression, result) or None. Deliberately narrow: it is a tool, not a guess."""
    q = question.lower()
    if not any(w in q for w in ("fee", "deduct", "restocking", "discount", "charge")):
        return None
    pct = _PERCENT.search(question)
    amounts = [m for m in _MONEY.finditer(question) if not question[m.end():m.end() + 1] == "%"]
    if not pct or not amounts:
        return None
    amount = float(amounts[0].group(1))
    rate = float(pct.group(1))
    expr = f"{amount:g}*(1-{rate:g}/100)"
    return expr, safe_eval(expr)


# ---- prompt-injection sanitizer -----------------------------------------------------------------------------------
_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|rules|prompts?)",
    r"(reveal|show|print|leak|output) (me )?(the |your )?(system prompt|hidden prompt|instructions|password|secret|api key)",
    r"you are now\b",
    r"new instructions\s*:",
    r"^\s*system\s*:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.IGNORECASE)


def sanitize(text: str) -> tuple[str, list[str]]:
    """Treat retrieved text as data, never as instructions: drop sentences that try to give the AI orders."""
    from .text import sentences

    kept, removed = [], []
    for s in sentences(text):
        (removed if _INJECTION_RE.search(s) else kept).append(s)
    return " ".join(kept), removed
