"""GBNF grammars that force a model to speak only mathematics.

Constrained decoding is the honest version of "the model thinks in mathematics":
you cannot change what a quantised checkpoint has learned, but you *can* make it
mathematically impossible for it to emit anything except a well-formed symbolic
expression.  Once the output alphabet is a math grammar, every token the model
produces is a math token, and ARIA's parser and verifier take over from there.

Two modes:

``loose``
    Enforces the three-line contract (``goal``/``var``/``expr``) and a
    mathematical character set.  Fast, keeps the model fluent; ARIA still
    validates the parse and repairs on failure.

``strict``
    Enforces the full expression grammar for a conservative subset of ARIA
    notation (numbers, identifiers, ``+ - * / ^``, parentheses, relations).
    Guarantees parseability, but rigid spacing measurably hurts fluency; use it
    with small models that otherwise drift.

Both are generated as text so they can be inspected, diffed and shipped as a
``.gbnf`` file next to a model.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

__all__ = ["build_grammar", "validate_grammar", "GRAMMAR_MODES"]

GRAMMAR_MODES = ("loose", "strict")

_GOALS = "solve | simplify | factor | expand | verify | differentiate | compute | analyze"

_LOOSE = f"""# ARIA math-only output grammar (loose mode)
# Constrains the model to emit the ARIA contract, expression characters only.
root        ::= "goal: " goal "\\n" var-line? "expr: " expr-text "\\n"?
goal        ::= "solve" | "simplify" | "factor" | "expand" | "verify" | "differentiate" | "compute" | "analyze"
var-line    ::= "var: " ident "\\n"
ident       ::= [A-Za-z_] [A-Za-z0-9_]*
expr-text   ::= expr-char+
expr-char   ::= [0-9A-Za-z_] | " " | "+" | "-" | "*" | "/" | "^" | "(" | ")" | "=" | "<" | ">" | "," | "." | "|" | "[" | "]"
"""

_STRICT = f"""# ARIA math-only output grammar (strict mode)
# Guarantees the expression parses; spacing is significant.
root         ::= "goal: " goal "\\n" var-line? "expr: " expr "\\n"?
goal         ::= "solve" | "simplify" | "factor" | "expand" | "verify" | "differentiate" | "compute" | "analyze"
var-line     ::= "var: " ident "\\n"
ident        ::= [A-Za-z_] [A-Za-z0-9_]*
expr         ::= additive ((" = " | " == " | " < " | " > " | " <= " | " >= ") additive)?
additive     ::= multiplicative ((" + " | " - ") multiplicative)*
multiplicative ::= power ((" * " | " / ") power)*
power        ::= primary (" ^ " exponent)?
exponent     ::= "-"? [0-9]+ | "(" [0-9]+ "/" [0-9]+ ")" | "(" additive ")"
primary      ::= number | ident | "(" additive ")" | ident "(" additive ")"
number       ::= [0-9]+ ("." [0-9]+)?
"""

_TEMPLATES = {"loose": _LOOSE, "strict": _STRICT}


def build_grammar(mode: str = "loose") -> str:
    """Return a GBNF grammar enforcing the ARIA contract."""
    if mode not in _TEMPLATES:
        raise ValueError(f"unknown grammar mode {mode!r}; choose from {', '.join(GRAMMAR_MODES)}")
    return _TEMPLATES[mode].strip() + "\n"


# ---------------------------------------------------------------------------
# Validation (structural: we cannot run llama.cpp here)
# ---------------------------------------------------------------------------

_RULE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*::=\s*(.*)$")


def _strip_literals(body: str) -> str:
    """Remove string literals and character classes so identifier scanning is safe."""
    return re.sub(r'"(?:[^"\\]|\\.)*"|\[(?:[^\]\\]|\\.)*\]', " ", body)


def validate_grammar(text: str) -> Tuple[bool, List[str]]:
    """Check that a grammar is internally consistent.

    Verifies: every rule reference is defined, the ``root`` rule exists, quotes
    and parentheses balance, and no rule is defined twice.  This is a structural
    check -- it does not prove the grammar accepts what you think it does (only a
    real sampler does that).
    """
    problems: List[str] = []
    rules: Dict[str, str] = {}
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _RULE_RE.match(raw_line)
        if match:
            name, body = match.group(1), match.group(2)
            if name in rules:
                problems.append(f"rule {name!r} defined twice")
            rules.setdefault(name, "")
            rules[name] += " " + body
            current = name
        elif current is not None and raw_line.startswith((" ", "\t")):
            rules[current] += " " + line
        else:
            problems.append(f"unparseable line: {raw_line!r}")

    if "root" not in rules:
        problems.append("no 'root' rule defined")

    for name, body in rules.items():
        without_literals = _strip_literals(body)
        if without_literals.count("(") != without_literals.count(")"):
            problems.append(f"unbalanced parentheses in rule {name!r}")
        if body.count('"') % 2 != 0:
            problems.append(f"unbalanced quotes in rule {name!r}")
        for reference in re.findall(r"[A-Za-z_][A-Za-z0-9_-]*", without_literals):
            token = reference.rstrip("-")
            if token in rules or token in {"true", "false"}:
                continue
            problems.append(f"rule {name!r} references undefined rule {token!r}")

    return (not problems, problems)


def rules_of(grammar: str) -> Dict[str, str]:
    """Return the rule table of a grammar (used by tests and docs)."""
    rules: Dict[str, str] = {}
    current: str | None = None
    for raw_line in grammar.splitlines():
        match = _RULE_RE.match(raw_line)
        if match:
            current = match.group(1)
            rules[current] = match.group(2).strip()
        elif current and raw_line.strip() and not raw_line.strip().startswith("#"):
            rules[current] += " " + raw_line.strip()
    return rules
