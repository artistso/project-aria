"""Parsing: notation and natural-language prompts in, symbolic state out.

Two front ends share a single recursive-descent core:

* :func:`parse` turns mathematical notation (``2x + 4 = 10``,
  ``log(a b)``, ``x ≡ 3 mod 7``, ``[[1, 2], [3, 4]]``) into an
  :class:`~aria.core.ast.Expr`.
* :func:`parse_problem` additionally recognises instructions
  ("solve for x in ...", "simplify ...", "differentiate ...") and returns a
  :class:`~aria.core.state.SymbolicState` with an explicit typed goal.

The parser is intentionally a *grammar*, not a language model: ambiguity in
natural-language mathematics is resolved by explicit production rules so that
parsing failures are loud instead of silently producing plausible nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .ast import (
    Expr,
    Number,
    Symbol,
    add,
    div,
    func,
    mul,
    num,
    power,
    sym,
)
from .ast import Congruence, Derivative, Equation, Matrix, Relation
from .state import Goal, GoalKind, SymbolicState

__all__ = ["ParseError", "parse", "parse_problem", "parse_equation", "tokenize"]


class ParseError(ValueError):
    """Raised when the input is not in the accepted grammar."""


#: Function names the parser treats as function application rather than
#: implicit multiplication.
FUNCTIONS = {
    "log",
    "ln",
    "exp",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceil",
    "max",
    "min",
    "gcd",
    "lcm",
    "mod",
}

#: Spelled-out Greek names and common multi-letter quantities that must stay
#: atomic even though they consist of single letters.
ATOMIC_NAMES = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
    "dx", "dy", "dt", "dz", "inf", "nan", "max", "min",
}

CONSTANTS: dict[str, str] = {
    "pi": "pi",
    "π": "pi",
    "tau": "tau",
    "e": "e",
    "i": "i",
    "oo": "inf",
    "inf": "inf",
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_UNICODE_MAP = {
    "≡": " ≡ ",
    "≤": " <= ",
    "≥": " >= ",
    "≠": " != ",
    "×": " * ",
    "·": " * ",
    "⋅": " * ",
    "÷": " / ",
    "−": " - ",
    "²": "^2",
    "³": "^3",
    "√": "sqrt",
    "π": "pi",
    "∞": "inf",
    "−": "-",
}


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int

    def __str__(self) -> str:
        return f"{self.kind}({self.text})"


_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_\u0370-\u03ff][A-Za-z_0-9\u0370-\u03ff]*)
  | (?P<op><=|>=|!=|==|\*\*|[-+*/^=<>(),\[\]{}])
  | (?P<mod>≡)
    """,
    re.VERBOSE,
)


def _normalise(text: str) -> str:
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    # Accept "x_1" style subscripts as ordinary names.
    return text.strip()


def tokenize(text: str) -> List[Token]:
    text = _normalise(text)
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            raise ParseError(f"unexpected character {text[pos]!r} at position {pos}")
        kind = match.lastgroup or ""
        value = match.group()
        pos = match.end()
        if kind == "space":
            continue
        if kind == "mod":
            tokens.append(Token("op", "≡", match.start()))
        else:
            tokens.append(Token(kind, value, match.start()))
    tokens.append(Token("eof", "", pos))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class Parser:
    def __init__(
        self,
        tokens: Sequence[Token],
        text: str = "",
        symbol_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.tokens = list(tokens)
        self.text = text
        self.index = 0
        self.symbol_names = set(symbol_names or ())

    # -- token helpers -----------------------------------------------------

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def peek(self, offset: int = 1) -> Token:
        idx = min(self.index + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def at(self, text: str) -> bool:
        return self.current.kind in ("op", "name") and self.current.text == text

    def accept(self, text: str) -> bool:
        if self.at(text):
            self.advance()
            return True
        return False

    def expect(self, text: str) -> Token:
        if not self.at(text):
            raise ParseError(
                f"expected {text!r} at position {self.current.pos}, found {self.current.text!r}"
            )
        return self.advance()

    # -- grammar -----------------------------------------------------------

    def parse_statement(self) -> Expr:
        expr = self.parse_relation()
        if self.at("mod"):
            self.advance()
            modulus_expr = self.parse_relation()
            if not isinstance(expr, Equation):
                raise ParseError("'mod' requires a congruence relation")
            return Congruence(expr.lhs, expr.rhs, modulus_expr)
        return expr

    def parse_relation(self) -> Expr:
        left = self.parse_additive()
        token = self.current
        if token.kind == "op" and token.text in ("=", "==", "<", ">", "<=", ">=", "!=", "≡"):
            self.advance()
            right = self.parse_additive()
            if self.at("mod") and token.text in ("=", "==", "≡"):
                self.advance()
                modulus = self.parse_additive()
                return Congruence(left, right, modulus)
            if token.text in ("=", "=="):
                return Equation(left, right)
            if token.text == "≡":
                return Congruence(left, right, Number(Fraction(1)))
            return Relation(token.text, left, right)
        return left

    def parse_additive(self) -> Expr:
        terms = [self.parse_multiplicative()]
        while self.current.kind == "op" and self.current.text in ("+", "-"):
            sign = self.advance().text
            rhs = self.parse_multiplicative()
            terms.append(rhs if sign == "+" else mul(-1, rhs))
        return add(*terms)

    def parse_multiplicative(self) -> Expr:
        value = self.parse_unary()
        while True:
            if self.current.kind == "op" and self.current.text == "*":
                self.advance()
                value = mul(value, self.parse_unary())
            elif self.current.kind == "op" and self.current.text == "/":
                self.advance()
                value = div(value, self.parse_unary())
            elif self._starts_implicit_factor():
                value = mul(value, self.parse_unary())
            else:
                return value

    def _starts_implicit_factor(self) -> bool:
        token = self.current
        if token.kind == "number":
            return False  # "2 3" is a syntax error, not 6
        if token.kind == "name":
            return not (token.text in ("mod",))
        return token.kind == "op" and token.text in ("(", "[")

    def parse_unary(self) -> Expr:
        if self.current.kind == "op" and self.current.text == "-":
            self.advance()
            return mul(-1, self.parse_unary())
        if self.current.kind == "op" and self.current.text == "+":
            self.advance()
            return self.parse_unary()
        return self.parse_power()

    def parse_power(self) -> Expr:
        base = self.parse_primary()
        if self.current.kind == "op" and self.current.text in ("^", "**"):
            self.advance()
            exponent = self.parse_unary()  # right associative
            return power(base, exponent)
        return base

    def parse_primary(self) -> Expr:
        token = self.current
        if token.kind == "number":
            self.advance()
            return _parse_number(token.text)
        if token.kind == "op" and token.text == "(":
            self.advance()
            inner = self.parse_statement()
            self.expect(")")
            return inner
        if token.kind == "op" and token.text == "[":
            return self.parse_matrix()
        if token.kind == "name":
            if token.text == "d" and self.peek().text == "/":
                derivative = self._try_parse_derivative()
                if derivative is not None:
                    return derivative
            self.advance()
            name = token.text
            if name in CONSTANTS:
                return sym(CONSTANTS[name])
            if name in FUNCTIONS and (
                self.at("(") or self.current.kind in ("name", "number")
            ):
                args = self._parse_call_arguments()
                return _apply_function(name, args)
            return self._symbol(name)
        raise ParseError(f"unexpected token {token.text!r} at position {token.pos}")

    def _symbol(self, name: str) -> Expr:
        """Resolve an identifier, splitting juxtaposed letters where that is intended.

        Mathematical notation writes products as ``2ab`` or ``xy``.  A bare
        multi-letter identifier that is not a known function or a declared
        symbol is therefore read as a product of single-letter variables, which
        matches how these expressions are actually written.  Names declared via
        ``parse(..., symbols=[...])`` stay atomic, as do Greek names and
        conventional multi-letter quantities.
        """
        if (
            len(name) >= 2
            and len(name) <= 3
            and name.isascii()
            and name.isalpha()
            and name.islower()
            and name not in ATOMIC_NAMES
            and name not in self.symbol_names
            and name not in FUNCTIONS
            and name not in CONSTANTS
        ):
            return mul(*(sym(letter) for letter in name))
        return sym(name)

    def _parse_call_arguments(self) -> List[Expr]:
        args: List[Expr] = []
        if self.accept("("):
            if self.accept(")"):
                raise ParseError("empty argument list")
            while True:
                args.append(self.parse_statement())
                if self.accept(","):
                    continue
                self.expect(")")
                break
            return args
        args.append(self.parse_power())
        return args

    def _try_parse_derivative(self) -> Optional[Expr]:
        # d/dx[expr]
        if not (self.peek(1).text == "/" and self.peek(2).kind == "name"):
            return None
        var_token = self.peek(2)
        if not var_token.text.startswith("d") or len(var_token.text) < 2:
            return None
        if self.peek(3).text != "[":
            return None
        self.index += 3  # consume d / dx
        self.expect("[")
        inner = self.parse_statement()
        self.expect("]")
        return Derivative(inner, Symbol(var_token.text[1:]))

    def parse_matrix(self) -> Expr:
        self.expect("[")
        rows: List[Tuple[Expr, ...]] = []
        if self.accept("]"):
            return Matrix(())
        while True:
            self.expect("[")
            row: List[Expr] = []
            while True:
                row.append(self.parse_statement())
                if self.accept(","):
                    continue
                self.expect("]")
                break
            rows.append(tuple(row))
            if self.accept(","):
                continue
            self.expect("]")
            break
        return Matrix(tuple(rows))


def _parse_number(text: str) -> Number:
    """Numeric literals are parsed *exactly* wherever possible."""
    if "e" in text or "E" in text:
        mantissa, exponent = re.split(r"[eE]", text)
        return Number(Fraction(mantissa) * Fraction(10) ** int(exponent))
    if "." in text:
        return Number(Fraction(text))
    return Number(Fraction(int(text)))


def _apply_function(name: str, args: Sequence[Expr]) -> Expr:
    if name == "sqrt" and len(args) == 1:
        return power(args[0], Number(Fraction(1, 2)))
    if name == "ln" and len(args) == 1:
        return func("log", args[0])
    return func(name, *args)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse(text: str, symbols: Optional[Sequence[str]] = None) -> Expr:
    """Parse mathematical notation into an expression tree.

    ``symbols`` lists identifiers that must be treated as single atomic
    variables (``parse("ab", symbols=["ab"])`` keeps ``ab`` intact instead of
    reading it as ``a * b``).
    """
    parser = Parser(tokenize(text), text, symbols)
    expr = parser.parse_statement()
    if parser.current.kind != "eof":
        raise ParseError(f"trailing input at position {parser.current.pos}: {text!r}")
    return expr


def parse_equation(text: str) -> Equation:
    expr = parse(text)
    if not isinstance(expr, Equation):
        raise ParseError(f"expected an equation, got {expr}")
    return expr


# ---------------------------------------------------------------------------
# Natural-language front end
# ---------------------------------------------------------------------------

_SOLVE_RE = re.compile(
    r"^\s*(?:solve|find)\s+(?:for\s+(?P<var1>[A-Za-z_]\w*)\s+(?:in|of)\s+(?P<body1>.+?)"
    r"|(?P<body2>.+?)\s+for\s+(?P<var2>[A-Za-z_]\w*))\s*[.?!]?\s*$",
    re.IGNORECASE,
)
_SOLVE_BARE_RE = re.compile(r"^\s*solve\s+(?P<body>.+?)\s*[.?!]?\s*$", re.I)
_SIMPLE_RE = re.compile(r"^\s*(?:simplify|reduce|evaluate|compute|calculate)\s+(?P<body>.+?)\s*[.?!]?\s*$", re.I)
_FACTOR_RE = re.compile(r"^\s*factor(?:ise|ize)?\s+(?P<body>.+?)\s*[.?!]?\s*$", re.I)
_EXPAND_RE = re.compile(r"^\s*(?:expand|distribute)\s+(?P<body>.+?)\s*[.?!]?\s*$", re.I)
_VERIFY_RE = re.compile(
    r"^\s*(?:verify|check|show|prove|is it true that|is)\s+(?P<body>.+?)(?:\s+true)?\s*[.?!]?\s*$", re.I
)
_DIFF_RE = re.compile(
    r"^\s*(?:differentiate|derive|derivative\s+of)\s+(?P<body>.+?)"
    r"(?:\s+with\s+respect\s+to\s+(?P<var>[A-Za-z_]\w*))?\s*[.?!]?\s*$",
    re.I,
)
_DERIVE_RE = re.compile(r"^\s*(?:derive|show\s+that)\s+(?P<body>.+?)\s+from\s+(?P<premises>.+?)\s*[.?!]?\s*$", re.I)


def _strip_wrapping(text: str) -> str:
    text = text.strip().rstrip(".")
    return text


def parse_problem(text: str, symbols: Optional[Sequence[str]] = None) -> SymbolicState:
    """Parse a natural-language or semi-formal prompt into a symbolic state."""
    cleaned = text.strip()
    if not cleaned:
        raise ParseError("empty input")

    match = _DIFF_RE.match(cleaned)
    if match:
        body = _strip_wrapping(match.group("body"))
        expr = parse(body, symbols)
        var_name = match.group("var")
        if var_name is None:
            symbols = expr.free_symbols
            if len(symbols) != 1:
                raise ParseError(
                    "cannot infer differentiation variable; say 'with respect to x'"
                )
            var = symbols[0]
        else:
            var = Symbol(var_name)
        return SymbolicState(
            (Derivative(expr, var),),
            Goal(GoalKind.DIFFERENTIATE, expr, var),
            source_text=text,
        )

    match = _SOLVE_RE.match(cleaned)
    if match:
        body = _strip_wrapping(match.group("body1") or match.group("body2") or "")
        var_name = match.group("var1") or match.group("var2")
        expr = parse(body, symbols)
        var = Symbol(var_name) if var_name else _infer_solve_var(expr)
        return SymbolicState((expr,), Goal(GoalKind.SOLVE, expr, var), source_text=text)

    match = _SOLVE_BARE_RE.match(cleaned)
    if match:
        expr = parse(_strip_wrapping(match.group("body")), symbols)
        var = _infer_solve_var(expr)
        return SymbolicState((expr,), Goal(GoalKind.SOLVE, expr, var), source_text=text)

    for regex, kind in ((_FACTOR_RE, GoalKind.FACTOR), (_EXPAND_RE, GoalKind.EXPAND), (_SIMPLE_RE, GoalKind.SIMPLIFY)):
        match = regex.match(cleaned)
        if match:
            expr = parse(_strip_wrapping(match.group("body")), symbols)
            return SymbolicState((expr,), Goal(kind, expr), source_text=text)

    match = _DERIVE_RE.match(cleaned)
    if match:
        target = parse(_strip_wrapping(match.group("body")), symbols)
        premises = tuple(parse(p.strip(), symbols) for p in match.group("premises").split(" and "))
        return SymbolicState(premises + (target,), Goal(GoalKind.DERIVE, target), source_text=text)

    match = _VERIFY_RE.match(cleaned)
    if match and any(token in match.group("body") for token in ("=", "≡", "<", ">")):
        expr = parse(_strip_wrapping(match.group("body")), symbols)
        return SymbolicState((expr,), Goal(GoalKind.VERIFY, expr), source_text=text)

    expr = parse(_strip_wrapping(cleaned), symbols)
    kind = GoalKind.VERIFY if isinstance(expr, (Equation, Congruence, Relation)) else GoalKind.SIMPLIFY
    return SymbolicState((expr,), Goal(kind, expr), source_text=text)


def _infer_solve_var(expr: Expr) -> Symbol:
    if isinstance(expr, (Equation, Relation, Congruence)):
        symbols = expr.lhs.free_symbols or expr.rhs.free_symbols
    else:
        symbols = expr.free_symbols
    if len(symbols) != 1:
        raise ParseError(
            f"cannot infer solve variable from {len(symbols)} symbols; say 'solve for x in ...'"
        )
    return symbols[0]
