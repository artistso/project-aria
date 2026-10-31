# ARIA Design Specification

A precise description of the accepted input language, the object model, the
semantics of verification, and the public API. This is the document to read
before extending the system.

---

## 1. Input grammar

### 1.1 Notation

```ebnf
statement   ::= relation [ "mod" expression ] ;
relation    ::= additive [ ( "=" | "==" | "<" | ">" | "<=" | ">=" | "!=" | "≡" ) additive ] ;
additive    ::= term { ( "+" | "-" ) term } ;
term        ::= unary { ( "*" | "/" | juxtaposition ) unary } ;
unary       ::= ( "-" | "+" ) unary | power ;
power       ::= primary [ ( "^" | "**" ) unary ] ;        (* right associative *)
primary     ::= number | identifier | "(" statement ")" | matrix | derivative ;
matrix      ::= "[" row { "," row } "]" ;
row         ::= "[" statement { "," statement } "]" ;
derivative  ::= "d" "/" "d" ident "[" statement "]" ;
number      ::= digit+ [ "." digit* ] [ ("e"|"E") ["+"|"-"] digit+ ]
              | "." digit+ ;
identifier  ::= letter { letter | digit | "_" } ;
```

Accepted Unicode is normalised first: `≡ ≤ ≥ ≠ × · ÷ − ² ³ √ π ∞`.

### 1.2 Juxtaposition rule (explicit, documented ambiguity resolution)

`2x`, `2(x+1)`, `(x+1)(x−1)`, `2 sin(x)` are products. A bare identifier of two or
three lowercase ASCII letters that is not a known function, constant, Greek name
or declared symbol is read as a product of single-letter variables:
`2ab` ⇒ `2·a·b`.

To keep a multi-letter identifier atomic, declare it:

```python
parse("ab", symbols=["ab"])                 # -> Symbol("ab")
parse_problem("solve ab = 5", symbols=["ab"])
```

The set of names that always stay atomic is `ATOMIC_NAMES` in `aria/core/parse.py`
(Greek names, `dx`, `dy`, `dt`, …).

### 1.3 Instruction grammar

| pattern | goal produced |
|---|---|
| `solve for x in <stmt>` / `solve <stmt> for x` / `solve <stmt>` | `Goal(SOLVE, var=x)` |
| `simplify \| reduce \| evaluate \| compute \| calculate <expr>` | `Goal(SIMPLIFY)` |
| `factor \| factorise \| factorize <expr>` | `Goal(FACTOR)` |
| `expand \| distribute <expr>` | `Goal(EXPAND)` |
| `verify \| check \| show \| prove \| is <stmt> [true]` | `Goal(VERIFY)` |
| `differentiate <expr> [with respect to x]` | `Goal(DIFFERENTIATE, var=x)` |
| `derive <stmt> from <premises>` | `Goal(DERIVE)` |
| anything else | `VERIFY` if the expression is relational, `SIMPLIFY` otherwise |

If a solve variable cannot be inferred unambiguously, the parser raises rather
than guessing.

---

## 2. Object model

| node | fields | notes |
|---|---|---|
| `Number` | `value: Fraction \| float` | exact unless constructed from a float |
| `Symbol` | `name`, `assumptions: frozenset` | assumptions: `positive`, `integer`, `nonzero`, `real`, … |
| `Add` | `terms: tuple[Expr, …]` | flattened, zero-free, canonically sorted |
| `Mul` | `factors: tuple[Expr, …]` | flattened, numeric coefficient first, like factors merged into powers, canonically sorted |
| `Pow` | `base`, `exp` | `x^0 → 1`, `x^1 → x`, `0^p → 0 (p>0)` |
| `Func` | `name`, `args` | `log`, `exp`, `sin`, `cos`, `tan`, `abs`, `sign`, … |
| `Derivative` | `expr`, `var` | `d/dx[…]` |
| `Equation` | `lhs`, `rhs` | `lhs = rhs` |
| `Relation` | `op`, `lhs`, `rhs` | `<`, `<=`, `>`, `>=`, `!=` |
| `Congruence` | `lhs`, `rhs`, `modulus` | `lhs ≡ rhs (mod n)` |
| `Matrix` | `rows: tuple[tuple[Expr, …], …]` | rectangular; ragged input rejected |

**Canonical form contract.** Two expressions that are equal by associativity,
commutativity and unit/zero laws have *identical* canonical keys. This is used
for hashing, deduplication and the structural target test in search. It does not
claim algebraic equivalence: `2x + 3x` and `5x` are different canonical forms
until `collect_like_terms` runs — which is exactly why that rewrite is a visible,
verifiable step.

---

## 3. Derivation semantics

A `DerivationStep` is `(rule, before, after, justification, check, verdict)`.

| `check.kind` | verification procedure | sound because |
|---|---|---|
| `equivalent` | `Verifier.equivalent(before, after)` | expression identity is preserved |
| `equation_scaled` | residuals `r_b = lhs_b − rhs_b`, `r_a = lhs_a − rhs_a`; require `r_a = k·r_b`, `k ≠ 0` | exactly the class of reversible equation manipulations; rejects squaring both sides |
| `both_sides_add` | `lhs_a − lhs_b ≡ rhs_a − rhs_b` | same quantity added to both sides |
| `both_sides_mul` | `lhs_a / lhs_b ≡ rhs_a / rhs_b` (with a zero guard) | same non-zero factor applied |
| `modular_equivalent` | enumerate solution sets of both linear congruences modulo `lcm(n₁, n₂)` and compare | decided by construction, no algebraic assumption |
| `root` | substitute the candidate value into the prior equation; require an exact (or symbolically simplified) zero residual | a root is defined by vanishing residual |
| `premise` | accepted without proof | only used for assumptions of the problem statement |

A derivation is `valid` iff every step has a `proven` verdict. `partial`,
`refuted` and `open` are distinct statuses and are never conflated with success.

---

## 4. Verification decision procedure

```
equivalent(a, b):
  a == b                                  -> proven   (structural)
  contains π, e, i, τ, ∞                  -> sympy only
  exact rational probe finds a ≠ b        -> refuted  (counterexample)
  float probe finds |a−b| > 1e-6·scale    -> refuted  (counterexample, refutation only)
  rational normal form decides a − b = 0  -> proven / refuted
  sympy normalisation strategies decide   -> proven / refuted
  otherwise                               -> unknown
```

Guarantees:

* `proven` implies identity (modulo backend soundness).
* `refuted` is accompanied by a witness (a point and the two values) or an
  explicit residual.
* `unknown` is reported as `unknown`. There is no "probably true".

---

## 5. Public API

```python
from aria import SymbolicEngine, Verifier, parse_problem, parse

state      = parse_problem("solve for x in 2x + 4 = 10")   # SymbolicState
engine     = SymbolicEngine(Verifier(backend="auto"))
derivation = engine.run(state)                              # Derivation

derivation.status            # 'solved' | 'partial' | 'open' | 'refuted'
derivation.solutions         # tuple[Expr, ...]
derivation.steps             # tuple[DerivationStep, ...]
derivation.valid             # bool: every step proven
derivation.trace_accuracy    # float
derivation.summary()         # JSON-serialisable dict
derivation.to_text() / .to_markdown()
```

Lower-level entry points:

```python
engine.solve(equation, var)          engine.solve_congruence(cong, var)
engine.simplify(expr)                engine.expand(expr)
engine.factor(expr)                  engine.differentiate(expr, var)
engine.prove(claim)                  verifier.equivalent(a, b)
check_law("newton_second")           derive_dimension("F = m a", dims, "a")
embed(expr, dim=128)                 PolicySearch().search(start, goal)
```

---

## 6. Error taxonomy

| condition | behaviour |
|---|---|
| unparsable input | `ParseError` with character position; CLI exit code 2 |
| ambiguous solve variable | `ParseError` asking for `solve for x in …` |
| unknown unit or symbol without a dimension | `DimensionError` |
| structurally unequal matrix rows | `ValueError` at construction |
| division by zero in canonicalisation | `ZeroDivisionError` raised, reported by the rule layer as "no candidate" |
| undecidable claim | `Verdict(status="unknown")`, derivation status `partial` |
| resource-heavy rewrite | capped: `MAX_TERMS`, `max_iterations`, `max_expansions` |

---

## 7. Determinism

Given identical inputs and backend, ARIA is deterministic: numeric probes are
seeded, rule application order is fixed, canonical ordering is total, and the
policy weights are a static JSON file. The benchmark is therefore reproducible
in CI without flaky tolerances.
