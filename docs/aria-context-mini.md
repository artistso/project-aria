# ARIA — Mini Context Pack

> Compact digest for models with a small context window (~2k tokens).
> Full version: `docs/aria-context-pack.md`.

## What it is

ARIA is a neuro-symbolic reasoning system where **typed expression trees, not
tokens, are the substrate**. Math is parsed into exact symbolic objects,
transformed by explicit justified rewrite rules, and **every step is verified**
before acceptance. A small learned policy ranks which rewrite to try next; it
never emits answers or bypasses verification.

Python 3.10+, zero required dependencies (SymPy optional, upgrades
transcendental identity checks from `unknown` to `proven`).

## Pipeline

```
text → parse → SymbolicState(exprs, Goal) → engine dispatch
     → rule-produced candidate rewrite → verifier checks the step invariant
     → accepted step appended → Derivation(status, steps, solutions)
```

## Representation

Nodes: `Number` (exact `Fraction`), `Symbol` (+assumptions), `Add`, `Mul`,
`Pow`, `Func`, `Derivative`, `Equation`, `Relation`, `Congruence`, `Matrix`.

Canonical forms: associativity/commutativity/unit laws produce identical
canonical keys (used for hashing, dedup, structural goal tests). This is **not**
algebraic equivalence — `2x + 3x` and `5x` differ until `collect_like_terms`
runs, which is why that rewrite is a visible verified step.

Parsing is a grammar, not a model: failures raise `ParseError` with a position.
`2ab` means `2·a·b` unless `ab` is declared atomic via `parse(..., symbols=[...])`.
Instruction patterns map to goals: solve / simplify / factor / expand / verify /
differentiate / derive.

## Verification (the core idea)

Three-valued verdicts: `proven`, `refuted`, `unknown`. `unknown` is never
reported as success.

Order of decision for `equivalent(a, b)`:
1. structural equality → proven
2. if `π, e, i` occur → SymPy only (constants are not variables; probing unsound)
3. exact rational probing → refuted with a witness
4. float probing → refutation only, never proof
5. exact rational normal form (polynomial/rational) → proven/refuted
6. SymPy with several sound normalisations
7. otherwise → unknown

Step invariants (`check.kind`):

| kind | check |
|---|---|
| `equivalent` | expressions identical |
| `equation_scaled` | residual `lhs−rhs` preserved up to non-zero constant `k` — automatically rejects squaring both sides |
| `both_sides_add` / `both_sides_mul` | same quantity applied to both sides |
| `modular_equivalent` | congruence solution sets equal (enumerated) |
| `root` | substitution makes the residual vanish |
| `premise` | accepted assumption |

`Derivation.valid` ⇔ every step `proven`. Statuses: `solved`, `partial`, `open`,
`refuted`. `partial` is first-class and always carries a reason.

## Engine

- **solve**: move all to one side → simplify → degree 1 isolate/divide;
  degree 2 factor (rational roots) or quadratic formula (radicals simplified,
  complex roots labelled); degree ≥3 rational-root factorisation else `partial`.
  All candidate solutions substituted back into the original equation.
- **simplify**: fixpoint over rules; changes caused only by canonical
  reconstruction are labelled `normalise`, not mis-attributed.
- **factor**: common factor (numeric + symbolic), difference of squares,
  quadratic factor.
- **differentiate**: rules → simplify → independent SymPy check.
- **modular**: reduce, divide by gcd when it divides, multiply by modular
  inverse, enumerate residue classes.
- Rules include log product/quotient/power, `exp∘log`, Pythagorean identity,
  rational cancellation via polynomial gcd, radical simplification.

## Science

Dimension = rational exponent vector over (M, L, T, I, Θ, N, J). Sums require
homogeneous terms; `log/exp/sin/cos` require dimensionless arguments.
`check_law`, `check_equation`, and `derive_dimension` (solve a law for an unknown
quantity's dimension). 9-law library + a malformed control law (`F = m a^2`) that
must be rejected.

## Neural component

Ranks candidate rewrites by structural features (size/depth delta, similarity to
current state and to the goal form, operator deltas, rule identity). Bradley–Terry
logistic ranker trained on preference pairs mined from *verified* derivations,
persisted as JSON. A bad policy costs search time, never correctness.

## API

```python
from aria import SymbolicEngine, Verifier, parse_problem
state = parse_problem("solve for x in 2x + 4 = 10")
d = SymbolicEngine(Verifier()).run(state)
d.status; d.solutions; d.steps; d.valid; d.trace_accuracy; d.to_text()
```

CLI: `aria run|solve|simplify|factor|expand|verify|differentiate|units|
derive-dim|bench|train`.

## Results (46-task suite, v0.1.0)

exact correctness 46/46 · invalid derivation rate 0.0 · verified trace rate 1.0 ·
unit consistency 1.0 (9 laws + control).

Caveat: the suite was authored alongside the engine — this measures "no
regressions, no unsound steps", not general mathematical ability.

Negative result kept on purpose: the learned policy is **slower** than the fixed
rule order (2.71 vs 1.86 mean expansions) at this scale; rationale in
`docs/EVALUATION.md`.

## Limitations

Notation + small instruction grammar, not general natural language. No Lean/Coq
or Z3 bridge yet. No domain side-conditions on rational cancellation. Logistic
ranker, not a tree transformer. No published LLM baseline comparison.
