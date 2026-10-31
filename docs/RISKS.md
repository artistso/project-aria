# Risk Assessment

Risks are grouped as in the proposal (technical, architectural, scientific),
with two additions that matter for this specific implementation: **verification
risk** (a verifier that is wrong is worse than none) and **evaluation risk**
(numbers that flatter the system).

Each entry: the risk, why it matters, the mitigation, and its current status in
this repository.

---

## 1. Technical risks

### 1.1 Symbolic parsing complexity — **Active, bounded**

*Risk.* Mathematical notation is context-sensitive: `2ab` may be `2·a·b` or a
variable `ab`; `sin x` is application, not multiplication; `d/dx` is not a
quotient.

*Mitigation.* A hand-written grammar with explicit, documented resolution rules
(`docs/DESIGN_SPEC.md` §1.2), plus a `symbols=` escape hatch for atomic names.
Parser failures raise with character positions; nothing is silently guessed.

*Residual risk.* Genuine ambiguity in free text ("the sum of x squared and 1")
is out of scope. An LLM front-end that emits `SymbolicState` is the intended
division of labour; its parse errors must then be caught by this grammar.

### 1.2 Ambiguity in natural-language mathematical notation — **Accepted boundary**

*Risk.* Users write mathematics informally; a grammar cannot cover it.

*Mitigation.* A small, well-defined instruction grammar for goals only; the
mathematical payload must be notational. Anything else fails loudly.

### 1.3 Combinatorial explosion in expression search — **Mitigated**

*Risk.* Rewriting search on large expressions is exponential.

*Mitigation.* Canonical forms give cheap state deduplication; `MAX_TERMS`,
`max_iterations`, `max_expansions` and a beam width bound every search;
expansion is restricted to small integer powers.

*Residual risk.* The current engine solves by *planning*, not search, for the
equation cases it supports; genuine search only appears in `PolicySearch`, which
is bounded.

### 1.4 Scaling to broad scientific domains — **Open**

*Risk.* Mechanics and electromagnetism are expressible with dimensions; chemistry,
thermodynamics with state variables, and field theories are not merely dimensional.

*Mitigation.* A law registry with explicit domain tags; `partial` outcomes with
reasons; the roadmap stages additions by domain.

---

## 2. Architectural risks

### 2.1 Over-reliance on neural approximation — **Structurally prevented**

*Risk.* A neural component that can emit answers will occasionally emit
confident nonsense.

*Mitigation.* The policy only **ranks** candidate rewrites produced by rules. It
cannot construct an expression, cannot skip verification, and cannot set the
final status. A misconfigured or adversarial policy can only waste search
budget. This is enforced by API shape: `StepPolicy.score(features) -> float`.

*Status.* Enforced by design and covered by tests
(`tests/test_neural.py::test_policy_prefers_goal_directed_steps` and the search
tests).

### 2.2 Insufficient verification — **Highest-severity risk, actively managed**

*Risk.* A verifier that returns `proven` for a false statement destroys the
entire value proposition.

*Mitigation.*

* Three-valued logic: `unknown` is never promoted to `proven`.
* Structural and exact-arithmetic checks run before any heuristic backend.
* Numeric probing can only **refute**, never prove; tolerances are loose enough
  that rounding cannot manufacture a counterexample.
* Constants (`π`, `e`, `i`) are excluded from probing, where treating them as
  free variables would be unsound.
* Equation steps are checked by *residual proportionality*, which rejects
  irreversible operations (squaring) rather than assuming they are fine.
* Congruence steps are checked by enumerating and comparing solution sets.
* Corrupted-step tests assert that tampering is detected.

*Residual risk.* SymPy itself is trusted for transcendental identities. A
defence-in-depth step — emitting Lean/Coq for independent checking — is on the
roadmap and is the correct long-term answer.

### 2.3 Inability to generalise outside training domains — **Open, expected**

*Risk.* Rules are domain-specific; the system fails gracefully outside them.

*Mitigation.* Reported as `partial` with an explicit reason (e.g. "polynomial of
degree 3 found, but only 1 rational root"). No silent fallback to guessing.

### 2.4 Trace quality vs. trace correctness — **Mitigated**

*Risk.* A step can be correct while its *label* is misleading (for example,
attributing a constant-folding change to a distribution rule).

*Mitigation.* Rules report a justification only when they actually fired; tree
changes caused by canonical reconstruction are labelled `normalise`.

---

## 3. Scientific risks

### 3.1 Physical laws requiring assumptions not captured by syntax — **Partially mitigated**

*Risk.* `F = ma` presumes an inertial frame, constant mass, and classical
regime. Dimension checking cannot see any of that.

*Mitigation.* Laws are registered with a domain tag (`classical mechanics`,
`relativity`, `quantum mechanics`). Dimensional consistency is presented as a
*necessary* condition, not a sufficient one, in both the API and docs.

### 3.2 Hidden domain constraints — **Open**

*Risk.* Thermodynamic sign conventions, gauge choices, boundary conditions.

*Mitigation.* Out of scope for the current layer; the roadmap proposes attaching
assumption sets to symbols (`SymbolicState.assumptions` already exists and is
threaded into SymPy symbols) and requiring them to be discharged.

### 3.3 Unit errors that are dimensionally consistent — **Open**

*Risk.* A dimensionally correct equation can still be physically wrong
(e.g. missing a factor of 2). Dimensional analysis cannot detect it.

*Mitigation.* Be explicit: the science layer answers "is this dimensionally
possible?", not "is this law true?". Canonical-case verification is roadmap work.

---

## 4. Evaluation risks

### 4.1 Benchmark authored with the system — **Acknowledged**

*Risk.* A 100% score on a self-authored suite is a regression signal, not
evidence of general ability.

*Mitigation.* Stated in `README.md` and in `docs/EVALUATION.md` §3; the report
distinguishes "no known regressions" from "solves arbitrary mathematics". A
frozen external split is roadmap work.

### 4.2 Selective reporting — **Mitigated**

*Risk.* Reporting only wins.

*Mitigation.* Negative results are published in `docs/EVALUATION.md` §4,
including the finding that the learned policy is slower than the fixed rule
order at this scale. Numbers that cannot be reproduced are not reported at all
(the LLM comparison is explicitly absent, with reasons).

### 4.3 Silent metric drift — **Mitigated**

*Risk.* Metrics quietly change definition between runs.

*Mitigation.* Metrics live in one place (`BenchmarkReport.metrics()`), are
asserted in tests (`tests/test_bench.py`), and the JSON report is machine
diffable.

---

## 5. Risk register summary

| # | Risk | Severity | Likelihood | Status |
|---|---|---|---|---|
| 2.2 | Verifier unsoundness | Critical | Low | Managed; Lean/Coq bridge pending |
| 1.3 | Search explosion | High | Medium | Mitigated by caps and planning |
| 4.1 | Self-authored benchmark | High | High | Disclosed; external split pending |
| 1.1 | Parse ambiguity | Medium | High | Bounded by grammar + documented rule |
| 2.1 | Neural overreach | Critical | Low | Prevented by architecture |
| 3.1 | Unstated physical assumptions | Medium | High | Disclosed; domain tags + assumption sets |
| 1.4 | Domain breadth | Medium | High | Open; staged roadmap |
| 4.3 | Metric drift | Medium | Low | Managed by tests |
| 2.3 | Generalisation | Medium | Medium | Open by design (`partial` status) |
| 3.2 | Hidden constraints | Low | Medium | Out of scope, tracked |
