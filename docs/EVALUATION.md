# ARIA Evaluation Protocol and Results

Evaluation is where a proposal like this earns or loses credibility. This
document states exactly what is measured, how, on what, and — importantly — what
the numbers do *not* show.

---

## 1. What is measured

| Metric | Definition | Why it matters |
|---|---|---|
| **Exact correctness** | Fraction of tasks whose *verified* answer matches an independently stated expectation (compared symbolically, not as strings) | Distinguishes right answers from plausible ones |
| **Invalid derivation rate** | Refuted steps ÷ total steps across all derivations | The soundness signal; a verified system must be 0 |
| **Verified trace rate** | Fraction of derivations in which every step carried a `proven` verdict | Whether the *reasoning* is checkable, not just the answer |
| **Mean trace accuracy** | Mean per-derivation fraction of steps proven | Partial credit for long derivations |
| **Symbolic equivalence** | Solutions compared by `Verifier.equivalent`, order-insensitive multisets | Rejects string-matching credit and accepts equivalent forms |
| **Unit consistency** | Fraction of dimensional checks passing, *including a deliberately malformed control law* | Proves the dimension checker discriminates rather than always agreeing |

Deliberately **not** reported as headline metrics:

* throughput / latency (not a goal at this stage),
* "reasoning quality" judged by another model (not reproducible),
* any number produced by comparing against an LLM without a fixed, published
  protocol — see §5.

---

## 2. Protocol

* **Task suite:** 46 tasks in `aria/bench/tasks.py`, plus 10 dimensional checks
  (9 library laws + 1 malformed control).
* **Ground truth:** every task states its expected solution(s), simplified form
  or verdict *independently of the engine*. Ground truth is never derived from
  the engine's output.
* **Scoring:** `aria/bench/runner.py`. Solutions are matched as an
  order-insensitive multiset under symbolic equivalence. For simplify/factor
  tasks the engine's result must be provably equivalent to the reference form
  *and* the derivation must be `solved`.
* **Backend:** SymPy 1.14 when available; the dependency-free backend is
  separately tested (`tests/test_verify.py::test_unknown_backend_reports_unknown_without_crashing`).
* **Determinism:** seeded probes, fixed rule order, static policy weights. The
  suite reproduces exactly in CI.
* **Command:** `aria bench --markdown benchmarks/report.md --json-out benchmarks/report.json`.

---

## 3. Results (this repository, current commit)

```
tasks                    : 46
exact_correctness        : 1.0000
invalid_derivation_rate  : 0.0000
verified_trace_rate      : 1.0000
mean_trace_accuracy      : 1.0000
unit_consistency         : 1.0000
```

| suite | tasks | correct | mean steps |
|---|---|---|---|
| solve | 12 | 12 | 3.33 |
| simplify | 7 | 7 | 1.71 |
| factor | 5 | 5 | 1.20 |
| log-exp | 6 | 6 | 1.00 |
| modular | 5 | 5 | 0.60 |
| calculus | 5 | 5 | 1.00 |
| verify | 6 | 6 | 0.00 |
| dimensional analysis | 10 checks | 10 | — |

Full per-task detail: [`benchmarks/report.md`](../benchmarks/report.md).

### Reading these numbers correctly

This suite was written alongside the engine. It is a **regression harness that
proves the machinery works**, not an estimate of general mathematical ability.
The tasks are drawn from the same distribution the rules were designed for, and
the suite is small. A 100% score here means "no known regressions and no unsound
steps", not "solves arbitrary mathematics".

What the numbers do support:

* the representation and parser handle the intended notation;
* every derivation in the suite is step-verified by an independent checker;
* the dimension checker accepts valid laws and rejects the malformed control;
* the pipeline is deterministic and reproducible.

What they do not support:

* performance on unseen domains (topology, analysis, linear algebra, statistics);
* robustness to genuinely ambiguous natural language;
* any claim about competition mathematics.

---

## 4. Ablations and negative results

**Verification ablation.** Disabling the verifier does not change the *answers*
the rules produce — the rules are deterministic algebraic identities. Its value
is demonstrated instead on corrupted input: `tests/test_verify.py::test_invalid_equation_step_is_caught`
and `tests/test_solve.py::test_literal_tampering_is_detected` show that
tampered steps are refuted rather than accepted. A system without step
verification would report those derivations as successful.

**Neural policy ablation (reported honestly, no gain observed).**

Same beam search, same verifier, three candidate orderings on the simplify
subset:

| ordering | solved | mean expansions |
|---|---|---|
| learned policy | 7 | 2.71 |
| fixed rule order (engine default) | 7 | 1.86 |
| seeded random shuffle | 7 | 2.29 |

All three solve the same tasks — as they must, because the verifier is shared.
The learned policy is **not** more efficient than the hand-ordered rule list at
this scale.

Why this result is unsurprising:

1. The training corpus is ~10² verifier-mined pairwise rule-ordering labels (111 at present), not human preference ratings; the model has ~20 parameters.
2. The fixed order is not arbitrary — the rules were written in dependency
   order, which is a strong prior that a small ranker must rediscover.
3. The tasks are 1–3 rewrites long, so there is little room for a heuristic to
   pay off.

What the policy result *does* establish: the training pipeline works end to end
(labels mined from verified traces, held-out ranking accuracy 73.9%, loss
monotonically decreasing), and the neural component cannot influence
correctness by construction. Scaling the corpus, the model, and the task length
is Phase 2 work, not a claim made here.

**Rate of `unknown`.** With the dependency-free backend, transcendental
identities return `unknown` instead of being proved. This is by design and is
tested explicitly; it is the behaviour a verification-first system should have.

---

## 5. Comparison with text-only models (not yet done)

The proposal calls for comparing against LLM baselines. That comparison is
**not** presented in this repository because a defensible version requires:

1. a frozen, published task set with held-out items;
2. identical prompts and identical scoring code for both systems;
3. published model identifiers, versions and dates;
4. separate reporting of final-answer correctness and step-level soundness
   (an LLM's trace cannot be verified by SymPy alone; each step must be
   machine-checked with the same `check` invariants ARIA uses);
5. an analysis of failure *types*: wrong final answer vs. correct answer with an
   invalid step — the second is the failure mode the proposal is about.

Without those, any number would be marketing. `docs/ROADMAP.md` Phase 4 tracks
building the harness.

---

## 6. Threats to validity

| Threat | Mitigation / status |
|---|---|
| Tasks written by the same author as the rules | Acknowledged; external held-out suite needed |
| SymPy backend soundness assumed | ARIA's own exact rational backend independently checks polynomial/rational cases |
| Numeric probes could, in principle, produce a false refutation | Tolerance is loose (1e-6 relative); probes can only refute, never prove; exact paths are tried first |
| Rule set has domain gaps (e.g. no `arcsin` antiderivatives) | Reported as `partial`, never as solved |
| Cancellation without domain conditions (`(x²−1)/(x−1)` at `x=1`) | Known limitation: side-conditions are not tracked; tracked on the roadmap |
| Benchmark suite may overfit to implemented rules | Reported in §3, and mitigated by the `partial` status discipline |

---

## 7. How to reproduce

```bash
pip install -e ".[dev]"
pytest                                    # 282 tests at this revision
aria bench --markdown benchmarks/report.md --json-out benchmarks/report.json
aria train --epochs 80 --eval             # policy metrics + ordering comparison
```
