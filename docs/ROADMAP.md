# ARIA Roadmap

The proposal's four phases, mapped onto concrete deliverables. Checked items are
implemented and covered by tests in this repository.

---

## Phase 1 — Foundation and data construction

| Deliverable | Status |
|---|---|
| Symbolic grammar for mathematical expressions | ✅ `docs/DESIGN_SPEC.md` §1, `aria/core/parse.py` |
| Expression-tree parser for algebraic and scientific notation | ✅ implicit products, fractions, powers, functions, matrices, congruences, `d/dx`, scientific notation, Greek symbols |
| Typed symbolic objects with canonical forms | ✅ `aria/core/ast.py` |
| Exact arithmetic (no float drift) | ✅ `Fraction` end-to-end; floats confined to refutation probes |
| Symbolic equivalence dataset | ✅ 40+ equivalence/refutation fixtures in `tests/test_verify.py`, plus benchmark tasks |
| Algebraic identity corpus | ✅ `aria/bench/tasks.py` (identities, log laws, trig identities) |
| Equation-solving corpus | ✅ linear, quadratic (rational/irrational/complex), higher-degree rational roots, congruences |
| Physics derivation corpus | ⏳ partial — law library + dimensional checks exist; multi-step derivations pending |
| Theorem-proving corpus | ❌ not started (depends on the Lean/Coq bridge) |

**Exit criterion:** parse → represent → verify round-trip on the algebra and
science suites, with zero invalid steps. **Met.**

---

## Phase 2 — Neuro-symbolic model development

| Deliverable | Status |
|---|---|
| Symbolic embedding layer (graph + vector) | ✅ `aria/embed/tree_embed.py` |
| Structural step features | ✅ `aria/neural/policy.py` |
| Learned policy trained on derivation steps | ✅ logistic ranker, ~74% held-out ranking accuracy |
| Policy-guided search over rewrites | ✅ `PolicySearch`, with ordered/random baselines |
| Integrated symbolic validation at every step | ✅ five invariant classes, checked per step |
| Benchmark against standard arithmetic and algebra tasks | ✅ `aria bench`, 46 tasks |
| Transformer/GNN over tree positions replacing the ranker | ⏳ next milestone |

**Exit criterion:** a learned component that improves search without ever
affecting correctness. **Correctness criterion met** (policy cannot emit
answers). **Improvement criterion not yet met** — see
[`EVALUATION.md`](EVALUATION.md) §4 for the honest result.

Next concrete steps:

1. Replace the logistic ranker with a tree transformer over `SymbolicGraph`
   node features; keep the same feature contract and the same verification
   boundary.
2. Add expert-iteration: search → verify → keep the shortest verified path →
   retrain.
3. Scale the mined corpus from ~10² to ~10⁶ preference pairs by generating
   synthetic expressions with a canonical target form.

---

## Phase 3 — Scientific reasoning extensions

| Deliverable | Status |
|---|---|
| Physical units and dimension algebra | ✅ exact rational exponent vectors over 7 SI base dimensions |
| Constraint models for physical laws | ✅ 9 laws, checked; plus a malformed control law to prove discrimination |
| Dimensional inversion ("solve the law for a dimension") | ✅ `derive_dimension` |
| Mechanics / electromagnetism / chemistry problem solving | ⏳ mechanics and EM law set complete; chemistry ontologies pending |
| Automatic unit propagation through derivations | ⏳ next milestone |
| Conservation-law and invariant checking | ⏳ designed, not implemented |

Next concrete steps:

1. Attach dimensions to `SymbolicState` symbols so `engine.solve` can reject
   dimensionally impossible equations before solving them.
2. Add unit-aware arithmetic (`Quantity(value, dimension)`) so numeric answers
   carry units.
3. Add chemistry: molar mass, stoichiometric balance as a linear system over
   element vectors.

---

## Phase 4 — Benchmarking and evaluation

| Deliverable | Status |
|---|---|
| Symbolic simplification benchmark | ✅ 7 tasks |
| Equation-solving benchmark | ✅ 12 tasks |
| Proof/identity verification benchmark | ✅ 6 tasks |
| Scientific derivation benchmark | ⏳ dimensional checks only (10 checks) |
| Exact correctness metric | ✅ |
| Invalid derivation rate | ✅ |
| Symbolic equivalence metric | ✅ (order-insensitive solution matching) |
| Reasoning-trace accuracy | ✅ |
| Unit-consistency metric | ✅ |
| Comparison against text-only LLM baselines | ❌ blocked on a fixed external-API evaluation harness |

Next concrete steps:

1. Publish a frozen test split, versioned, that contributors never train on.
2. Build the LLM comparison harness: same prompts, same scoring code, published
   model/version/date, plus a "solution-only" vs "solution+trace" split.
3. Add adversarial tasks: near-miss identities, off-by-one coefficients,
   dimensionally plausible but wrong laws — the cases where a token model is
   most likely to look right.

---

## Cross-cutting milestones

| Milestone | Rationale | Status |
|---|---|---|
| Lean/Coq export for verified summaries | Converts ARIA's step verdicts into independently checkable proofs | Not started |
| Z3/SMT bridge | Non-linear constraints, inequalities, integer feasibility | Not started |
| Assumption-carrying rewriting (`x ≠ 1` when cancelling `x − 1`) | Current rational cancellation is generic; domain conditions are not tracked as side-conditions | Design needed |
| Proof-state search (backward chaining) | Required for real theorem proving, as opposed to identity checking | Not started |
| Published corpus + reproducibility script | Prerequisite for credible external claims | Partial (`aria bench --json-out`) |

---

## Non-goals

* Natural-language understanding beyond the instruction grammar. ARIA is a
  reasoning backend, not a chatbot; an LLM front-end is expected to translate
  free text into `SymbolicState` and to explain ARIA's verified output.
* Replacing computer algebra systems. ARIA orchestrates them behind a
  verification boundary.
* Pretraining a foundation model. The neural component is trained on verified
  symbolic traces, which is a small, curated, and self-labelling dataset.
