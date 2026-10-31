# ARIA — Algebraic Reasoning & Inference Architecture

**A neuro-symbolic architecture in which mathematical objects, not text tokens, are the substrate for reasoning.**

ARIA implements the research direction outlined in [the project proposal](#proposal-summary): instead of asking a language model to *sound* like it is doing algebra, ARIA represents mathematics as typed expression trees, transforms them with explicit rules, verifies every transformation with a symbolic engine, and uses a learned policy to decide which transformation to attempt next. Answers arrive with a machine-checkable derivation rather than a plausible-looking string.

```
   input ──▶ representation ──▶ symbolic embedding ──▶ reasoning core ──▶ verification ──▶ answer
             (parse to AST)      (graph + vector)       (rules + search)   (prove/refute)     + trace
                    ▲                                                                          │
                    └────────────────────  scientific layer (units, laws) ◀───────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/artistso/project-aria
cd project-aria
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # SymPy is optional but strongly recommended
```

```console
$ aria run "solve for x in 2x + 4 = 10"
Goal: Goal(solve(x))

  0.  4 + 2 x = 10
  1.  -10 + 4 + 2 x = 0      ✓ move_all_to_one_side — subtract the right-hand side from both sides
  2.  -6 + 2 x = 0           ✓ normalise — canonicalise by folding constants and identities
  3.  2 x = 6                ✓ isolate_variable_term — add 6 to both sides
  4.  x = 3                  ✓ divide_by_coefficient — divide both sides by 2

Solutions: x = 3
Status: solved  (trace accuracy 100%)
```

Every `✓` means an independent symbolic check confirmed that the step preserves
the truth of the statement — not that the step "looks reasonable".

```console
$ aria simplify "(x+1)^3"
  0.  (1 + x)^3
  1.  1 + x + x + x + x^2 + x^2 + x^2 + x^3      ✓ distribute — expand the power (1 + x)^3
  2.  1 + 3 x + 3 x^2 + x^3                      ✓ collect_like_terms — collect like terms

$ aria verify "log(a*b) = log(a) + log(b)"
Status: solved      verified: proven via sympy-simplify

$ aria verify "log(a+b) = log(a) + log(b)"
Status: refuted     refuted: 1.386294 != 2.484907 at a=0.9431, b=2.1701

$ aria solve "x^2 + 2x + 5 = 0"
Solutions: x = -1 + 2 i, x = -1 - 2 i
Status: solved      Notes: the discriminant is negative, so the solutions are complex

$ aria solve "5x ≡ 3 mod 7"
  0.  5 x ≡ 3 (mod 7)
  1.  x ≡ 2 (mod 7)          ✓ multiply_by_inverse — multiply both sides by the modular inverse 3
Solutions: x = 2
```

Dimensional analysis, from the scientific reasoning layer:

```console
$ aria units --law newton_second
Equation: F = m a
  left  dimension: kg m / s^2
  right dimension: kg m / s^2
  verdict: CONSISTENT — both sides have the same dimension

$ aria units "F = m a^2" --dims F=kg*m/s^2 --dims m=kg --dims a=m/s^2
  verdict: INCONSISTENT — kg m / s^2 != kg m^2 / s^4

$ aria derive-dim "F = m a" --unknown a --dims F=kg*m/s^2 --dims m=kg
Solving F = m a for [a] gives [a] = m / s^2
```

---

## Proposal summary

> Current generative systems predict tokens; they do not possess a native,
> verifiable representation of mathematical objects, rules, or physical laws.
> A system that can *appear* intelligent while lacking formal grounding is
> insufficient for science.

ARIA takes that critique as an engineering specification and answers it layer by
layer:

| Proposal layer | ARIA implementation |
|---|---|
| **5.1 Representation** | Typed expression trees (`aria/core/ast.py`), a recursive-descent grammar for notation and instruction phrases (`aria/core/parse.py`), symbolic states with explicit goals (`aria/core/state.py`) |
| **5.2 Symbolic embedding** | Node/edge graph + fixed-width hash-bucketed structural vectors (`aria/embed/tree_embed.py`) |
| **5.3 Neural reasoning core** | Structural step features, a logistic ranker trained on verified derivations, and policy-ordered beam search over rewrites (`aria/neural/`) |
| **5.4 Verification** | Three-valued verifier — *proven / refuted / unknown* — with a SymPy backend, an exact rational-function backend that needs no dependencies, and step-level invariant checks (`aria/verify/`) |
| **5.5 Scientific reasoning** | Exact dimensional algebra over the seven SI base dimensions, a physical-law library, and dimensional inversion (`aria/science/`) |

Everything above is implemented and tested. Anything not yet built is listed
under [Status](#status-what-is-and-is-not-implemented) rather than implied.

---

## Design principles

**1. Exactness by construction.** Numeric literals parse to `Fraction`, never to
`float`. `2x + 4 = 10` is `Add(2·x, 4)`, and `x^2 - 1` never becomes `0.999…`.
Inexact floats exist only inside the *refutation* probe, where they can never
prove anything.

**2. Three-valued verification.** A verifier that only says "yes" is useless.
Every claim resolves to `proven`, `refuted` or `unknown`, and `unknown` is
reported as `unknown` — never quietly upgraded to success.

**3. Steps carry invariants, not adjectives.** Each derivation step declares
*why* it is admissible, and the verifier checks that specific invariant:

| step kind | invariant checked |
|---|---|
| `equivalent` | the two expressions are identically equal |
| `equation_scaled` | the equation residual is preserved up to a non-zero factor — this is what makes an algebraic rewrite reversible, and it automatically rejects squaring both sides |
| `both_sides_add` / `both_sides_mul` | the same quantity was added to / multiplied through both sides |
| `modular_equivalent` | the two congruences have identical solution sets (enumerated, not assumed) |
| `root` | substituting the candidate value makes the residual vanish exactly |

**4. The neural component may cost time, never correctness.** The policy ranks
candidate transformations; it cannot emit expressions and it cannot bypass
verification. A badly trained policy slows the search down; it cannot produce a
false claim.

**5. Physics is a constraint system.** `F = m a^2` is rejected by dimensional
analysis before any symbolic manipulation happens.

---

## Architecture

```
aria/
├── core/          symbolic objects, grammar, states, exact polynomial algebra
│   ├── ast.py         Number/Symbol/Add/Mul/Pow/Func/Derivative/Equation/Congruence/Matrix
│   ├── parse.py       notation + natural-language goal extraction
│   ├── poly.py        rational normal forms, polynomial gcd/division, rational roots
│   └── state.py       SymbolicState, Goal(Solve(x)), assumptions
├── embed/         structural graph + vector encodings
├── neural/        step features, logistic ranker, policy-guided search, training
├── reason/        rewrite rules, derivations, the planning engine
│   ├── rules.py       justified local rewrites (algebra, logs, trig, calculus)
│   ├── derivation.py  steps with verdicts, trace accuracy, rendering
│   └── engine.py      solve / simplify / factor / expand / differentiate / prove / modular
├── verify/        three-valued verification (SymPy + dependency-free backends)
├── science/       dimensions, units, physical laws, dimensional inversion
├── bench/         task suite, metrics, markdown/JSON reports
└── cli.py         `aria` command-line interface
```

Detailed design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
Formal specification: [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md) ·
Prior art and positioning: [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) ·
Local-model integration: [`docs/LLM_INTEGRATION.md`](docs/LLM_INTEGRATION.md) ·
Weights and training: [`docs/WEIGHT_REWRITING.md`](docs/WEIGHT_REWRITING.md) ·
The ledger: [`docs/LEDGER.md`](docs/LEDGER.md) ·
What the account already has: [`docs/SCAN_FINDINGS.md`](docs/SCAN_FINDINGS.md) ·
Reading it with an offline LLM: [`docs/OFFLINE_LLM_GUIDE.md`](docs/OFFLINE_LLM_GUIDE.md).

---

## Results

Measured by `aria bench` on the 46-task suite in `aria/bench/tasks.py`
(`benchmarks/report.md`, regenerated by CI):

| metric | value |
|---|---|
| exact correctness | **100%** (46/46) |
| invalid derivation rate | **0.0** |
| verified trace rate | **100%** |
| mean trace accuracy | **100%** |
| unit consistency (9 laws + 1 control) | **100%** |

| suite | tasks | correct | mean steps |
|---|---|---|---|
| solve (linear, quadratic, irrational, complex, identity, contradiction) | 12 | 12 | 3.33 |
| simplify | 7 | 7 | 1.71 |
| factor | 5 | 5 | 1.20 |
| log / exp | 6 | 6 | 1.00 |
| modular | 5 | 5 | 0.60 |
| calculus | 5 | 5 | 1.00 |
| identity verification | 6 | 6 | 0.00 |
| dimensional analysis | 10 checks | 10 | — |

These numbers describe a **small, deliberately scoped suite** that the system was
built against — they are not a claim of general mathematical competence. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the protocol, the negative
results (the learned policy does *not* beat the hand-ordered baseline at this
scale), and what a credible external evaluation would require.

The neural component, measured separately:

```
preference pairs mined from verified derivations : 111
held-out ranking accuracy                        : 73.9%
```

---

## Local models: the math-only channel

ARIA can sit behind a local LLM so that **mathematics is the only intermediate
language**. The model is grammar-constrained to emit a three-line contract
(`goal` / `var` / `expr`) and nothing else; ARIA parses it, proves every step, and
the model then narrates the verified trace. The model never states a mathematical
fact, so it cannot invent one.

```bash
aria grammar --mode loose --out aria-loose.gbnf      # the output alphabet
llama-server -m qwen3-4b-instruct-q4_k_m.gguf --port 8080 --grammar-file aria-loose.gbnf
aria ask "solve for x in 2x + 4 = 10" --endpoint http://127.0.0.1:8080/v1/chat/completions
aria ask "solve for x in 2x + 4 = 10" --dry-run      # no model needed
```

A 3–8B model is sufficient because the model only translates; the mathematics is
done exactly. Full guide, including why a 2.8T model is the wrong target for a
tablet and what can only be achieved by retraining:
[`docs/LLM_INTEGRATION.md`](docs/LLM_INTEGRATION.md).

---

## Rewriting the weights

Everything above changes *behaviour* without touching a parameter. The next step
up is to change the weights — and ARIA specifies exactly how, because the
verifier is already a reward function:

| completion | reward |
|---|---|
| invalid contract (didn't speak mathematics) | 0.00 |
| parsed, rejected by the symbolic layer | 0.05 |
| `partial` (some steps proven) | 0.30 + 0.40 × trace accuracy |
| `refuted` (a decided counterexample) | 0.50 |
| `solved`, every step proven | **1.00** |

No labels, no human preference data. The rewrite rules become the reward and the
reward becomes the gradient, so a weight update is steered toward provable
mathematics.

```bash
aria dataset sft --out data/sft.jsonl      # 136 question → contract pairs
aria dataset rl  --out data/rl.jsonl       # 46 prompts + reward id
aria reward --completion "goal: solve
var: x
expr: 2x + 4 = 10"                        # -> 1.00 solved
python training/sft_qlora.py --dry-run     # data check, no GPU needed
```

Reference QLoRA and GRPO scripts live in [`training/`](training/README.md); the
reasoning — what a weight is, why model editing cannot install a procedure, and
which hardware tier you actually need — is in
[`docs/WEIGHT_REWRITING.md`](docs/WEIGHT_REWRITING.md).

---

## The ledger: a record instead of a reward

A reward is a number somebody hands you; it can be maximised without being understood.
ARIA keeps a file instead — one line per attempt, written by the system, for the system:

```json
{"seq": 7, "question": "solve 9x - 3 = 24", "skill": "linear_equation",
 "outcome": "solved", "verified": true, "demonstrated": true,
 "steps": [{"rule": "move_all_to_one_side", "ok": true, ...}],
 "first_error": "", "note": "subtract 24, then divide by 9",
 "author": {"verdict": "verifier", "note": "model"}}
```

The verifier writes `outcome`, `steps` and `first_error`; the model writes only `note`.
Nothing is scored. An attempt is *demonstrated* when every step the engine took verified
and the run reached a decision — so there is no partial credit to farm, and
`solve 9x - 3 = 24` repeated five times holds nothing, because the ledger counts **distinct
problems**.

Advancement is a school, not a staircase: twelve rungs (arithmetic → simplify → expand →
linear → factor → quadratic → derivatives → log/exp → trig → identities → congruences),
each held by three distinct proven problems, then college. A failure never costs a grade —
it opens a *repair*, and advancement waits until the repair is clean. A rung the engine
cannot decide is marked **dark**, not failed.

```bash
aria ledger next                        # the problem to attempt, plus its contract
aria ledger record --question "solve 9x - 3 = 24" \
                   --completion "goal: solve\nvar: x\nexpr: 9x - 3 = 24"
aria ledger report                      # the grade card
aria ledger sft --out data/ledger.jsonl # only the processes that were proven
```

The motivating measurement: the shaped reward pays **0.90** for `9/x - 3 = 24`, whose final
step fails verification. The ledger refuses it. That comparison is a test
(`test_the_scalar_reward_can_pay_for_a_process_that_fails`), not a claim.
Full reasoning and falsifiers: [`docs/LEDGER.md`](docs/LEDGER.md).

---

## Positioning against prior art

**This is not a new idea, and the README should not imply that it is.** ARIA
continues four well-established lines of work: computer algebra and automated
theorem proving (1960s–), neuro-symbolic AI, neural theorem proving in the
AlphaGeometry / AlphaProof / DeepSeek-Prover tradition, and neural-guided
symbolic search in the DreamCoder tradition. In all of those systems a neural
component proposes and a symbolic engine verifies — the same split ARIA uses.

What is modestly distinctive here is not capability but discipline:

- **step-level invariants** — each step declares *which* algebraic property makes
  it admissible, and that property is checked (this is what rejects squaring both
  sides);
- **three-valued verdicts as a headline metric** — `unknown` is reported, and the
  benchmark's headline number is the *invalid derivation rate*;
- **a hard boundary the neural component cannot cross** — it ranks rule outputs,
  it cannot emit expressions;
- **dependency-free reproducibility** — ~6,300 lines, stdlib only, deterministic.

ARIA is far weaker than the state of the art in capability: no Lean/Coq backend,
no proof search, no pretrained model, no IMO-scale results, and a 46-task
self-authored benchmark. Full lineage and the blunt comparison:
[`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

---

## Status: what is and is not implemented

**Implemented and tested**

- Exact symbolic representation and canonical forms
- Notation parser: implicit multiplication, fractions, powers, functions, matrices, congruence, derivatives, scientific notation, Greek symbols
- Goal extraction from instructional text ("solve for x in …", "simplify …", "verify …")
- Rewrite rules for algebra, logarithms/exponentials, trigonometry, factoring and calculus
- Equation solving: linear, quadratic (rational/irrational/complex roots), higher degree via rational roots, linear congruences
- Step-level verification with five distinct invariant classes
- Dimensional analysis, nine physical laws, dimensional inversion
- Structural embeddings and a trained step-ranking policy with policy-guided search
- Benchmark harness with five reported metrics
- CLI, 208 tests, CI workflow

**Partial / deliberately bounded**

- The parser handles mathematical *notation* plus a small instruction grammar — not general natural language. Ambiguous juxtaposition (`2ab`) is resolved by an explicit documented rule, not by a language model.
- Proof goals return a verified verdict on an identity; multi-premise derivations are parsed and type-checked but not discharged (no Lean/Coq bridge yet).
- The neural core is a compact logistic ranker over structural features, not a transformer over tree positions.

**Not implemented** (tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md))

- Lean/Coq export and proof-state search
- Z3/SMT bridge for non-linear constraints
- Tensor/linear-algebra reasoning beyond matrix parsing
- Training data corpora at scale (the current policy trains on ~100 mined pairs)
- Comparative evaluation against LLM baselines through a fixed API

---

## Development

```bash
pytest                                  # 208 tests
aria bench                              # full suite + markdown report
aria bench --suite solve --markdown benchmarks/report.md
aria train --epochs 80 --eval           # retrain the step policy
python -m aria --help
```

ARIA runs without SymPy. In that mode the verifier proves polynomial and
rational-function equivalences exactly, refutes by exact/float probing, and
honestly returns `unknown` for transcendental identities:

```console
$ python -m aria --backend internal verify "log(a*b) = log(a) + log(b)"
Status: partial    undecided: unknown via no-backend (...)
```

Install `sympy` to promote those cases to proofs.

---

## License

MIT — see [LICENSE](LICENSE).
