# Using ARIA with an Offline LLM (tablet)

Practical guide for working on ARIA on a Galaxy Tab S10+ (or any device) with a
local model and no network. The short version: **an offline LLM cannot run
ARIA, but it can read it** — so what you upload decides what you get back.

---

## 1. What the model can and cannot do

| Task | Offline LLM | Notes |
|---|---|---|
| Explain / critique the design | ✅ good | Use the context pack |
| Review specific functions for bugs | ✅ good | Give it the file, not the zip |
| Invent new rewrite rules or benchmark tasks | ✅ good | Ask for the exact `@rule` signature |
| Explain a derivation ARIA produced | ✅ good | Paste the CLI output + context pack |
| *Run* ARIA, solve an equation for you | ❌ no Python | It will hallucinate derivations; do not trust them |
| Verify a claim | ❌ no | Only the SymbolicEngine verifies, not the model |
| Keep a large codebase in mind at once | ⚠️ limited | 6,300 LOC ≈ 60k+ tokens |

**Fidelity warning.** A local model will happily invent functions that do not
exist (`aria.solve_equation(...)`). The context pack is the authoritative
interface list. If an answer references something not in the pack, it is
hallucinated.

---

## 2. Pick a bundle by context size

| Model context | Upload | Size |
|---|---|---|
| ~8k tokens (4k–8k) | `aria-context-mini.md` | ~5 KB |
| 16k–32k | `aria-context-pack.md` | ~25 KB |
| 64k–128k | `aria-context-pack.md` + `README.md` + `docs/DESIGN_SPEC.md` | ~45 KB |
| 200k+ | the whole `aria/` source tree, or `aria-offline-llm.zip` | ~760 KB |

Files at the download location:

```
aria-context-mini.md      small digest, start here
aria-context-pack.md      full single-file digest (all five layers)
aria-offline-guide.md     this guide
aria-docs.zip             README + LICENSE + every doc (~80 KB)
aria-0.1.0.zip            clean, installable source tree (~120 KB)
```

Plain `.md` usually uploads more reliably than `.zip` on Android LLM apps; some
apps extract zips, many do not.

---

## 3. Prompts that work

### 3.1 Design review (upload: context pack)

> You are reviewing ARIA, a neuro-symbolic reasoning system described in the
> attached context pack. Focus on the verification layer. (1) Identify the three
> weakest points in the decision procedure for `equivalent(a, b)` and what input
> class could produce a wrong `proven`. (2) For each step invariant in §4.2, name
> an operation that is *sound* but would be rejected, and one that is unsound but
> might slip through. (3) Propose the smallest change that would make the system
> defend itself against a hostile rewrite rule. Quote the relevant sections.

### 3.2 Rule extension (upload: context pack; better: `aria/reason/rules.py`)

> Here is ARIA's rule file. Add a rule for `a^2 - b^2` style factoring of
> expressions where the coefficient of the squared term is not 1 (e.g.
> `4x^2 - 9y^2`), matching the existing `@rule(name, kind, description)`
> decorator and returning `(new_expr, justification)` or `None`. It must be
> a sound identity, must not fire when nothing changes, and must be added to
> `FACTOR_RULES`. Then write the pytest test that asserts the engine factors it
> and that the derivation is `.valid`.

### 3.3 Benchmark expansion (upload: `aria/bench/tasks.py`)

> Here is ARIA's benchmark suite. Add 10 tasks in the same `Task(...)` style that
> test *soundness rather than accuracy*: cases where a careless system would
> report a wrong step. Each task needs an independently stated expectation.
> Do not add tasks the current rule set cannot already solve, unless you mark
> them with a `note=` explaining the expected `partial` outcome.

### 3.4 Derivation audit (upload: context pack + a pasted CLI trace)

> Below is a derivation produced by ARIA. For each numbered step: state the
> invariant that must hold, check by hand whether it holds for these expressions,
> and flag any step whose *stated rule* does not match what actually changed.
> Do not re-solve the problem; audit the trace.

### 3.5 Port for the tablet (upload: context pack)

> ARIA's core (parser, AST, verifier's exact rational path) is pure Python with no
> dependencies. Produce a faithful Kotlin (or Swift) port plan for the subset
> needed to run `aria solve "2x + 4 = 10"` offline: module boundaries, the AST
> types, where `Fraction` maps to, and which verifier branches can be dropped
> without silently upgrading `unknown` to `proven`. Call out anything that cannot
> be ported soundly.

### 3.6 Proposal critique (upload: `README.md`)

> Critique this README as a skeptical reviewer. Where does it overclaim? Which
> sentences would not survive peer review? Which numbers are presented without
> the caveat they need? Be specific and quote.

---

## 4. If you want to actually run ARIA on the tablet

The core is dependency-free, so a Python runtime on Android is enough:

1. Install **Termux** (F-Droid) and run `pkg install python`.
2. `unzip aria-0.1.0.zip && cd aria-0.1.0`
3. `python -m aria run "solve for x in 2x + 4 = 10"`
4. Optional, for the stronger verifier: `pip install sympy` (needs network once).

### Measured profile without SymPy (stdlib only)

Verified by extracting the shipped zip and running it on a bare Python 3.11:

| metric | with SymPy | stdlib only |
|---|---|---|
| exact correctness | 46/46 (100%) | **34/46 (73.9%)** |
| invalid derivation rate | 0.0 | 0.0 |
| verified trace rate | 1.0 | 0.739 |
| mean trace accuracy | 1.0 | 0.804 |
| unit consistency | 1.0 | 1.0 |

This drop is the system behaving correctly, not breaking. What degrades:

- identities involving `log`, `exp`, `sin`, `cos` return `partial` / `unknown`,
  because the dependency-free verifier only decides rational functions;
- calculus tasks return `partial`, because the *independent* derivative check
  uses SymPy (the derivative itself is still computed and simplified by ARIA);
- irrational-root verification returns `partial` when the substitution cannot be
  reduced exactly.

What never changes: the **invalid derivation rate stays 0.0**, and no undecided
claim is ever reported as `proven`. That is the difference between a system that
degrades and one that starts guessing.

---

## 5. Working within a small context: what to upload per question

| Question | Upload |
|---|---|
| "Is the verifier sound?" | `aria-context-pack.md` (or §4 alone) |
| "Why did this trace fail?" | context pack + the CLI output |
| "Fix this function" | the single `.py` file |
| "Review the whole architecture" | context pack + `docs/ARCHITECTURE.md` |
| "Add tasks / rules" | `aria/bench/tasks.py` or `aria/reason/rules.py` |
| "Is this project credible?" | `README.md` + `docs/EVALUATION.md` + `docs/RISKS.md` |

Rule of thumb: **one question, one file, one caveat.** If you must paste code,
prefer the digest over the source; the digest was written to be read linearly.

---

## 6. Keep the numbers honest

When a model summarizes ARIA back to you, check that it repeats these caveats:

- 46/46 on a **self-authored** suite means "no regressions, no unsound steps",
  not general mathematical ability.
- The learned policy is **slower** than the fixed rule order at this scale —
  a published negative result, not a hidden flaw.
- No LLM baseline comparison exists yet, and the docs say why.
- `unknown` verdicts are a feature; a system that never says "unknown" is lying.

If a summary drops those, the summary is wrong even if it sounds confident.
