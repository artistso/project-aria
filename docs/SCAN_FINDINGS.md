# Scan findings — what already exists, and what ARIA should take from it

Written after reading the account's GitHub repositories, Google Drive, and Google Docs for
material that bears on Project ARIA. Follows the convention of the `00 — START HERE` /
`03 — Change Log` / `05 — Scan of source Google Docs` set already in Drive: say what was
looked at, what it is worth, and what it is not.

**Scope actually covered.** 100 repositories listed on GitHub, READMEs of 12 read and file
trees inspected. 100 most recent Drive files listed, 5 read in full, 3 name searches run.
60 Google Docs listed by title and date, none read in full. This is a survey, not an
exhaustive audit; anything marked *unread* below should be treated as a lead, not a fact.

---

## 1. The one that matters most: `aura-cal` and AGF

`artistso/aura-cal` is a Kotlin/Compose mathematical canvas (in closed testing on Play,
version 3.0) with something ARIA needs: **Aura Graph Format, draft 0.1** — a documented
format for a *typed, port-bound hypergraph* of mathematical objects, shipped with a JSON
Schema, a conformance suite (60 kernel specs), and an **independent Python implementation**
in `python/` covering six conformance classes plus LaTeX projection (203 checks).

Why it matters: AGF's stated purpose includes "attaching machine-checkable claims (types,
diagnostics, **rewrite steps, verification verdicts**) to a stable document identity". That
is ARIA's `Derivation` object, almost line for line — goal, ordered steps with `before` /
`after` / `rule` / `justification` / `verdict`, and a final status. ARIA currently
serialises a derivation to text and markdown and nothing else.

What to take:

1. **AGF as a second serialisation of `Derivation`.** `aria export --format agf` mapping
   `Expr` → vertices/ports and `DerivationStep` → a rewrite edge carrying its verdict. The
   payoff is that ARIA's proofs become openable in a canvas app the user already ships,
   and AGF gains a second producer that is not a copy of the first.
2. **The type lattice.** Aura treats every object as typed and every connection as a port
   binding governed by a lattice. ARIA's symbolic embedding (`aria/embed/`) is numeric and
   tree-shape based; a type lattice is a cheaper, checkable signal that a rewrite is even
   *well-typed* before the verifier is asked.
3. **The audit habit.** `docs/benchmarks/` contains a reachability audit that found three
   symbols the ink classifier can never emit. That is the same species of finding as this
   ledger's dark rung: a part of the system that silently cannot work. Both projects should
   keep publishing them.

## 2. `quadrature-time` — the status table

A physics paper repository with 24 pytest checks that reproduce every number in the paper,
and a README table that sorts claims into **Established (standard, attributed)** / **New
and correct** / **Model-dependent** / **Heuristic or open**, citing what is ruled out
("with the natural energy identification it is already ruled out by transmon T₂ data").

Take the table, not the physics. ARIA's docs already gesture at this in
`docs/WEIGHT_REWRITING.md` §7; the four-row form is better than prose because it forces a
verdict on every claim, including the flattering ones.

## 3. `amcep` — falsification-first, and a warning about scores

`artistso/amcep` opens by refusing to be what its own source material claims. It defines
`S_n(ρ,T,x,E) = (ρ + T - ρ·xⁿ)/E`, differentiates it, shows `∂S/∂ρ > 0`, and concludes that
this **falsifies the stated contradiction-penalty principle** — the score goes *up* when
contradiction goes up.

That is the same failure mode as the ledger's motivating example, one level up: a formula
that looks like it measures the right thing, is easy to compute, and rewards the opposite
of what it claims. ARIA's shaped reward (`9/x - 3 = 24` earning 0.90 for a wrong answer) is
the small version of it. Both were caught the same way — by differentiating / auditing the
quantity instead of trusting it. Keep `docs/PRIOR_ART.md` honest about this being a pattern
in the account's own work, not a mistake unique to language models.

## 4. `synapshield` — the correction document

Contains a section titled *Important correction* that retracts attributed quotations and
health claims, an `MODEL_LIMITATIONS.md`, and a change log explaining that the previous
release kinetics implied a **12-hour half-life** against a claimed 10–15 years. This is the
house standard for retracting your own claim in public. ARIA's equivalents are the ledger's
`blocked` rung and `docs/RISKS.md`; they should read like this when ARIA is wrong.

## 5. AHSG — the screen ARIA would live behind

`artistso/AHSG` (Kotlin, "math proposal for an app") has `AHSGPhysicsEngine.kt` plus UI
components named `DerivationCard`, `SpectraCharts`, and screens `DerivationsScreen` /
`AuditsScreen`. If ARIA ever gets a tablet face, this is the existing shape for it: a
derivation card with an adjacent audit view. ARIA's `Derivation.to_markdown()` is already
close to what a `DerivationCard` renders.

## 6. The Resonance Engine set (Drive) — the ancestor of the ledger

Five files in Drive are the design of a previous attempt at this problem:

| file | content |
|---|---|
| `loss_function.md` | `L_total = L_knowledge + L_egalitarian + L_consistency`; a hinge floor `P(σⱼ) ≥ α`, a KL to the uniform prior, and a norm-conservation term between a "hippocampal singularity" and orbital knowledge vectors |
| `training_schedule.md` | five phases over 17+ weeks, one loss term dominant per phase |
| `tokenization.md` | a codon layer (`A`/`T`/`C`/`G` = amplitude / phase / cultural context / geometry) plus reserved structural tokens |
| `inference_log.md` | one record per query: four layers, gate values, phase cosines, a final composite loss, a collapsed output |
| `resonance_engine.py`, `data_pipeline.py`, `resonance_engine_v1_state.json`, per-question `query`/`answer` JSON | the executable part |

Two things follow, and they are the reason the ledger looks the way it does.

**`inference_log.md` is the ledger's ancestor.** It already recorded every attempt in a
file, with the query, the answer, and a verdict. What it recorded as the verdict was
`L_total = 0.015371` and a gate of 0.957 — similarities and a composite loss. It could not
distinguish a wrong process from a right one, because nothing in the record was a check.
The ledger is that idea, with the symbolic engine's per-step verdicts in place of the
similarity score.

**The egalitarian loss was not importable.** Constraining per-token probabilities to a
floor and pulling them toward uniform is a training objective for *generation*, and ARIA's
translator emits a three-line contract; a uniform prior over response modes would actively
degrade it. The one idea from that design that survived is the part that is checkable:
**distinctness** — three *different* problems must be proven, not one problem three times.

## 7. Leads not yet taken

* **Lean 4.** `MathlibDemo.lean_182_0_No info found…md` and two "AMCEP FULLY VERIFIED IN
  LEAN 4" documents show the account has done Lean experiments. Exporting ARIA's dark rung
  (congruences) as Lean obligations is the principled way to light it up, and it is the
  natural upgrade path for the verifier: SymPy decides most of the current ladder, Lean
  would decide what SymPy cannot.
* **`Anharmonic-Space`** (`src/engine/topologicalCore.ts`, "llm learning ocr") — unread
  beyond the file tree. Possibly relevant to input ingestion, not to verification.
* **Physics corpus** (zero-rings, relational thermodynamics, quadrature time) — not needed
  for ARIA's core; the *practice* in those repos is, and is taken above.
* **No repository in the account contains a training loop that runs on the tablet**, which
  is consistent with the hardware numbers in `docs/WEIGHT_REWRITING.md`.

## 8. What this changes in the plan

1. `aria export --format agf` — Derivation → AGF, so proofs are portable into `aura-cal`.
2. Lean obligation export for rungs the SymPy verifier cannot close (currently `modular`).
3. Adopt the four-row status table as the standard shape for any ARIA claim document.
4. Retire the Resonance Engine's composite loss in favour of the ledger, in writing, so the
   decision is on record rather than silently dropped.
