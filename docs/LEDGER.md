# The ledger — ARIA's own record of what it got right and wrong

> "Reward can be duplicitous and deceiving and it brings about terrible structure. The
> reward for doing good is the knowledge and understanding that one has done something
> good, and it builds the structure going forward. I think the best reward system would be
> to allow the AI to build out a separate file for themselves that keeps track of their
> correct and incorrect processes, and we utilize it just like a grade school system
> advancing up to college."

This document implements the process-record and school parts of that paragraph. It
rejects scalar reward-maximisation as the curriculum; it does **not** reject human
appreciation, spontaneous validation, personal preferences, or private self-reward. Those
are optional human signals, not proof of mathematics and never a condition for progress.

---

## 1. Why the scalar had to go

The earlier implementation shipped `reward()`: a 0–1 verifier score. The problem is not
that human feedback is worthless or must be eliminated; it is that one scalar is a lossy,
optimizable substitute for process evidence and human context. A person's spontaneous
appreciation or preference may be useful interaction information, but it cannot certify a
proof, and its absence says nothing. The scalar was unsuitable as ARIA's organizing
training objective for reasons we can inspect rather than hand-wave.

**It pays for a broken process.** Take the contract `goal: solve / var: x / expr: 9/x - 3 = 24`.
ARIA's engine emits three steps, the last of which *fails verification*, and then declares
`solved`. The reported value `x = 1/3` happens to satisfy the equation, but the step that
produces it has an invalid justification and is not verified. The shaped reward returns
**0.90** (`0.70 + 0.30 × trace_accuracy`) because two of the three steps checked out. It
pays for an unchecked process even when the final value happens to be right. A model
trained on this could learn that a plausible-looking trace is worth more than a checked
one.

**It compresses a process into a point.** A single number cannot distinguish "solved it in
four verified steps" from "guessed the answer and back-filled one step that happened to
check". Both can score 1.0. Once the number is the target, the cheapest way to raise it is
to stop doing the thing it was supposed to measure.

**It erases context.** A score cannot retain whether a person was expressing delight,
requesting a different name, correcting a preference, or asking for a logic review. Keep
those statements in their own sourced, inspectable channel rather than converting them to
points. The ledger keeps what the scalar threw away: **which rule was applied, whether the
engine could check it, and where it first broke.**

## 2. What replaces it

`aria/llm/ledger.py` appends one JSON line per attempt to a file the system owns
(`~/.aria/ledger.jsonl`, or `$ARIA_LEDGER`).

```json
{"seq": 7, "when": "2026-10-31T18:04:11+00:00",
 "question": "solve 9x - 3 = 24", "skill": "linear_equation",
 "contract": "goal: solve\nvar: x\nexpr: -3 + 9 x = 24",
 "outcome": "solved", "verified": true, "demonstrated": true,
 "steps": [{"rule": "move_all_to_one_side", "before": "-3 + 9 x = 24",
            "after": "-24 - 3 + 9 x = 0", "ok": true, ...}],
 "first_error": "", "note": "subtract 24, then divide by 9",
 "author": {"verdict": "verifier", "note": "model"}}
```

Four rules make the file trustworthy.

1. **The verifier writes the verdicts; the model writes the reflection.** `outcome`,
   `steps`, `verified` and `first_error` come from the symbolic engine. Only `note` is
   authored by the model. A self-report nobody checks is not a record, it is a story — so
   the record is split by authorship and the split is stored in the file.
2. **Nothing is scored.** An attempt is *demonstrated* when the contract parsed, every step
   the engine took verified, and the run reached a decision. There is no partial credit, so
   there is nothing to farm.
3. **Advancement is by demonstrated process, held over distinct problems.** Three clean,
   distinct proofs at a rung hold it. Not three attempts: `solve 9x - 3 = 24` repeated five
   times holds nothing, because the ledger counts distinct expressions.
4. **A rung the engine cannot decide is marked `blocked`, not failed.** Congruences come
   back `partial` — ARIA has no congruence solver. The card says `DARK` and the reason,
   instead of blaming the learner for a missing rule.

### Human signals are optional and non-punitive

A person may spontaneously say “I liked that cheesecake; please offer a variation,” ask
for a different name, correct a prior preference, or request an inspection of some logic.
These are different kinds of human communication, not a scalar score. User-stated likes and
dislikes can be saved locally as `memory` entries of kind `preference`, kept as subjective,
correctable text with provenance. A request to inspect logic is a task to handle, not a
reward. A stated preference is not evidence that a mathematical claim is true.

There is no requirement to praise, rate, or validate each attempt. **No feedback means no
observation**—not dislike, failure, or a missing reward that the system should punish. A
person remains free to give themself a private reward without asking another person or the
system; it is not a grade and need not be logged. Mistakes open repairs while held grades
remain; they are opportunities to improve the foundation, not grounds for a penalty.

“Monte Carlo” is a useful analogy for sparse, stochastic observations and occasional
checks—not a claim that every human interaction is random or measurable. An optional
sampled feedback prompt could be explored later, but no solicitation/sampling routine is
implemented here, and no sampled appreciation would replace the independent verifier.

## 3. The school

| level | rung | what it means | problems available |
|---|---|---|---|
| 1 | `arithmetic` | evaluate a number, or check an arithmetic claim | 4 |
| 2 | `simplify` | rewrite into canonical form | 17 |
| 3 | `expand` | distribute products | 4 |
| 4 | `linear_equation` | solve a degree-1 equation | 10 |
| 5 | `factor` | factor a polynomial | 8 |
| 6 | `quadratic_equation` | solve a degree-2 equation | 8 |
| 7 | `derivative_polynomial` | differentiate a polynomial | 4 |
| 8 | `derivative_transcendental` | differentiate through a function | 6 |
| 9 | `log_exp` | logarithms and exponentials | 6 |
| 10 | `trig` | trigonometric expressions | 4 |
| 11 | `identity` | decide a statement for all values | 6 |
| 12 | `modular` | congruences and divisibility | **dark** |
| — | college | every gradable rung held | — |

Every problem in that table is *executable*: it is either drawn from the benchmark suite or
from the seed list in `CURRICULUM`, and a test asserts each seed classifies to its own rung
and is demonstrated by the engine. A rung you cannot demonstrate is not a rung.

**States.** `learning` → `held` (three clean attempts over at least three distinct
expressions) → `repair` if a later attempt on that rung fails. A failure never erases the
historical held grade; it opens a *repair*, and the repair closes after two new clean
attempts on distinct expressions. The repair is prioritized before new material, and
college graduation is suspended while any repair is open. `blocked` is the engine's
limitation, not the learner's.

**Graduation** is not grade 13. It is the point where every gradable rung is held, the dark
rungs are named, and `next_task()` returns `None` — there is nothing left to be taught by
this verifier, only claims left to attack.

## 4. Using it

```bash
aria ledger next                     # the problem to attempt, with its contract
aria ledger record --question "solve 9x - 3 = 24" \
                   --completion "goal: solve\nvar: x\nexpr: 9x - 3 = 24" \
                   --note "subtract 24, then divide by 9"
aria ledger report                   # the grade card
aria ledger sft --out data/ledger.jsonl   # demonstrated processes from held rungs only
```

The card, after working the ladder for thirty-four attempts:

```
ARIA report card -- 34 record(s), graduated (college) -- 11 of 12 rungs held, 1 dark

   1  arithmetic               HELD    3/3 proven, 3 distinct, streak 3
   4  linear_equation          HELD    3/3 proven, 3 distinct, streak 3
   ...
  12  modular                  DARK    0/1 proven
      the engine returns `partial` here: this rung cannot be graded yet
```

### What it feeds

* **SFT** — `ledger sft` exports only demonstrated processes from rungs the school already
  holds. A failed attempt is never a training target; if a later attempt on the same
  expression succeeds, the corrected pair is eligible only when that rung is held. The
  error stays in the file as inspectable history, not as a lesson. `--include-learning` is
  an explicit bootstrap escape hatch, not the default curriculum.
* **Legacy scalar experiment** — `training/grpo_verifier.py` now refuses training unless
  `--experimental-allow-scalar-reward` is passed. Its binary and shaped signals remain for
  historical comparison only; they are not the ledger or the accepted training objective.
* **Curriculum** — `next_task()` picks repairs first, then the lowest unheld rung, and skips
  problems already filed under that rung. The order of study is derived from the record, not
  set by hand.

## 5. What this does *not* solve

**The ledger is only as honest as its verifier.** The engine can print `solved` while a
produced step remains unchecked. The `9/x - 3 = 24` case demonstrates the distinction: the
reported value is right, but the ledger refuses to call the process demonstrated because
one step failed verification. Fixing the solver is separate work; the ledger's job is to
refuse to count the unchecked process in the meantime.

**`blocked` can hide a bad curriculum.** If a rung is dark because the problems are bad
rather than the engine, the ledger will excuse it forever. The guard is that `blocked`
requires attempts and only `partial`/`open`/`rejected` outcomes — a wrong answer is always
`learning` or `repair`, never `blocked`.

**Three-of-three is a policy, not a law.** It is deliberately small so a run finishes in
half an hour, and deliberately about *distinct* problems so repetition cannot substitute
for breadth.

**The note field can rot.** Nothing reads the model's own reflection yet. It is stored
because it is the only place the system says *why*; the moment it is used to propose the
next repair, it stops being decoration and becomes the feedback channel it was meant to be.

## 6. What would falsify this

1. **Held rungs do not predict held-out performance.** Take the ledger after graduation and
   run `aria bench`. If exact correctness on the suites behind held rungs is not
   substantially higher than on dark or unheld ones, mastery is miscalibrated.
2. **The file grows and the grade does not move.** A ledger with hundreds of entries and a
   stuck grade means the repair loop is not teaching anything; it is a diary.
3. **`blocked` swallows rungs.** If more than one rung is dark on a verifier that can decide
   most mathematics, the classification is wrong, not the engine.
4. **The legacy scalar experiment is not useful.** If either GRPO signal encourages
   trivial-but-valid contracts or raises its scalar while held-out task results do not
   improve, the experiment has failed. It must not alter school advancement or replace
   ledger-backed SFT; the current code requires an explicit opt-in to run it.
