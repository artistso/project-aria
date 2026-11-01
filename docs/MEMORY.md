# The text memory — a file the system can look into

Weights are knowledge that cannot be read, cited, or corrected one item at a time.
This is the other kind: a file of short text entries that ARIA writes, searches, and
can be argued with. It sits beside the ledger. The **ledger** records what the system
*did*; the **memory** records what it *believes*, and where each belief came from.

```bash
aria memory add "the Schwarzschild singularity at r = 0 is spacelike" \
    --tags penrose,singularity --source aria/geometry/causal.py --confidence verified
aria memory add "I liked the cheesecake; offer a variation again" \
    --kind preference --source "user-stated" --confidence stated
aria memory find "where is the horizon"        # ranked, explainable
aria memory correct 7 "corrected statement" --source "geometry check"
aria memory context "penrose singularity" --max-chars 1200
aria memory audit                              # active entries with nothing holding them up
aria memory export --out memory.md             # active entries plus supersession history
```

## 1. Five kinds of entry, and one rule

| kind | what it is | where it comes from |
|---|---|---|
| `fact` | a statement that is true or false — *a cube has six faces* | documents, geometry checks, scanned sources |
| `procedure` | something that worked — *this contract solved this problem* | the ledger, automatically |
| `correction` | something believed before that turned out wrong | a broken ledger attempt |
| `source` | a pointer to material worth rereading | GitHub, Drive, Docs |
| `preference` | a subjective, user-stated like, dislike, name, or requested variation | direct user statement; confidence remains `stated` unless separately tested |

The rule: **every entry carries a source and a confidence**, and confidence is one of
`unverified` → `stated` → `tested` → `verified`.

An entry with no source is not deleted and not hidden; it is marked `unverified` and
`aria memory audit` lists it. `aria memory correct <id> <text>` appends a sourced
replacement with a `supersedes` link. When the original is a preference, the replacement
retains `kind=preference` unless explicitly changed. The original remains in the file as
history, while retrieval and audit use the active replacement. Corrections cannot target
an already superseded record; correct the current replacement instead.

Human appreciation and preference are real interaction signals, but they are not
mathematical proof or a scalar training objective. A person may spontaneously say what
they liked, disliked, or want changed; the system must not require praise as a protocol.
When deliberately saved as a `preference`, the entry remains subjective and correctable.
No feedback means no observation—not dislike, failure, or a penalty. A person may also
choose a local self-reward without asking another person or asking the system; that act
does not need to be logged or validated. This repository has no automatic preference
capture or feedback solicitation, and `aria ask` does not inject memory automatically.

That is a guard against a text memory turning into confabulation, not a proof that every
source is true. The current test suite checks the memory mechanics and geometry routines;
it does not establish the truth of every user-supplied or imported memory.

## 2. Looking into it

Retrieval is lexical — IDF-weighted token overlap over the text and tags, with a
coverage bonus and a confidence weighting — deliberately, because:

* it runs on a tablet with no GPU and no embedding server;
* it is explainable: you can see *why* an entry came back, which matters when the
  entry is going to condition a model;
* it cannot drift. An embedding index silently changes when the model changes; a term
  match does not.

`aria memory context "..."` renders the result as a block:

```
What ARIA remembers that bears on 'singularity spacelike' (2 of 36 entries):
- (fact, verified) schwarzschild Penrose diagram: r = 0 is singularity at UV = 1;
  spacelike boundary: a moment in time, not a place in space
  [geometry, penrose, schwarzschild] (aria/geometry/causal.py)
Use an entry only if it applies; each carries its source, and `unverified` means
nobody has checked it. A `preference` is subjective user guidance, not a fact, consent,
or verifier result. No feedback means no observation, not dislike or failure; do not
infer a penalty.
```

That last line is part of the block on purpose. The model is told that memory is
evidence, not authority. `--max-chars` omits only whole entries and prints a reduction
notice; JSON output contains the complete source-entry manifest. If even the mandatory
header and notice do not fit, the CLI returns `HELD` with recoverable entries and asks for
a larger budget. No entry is clipped or silently discarded.

## 3. The three uses

**Retrieval → prompt block.** `aria memory context` produces a sourced block for a caller
to inspect and place into a prompt. The current `aria ask` / `MathChannel` does **not yet**
automatically retrieve or inject that block, so memory-conditioned model behavior is not
claimed as an integrated feature.

**Work → memory.** `aria memory import --from ledger` turns each demonstrated attempt into
a `procedure` (confidence `verified`, source `ledger#N`) and attempts with a recorded
`first_error` into a `correction` (confidence `tested`). These imported correction records
are history; explicit `memory correct` is the operation that supersedes a particular
active entry. Identical entries are de-duplicated.

**Ledger → weights.** The working training path is ledger-backed SFT after school
advancement. `aria ledger sft` exports question/contract pairs from held rungs by default;
it does not currently attach a retrieved memory block or claim that memory caused the
model to learn a fact. The actual gradient step requires running `training/sft_qlora.py`
on a compatible, explicitly selected base model. Until retrieval provenance is carried
into that pipeline, the honest path is ledger → held rung → SFT, not memory → gradient.

## 4. What is not built yet, and should be

* **Automatic ask-time retrieval.** The memory command can build a prompt block, but
  `MathChannel` does not yet read the memory store or track which entries were used.
* **Citation counting and provenance to SFT.** Nothing tracks retrieval frequency or
  carries memory IDs into a later ledger/SFT record, so memory-attributed learning is not
  established.
* **Automatic demotion.** A failed ledger attempt can be imported as a correction record,
  but it does not automatically supersede a specific memory. Use `memory correct` to
  supersede one explicitly and retain its history.
* **Semantic search.** If lexical retrieval misses paraphrases in practice, evaluate a
  local embedding model (see `docs/HF_MANIFEST.md`) rather than silently switching to a
  cloud call.

## 5. What would falsify this

1. **Retrieved memory does not change outcomes.** After ask-time integration and
   provenance logging exist, run the same task set with and without memory context. If
   contract validity and verified-step outcomes do not change, the memory is decoration.
2. **The audit stays green while the memory is wrong.** Spot-check ten `verified`
   entries against their sources; if any are false, the confidence ladder is being
   applied to the wrong things.
3. **The file grows faster than the grade.** Same test as the ledger: memory that never
   changes behaviour is a diary with headings.
