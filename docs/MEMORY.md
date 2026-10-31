# The text memory — a file the system can look into

Weights are knowledge that cannot be read, cited, or corrected one item at a time.
This is the other kind: a file of short text entries that ARIA writes, searches, and
can be argued with. It sits beside the ledger. The **ledger** records what the system
*did*; the **memory** records what it *believes*, and where each belief came from.

```bash
aria memory add "the Schwarzschild singularity at r = 0 is spacelike" \
    --tags penrose,singularity --source aria/geometry/causal.py --confidence verified
aria memory find "where is the horizon"        # ranked, explainable
aria memory context "penrose singularity"      # a block to put in front of a model
aria memory audit                              # entries with nothing holding them up
aria memory export --out memory.md             # the human-readable file
```

## 1. Three kinds of entry, and one rule

| kind | what it is | where it comes from |
|---|---|---|
| `fact` | a statement that is true or false — *a cube has six faces* | documents, geometry checks, scanned sources |
| `procedure` | something that worked — *this contract solved this problem* | the ledger, automatically |
| `correction` | something believed before that turned out wrong | a broken ledger attempt |
| `source` | a pointer to material worth rereading | GitHub, Drive, Docs |

The rule: **every entry carries a source and a confidence**, and confidence is one of
`unverified` → `stated` → `tested` → `verified`.

An entry with no source is not deleted and not hidden; it is marked `unverified` and
`aria memory audit` lists it. That is the whole defence against a text memory turning
into confabulation: a guess is allowed to exist, but it is not allowed to *look* like
knowledge. In the current test suite, the geometry import produces 36 verified entries
and zero unverified ones, and adding one unsourced guess makes the audit name exactly
that one.

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
nobody has checked it.
```

That last line is part of the block on purpose. The model is told that memory is
evidence, not authority.

## 3. The three uses

**Retrieval → the model.** The block above goes in front of the translator prompt. This
is the "look into it" behaviour: it is the difference between a model that has read
something once during training and a system that can go and check what it knows.

**Work → memory.** `aria memory import --from ledger` turns every demonstrated attempt
into a `procedure` (confidence `verified`, source `ledger#N`) and every broken attempt
into a `correction` (confidence `tested`). Importing twice adds nothing, because
identical text is de-duplicated. So the record of what the system did becomes the
representation of what it knows, without a human writing summaries.

**Memory → weights.** This is the "text-based understanding as a rewrite" part, and it
is the part to be careful about. A memory becomes a training target only through the
ledger, never directly:

1. A memory retrieved for a task is put in the prompt.
2. If the model then produces a contract that the engine *proves*, the ledger records
   it, and `aria ledger sft` exports it — with the memory in the prompt and the proven
   contract as the target.
3. If it produces a contract that fails, the ledger records the break, and the memory
   that misled it is a candidate for demotion.

So the pipeline is memory → prompt → ledger → gradient, and the gradient only ever sees
*proven* work. That is what makes "text as a rewrite" safe: the text proposes, the
verifier disposes.

## 4. What is not built yet, and should be

* **Citation counting.** Nothing tracks how often an entry is retrieved, so "frequently
  used and never contradicted" is a stated rule with no implementation. The data model
  has `links` for it; the counter is the next piece of work.
* **Demotion.** A memory contradicted by a failed attempt is flagged in the ledger but
  is not yet moved to `correction` automatically.
* **Semantic search.** If lexical retrieval turns out to miss paraphrases, the honest
  upgrade is a small embedding model run locally (see `docs/HF_MANIFEST.md`), not a
  cloud call — the memory file must stay usable offline.

## 5. What would falsify this

1. **Retrieved memory does not change outcomes.** Run the same task set with and
   without `memory context` prepended. If contract validity and proven-step rate do not
   move, the memory is decoration.
2. **The audit stays green while the memory is wrong.** Spot-check ten `verified`
   entries against their sources; if any are false, the confidence ladder is being
   applied to the wrong things.
3. **The file grows faster than the grade.** Same test as the ledger: memory that never
   changes behaviour is a diary with headings.
