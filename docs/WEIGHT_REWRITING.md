# Changing What a Weight Means

> "We have to change the weights — we have to change what it means to be a
> weight — and use our logic here as a rewrite."

Changing model weights remains a requirement. A grammar or interface alone does
not satisfy it. The accepted organising path in ARIA is now **ledger + staged
school advancement + supervised fine-tuning (SFT)**. Scalar reward-maximisation
was explored, measured, and rejected as the curriculum or organising principle.

This document separates what can change today from the still-open research
claim about weights themselves.

---

## 1. Four meanings of "weight rewriting"

| # | Operation | What actually changes | Is it a weight change? |
|---|---|---|---|
| 1 | Interface — grammar, tools, contracts | nothing in the model weights | no |
| 2 | Training — SFT or continued pretraining | parameters/adapters and the function they implement | yes |
| 3 | Weight-space editing — ROME/MEMIT, task arithmetic, merging | selected associations or behaviour directions | yes, but not a reliable way to install a multi-step procedure |
| 4 | Architecture substitution — symbolic layers or generated weights | the function class and what a parameter can represent | research-grade |

ARIA has the translator interface today and reference QLoRA code for actual
parameter updates. The ledger-backed training path must be run on a compatible
model and GPU before any claim that weights have changed is justified.

The user-specified model source is Hugging Face
`meshllm/Kimi-K3-UD-Q4_K_XL-layers`, with retrieval requested by `git clone`:

```bash
git clone https://huggingface.co/meshllm/Kimi-K3-UD-Q4_K_XL-layers
```

That exact repository path, its existence, contents, license, and runtime are
**unverified here**: Hugging Face was unreachable from this sandbox. The name is
preserved as requested; it is not evidence that the repository exists. A clone
may contain Git LFS pointer files rather than model weights, so inspect the
actual files before using them. Do not silently substitute another source.

---

## 2. Why direct weight editing is not the current route

A fact edit and a procedure update are different problems. A procedure such as
solving an equation is a composition of steps; changing a few factual
associations does not install a dependable multi-step process. ROME/MEMIT,
task arithmetic, and merging may be useful for bounded vocabulary or behaviour
experiments, but they do not replace the verifier or the school.

ARIA's natural-language model remains at the input/output edges. The symbolic
runtime is the representation and reasoning substrate between those edges.
The LLM still matters: the translator's weights can be updated to better map
natural-language requests into ARIA's contract, and the narrator can be trained
to express only the checked frame.

---

## 3. Accepted update loop: ledger, school, SFT

```text
question
  → model with current weights emits a contract
  → ARIA parses and verifies every symbolic step
  → ledger records the process, failures, and corrections
  → distinct verified work holds a school rung
  → only held-rung processes are exported for SFT by default
  → QLoRA changes an adapter/weights
  → held-out phrasings and ARIA's independent benchmarks evaluate the change
```

The ledger is not a scalar objective. Its visible state is the work record and
the staged advancement rule:

* each attempt retains its question, contract, outcome, checked steps, and first
  break;
* an attempt is demonstrated only when the engine reaches a decision and no
  produced step is unverified or refuted;
* three clean, distinct examples hold a rung; repeated attempts on one problem
  do not substitute for breadth;
* failure opens repair without erasing the previous grade;
* an undecidable rung is blocked/held, not scored as a learner failure;
* `aria ledger sft` exports demonstrated processes from held rungs by default.

Example:

```bash
aria ledger next
aria ledger record --question "solve 9x - 3 = 24" \
    --completion "goal: solve\nvar: x\nexpr: 9x - 3 = 24"
aria ledger report
aria ledger sft --out data/grade-04.jsonl --max-level 4
python training/sft_qlora.py \
    --model <compatible-local-or-HF-base> \
    --data data/grade-04.jsonl \
    --out out/aria-grade-04
```

`--include-learning` is an explicit bootstrap option; it is not the default
school path. The training script updates the model adapter; the ledger decides
what qualifies as a lesson. Neither a successful training run nor a high school
rung proves general mathematical competence. Evaluation on held-out phrasings
and independent tasks remains necessary.

---

## 4. Why the scalar-reward path was rejected

The old scalar was a compact score from the verifier. It was not an adequate
organising principle because it compresses distinct process histories into one
number and can pay for a broken process. The concrete regression is retained as
a diagnostic: `9/x - 3 = 24` can earn **0.90** from the shaped score even though
one of its steps fails verification. The reported value happens to satisfy the
equation, but the ledger refuses to count the unchecked process as demonstrated.

`aria reward`, `aria dataset rl`, and `training/grpo_verifier.py` remain only as
legacy diagnostics/research scaffolding. GRPO training now requires an explicit
`--experimental-allow-scalar-reward` acknowledgement. They do not determine
school advancement and are not the accepted weight-update path. A binary scalar
is still a scalar objective; it does not become the ledger merely because it
uses the verifier.

Rejecting that objective does not mean rejecting human feedback. Appreciation,
likes/dislikes, corrections, and requests for review may be voluntary, spontaneous,
and useful as subjective interaction context. They do not prove mathematics, change
school advancement, or become negative reward when absent. A person may choose a
private self-reward without external validation; mistakes lead to repair rather than
loss of a held grade. Current local memory can store a user-stated preference, but
automatic feedback capture or solicitation is not implemented.

The important distinction is:

* **ledger:** a persistent, inspectable record of which process was checked and
  what was learned or left unresolved;
* **school:** a discrete advancement state derived from distinct demonstrated
  work;
* **SFT:** an actual parameter/adapter update on exported held-stage records;
* **benchmark:** a separate evaluation of whether the changed weights improved
  the requested behavior.

---

## 5. Model and hardware constraints

The tablet is an inference device, not the machine that runs QLoRA. A Q4 GGUF
is an inference artifact; QLoRA needs a compatible training base and GPU. The
current `training/sft_qlora.py` is a reference, not evidence of a completed
training run. The specified Kimi K3 layers path is not verified and must not be
assumed to be trainable from its name alone.

The 1.7B-versus-4B choice and whether to rent a GPU remain open deployment
decisions. Measure the base model's contract validity, repair count, peak memory,
and held-out performance before selecting a training tier. Do not report a
model-size or hardware estimate as measured unless the actual model files and
runtime have been inspected.

---

## 6. Literal research directions beyond SFT

If "change what a weight means" means making parameters themselves symbolic
operators, that is a separate research programme. Candidate directions include
symbolic arithmetic layers, hypernetworks/fast weights, rule-conditioned
adapters, and continued pretraining on formal corpora. They require explicit
hypotheses, baselines, falsifiers, and complete weight provenance. No claim that
any of these has solved ARIA's target should be made without such evidence.

The immediate testable path remains: preserve the verified process in the
ledger, advance by the staged school, change weights through SFT, then measure
held-out behavior. The older scalar objective is not quietly reinstated under a
new name.
