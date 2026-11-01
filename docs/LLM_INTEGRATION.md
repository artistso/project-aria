# The Math Channel — running a local model with ARIA

This document is about the specific architecture: *text in → mathematics as the
only intermediate language → text out*, with a local model at both boundaries and
ARIA in the middle.

---

## 1. "Thinking in pure mathematics" — three different things

The phrase covers three different engineering problems; the symbolic interface
and grammar can be run as code, while model training and model fit depend on the
actual weights and hardware.

| Tier | What it means | What it takes | Reachable today? |
|---|---|---|---|
| **A. Reasoning substrate** | the *state* the system manipulates is formal, not prose | a symbolic engine + verifier + a protocol that asks for a contract and rejects malformed or extra output | **yes** — this is `aria/llm/` |
| **B. Output alphabet** | the model may only *emit* mathematical tokens | constrained decoding with a grammar (GBNF), when the selected server honors it | **yes** — `aria/llm/grammar.py` |
| **C. Weight-level numeracy** | the *residual stream* encodes symbolic structure because the model was trained on formal corpora | continued pretraining with full-precision weights, a formal corpus, and serious compute | **no** — research scale |

ARIA implements a symbolic reasoning substrate, a contract protocol, and an
optional constrained output alphabet when the inference endpoint supports GBNF.
The language model is still an actual model at the natural-language edges;
`--dry-run` is only a parser stand-in for tests. Weight-
level numeracy is a separate research question, not something established by the
grammar.

The architecture does not require the LLM to perform the symbolic derivation,
but translation quality remains an empirical dependency: the model can misread
intent, and a valid contract can still be the wrong contract. No model-size
sufficiency or superiority claim is made here; compare the specified source and
any reference model on held-out phrasings and the same independent checks.

---

## 2. The specified model source and hardware status

The project-specified source is Hugging Face
`meshllm/Kimi-K3-UD-Q4_K_XL-layers`. The requested retrieval method is:

```bash
git clone https://huggingface.co/meshllm/Kimi-K3-UD-Q4_K_XL-layers
```

**Not verified in this sandbox:** I could not reach Hugging Face, so I did not
confirm that the repository exists, inspect its files or license, identify its
format, measure its size, or test inference/training. Family-level Kimi K3 specs
and the suffix `Q4_K_XL-layers` do not establish the contents or compatibility of
this exact path. Do not silently replace it with a Qwen model or claim that this
repository exists from the identifier alone. See [`HF_MANIFEST.md`](HF_MANIFEST.md)
for the explicit verification checklist.

The code can talk to an OpenAI-compatible model server. A Qwen3 GGUF command
below is a **separate integration example**, not the specified source and not a
measured recommendation. A 4-bit GGUF is ordinarily an inference artifact; the
QLoRA reference requires a compatible training base and an explicit `--model`
path. Inspect the actual clone before selecting an inference or training route.
No local fit, GPU tier, or run-time estimate is asserted for the specified path.

---

## 3. Architecture

```
  user text
      │
      ▼
┌──────────────────────────────────────────────────┐
│ 1. translator (actual model; grammar if supported)│   asked for a contract:
│    "goal: solve / var: x / expr: 2x+4 = 10"       │   goal tag + expression
└──────────────────────────────────────────────────┘
      │  strict contract validation, then parse
      ▼
┌─────────────────────────────────────────────┐
│ 2. ARIA symbolic runtime                    │   parser → rewrite rules →
│    symbolic transforms                      │   verifier → derivation + audit frame
└─────────────────────────────────────────────┘
      │  frame: audit pass, fail, or held
      ▼
┌─────────────────────────────────────────────┐
│ 3. narrator  (actual model, free text)      │   prompted with the ARIA frame
│    "x = 3, because ..."                     │   prose is not a verification step
└─────────────────────────────────────────────┘
      │
      ▼
  user text
```

Two protocol rules define the intended boundary, not a guarantee of model
behavior:

1. **The translator is asked for intent and an expression, not an answer.** A
   valid contract can still misrepresent the user's request; the verifier checks
   the symbolic task it received, not intent fidelity.
2. **The narrator is given the checked frame and asked not to add claims.** A
   prompt is not a sandbox: natural-language narration can still be wrong. The
   independently checked derivation and its audit state remain separate from the
   prose.

### 3.1 Cross-scale frame and held-state discipline

The design reference is the user's self-described advanced hyperthymnesia: a
personal analogy for considering micro, meso, and macro views simultaneously,
branching without premature collapse, preserving irreducible evidence, and
compiling checked branches. It is not a clinical premise or scientific
validation. Rust-engine idioms and bra-ket/Dirac notation are design inspirations
only; the `|1⟩`, `|0⟩`, and `|?⟩` labels in code are display aliases, not quantum
states.

`aria.llm.context` now represents:

* **macro:** task outcome, separate process-audit state, result, assumptions,
  held register, and open/rejected obligations;
* **meso:** branches/checkpoints with an explicit `all` or `any` compilation rule;
* **micro:** atomic claims, rewrite steps, verifier evidence, and provenance.

The current `MathChannel` projects its selected symbolic derivation into one
meso branch. The types can compile multiple branches, but multi-candidate search
and route selection are not yet wired into the symbolic engine. The
least-resistance helper only ranks candidates whose audit passed and whose
verifier evidence reference is present; it returns no route if none pass. It
cannot promote an unresolved or failed path because it is cheaper.

The held register changes only on a verifier event with an evidence reference.
Narration, context reduction, and context recovery do not change it. Required
context overflow holds before the model call. Optional details are omitted as
whole blocks; the full source text remains in the result object and the `--json`
manifest, from which a larger-budget pack can be rebuilt. Reduced narration is
instructed not to infer omitted details.

### 3.2 Human feedback: recognition, preference, and review

Human feedback is not the verifier and not a scalar objective. A person may
spontaneously appreciate a result, request a variation, correct a name/preference,
or ask for an inspection of logic. Such language can guide interaction when its
meaning is explicit, but it cannot establish mathematical truth. Silence is no
observation—not dislike, failure, or a missing reward. A mistake opens a repair
without erasing a held grade.

The existing text memory can store a stated preference with provenance as
`kind=preference`; it remains subjective and can be corrected with supersession
history. `aria ask` does not automatically collect or inject this memory, and no
feedback-prompt/Monte-Carlo sampling routine is implemented. Any later optional
sampling would need an easy skip and must never gate progress. A person may give
themself a local reward without system approval or external consent; the runtime
does not score or authorize that choice.

---

## 4. The contract

Three lines, fixed order:

```
goal: <solve|simplify|factor|expand|verify|differentiate|compute|analyze>
var: <single variable, only for solve and differentiate>
expr: <expression in ARIA notation>
```

Three layers of checks, with distinct limits:

| layer | mechanism | failure mode |
|---|---|---|
| grammar | GBNF constrains output if the endpoint honors the field | unsupported servers may ignore it |
| protocol | `parse_contract` rejects extra text, unknown/duplicate/misordered fields, missing values, and invalid expressions | `ContractError` with the reason; user-intent fidelity is still unverified |
| verifier | checks the symbolic steps it can decide; task outcome and process audit remain separate | `partial`/`held` when evidence is insufficient; narration is not an extra proof |

The repair loop appends the actual error and asks again:

```
user:       That output violated the contract: expression does not parse
            (unexpected end of input at position 5); received '2x +' .
            Reply with the three-line contract only, no other text.
```

```python
from aria.llm import MathChannel, OpenAICompatClient

client = OpenAICompatClient("http://127.0.0.1:8080/v1/chat/completions", model="local")
result = MathChannel(client).ask("solve for x in 2x + 4 = 10")
print(result.to_text())        # contract + symbolic trace/frame + narration
print(result.derivation.valid) # True only if every step was proven
```

---

## 5. Running it

### 5.1 Start a local server (llama.cpp)

```bash
# dump the grammar next to your model
aria grammar --mode loose --out aria-loose.gbnf

llama-server -m qwen3-4b-instruct-q4_k_m.gguf \
             --port 8080 \
             --grammar-file aria-loose.gbnf \
             -c 4096
```

`--grammar-file` applies constrained decoding server-side for every request, so
even a model that ignores instructions cannot emit prose. ARIA also sends the
grammar per-request (llama.cpp reads the `grammar` field), which is what makes it
work with servers started without the flag.

### 5.2 Talk to it

```bash
aria ask "solve for x in 2x + 4 = 10" --endpoint http://127.0.0.1:8080/v1/chat/completions
aria ask "is log(a*b) the same as log(a)+log(b)?" --grammar strict
aria --json ask "factor x^2 - 9"
```

### 5.3 No model? Still prove the loop works

```bash
aria ask "solve for x in 2x + 4 = 10" --dry-run
```

`--dry-run` swaps the model for ARIA's own instruction parser. It demonstrates the
exact contract a real model must produce, and it is what the test-suite uses so
that CI never needs weights.

### 5.4 Other servers

Any OpenAI-compatible endpoint works — LM Studio (`http://127.0.0.1:1234/v1/chat/completions`),
Ollama (`http://127.0.0.1:11434/v1/chat/completions`), vLLM. Servers that do not
support the `grammar` field ignore it; the protocol layer still validates.

---

## 6. What the grammar guarantees — and what it does not

**Guarantees**

* the protocol accepts only the ordered contract fields, rejects extra prose,
  and validates the goal, variable, and expression syntax;
* when the selected server enforces the supplied GBNF, the emitted alphabet is
  constrained to the grammar;
* with an enforced `--mode strict` grammar, expressions in its supported subset
  are parseable by construction.

**Does not guarantee**

* that the expression is the *right* one (the model can misread the user);
* that the mathematics is deep enough for the question (ARIA reports `partial`);
* freedom from nonsense variables (`x`, `y` only).

The strict grammar trades fluency for parseability; on small models it often
produces grammatically valid but awkward expressions. Start loose, tighten only if
your model drifts.

---

## 7. What you cannot get without retraining

1. **A quantised GGUF is an inference artifact, not a QLoRA training base.**
   The specified `meshllm/Kimi-K3-UD-Q4_K_XL-layers` path has not been fetched
   or inspected here; do not infer its format, size, or trainability from its
   name. The QLoRA reference script needs a compatible model directory and GPU.
2. **Weight-level numeracy beyond contract translation** would require research
   and evaluation on formal corpora (Lean/mathlib, Isabelle, symbolic derivation
   traces) or another explicit training design. Hardware and memory requirements
   depend on the actual base model and must be measured, not asserted from a
   model label.
3. **A tokenizer change affects the embedding interface** and may require
   substantial retraining. If the goal is "mathematical tokens instead of word
   tokens", validate that intervention against an unchanged-tokenizer baseline.

---

## 8. Deeper integration, in order of value per unit of effort

| step | effect | status / effort |
|---|---|---|
| grammar + repair loop | constrained contract translation with protocol validation | implemented; does not validate user-intent fidelity |
| derivation-conditioned prompting | show the checked frame to the output narrator | implemented in the channel; the narration model can still be wrong |
| ledger-backed SFT after a school rung is held | change adapter/weights using distinct demonstrated processes | reference path implemented; actual training and held-out gains not yet established |
| continued pretraining on formal corpora with a math-aware tokenizer | teach a model broader formal structure | research programme |
| symbolic/rule-conditioned parameter experiments | test literal changes to what weights represent | research programme; not established by this repository |

Scalar-reward GRPO is retained only as explicitly opted-in legacy scaffolding.
It is not the recommended or accepted school path: scalar maximisation is not
ledger advancement, even when the scalar is binary. ARIA's training data and
school decisions preserve the record of a process instead of compressing it to
a score.

---

## 9. Privacy, determinism, and the proposal's wording

* A localhost endpoint keeps model traffic on the same machine **when the caller
  actually selects a local server**. `--endpoint` can also name a remote service;
  this client does not guarantee that nothing leaves the device.
* ARIA's symbolic engine is deterministic for a fixed request and configuration.
  The translation model may be stochastic or mistranslate the user's intent;
  narration may vary or overstate. The checked derivation, not prose, is the
  authority for the symbolic task outcome.
* The proposal's wording is preserved verbatim: **"translates back into text by
  doing its cryptographic work."** It is a load-bearing design statement, not a
  synonym to silently replace with "decode." The implementation in this
  repository does not yet define or demonstrate the claimed cryptographic work.
  Current text generation uses the selected model's inference path; no
  cryptographic reasoning mechanism is implemented or claimed here. This is an
  open specification/engineering requirement, not evidence of a scientific or
  cryptographic result.

---

## 10. Failure modes

| symptom | cause | fix |
|---|---|---|
| `contract rejected after N attempts` | model cannot follow the format | use `--grammar strict`, or a larger model |
| derivation status `partial` | ARIA cannot prove the claim (e.g. transcendentals without SymPy) | install SymPy; or accept the honest `partial` |
| wrong expression, valid contract | translation error | tighten the prompt; add few-shot examples |
| `connection refused` | server not running / wrong port | check `llama-server` output; try `/v1/models` |
| narration contradicts the trace | model ignoring its context | ARIA's derivation is authoritative; keep `--json` output |
