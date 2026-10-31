# The Math Channel — running a local model with ARIA

This document is about the specific architecture: *text in → mathematics as the
only intermediate language → text out*, with a local model at both boundaries and
ARIA in the middle.

---

## 1. "Thinking in pure mathematics" — three different things

The phrase covers three very different engineering problems. Only one of them is
reachable on a tablet.

| Tier | What it means | What it takes | Reachable today? |
|---|---|---|---|
| **A. Reasoning substrate** | the *state* the system manipulates is formal, not prose | a symbolic engine + verifier + a protocol that forbids the model from answering | **yes** — this is `aria/llm/` |
| **B. Output alphabet** | the model may only *emit* mathematical tokens | constrained decoding with a grammar (GBNF) | **yes** — `aria/llm/grammar.py` |
| **C. Weight-level numeracy** | the *residual stream* encodes symbolic structure because the model was trained on formal corpora | continued pretraining with full-precision weights, a formal corpus, and serious compute | **no** — research scale |

ARIA implements A and B, and documents C without pretending it is done.

The important consequence: **you do not need a 2.8-trillion-parameter model to get
mathematics as the logic of a system.** You need a small model that can reliably
translate a sentence into a contract, and a runtime that is actually mathematical.
A 3–8B model plus ARIA is more mathematically reliable than a frontier model
answering unaided, because the frontier model is still guessing while ARIA is
not.

---

## 2. Hardware reality

The model this project was pointed at is Kimi K3:

| property | value |
|---|---|
| parameters | 2.8 T total, ~104 B activated (MoE, 16 of 896 experts) |
| layers | 93 |
| context | 1,048,576 tokens |
| released | 16 July 2026 (weights 27 July 2026) |
| 4-bit weight footprint | **~1.4 TB** before runtime overhead |
| Moonshot's deployment guidance | supernode with **64+ accelerators** |

That does not run on a Galaxy Tab S10+ (12–16 GB RAM), a laptop, or a single
workstation GPU. It is a datacentre model.

What actually runs locally, with llama.cpp and 4-bit weights:

| model class | Q4 file size | tablet feasibility |
|---|---|---|
| 1–2 B (Llama 3.2 1B/3B, Qwen3 1.7B) | 0.8–2 GB | comfortable |
| 3–4 B (Qwen3 4B, Phi-4-mini) | 2.5–3 GB | good |
| 7–8 B (Qwen3 8B, Llama 3.1 8B) | 4.5–5.5 GB | workable, slower |
| 14–32 B | 9–20 GB | not on 12 GB |

For the math channel, **translation quality is the only thing that matters**, and
that is a small-model task: turn one sentence into three lines. A 4B model does
this well. If it fails, ARIA's repair loop feeds back the parse error and it tries
again.

> **Practical note on downloading weights:** don't `git clone` a GGUF repository.
> The files are stored in Git LFS and a clone pulls pointers or gigabytes of
> history. Use `hf download <repo> --include "*.gguf"` (or `huggingface-cli
> download`) with a pattern, and check the model card's shard list — split repos
> (names containing `-layers`) usually require every shard to be present before
> loading.

---

## 3. Architecture

```
  user text
      │
      ▼
┌─────────────────────────────────────────────┐
│ 1. translator  (local model, grammar-bound) │   may emit ONLY:
│    "goal: solve / var: x / expr: 2x+4 = 10" │   a goal tag + an expression
└─────────────────────────────────────────────┘
      │  contract (validated, then parsed)
      ▼
┌─────────────────────────────────────────────┐
│ 2. ARIA symbolic runtime                    │   parser → rewrite rules →
│    exact arithmetic, no sampling            │   verifier → derivation
└─────────────────────────────────────────────┘
      │  verified trace (every step proven)
      ▼
┌─────────────────────────────────────────────┐
│ 3. narrator  (local model, free text)       │   may only describe the trace
│    "x = 3, because ..."                     │   it cannot add or recompute
└─────────────────────────────────────────────┘
      │
      ▼
  user text
```

Two rules make this work:

1. **The model never states a mathematical fact.** It states an *intent* and an
   *expression*. Everything else is computed and checked by ARIA.
2. **The narrator is sandboxed by the trace.** Its prompt contains the verified
   steps and an instruction not to add claims; if it drifts, the derivation is
   still correct and printed alongside.

---

## 4. The contract

Three lines, fixed order:

```
goal: <solve|simplify|factor|expand|verify|differentiate|compute|analyze>
var: <single variable, only for solve and differentiate>
expr: <expression in ARIA notation>
```

Enforced three times over:

| layer | mechanism | failure mode |
|---|---|---|
| grammar | GBNF makes malformed output unrepresentable | model cannot emit it |
| protocol | `parse_contract` re-validates and parses the expression | `ContractError` with the reason |
| verifier | every mathematical step is proven or reported `unknown` | `partial` status, never a false claim |

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
print(result.to_text())        # contract + verified trace + explanation
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

* output is exactly the three-line contract with a legal goal tag;
* expression characters are limited to mathematics (no prose can leak through);
* with `--mode strict`, the expression trees the model can emit are parseable by
  construction.

**Does not guarantee**

* that the expression is the *right* one (the model can misread the user);
* that the mathematics is deep enough for the question (ARIA reports `partial`);
* freedom from nonsense variables (`x`, `y` only).

The strict grammar trades fluency for parseability; on small models it often
produces grammatically valid but awkward expressions. Start loose, tighten only if
your model drifts.

---

## 7. What you cannot get without retraining

1. **A quantised GGUF cannot be trained.** It is an inference artifact. LoRA
   adapters exist for llama.cpp, but they are trained separately on full-precision
   weights and merged; you cannot fine-tune a Q4 file into a math-native model.
2. **Weight-level numeracy requires continued pretraining** on formal corpora
   (Lean/mathlib, Isabelle, symbolic derivation traces) at full precision, on
   hardware in the same class as the model itself. For a 3B model on a workstation
   that is a project; for Kimi K3 it is a datacentre.
3. **A tokenizer change invalidates embeddings** and forces full pretraining. If
   the goal is "mathematical tokens instead of word tokens", that is the price.

This is why tier C stays on the roadmap while tiers A and B ship today.

---

## 8. Deeper integration, in order of value per unit of effort

| step | effect | effort |
|---|---|---|
| grammar + repair loop (done) | model speaks only mathematics | hours |
| derivation-conditioned prompting: feed the verified trace back as few-shot examples | model learns ARIA's notation and failure modes | days |
| LoRA fine-tune a 3–8B model on `(question → contract)` pairs | higher translation accuracy, fewer repairs | days–weeks on one GPU |
| verifier-in-the-loop RL: reward = fraction of steps `proven` | model learns to propose *provable* goals | research project |
| continued pretraining on formal corpora with a math-aware tokenizer | tier C | research programme |

The AI-native version of the plan is the fourth row: ARIA already emits a
per-step reward signal (`Derivation.trace_accuracy`, `invalid_derivation_rate`),
which is exactly what a policy-gradient loop needs.

---

## 9. Privacy and determinism

* Nothing leaves the device: the client speaks to `127.0.0.1`, stdlib only.
* Decoding is temperature-0 with a fixed grammar, and ARIA is deterministic, so
  the same question yields the same derivation. The narration may vary; the
  mathematics may not.
* No cryptography is involved in either direction. Decoding is sampling from a
  probability distribution — there is no encryption or decryption step in an
  autoregressive model. Reproducibility comes from determinism, and correctness
  comes from verification.

---

## 10. Failure modes

| symptom | cause | fix |
|---|---|---|
| `contract rejected after N attempts` | model cannot follow the format | use `--grammar strict`, or a larger model |
| derivation status `partial` | ARIA cannot prove the claim (e.g. transcendentals without SymPy) | install SymPy; or accept the honest `partial` |
| wrong expression, valid contract | translation error | tighten the prompt; add few-shot examples |
| `connection refused` | server not running / wrong port | check `llama-server` output; try `/v1/models` |
| narration contradicts the trace | model ignoring its context | ARIA's derivation is authoritative; keep `--json` output |
