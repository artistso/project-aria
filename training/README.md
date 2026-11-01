# Training ARIA's language edge without replacing its reasoning core

These scripts change model weights; ARIA's symbolic engine remains the authority
for mathematics. The accepted training curriculum is **ledger + staged school
advancement**, not scalar reward maximisation. A rung becomes eligible for SFT
only after the ledger holds it through three clean, distinct problems.

> **Specified model source:** Hugging Face `meshllm/Kimi-K3-UD-Q4_K_XL-layers`.
> The requested retrieval route is `git clone https://huggingface.co/meshllm/Kimi-K3-UD-Q4_K_XL-layers`. This sandbox cannot
> reach Hugging Face, so repository existence, files, license, and runtime were
> not verified here. Do not infer that the repo exists from this name or from
> these instructions. A clone can still contain LFS pointer files; inspect what
> was actually fetched before treating it as weights. The small Qwen example
> below is a trainable reference experiment, not a silent replacement for the
> specified source.

| file | purpose | hardware |
|---|---|---|
| `../aria/llm/ledger.py` | verified process record, school state, staged SFT export | none |
| `sft_qlora.py` | QLoRA SFT on ledger-exported `messages` | CUDA GPU; depends on base-model size |
| `grpo_verifier.py` | legacy scalar-reward experiment; disabled unless explicitly opted in | unbenchmarked; outside the accepted path |
| `eval_contract.py` | measure contract validity against a live endpoint; any machine | none |

## 1. Work the school first

```bash
aria ledger next
aria ledger record --question "solve 9x - 3 = 24" \
    --completion "goal: solve\nvar: x\nexpr: 9x - 3 = 24"
aria ledger report
```

The ledger records the question, contract, engine outcome, and each verified
step. It does not calculate a scalar training reward. Three clean, distinct
proofs hold a rung; failures open repair, and undecided work stays blocked/held
rather than becoming a negative grade.

## 2. Export only held stages

```bash
aria ledger sft --out data/grade-04.jsonl --max-level 4
# Only after the specified clone is verified to contain a compatible training base:
python training/sft_qlora.py \
    --model ./Kimi-K3-UD-Q4_K_XL-layers \
    --data data/grade-04.jsonl \
    --out out/aria-grade-04
```

The model path is conditional, not a claim that the clone exists or is trainable.
For a separate code-path smoke test, pass an explicit compatible reference model
such as `Qwen/Qwen3-1.7B`; that is not a replacement for the specified source.

`aria ledger sft` defaults to held rungs only. `--max-level` caps the stage;
`--include-learning` is an explicit bootstrap escape hatch for proven examples
from rungs that are not held yet. A failed example is never an SFT target. If a
later attempt corrects the same expression, only the corrected contract is
exported; the failed attempt remains in the ledger as inspectable history.

The Qwen command is a small reference for a first trainable experiment. It does
not establish that Kimi K3 is locally trainable or runnable. QLoRA needs a
compatible full-precision base (or a supported training quantisation); a
Q4_K_XL GGUF is an inference artifact, not a fine-tuning input.

## 3. Evaluate the changed weights

```bash
python training/eval_contract.py \
    --endpoint http://127.0.0.1:8080/v1/chat/completions \
    --grammar loose
aria bench --json-out benchmarks/post-train.json
```

Compare the base and adapter on held-out **phrasings**, not only the curriculum
examples. Report contract validity, repair count, task correctness, invalid
steps, and unresolved/held results separately. A ledger rung is a curriculum
state, not evidence of general capability.

## 4. Scalar-reward path: legacy, not the school

`aria reward`, `aria dataset rl`, and `training/grpo_verifier.py` are retained as
historical diagnostics and a research-only experiment. They do not define
advancement or the accepted weight-update path. GRPO training now requires the
explicit `--experimental-allow-scalar-reward` acknowledgement. The old shaped
score is known to give `0.90` to a process with an unchecked step; its reported
value happens to be right, but the proof process is not established. That is why
a scalar is not the organizing principle.

Static `aria dataset sft` remains useful for a format smoke test, but staged
learning should use `aria ledger sft` so training targets are attached to
verified, held school rungs.

## Notes

* The training scripts are reference implementations; CI syntax-checks them but
  does not execute CUDA training.
* 1.7B vs 4B and GPU rental remain deployment choices, not claims made by this
  training guide. Measure the translation baseline and memory use before
  renting hardware.
