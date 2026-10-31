# Training a translator for the math channel

These scripts change weights — the one thing the tablet cannot do. Everything here
needs a CUDA GPU; nothing in the ARIA library does.

The goal is narrow and deliberate: **teach a small model to translate a question
into ARIA's three-line contract**, and then reward it with the verifier. You are
not training it to do mathematics. You are training it to *speak* mathematics so
the engine can.

| file | purpose | hardware |
|---|---|---|
| `../aria/llm/dataset.py` | miner: SFT pairs, RL prompts, verifier reward | none |
| `sft_qlora.py` | QLoRA supervised fine-tune on `data/sft.jsonl` | 1× 16–24 GB |
| `grpo_verifier.py` | GRPO with `aria.llm.dataset:reward` | 1× 24–80 GB |
| `eval_contract.py` | measure contract validity / reward against a live endpoint | GPU not required |

## 1. Generate data (no GPU)

```bash
aria dataset sft --out data/sft.jsonl --paraphrases 2
aria dataset rl  --out data/rl.jsonl
wc -l data/sft.jsonl data/rl.jsonl
```

`sft.jsonl` is `{"messages": [system, user, assistant]}` — the assistant turn is
always a contract (`goal:` / `var:` / `expr:`), never an answer. `rl.jsonl` is
prompts only, tagged with the reward import path.

## 2. Supervised fine-tune

```bash
pip install "transformers>=4.44" "peft>=0.12" "trl>=0.12" datasets accelerate bitsandbytes
python training/sft_qlora.py \
    --model Qwen/Qwen3-1.7B \
    --data data/sft.jsonl \
    --out out/qwen3-1.7b-aria-sft
```

Target: contract validity ≥ 98% on held-out phrasings, repairs < 0.2 per query.

## 3. Verifier-reward RL (GRPO)

```bash
python training/grpo_verifier.py \
    --model out/qwen3-1.7b-aria-sft \
    --data data/rl.jsonl \
    --out out/qwen3-1.7b-aria-grpo
```

The reward function *is* the verifier (`aria.llm.dataset:reward`). A completion
scores 0 unless ARIA can parse it, and 1.0 only if every derivation step is
proven. Reward hacking here means "the model learned to ask provable questions",
which is the goal.

## 4. Evaluate

```bash
# contract validity + mean reward against a running llama.cpp server
python training/eval_contract.py \
    --endpoint http://127.0.0.1:8080/v1/chat/completions \
    --grammar aria-loose.gbnf

# and the mathematics benchmark, before vs after
aria bench --json-out benchmarks/post-train.json
```

## Notes

* Iterate at 1.7B. Move to 4B only when the small model saturates; GRPO on 8B+
  is memory-hungry because of the group sampling.
* Keep a held-out split of *phrasings*, not just tasks — otherwise you measure
  memorisation.
* If the reward climbs while `aria bench` accuracy does not, the model has learned
  to pick easy goals. Add goal-diversity to the RL prompt set and re-run.
* These two GPU scripts are reference implementations; they are syntax-checked in
  CI but not executed there (no GPU).


## Which signal?

`grpo_verifier.py` defaults to `--signal pass`: 1.0 for a *demonstrated* process (contract
parsed, every engine step verified, run decided), 0.0 otherwise. There is no partial credit,
so there is nothing to farm. `--signal shaped` keeps the earlier 0.0-1.0 scale for
continuity; its dry run prints both:

```
python training/grpo_verifier.py --dry-run --data data/rl.jsonl
  signal: pass
    perfect contract   pass=1.0 shaped=1.00
    wrong process      pass=0.0 shaped=0.90
```

The second line is why `pass` is the default. See `docs/LEDGER.md`.
