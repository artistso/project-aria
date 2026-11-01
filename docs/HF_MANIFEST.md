# Model source and weight-file manifest

## Specified ARIA source

The project-specified model source is Hugging Face
`meshllm/Kimi-K3-UD-Q4_K_XL-layers`. The requested retrieval method is `git clone`:

```bash
git clone https://huggingface.co/meshllm/Kimi-K3-UD-Q4_K_XL-layers
```

**Unverified:** Hugging Face was unreachable from this sandbox. I did not verify
that this repository exists, inspect its files or license, confirm its size or
format, or run it. The identifier and command preserve the requested source; they
are not a claim that the repository exists. Do not substitute another model
silently.

If the clone succeeds, inspect the resulting directory before using it. A Git
LFS-managed repository may initially contain pointer files; a pointer is not a
weight shard. If LFS pointers are present and Git LFS is available, fetch the
large-file contents from the cloned repository, then check file sizes, the model
configuration, tokenizer, shard index, and license. The repository name alone
does not establish whether it is a GGUF, a trainable Hugging Face checkpoint,
or an inference-only artifact.

```bash
cd Kimi-K3-UD-Q4_K_XL-layers
# Only if the checked-out files are Git LFS pointers:
git lfs pull
```

Do not commit model weights to this repository. Use external storage for large
artifacts and keep a text manifest (source revision, file sizes, hashes, license
and training/inference compatibility) with the experiment.

## Separate reference models

The repository has used Qwen3 models as small **reference experiments** for the
LLM contract interface and for exercising the QLoRA script. Those examples do
not replace the specified Kimi source. A Qwen base can be used only when the
experiment is explicitly labelled as such.

| use | example | required distinction |
|---|---|---|
| local contract-translation smoke test | a compatible Qwen GGUF served by llama.cpp | inference example, not the specified ARIA source |
| SFT / QLoRA code-path reference | `Qwen/Qwen3-1.7B` or another compatible full model | separate base; requires explicit `--model` and a non-empty held-rung SFT export |

A quantised GGUF is an inference artifact, not the input expected by the current
QLoRA reference. Training updates an adapter/parameters only when run against a
compatible base with the required dependencies and hardware. No SFT run or
weight update is claimed by this manifest.

## Inspect and verify before use

* For a GGUF file, `aria model inspect <file>` reads the local header and can
  identify an LFS pointer instead of weights. This tool does not inspect an HF
  checkpoint directory.
* For a Hugging Face model directory, check the model configuration and
  tokenizer, ensure every shard named in the index exists and has a plausible
  size, and record the repository revision and license. Do not infer details
  from a model name.
* The current `training/sft_qlora.py` requires `--model` explicitly and accepts
  only a non-empty JSONL dataset with `messages` records. For the default school
  path, produce that dataset using `aria ledger sft`; it exports demonstrated
  processes only from rungs already held by the staged school.
* `aria ask --context-budget-chars N` bounds prompt context by character count
  unless a tokenizer-based measure is supplied by an embedding application.
  Model context-window limits are distinct and should be checked for the actual
  serving model.

No model size, local device fit, GPU tier, or run time is asserted here for the
specified Kimi path. Those measurements require the actual fetched files and a
working runtime. See [`LLM_INTEGRATION.md`](LLM_INTEGRATION.md),
[`WEIGHT_REWRITING.md`](WEIGHT_REWRITING.md), and [`../training/README.md`](../training/README.md)
for the current channel and training status.
