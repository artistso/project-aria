# What to download, and where to put it

A short, concrete list — sized to the hardware that was actually measured (11.72 GB RAM,
~2 GB VRAM, Mali-G720 GPU with no CUDA) and to what ARIA needs a model to do: **emit a
three-line contract, not solve anything.**

Three rules decide everything below.

1. **The tablet infers. A rented GPU trains.** No mobile GPU trains a transformer; the
   Mali chip is not reachable from llama.cpp either, so tablet inference is CPU-only.
2. **A 4-bit GGUF cannot be trained.** QLoRA starts from full-precision weights and
   quantises on the fly. GGUFs are for running, `safetensors` are for training.
3. **`git clone` of a model repo downloads LFS pointers**, ~130-byte text files that look
   like weights. Use the downloader, and check the result with `aria model inspect`.

---

## 1. The list

### Tier A — run on the tablet (llama.cpp, one GGUF each)

| purpose | repo | file to take | ≈ size |
|---|---|---|---|
| **translate** (the default) | `Qwen/Qwen3-1.7B-GGUF` (or `unsloth/Qwen3-1.7B-GGUF`) | `Qwen3-1.7B-Q4_K_M.gguf` | 1.1 GB |
| better translation, still fits | `Qwen/Qwen3-4B-GGUF` | `Qwen3-4B-Q4_K_M.gguf` | 2.5 GB |
| smoke tests / low memory | `Qwen/Qwen3-0.6B-GGUF` | `Qwen3-0.6B-Q4_K_M.gguf` | 0.4 GB |
| fallback family | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.0 GB |

Take **one** of the first two. File names follow the standard pattern; if a repo names
them differently there is still exactly one `Q4_K_M` file — take that one.

Note on Qwen3: it defaults to "thinking" mode. A translator must not spend 400 tokens
reasoning before writing three lines, so run it with thinking disabled
(`enable_thinking: false` in the template, or `/no_think`).

### Tier B — train (rented GPU: one 16–24 GB card)

| purpose | repo | files to take | ≈ size |
|---|---|---|---|
| QLoRA SFT + GRPO base | `Qwen/Qwen3-1.7B` (**not** GGUF) | `config.json`, `generation_config.json`, `model-*.safetensors` (all shards), `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt` | 3.4 GB |
| faster 4-bit load (optional) | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` | the same set | ~1.2 GB |
| 4B tier, if you go bigger | `Qwen/Qwen3-4B` | the same set | ~8 GB |

**All shards or nothing.** A `model-00002-of-00003.safetensors` without `00001` and
`00003` will not load, and the error arrives four minutes into a paid run.

### Tier C — only if memory retrieval proves too blunt

| purpose | repo | file | ≈ size |
|---|---|---|---|
| semantic search over the memory file | `nomic-ai/nomic-embed-text-v1.5-GGUF` | `nomic-embed-text-v1.5.Q4_K_M.gguf` (or F16) | 0.09–0.27 GB |

Not needed yet. Lexical retrieval (`aria memory find`) is explainable and runs anywhere;
add this only if paraphrases are actually being missed.

### What not to download

* **Kimi K3 in any form.** 2.8 T parameters, ~1.4 TB, 64+ accelerators. It is a teacher to
  distil *from*, never a file to fetch here.
* **Any GGUF for training.** See rule 2.
* **70B / 32B quantised "just in case".** It will not fit in 11.7 GB with the OS running.
* **Vision/mmproj files.** Nothing in ARIA reads images yet.

---

## 2. Where to put it

| what | destination | why |
|---|---|---|
| GGUF files (Tier A) | **Google Drive**, folder `ARIA/models/` | too big for GitHub; I can list Drive and read sizes directly |
| full-precision shards (Tier B) | **the training machine**, not Drive | 3.4 GB round-trips are wasted time; upload only if you want me to check them |
| manifests, `*.jsonl`, docs, memory, ledger | **GitHub** (`artistso/project-aria`) | small text, versioned, works with `git` |
| things you want me to read and cite | **Drive or Docs** | the connector reads both |

Suggested Drive layout:

```
ARIA/
  models/     Qwen3-1.7B-Q4_K_M.gguf
  data/       ledger.jsonl, memory.jsonl, sft.jsonl, rl.jsonl
  docs/       aria-context-pack.md, aria-report-card.md
```

GitHub limits matter: a single file over 100 MB is rejected, and the LFS workaround puts
**pointers** in the clone — the exact trap that produced the `gitattributes.txt` earlier in
this project. So: weights in Drive, text in Git, and never the two mixed.

---

## 3. How I check what you send

1. **Size, from Drive metadata.** An LFS pointer is ~130 bytes; a real 1.7B Q4 file is
   ~1.1 GB. This alone catches the common failure.
2. **Header, when a local copy exists.** `aria model inspect <file>` parses the GGUF
   header with no GPU and no runtime and prints the parameter count, tensor count,
   dominant quantisation, architecture and context length:

   ```
   $ aria model inspect Qwen3-1.7B-Q4_K_M.gguf
   Qwen3-1.7B  (qwen3)
     parameters     1.72B  (1,720,000,000)
     tensors        290
     quantisation   q4_k
     context        40960
   ```

   If it says "this is a Git LFS pointer, not a model", the download was a clone.
3. **Shard completeness, for Tier B.** The count of `model-*.safetensors` must equal the
   count in `model.safetensors.index.json`.

I cannot reach Hugging Face from this sandbox (the TLS handshake is blocked), so
**anything that must come from HF has to be fetched by you** — Drive is the reliable route
for me to see it.

---

## 4. What the run will cost, roughly

| stage | hardware | wall clock |
|---|---|---|
| SFT on the ledger's proven work (a few hundred pairs, 2 epochs) | 1× 16 GB (T4/A4000 class) | 20–40 min |
| GRPO with `--signal pass` (46 prompts, 8 generations) | 1× 24 GB (A10/L4) | 1–3 h |
| evaluation (`eval_contract.py` against the endpoint) | any | minutes |

Iterate at 1.7B. Move to 4B only once the 1.7B translator holds a contract-validity rate
above ~95% on `aria bench`.

---

## 5. The decision I need from you

Pick the **runtime tier** — 1.7B (safe, ~1.1 GB, comfortable on the tablet) or 4B (better
translation, ~2.5 GB, tighter but still fine) — and whether you want to train on a rented
GPU at all this round. Everything else on this page follows from those two answers.
