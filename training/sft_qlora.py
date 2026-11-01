"""QLoRA supervised fine-tune of a translator for ARIA's contract.

Reference implementation. Requires transformers / peft / trl / bitsandbytes and a
CUDA GPU; it is syntax-checked in CI but never executed there. The model must be
selected explicitly; the Qwen command below is a separate smoke-test base, not a
substitute for the specified Kimi source.

    python training/sft_qlora.py --model Qwen/Qwen3-1.7B \
        --data data/grade-04.jsonl --out out/qwen3-1.7b-aria-grade-04
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `aria` importable when this script runs from a plain copy of the tree
# (no `pip install -e .` needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for the ARIA contract")
    parser.add_argument(
        "--model",
        required=True,
        help="explicit local model directory or Hub ID; no unverified base is selected by default",
    )
    parser.add_argument("--data", default="data/sft.jsonl")
    parser.add_argument("--out", default="out/aria-sft")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="validate config and data only")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.dry_run:
        # Validate the data with the standard library only: no torch required.
        import json
        from collections import Counter

        rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
        if not rows:
            raise ValueError("refusing an empty SFT dataset; hold a school rung before export")
        for index, row in enumerate(rows, start=1):
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"SFT row {index} needs a non-empty `messages` list")
            if any(not isinstance(message, dict) or "role" not in message or "content" not in message for message in messages):
                raise ValueError(f"SFT row {index} messages need `role` and `content` fields")
        roles = Counter(
            tuple(message["role"] for message in row["messages"]) for row in rows
        )
        print(f"training examples: {len(rows)}")
        print("message shapes:", dict(roles))
        print("first target:", rows[0]["messages"][-1]["content"] if rows else "<empty>")
        return 0

    # Imported lazily so the dry run works on a machine with no GPU stack.
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")
    if not len(dataset):
        raise ValueError("refusing an empty SFT dataset; hold a school rung before export")
    if "messages" not in dataset.column_names:
        raise ValueError("SFT data must contain a `messages` column (use `aria ledger sft`)")
    extra_columns = [column for column in dataset.column_names if column != "messages"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)
    print(f"training examples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantisation, device_map="auto", trust_remote_code=True
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=args.max_seq_len,
        packing=False,
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
