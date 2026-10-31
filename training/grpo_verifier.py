"""GRPO with the ARIA verifier as the reward function.

The reward is not a preference model: it parses the completion, runs the symbolic
engine, and scores how much of the derivation was proven. See
``aria/llm/dataset.py::score_completion`` for the exact scale.

    python training/grpo_verifier.py --model out/qwen3-1.7b-aria-sft \
        --data data/rl.jsonl --out out/qwen3-1.7b-aria-grpo
"""

from __future__ import annotations

import argparse
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO with a verified-mathematics reward")
    parser.add_argument("--model", default="out/aria-sft")
    parser.add_argument("--data", default="data/rl.jsonl")
    parser.add_argument("--out", default="out/aria-grpo")
    parser.add_argument("--generations", type=int, default=8, help="group size for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    import json

    from aria.llm.dataset import reward as aria_reward
    from aria.llm.grammar import build_grammar

    if args.dry_run:
        records = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
        print(f"rl prompts: {len(records)}")
        print("reward of a perfect contract:", aria_reward("goal: solve\nvar: x\nexpr: 2x + 4 = 10"))
        print("reward of an invalid contract:", aria_reward("goal: solve\nexpr: 2x +"))
        print("sample prompt:", records[0]["prompt"] if records else "<empty>")
        return 0

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"rl prompts: {len(dataset)}")

    def verifier_reward(completions: List[str], **_: object) -> List[float]:
        """TRL reward hook: map each sampled completion to [0, 1]."""
        return [aria_reward(text) for text in completions]

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.generations,
        gradient_accumulation_steps=1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[verifier_reward],
    )
    # The grammar is enforced at inference time by the server; during training the
    # verifier reward is what teaches the model to stay inside it.
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved to {args.out}; grammar used at inference:\n{build_grammar('loose')[:200]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
