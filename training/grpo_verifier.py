"""Legacy scalar-reward GRPO experiment; opt-in only.

This script is not ARIA's accepted training curriculum. The accepted route is
ledger-recorded verification, staged school advancement, then SFT from held
rungs (``aria ledger sft``). GRPO is retained only for explicit experiments
because even a binary verifier signal is a scalar objective, and the shaped
version is known to reward a broken process.

To run the experiment intentionally, pass
``--experimental-allow-scalar-reward``. The default is to refuse training.

    python training/grpo_verifier.py --model out/qwen3-1.7b-aria-sft \
        --data data/rl.jsonl --out out/qwen3-1.7b-aria-grpo \
        --signal pass --experimental-allow-scalar-reward
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `aria` importable when this script runs from a plain copy of the tree
# (no `pip install -e .` needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legacy scalar-reward GRPO experiment (training requires explicit opt-in)"
    )
    parser.add_argument("--model", default="out/aria-sft")
    parser.add_argument("--data", default="data/rl.jsonl")
    parser.add_argument("--out", default="out/aria-grpo")
    parser.add_argument("--generations", type=int, default=8, help="group size for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--signal",
        choices=("pass", "shaped"),
        default="pass",
        help="legacy GRPO comparison signal: binary pass/fail or the shaped diagnostic scale",
    )
    parser.add_argument(
        "--experimental-allow-scalar-reward",
        action="store_true",
        help="explicitly opt in to this legacy scalar-objective experiment",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.dry_run and not args.experimental_allow_scalar_reward:
        parser.error(
            "scalar-reward training is not ARIA's curriculum; use ledger-backed SFT, "
            "or pass --experimental-allow-scalar-reward to run this legacy experiment"
        )

    import json

    from aria.llm.dataset import reward as aria_reward
    from aria.llm.grammar import build_grammar
    from aria.llm.ledger import judge

    def pass_signal(text: str) -> float:
        return 1.0 if judge(text).demonstrated else 0.0

    if args.dry_run:
        records = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
        good = "goal: solve\nvar: x\nexpr: 2x + 4 = 10"
        broken = "goal: solve\nvar: x\nexpr: 9/x - 3 = 24"
        print(f"rl prompts: {len(records)}")
        print(f"signal: {args.signal}")
        print("  perfect contract   pass=%.1f shaped=%.2f" % (pass_signal(good), aria_reward(good)))
        print("  wrong process      pass=%.1f shaped=%.2f" % (pass_signal(broken), aria_reward(broken)))
        print("sample prompt:", records[0]["prompt"] if records else "<empty>")
        return 0

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"rl prompts: {len(dataset)}")

    score = pass_signal if args.signal == "pass" else aria_reward

    def verifier_reward(completions: List[str], **_: object) -> List[float]:
        """TRL reward hook: did the sampled completion prove its own process?"""
        return [score(text) for text in completions]

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
