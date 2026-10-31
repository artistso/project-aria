"""GRPO with the ARIA verifier -- binary first, shaped only if you must.

Two signals are available:

``--signal pass`` (default)
    ``1.0`` when the completion is a *demonstrated* process -- the contract
    parsed, every step the engine took was verified, and the run reached a
    decision -- otherwise ``0.0``. This is the ledger's rule
    (``aria/llm/ledger.py``) and it is the one to use: there is no partial
    credit, so there is nothing to farm.

``--signal shaped``
    The 0.0-1.0 scale in ``aria/llm/dataset.py::score_completion``. It is kept
    for continuity, and it carries a known defect: a wrong rational solve
    (``9/x - 3 = 24``) earns 0.90 because two of its three steps verified. The
    ledger refuses it. Prefer ``pass``.

    python training/grpo_verifier.py --model out/qwen3-1.7b-aria-sft \
        --data data/rl.jsonl --out out/qwen3-1.7b-aria-grpo --signal pass
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
    parser.add_argument(
        "--signal",
        choices=("pass", "shaped"),
        default="pass",
        help="pass = binary demonstrated/not (recommended); shaped = legacy 0..1 scale",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

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
