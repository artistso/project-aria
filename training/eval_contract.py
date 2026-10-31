"""Measure a live model's contract quality with ARIA's verifier.

Works against any OpenAI-compatible server (llama.cpp, LM Studio, Ollama) and
needs no GPU on the machine running the evaluation.

    python training/eval_contract.py --endpoint http://127.0.0.1:8080/v1/chat/completions
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.bench.tasks import TASKS, tasks_for  # noqa: E402
from aria.llm.client import ClientError, OpenAICompatClient  # noqa: E402
from aria.llm.dataset import reward, score_completion  # noqa: E402
from aria.llm.grammar import build_grammar  # noqa: E402
from aria.llm.orchestrator import MathChannel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate contract quality of a local model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="local")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--grammar", choices=("loose", "strict", "off"), default="loose")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = OpenAICompatClient(endpoint=args.endpoint, model=args.model, timeout=120)
    grammar_mode = None if args.grammar == "off" else args.grammar
    channel = MathChannel(client, grammar_mode=grammar_mode, narrate=False)

    tasks = [t for t in tasks_for(args.suite) if t.suite != "units"]
    if args.limit:
        tasks = tasks[: args.limit]

    valid = 0
    solved = 0
    repairs = 0
    rewards = []
    rows = []

    for task in tasks:
        try:
            result = channel.ask(task.prompt)
        except ClientError as error:  # pragma: no cover - network dependent
            print(f"server error: {error}", file=sys.stderr)
            return 2
        breakdown = score_completion(result.contract_text) if result.contract_text else None
        score = reward(result.contract_text) if result.contract_text else 0.0
        rewards.append(score)
        valid += 1 if result.request is not None else 0
        solved += 1 if (result.derivation and result.derivation.status == "solved") else 0
        repairs += result.repairs
        rows.append((task.task_id, result.request is not None, score, result.error))

    total = max(1, len(tasks))
    print(f"tasks                : {len(tasks)}")
    print(f"contract validity    : {valid / total:.1%} ({valid}/{total})")
    print(f"solved by the engine : {solved / total:.1%}")
    print(f"mean reward          : {statistics.mean(rewards) if rewards else 0.0:.3f}")
    print(f"mean repairs/query   : {repairs / total:.2f}")
    print()
    for task_id, ok, score, error in rows:
        flag = "ok " if ok else "FAIL"
        print(f"  {flag} {task_id:10s} reward={score:.2f}" + (f"  {error[:70]}" if error else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
