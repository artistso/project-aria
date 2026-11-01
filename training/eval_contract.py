"""Evaluate an LLM translator without reducing checked work to a scalar reward.

Works against any OpenAI-compatible server (llama.cpp, LM Studio, Ollama) and
needs no GPU on the machine running the evaluation. Reports translation
validity, repair load, engine outcomes, and the separate process-audit state.

    python training/eval_contract.py --endpoint http://127.0.0.1:8080/v1/chat/completions
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.bench.tasks import tasks_for  # noqa: E402
from aria.llm.client import ClientError, OpenAICompatClient  # noqa: E402
from aria.llm.context import AuditState  # noqa: E402
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

    tasks = [task for task in tasks_for(args.suite) if task.suite != "units"]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    valid_contracts = 0
    decisions = 0
    repairs = 0
    audit_counts = Counter({state.value: 0 for state in AuditState})
    outcomes: Counter[str] = Counter()
    rows = []

    for task in tasks:
        try:
            result = channel.ask(task.prompt)
        except ClientError as error:  # pragma: no cover - network dependent
            print(f"server error: {error}", file=sys.stderr)
            return 2

        valid = result.request is not None
        valid_contracts += int(valid)
        repairs += result.repairs
        outcome = result.derivation.status if result.derivation else "not-run"
        outcomes[outcome] += 1
        decisions += int(outcome in {"solved", "refuted"})
        audit = result.frame.audit_state.value if result.frame else AuditState.HELD.value
        audit_counts[audit] += 1
        rows.append((task.task_id, valid, outcome, audit, result.repairs, result.error))

    total = len(tasks)
    denominator = max(1, total)
    print(f"tasks                  : {total}")
    print(f"contract validity      : {valid_contracts / denominator:.1%} ({valid_contracts}/{total})")
    print(f"engine decisions       : {decisions / denominator:.1%} ({decisions}/{total})")
    print(f"process audit pass     : {audit_counts[AuditState.PASS.value]}")
    print(f"process audit fail     : {audit_counts[AuditState.FAIL.value]}")
    print(f"process audit held     : {audit_counts[AuditState.HELD.value]}")
    print(f"mean repairs/query     : {repairs / denominator:.2f}")
    print(f"task outcomes          : {dict(sorted(outcomes.items()))}")
    print()
    for task_id, valid, outcome, audit, repair_count, error in rows:
        flag = "ok " if valid else "FAIL"
        message = f"  {flag} {task_id:10s} outcome={outcome:<8s} audit={audit:<4s} repairs={repair_count}"
        if error:
            message += f"  {error[:70]}"
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
