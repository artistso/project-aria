"""Command-line interface for ARIA.

    aria run "solve for x in 2x + 4 = 10"
    aria simplify "(x+1)^2 - x^2"
    aria verify "log(a*b) = log(a) + log(b)"
    aria units --law newton_second
    aria units "F = m a" --dims "F=kg*m/s^2,m=kg,a=m/s^2"
    aria derive-dim "F = m a" --unknown a
    aria bench --markdown benchmarks/report.md
    aria train --eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import __version__
from .bench.runner import run_benchmark
from .core.parse import ParseError, parse, parse_problem
from .core.state import Goal, GoalKind, SymbolicState
from .embed.tree_embed import cosine, embed
from .reason.engine import SymbolicEngine
from .science.laws import LAW_LIBRARY, check_law, derive_dimension, explain_derivation, list_laws
from .science.units import DimensionError, check_equation, parse_unit
from .verify.verifier import Verifier


def _parse_dims(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    dimensions: Dict[str, str] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit(f"malformed dimension declaration {pair!r}; expected name=unit")
        name, unit = pair.split("=", 1)
        dimensions[name.strip()] = unit.strip()
    return dimensions


def _emit(derivation, as_json: bool) -> None:
    if as_json:
        print(json.dumps(derivation.summary(), indent=2))
    else:
        print(derivation.to_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aria",
        description="ARIA - symbolic mathematical reasoning with verified derivations",
    )
    parser.add_argument("--version", action="version", version=f"aria {__version__}")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summaries")
    parser.add_argument(
        "--backend",
        choices=("auto", "sympy", "internal"),
        default="auto",
        help="verification backend (default: auto)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="parse a prompt and execute its goal")
    run.add_argument("prompt", help='e.g. "solve for x in 2x + 4 = 10"')

    parse_cmd = subparsers.add_parser("parse", help="show the parsed symbolic state")
    parse_cmd.add_argument("prompt")

    for name, help_text in (
        ("solve", "solve an equation or congruence"),
        ("simplify", "simplify an expression"),
        ("expand", "expand an expression"),
        ("factor", "factor an expression"),
        ("verify", "verify a claim"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("expression")
        if name == "solve":
            sub.add_argument("--for", dest="variable", default=None, help="variable to solve for")

    diff = subparsers.add_parser("differentiate", help="differentiate an expression")
    diff.add_argument("expression")
    diff.add_argument("--var", default="x")

    emb = subparsers.add_parser("embed", help="show the structural embedding of an expression")
    emb.add_argument("expression")
    emb.add_argument("--dim", type=int, default=32)
    emb.add_argument("--compare", default=None, help="second expression to compare against")

    units = subparsers.add_parser("units", help="dimensional analysis of a physical equation")
    units.add_argument("equation", nargs="?", default=None, help='e.g. "F = m a"')
    units.add_argument("--dims", action="append", default=[], help="name=unit, repeatable")
    units.add_argument("--law", default=None, help="check a named law from the library")
    units.add_argument("--list", action="store_true", help="list known laws and units")

    derive = subparsers.add_parser("derive-dim", help="solve a law for an unknown dimension")
    derive.add_argument("equation")
    derive.add_argument("--unknown", required=True)
    derive.add_argument("--dims", action="append", default=[])

    bench = subparsers.add_parser("bench", help="run the benchmark suite")
    bench.add_argument("--suite", default=None, choices=None)
    bench.add_argument("--markdown", default=None, help="write a markdown report")
    bench.add_argument("--json-out", default=None, help="write a JSON report")

    ask = subparsers.add_parser(
        "ask", help="run the math channel against a local OpenAI-compatible model"
    )
    ask.add_argument("prompt")
    ask.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080/v1/chat/completions",
        help="OpenAI-compatible chat completions URL (llama.cpp / LM Studio / Ollama)",
    )
    ask.add_argument("--model", default="local")
    ask.add_argument(
        "--grammar",
        choices=("loose", "strict", "off"),
        default="loose",
        help="constrained-decoding grammar applied to the model's output",
    )
    ask.add_argument("--no-narrate", action="store_true", help="skip the prose explanation")
    ask.add_argument(
        "--dry-run",
        action="store_true",
        help="use ARIA's parser as the translator (no model needed)",
    )
    ask.add_argument("--show-trace", action="store_true", help="show raw model exchanges")

    grammar_cmd = subparsers.add_parser("grammar", help="print the GBNF grammar for a local model")
    grammar_cmd.add_argument("--mode", choices=("loose", "strict"), default="loose")
    grammar_cmd.add_argument("--out", default=None, help="write to a .gbnf file")

    dataset = subparsers.add_parser(
        "dataset", help="export training data mined from verified mathematics"
    )
    dataset.add_argument("kind", choices=("sft", "rl"))
    dataset.add_argument("--out", required=True, help="output .jsonl path")
    dataset.add_argument("--paraphrases", type=int, default=2, help="paraphrases per task")
    dataset.add_argument("--limit", type=int, default=None)

    reward_cmd = subparsers.add_parser(
        "reward", help="score a model completion with the verifier (RL reward)"
    )
    reward_cmd.add_argument("--completion", default=None, help="contract text to score")
    reward_cmd.add_argument("--file", default=None, help="file containing completions (one per line)")

    train = subparsers.add_parser("train", help="train the step-ranking policy")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--save", default="aria/neural/weights.json")
    train.add_argument("--eval", action="store_true", help="compare search orderings")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verifier = Verifier(backend=args.backend)
    engine = SymbolicEngine(verifier)

    try:
        if args.command == "run":
            state = parse_problem(args.prompt)
            _emit(engine.run(state), args.json)

        elif args.command == "parse":
            state = parse_problem(args.prompt)
            payload = {
                "expressions": [str(e) for e in state.expressions],
                "goal": str(state.goal),
                "symbols": [s.name for s in state.symbols],
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"expressions: {payload['expressions']}")
                print(f"goal       : {payload['goal']}")
                print(f"symbols    : {payload['symbols'] or '-'}")

        elif args.command == "solve":
            expression = parse(args.expression)
            from .core.ast import Congruence, Equation, Symbol

            if isinstance(expression, Congruence):
                var = Symbol(args.variable) if args.variable else expression.lhs.free_symbols[0]
                _emit(engine.solve_congruence(expression, var), args.json)
            else:
                equation = expression if isinstance(expression, Equation) else Equation(expression, 0)
                var = (
                    Symbol(args.variable)
                    if args.variable
                    else (equation.lhs.free_symbols or equation.rhs.free_symbols)[0]
                )
                _emit(engine.solve(equation, var), args.json)

        elif args.command == "simplify":
            _emit(engine.simplify(parse(args.expression)), args.json)

        elif args.command == "expand":
            _emit(engine.expand(parse(args.expression)), args.json)

        elif args.command == "factor":
            _emit(engine.factor(parse(args.expression)), args.json)

        elif args.command == "verify":
            _emit(engine.prove(parse(args.expression)), args.json)

        elif args.command == "differentiate":
            from .core.ast import Symbol

            _emit(engine.differentiate(parse(args.expression), Symbol(args.var)), args.json)

        elif args.command == "embed":
            first = parse(args.expression)
            vector = embed(first, dim=args.dim)
            if args.compare:
                second = parse(args.compare)
                other = embed(second, dim=args.dim)
                similarity = cosine(vector.vector, other.vector)
                if args.json:
                    print(json.dumps({
                        "left": str(first),
                        "right": str(second),
                        "cosine": round(similarity, 4),
                        "vector": [round(v, 4) for v in vector.vector],
                    }, indent=2))
                else:
                    print(f"cosine similarity: {similarity:.4f}")
            elif args.json:
                print(json.dumps({"expression": str(first), "vector": [round(v, 4) for v in vector.vector]}))
            else:
                print(f"expression: {first}")
                print(f"dimension : {len(vector)}")
                print("vector    : " + " ".join(f"{v:+.3f}" for v in vector.vector))

        elif args.command == "units":
            if args.list:
                print("Known laws:")
                for law in list_laws():
                    print(f"  {law.name:18s} {law.equation:28s} [{law.domain}]")
                print("\nBase units: kg, m, s, A, K, mol, cd")
                print("Derived   : N, J, W, Pa, Hz, C, V, ohm, F, T, Wb, rad, sr")
                return 0
            if args.law:
                report = check_law(args.law)
            elif args.equation:
                dimensions = _parse_dims(args.dims)
                if not dimensions and args.equation in LAW_LIBRARY:
                    report = check_law(args.equation)
                else:
                    report = check_equation(args.equation, dimensions)
            else:
                parser.error("provide an equation, --law NAME, or --list")
            if args.json:
                print(json.dumps({
                    "equation": report.equation,
                    "left": str(report.left) if report.left else None,
                    "right": str(report.right) if report.right else None,
                    "consistent": report.consistent,
                    "detail": report.detail,
                }, indent=2))
            else:
                print(report.to_text())
            return 0 if report.consistent else 1

        elif args.command == "derive-dim":
            print(explain_derivation(args.equation, _parse_dims(args.dims), args.unknown))

        elif args.command == "bench":
            report = run_benchmark(engine, suite=args.suite)
            print(report.to_markdown())
            if args.markdown:
                Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
                Path(args.markdown).write_text(report.to_markdown())
                print(f"\nwrote {args.markdown}")
            if args.json_out:
                Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json_out).write_text(report.to_json())
                print(f"wrote {args.json_out}")

        elif args.command == "ask":
            from .llm.client import OpenAICompatClient
            from .llm.orchestrator import MathChannel, dry_run_client

            grammar_mode = None if args.grammar == "off" else args.grammar
            client = (
                dry_run_client()
                if args.dry_run
                else OpenAICompatClient(endpoint=args.endpoint, model=args.model)
            )
            channel = MathChannel(
                client, engine, grammar_mode=grammar_mode, narrate=not args.no_narrate and not args.dry_run
            )
            result = channel.ask(args.prompt)
            if args.json:
                payload = {
                    "user_text": result.user_text,
                    "contract": result.contract_text,
                    "status": result.derivation.status if result.derivation else None,
                    "solutions": [str(v) for v in result.derivation.solutions] if result.derivation else [],
                    "explanation": result.explanation,
                    "repairs": result.repairs,
                    "error": result.error,
                }
                print(json.dumps(payload, indent=2))
            else:
                print(result.to_text())
                if args.show_trace and result.trace:
                    print("\nModel exchanges:")
                    for line in result.trace:
                        print(f"  {line}")
            return 0 if result.ok else 1

        elif args.command == "grammar":
            from .llm.grammar import build_grammar, validate_grammar

            text = build_grammar(args.mode)
            ok, problems = validate_grammar(text)
            if not ok:
                print("grammar failed structural validation:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                return 1
            if args.out:
                Path(args.out).write_text(text)
                print(f"wrote {args.out} ({len(text.splitlines())} lines, {args.mode})")
            else:
                print(text, end="")
            return 0

        elif args.command == "dataset":
            from .llm.dataset import build_rl_prompts, build_sft_examples, to_chat_jsonl

            if args.kind == "sft":
                examples = build_sft_examples(paraphrases_per_task=args.paraphrases)
                if args.limit:
                    examples = examples[: args.limit]
                payload = to_chat_jsonl(examples)
                count = len(examples)
            else:
                records = build_rl_prompts()
                if args.limit:
                    records = records[: args.limit]
                payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
                count = len(records)
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload)
            print(f"wrote {count} {args.kind} record(s) to {target}")
            return 0

        elif args.command == "reward":
            from .llm.dataset import score_completion

            if args.file:
                texts = [
                    line.strip()
                    for line in Path(args.file).read_text().splitlines()
                    if line.strip()
                ]
            elif args.completion:
                texts = [args.completion.replace("\\n", "\n")]
            else:
                texts = [sys.stdin.read()]

            scored = [score_completion(text, engine) for text in texts]
            if args.json:
                print(json.dumps([b.as_dict() for b in scored], indent=2))
            else:
                for text, breakdown in zip(texts, scored):
                    head = text.strip().splitlines()[0][:60] if text.strip() else "<empty>"
                    print(f"{breakdown.total:.2f}  {breakdown.status:9s} {head}")
            return 0

        elif args.command == "train":
            from .neural.train import evaluate_search, train_policy

            policy, report = train_policy(epochs=args.epochs, save_path=Path(args.save))
            print(report.to_text())
            if args.eval:
                print()
                print(json.dumps(evaluate_search(policy=policy), indent=2))

        return 0

    except (ParseError, DimensionError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
