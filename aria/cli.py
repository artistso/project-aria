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
    ask.add_argument(
        "--context-budget-chars",
        type=int,
        default=None,
        help=(
            "maximum input-context characters; mandatory overflow holds before inference, "
            "optional narration detail is dropped only as whole, recoverable blocks"
        ),
    )

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
        "reward", help="legacy scalar diagnostic for a completion (not a training objective)"
    )
    reward_cmd.add_argument("--completion", default=None, help="contract text to score")
    reward_cmd.add_argument("--file", default=None, help="file containing completions (one per line)")

    ledger_cmd = subparsers.add_parser(
        "ledger", help="the record ARIA keeps of its own correct and incorrect work"
    )
    ledger_sub = ledger_cmd.add_subparsers(dest="ledger_action", required=True)

    rec = ledger_sub.add_parser("record", help="attempt a problem and file the result")
    rec.add_argument("--question", required=True, help="the problem, in words")
    rec.add_argument("--completion", default=None, help="the contract that was produced")
    rec.add_argument("--file", default=None, help="file holding the contract")
    rec.add_argument(
        "--note",
        default="",
        help="the model's own reflection -- the only field it authors",
    )
    rec.add_argument("--ledger", default=None, help="ledger path (default $ARIA_LEDGER)")

    rep = ledger_sub.add_parser("report", help="print the grade card")
    rep.add_argument("--ledger", default=None)

    nxt = ledger_sub.add_parser("next", help="the problem to attempt next")
    nxt.add_argument("--ledger", default=None)

    led_sft = ledger_sub.add_parser(
        "sft", help="export proven processes from rungs held by the school"
    )
    led_sft.add_argument("--out", required=True)
    led_sft.add_argument("--ledger", default=None)
    led_sft.add_argument("--max-level", type=int, default=None, help="cap the export at this rung")
    led_sft.add_argument(
        "--include-learning",
        action="store_true",
        help="include proven examples from rungs not yet held (explicit bootstrap only)",
    )

    memory_cmd = subparsers.add_parser("memory", help="the text memory ARIA keeps for itself")
    memory_sub = memory_cmd.add_subparsers(dest="memory_action", required=True)

    mem_add = memory_sub.add_parser("add", help="remember something")
    mem_add.add_argument("text", help="what to remember")
    mem_add.add_argument(
        "--kind",
        choices=("fact", "procedure", "correction", "source", "preference"),
        default="fact",
    )
    mem_add.add_argument("--tags", default="", help="comma-separated tags")
    mem_add.add_argument("--source", default="", help="where it came from: file, doc, ledger#N")
    mem_add.add_argument(
        "--confidence",
        choices=("unverified", "stated", "tested", "verified"),
        default="unverified",
    )
    mem_add.add_argument("--memory", default=None)

    mem_correct = memory_sub.add_parser(
        "correct", help="append a sourced correction that supersedes one memory"
    )
    mem_correct.add_argument("memory_id", type=int)
    mem_correct.add_argument("text", help="the corrected memory text")
    mem_correct.add_argument("--source", default="user correction")
    mem_correct.add_argument("--tags", default="", help="comma-separated tags")
    mem_correct.add_argument(
        "--kind",
        choices=("fact", "procedure", "correction", "source", "preference"),
        default=None,
        help="replacement kind (defaults to preference when correcting a preference)",
    )
    mem_correct.add_argument(
        "--confidence",
        choices=("unverified", "stated", "tested", "verified"),
        default="stated",
    )
    mem_correct.add_argument("--memory", default=None)

    mem_find = memory_sub.add_parser("find", help="search memory")
    mem_find.add_argument("query")
    mem_find.add_argument("-k", "--k", dest="k", type=int, default=5)
    mem_find.add_argument("--memory", default=None)

    mem_ctx = memory_sub.add_parser("context", help="a block of memory to put in front of a model")
    mem_ctx.add_argument("query")
    mem_ctx.add_argument("-k", "--k", dest="k", type=int, default=6)
    mem_ctx.add_argument("--max-chars", type=int, default=1600)
    mem_ctx.add_argument("--memory", default=None)

    mem_digest = memory_sub.add_parser("digest", help="counts by kind, confidence and tag")
    mem_digest.add_argument("--memory", default=None)

    mem_audit = memory_sub.add_parser("audit", help="entries with no source, or unverified")
    mem_audit.add_argument("--memory", default=None)

    mem_import = memory_sub.add_parser("import", help="seed memory from the ledger or geometry")
    mem_import.add_argument(
        "--from", dest="import_from", choices=("ledger", "geometry"), required=True
    )
    mem_import.add_argument("--ledger", default=None, help="ledger path, for --from ledger")
    mem_import.add_argument("--memory", default=None)

    mem_export = memory_sub.add_parser("export", help="write the human-readable memory file")
    mem_export.add_argument("--out", required=True)
    mem_export.add_argument("--memory", default=None)

    geometry_cmd = subparsers.add_parser("geometry", help="shapes and causal diagrams ARIA can check")
    geometry_sub = geometry_cmd.add_subparsers(dest="geometry_action", required=True)

    geo_shape = geometry_sub.add_parser("shape", help="properties of a shape")
    geo_shape.add_argument("name")
    geo_shape.add_argument("--size", type=float, default=1.0, help="edge length or radius")
    geo_shape.add_argument("--sides", type=int, default=5, help="for a regular polygon")
    geometry_sub.add_parser("list", help="the shapes that are known")

    geo_penrose = geometry_sub.add_parser("penrose", help="structure of a Penrose diagram")
    geo_penrose.add_argument("model", choices=("minkowski", "schwarzschild", "frw"))

    geo_check = geometry_sub.add_parser("check", help="run every geometric identity")
    geo_check.add_argument(
        "--remember",
        action="store_true",
        help="file the verified facts into ARIA's memory",
    )
    geo_check.add_argument("--memory", default=None)

    model_cmd = subparsers.add_parser(
        "model", help="inspect a model file before trusting or training it"
    )
    model_sub = model_cmd.add_subparsers(dest="model_action", required=True)
    model_inspect = model_sub.add_parser("inspect", help="read a GGUF header (no GPU needed)")
    model_inspect.add_argument("path")

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
                client,
                engine,
                grammar_mode=grammar_mode,
                narrate=not args.no_narrate and not args.dry_run,
                context_budget=args.context_budget_chars,
            )
            result = channel.ask(args.prompt)
            if args.json:
                print(json.dumps(result.as_dict(), indent=2))
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

        elif args.command == "ledger":
            from .llm import ledger as ledger_module

            book = ledger_module.Ledger(Path(args.ledger) if args.ledger else None)

            if args.ledger_action == "record":
                if args.file:
                    completion = Path(args.file).read_text()
                elif args.completion:
                    completion = args.completion.replace("\\n", "\n")
                else:
                    completion = sys.stdin.read()
                entry = book.record(args.question, completion, note=args.note, engine=engine)
                if args.json:
                    print(json.dumps(entry.as_dict(), indent=2))
                else:
                    mark = "proven" if entry.demonstrated else entry.outcome
                    print(f"#{entry.seq}  {entry.skill:<24} {mark}")
                    if entry.first_error:
                        print(f"  first break: {entry.first_error}")
                    print(f"  filed in {book.path}")
                return 0

            if args.ledger_action == "report":
                if args.json:
                    print(json.dumps(book.card(), indent=2))
                else:
                    print(ledger_module.report(book))
                return 0

            if args.ledger_action == "next":
                following = book.next_task()
                if args.json:
                    print(json.dumps(following, indent=2))
                    return 0
                if following is None:
                    print("nothing left to be taught; write a general claim and try to break it.")
                elif following["prompt"]:
                    print(f"[{following['skill']}] {following['prompt']}")
                    contract = ledger_module.contract_of(following["prompt"])
                    if contract:
                        print()
                        print(contract)
                else:
                    print(f"[{following['skill']}] no unattempted problem left -- author one.")
                return 0

            records = book.to_sft_records(
                held_only=not args.include_learning,
                max_level=args.max_level,
            )
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
            )
            stage = f"through rung {args.max_level}" if args.max_level is not None else "all held rungs"
            if args.include_learning:
                stage += " plus learning rungs"
            print(f"wrote {len(records)} proven example(s) from {stage} to {target}")
            return 0

        elif args.command == "memory":
            from .llm import memory as memory_module

            store = memory_module.MemoryStore(Path(args.memory) if args.memory else None)

            if args.memory_action == "add":
                entry = store.add(
                    args.text,
                    kind=args.kind,
                    tags=[tag for tag in args.tags.split(",") if tag.strip()],
                    source=args.source,
                    confidence=args.confidence,
                )
                if args.json:
                    print(json.dumps(entry.as_dict(), indent=2))
                else:
                    print(f"#{entry.id} ({entry.kind}, {entry.confidence}) {entry.text}")
                    if not entry.source:
                        print("  no source: `memory audit` will flag this until something verifies it")
                return 0

            if args.memory_action == "correct":
                entry = store.correct(
                    args.memory_id,
                    args.text,
                    source=args.source,
                    confidence=args.confidence,
                    tags=[tag for tag in args.tags.split(",") if tag.strip()],
                    kind=args.kind,
                )
                if args.json:
                    print(json.dumps(entry.as_dict(), indent=2))
                else:
                    print(
                        f"#{entry.id} corrects memory #{entry.supersedes}: "
                        f"({entry.kind}, {entry.confidence}) {entry.text}"
                    )
                return 0

            if args.memory_action == "find":
                hits = store.find(args.query, k=args.k)
                if args.json:
                    print(json.dumps([entry.as_dict() for entry in hits], indent=2))
                elif not hits:
                    print("nothing remembered matches that yet.")
                else:
                    for entry in hits:
                        print(entry.line())
                return 0

            if args.memory_action == "context":
                try:
                    pack = store.context_pack(args.query, k=args.k, max_chars=args.max_chars)
                except memory_module.MemoryContextBudgetError as error:
                    if args.json:
                        print(json.dumps(error.as_dict(), indent=2))
                    else:
                        print(str(error), file=sys.stderr)
                    return 2
                if args.json:
                    print(json.dumps(pack.as_dict(), indent=2))
                elif pack.text:
                    print(pack.text)
                    if pack.omitted:
                        ids = ", ".join(f"#{entry.id}" for entry in pack.omitted)
                        print(f"\nOmitted memory ids: {ids}; rerun with a larger --max-chars to recover.")
                else:
                    print("nothing remembered matches that yet.")
                return 0

            if args.memory_action == "digest":
                print(json.dumps(store.digest(), indent=2))
                return 0

            if args.memory_action == "audit":
                flagged = store.audit()
                if args.json:
                    print(json.dumps([entry.as_dict() for entry in flagged], indent=2))
                elif not flagged:
                    print("every memory has a source and a confidence above unverified.")
                else:
                    print(f"{len(flagged)} memory item(s) need backing:")
                    for entry in flagged:
                        print(f"  #{entry.id} ({entry.confidence}) {entry.text}")
                return 0

            if args.memory_action == "import":
                if args.import_from == "ledger":
                    from .llm.ledger import Ledger

                    book = Ledger(Path(args.ledger) if args.ledger else None)
                    added = store.import_ledger(book)
                else:
                    added = store.import_geometry()
                fresh = len({entry.id for entry in added})
                print(f"imported {fresh} memory item(s) into {store.path}")
                return 0

            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(store.render_markdown())
            print(f"wrote {len(store.entries())} memories to {target}")
            return 0

        elif args.command == "geometry":
            from .geometry import PLATONIC, causal_checks, describe, diagram_text, shape_checks

            if args.geometry_action == "list":
                for solid in PLATONIC:
                    print(f"{solid.name:<14} {solid.schlafli:<8} V={solid.vertices:<3} E={solid.edges:<3} F={solid.faces}")
                print("circle, sphere, polygon (regular n-gon)")
                return 0

            if args.geometry_action == "shape":
                try:
                    data = describe(args.name, size=args.size, sides=args.sides)
                except ValueError as error:
                    print(f"error: {error}", file=sys.stderr)
                    return 2
                if args.json:
                    print(json.dumps(data, indent=2))
                else:
                    width = max(len(key) for key in data)
                    for key, value in data.items():
                        print(f"  {key:<{width}}  {value}")
                return 0

            if args.geometry_action == "penrose":
                print(diagram_text(args.model))
                return 0

            results = [("shape", claim, ok) for claim, ok in shape_checks()]
            results += [("causal", claim, ok) for claim, ok in causal_checks()]
            failures = [claim for _, claim, ok in results if not ok]
            for group, claim, ok in results:
                print(f"{'PASS' if ok else 'FAIL'}  [{group}] {claim}")
            if args.remember and not failures:
                from .llm.memory import MemoryStore

                store = MemoryStore(Path(args.memory) if args.memory else None)
                store.import_geometry()
                print(f"filed {len(results)} verified facts in {store.path}")
            return 0 if not failures else 1

        elif args.command == "model":
            from .llm.gguf import GGUFError, inspect_gguf

            try:
                info = inspect_gguf(args.path)
            except (GGUFError, OSError) as error:
                print(f"error: {error}", file=sys.stderr)
                return 2

            summary = info.summary()
            if args.json:
                print(json.dumps(summary, indent=2))
                return 0
            print(f"{summary['name'] or '<unnamed>'}  ({summary['architecture'] or 'unknown arch'})")
            print(f"  parameters     {summary['parameters_human']}  ({summary['parameters']:,})")
            print(f"  tensors        {summary['tensors']}")
            print(f"  quantisation   {summary['dominant_quantisation']}  {summary['quantisations']}")
            if summary["context_length"]:
                print(f"  context        {summary['context_length']}")
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
