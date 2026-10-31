"""Benchmark runner and metrics.

Reported metrics follow the evaluation plan in the proposal:

* **exact correctness** -- does the verified result match the independently
  stated answer (compared symbolically, not as text)?
* **invalid derivation rate** -- fraction of derivation steps the verifier
  refuted (the soundness signal);
* **verified trace rate** -- fraction of derivations whose every step verified;
* **symbolic equivalence rate** -- results that are provably equivalent even when
  not structurally identical to the reference form;
* **unit consistency** -- dimensional checks passed in the physics suite.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.parse import ParseError, parse, parse_problem
from ..core.state import GoalKind
from ..reason.derivation import SOLVED, Derivation
from ..reason.engine import SymbolicEngine
from ..science.laws import LAW_LIBRARY, check_law
from ..verify.verifier import Verifier
from .tasks import TASKS, Task, tasks_for

__all__ = ["TaskResult", "BenchmarkReport", "run_benchmark"]


@dataclass
class TaskResult:
    task: Task
    correct: bool
    status: str
    steps: int
    invalid_steps: int
    trace_accuracy: float
    verified_trace: bool
    result: Optional[str] = None
    solutions: Tuple[str, ...] = ()
    error: Optional[str] = None
    duration_ms: float = 0.0
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        return {
            "task": self.task.task_id,
            "suite": self.task.suite,
            "correct": self.correct,
            "status": self.status,
            "steps": self.steps,
            "invalid_steps": self.invalid_steps,
            "trace_accuracy": round(self.trace_accuracy, 3),
            "result": self.result,
            "solutions": list(self.solutions),
            "error": self.error,
            "ms": round(self.duration_ms, 2),
        }


@dataclass
class BenchmarkReport:
    results: List[TaskResult] = field(default_factory=list)
    unit_checks: List[Tuple[str, bool, str]] = field(default_factory=list)

    # -- metrics -----------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def exact_correctness(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def total_steps(self) -> int:
        return sum(r.steps for r in self.results)

    @property
    def invalid_derivation_rate(self) -> float:
        return (
            sum(r.invalid_steps for r in self.results) / self.total_steps
            if self.total_steps
            else 0.0
        )

    @property
    def verified_trace_rate(self) -> float:
        return (
            sum(1 for r in self.results if r.verified_trace) / self.total
            if self.total
            else 0.0
        )

    @property
    def mean_trace_accuracy(self) -> float:
        return (
            sum(r.trace_accuracy for r in self.results) / self.total if self.total else 0.0
        )

    @property
    def unit_consistency(self) -> float:
        if not self.unit_checks:
            return 0.0
        return sum(1 for _, ok, _ in self.unit_checks if ok) / len(self.unit_checks)

    def by_suite(self) -> Dict[str, Dict[str, float]]:
        suites: Dict[str, List[TaskResult]] = {}
        for result in self.results:
            suites.setdefault(result.task.suite, []).append(result)
        summary: Dict[str, Dict[str, float]] = {}
        for suite, results in sorted(suites.items()):
            steps = sum(r.steps for r in results)
            summary[suite] = {
                "tasks": len(results),
                "correct": sum(1 for r in results if r.correct),
                "correctness": sum(1 for r in results if r.correct) / len(results),
                "mean_steps": steps / len(results) if results else 0.0,
                "verified_trace_rate": sum(1 for r in results if r.verified_trace) / len(results),
            }
        return summary

    # -- rendering ---------------------------------------------------------

    def metrics(self) -> Dict[str, float]:
        return {
            "tasks": float(self.total),
            "exact_correctness": round(self.exact_correctness, 4),
            "invalid_derivation_rate": round(self.invalid_derivation_rate, 4),
            "verified_trace_rate": round(self.verified_trace_rate, 4),
            "mean_trace_accuracy": round(self.mean_trace_accuracy, 4),
            "unit_consistency": round(self.unit_consistency, 4),
            "total_steps": float(self.total_steps),
        }

    def to_markdown(self) -> str:
        metrics = self.metrics()
        lines = [
            "# ARIA benchmark report",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for name, value in metrics.items():
            lines.append(f"| {name} | {value} |")
        lines += ["", "## By suite", "", "| suite | tasks | correct | correctness | mean steps | verified traces |", "|---|---|---|---|---|---|"]
        for suite, data in self.by_suite().items():
            lines.append(
                f"| {suite} | {int(data['tasks'])} | {int(data['correct'])} | "
                f"{data['correctness']:.0%} | {data['mean_steps']:.2f} | {data['verified_trace_rate']:.0%} |"
            )
        lines += ["", "## Task detail", "", "| task | correct | status | steps | trace | result |", "|---|---|---|---|---|---|"]
        for result in self.results:
            outcome = "✓" if result.correct else "✗"
            detail = result.result or (", ".join(result.solutions) or result.error or "")
            lines.append(
                f"| {result.task.task_id} | {outcome} | {result.status} | {result.steps} | "
                f"{result.trace_accuracy:.0%} | `{detail}` |"
            )
        if self.unit_checks:
            lines += ["", "## Dimensional analysis", "", "| law | consistent | detail |", "|---|---|---|"]
            for name, ok, detail in self.unit_checks:
                lines.append(f"| {name} | {'✓' if ok else '✗'} | {detail} |")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "metrics": self.metrics(),
                "by_suite": self.by_suite(),
                "results": [r.as_row() for r in self.results],
                "unit_checks": [
                    {"law": name, "consistent": ok, "detail": detail}
                    for name, ok, detail in self.unit_checks
                ],
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _multiset_match(
    produced: Sequence, expected: Sequence[str], verifier: Verifier
) -> bool:
    """Order-insensitive comparison of solution sets with symbolic equality."""
    if len(produced) != len(expected):
        return False
    remaining = list(produced)
    for expected_text in expected:
        target = parse(expected_text)
        for index, candidate in enumerate(remaining):
            if verifier.equivalent(candidate, target).ok:
                remaining.pop(index)
                break
        else:
            return False
    return True


def run_benchmark(
    engine: Optional[SymbolicEngine] = None,
    suite: Optional[str] = None,
    include_units: bool = True,
) -> BenchmarkReport:
    engine = engine or SymbolicEngine()
    verifier = engine.verifier
    report = BenchmarkReport()

    for task in tasks_for(suite):
        started = time.perf_counter()
        try:
            state = parse_problem(task.prompt)
            derivation = engine.run(state)
            duration = (time.perf_counter() - started) * 1000
            result = _score_task(task, derivation, verifier)
            result.duration_ms = duration
            report.results.append(result)
        except (ParseError, ValueError, TypeError, ZeroDivisionError) as error:
            report.results.append(
                TaskResult(
                    task=task,
                    correct=False,
                    status="error",
                    steps=0,
                    invalid_steps=0,
                    trace_accuracy=0.0,
                    verified_trace=False,
                    error=f"{type(error).__name__}: {error}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )

    if include_units and suite in (None, "units"):
        for name in sorted(LAW_LIBRARY):
            law_report = check_law(name)
            report.unit_checks.append((name, law_report.consistent, law_report.detail))
        # A deliberately malformed law must be rejected, otherwise the check is vacuous.
        broken = check_law("newton_second")
        from ..science.units import check_equation

        bad = check_equation("F = m a^2", {"F": "kg*m/s^2", "m": "kg", "a": "m/s^2"})
        report.unit_checks.append(
            ("control_broken_law", not bad.consistent, "F = m a^2 must be rejected")
        )

    return report


def _score_task(task: Task, derivation: Derivation, verifier: Verifier) -> TaskResult:
    steps = len(derivation.steps)
    invalid = len(derivation.invalid_steps)
    # Proof-style goals produce a verdict rather than a step chain; a decided
    # verdict is what "verified" means for them.
    verified_trace = (
        derivation.valid and steps > 0
        if steps
        else derivation.status in (SOLVED, "refuted")
    )
    result_text = str(derivation.result) if derivation.result is not None else None
    solutions = tuple(str(value) for value in derivation.solutions)
    correct = False
    note = derivation.notes

    if task.expected_status is not None:
        correct = derivation.status == task.expected_status
    elif task.expected_solutions:
        correct = derivation.status == SOLVED and _multiset_match(
            derivation.solutions, task.expected_solutions, verifier
        )
    elif task.expected_result is not None:
        target = parse(task.expected_result)
        produced = derivation.result
        if produced is not None:
            equivalence = verifier.equivalent(produced, target)
            correct = equivalence.ok and derivation.status == SOLVED
    elif task.suite == "modular":
        correct = derivation.status == SOLVED and not derivation.solutions

    if task.suite == "units":
        report = check_law(task.prompt if task.prompt in LAW_LIBRARY else task.task_id)
        correct = report.consistent

    return TaskResult(
        task=task,
        correct=correct,
        status=derivation.status,
        steps=steps,
        invalid_steps=invalid,
        trace_accuracy=derivation.trace_accuracy,
        verified_trace=verified_trace,
        result=result_text,
        solutions=solutions,
        note=note,
    )
