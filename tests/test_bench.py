"""Benchmark harness and end-to-end accuracy."""

from aria.bench.runner import run_benchmark
from aria.bench.tasks import TASKS, tasks_for


def test_suite_is_well_formed():
    ids = [task.task_id for task in TASKS]
    assert len(ids) == len(set(ids))
    assert tasks_for("solve")
    for task in TASKS:
        # An empty solution tuple is a meaningful expectation: "no solution".
        assert (
            task.expected_solutions
            or task.expected_result
            or task.expected_status
            or task.suite in {"units", "modular"}
        )


def test_benchmark_runs_and_scores():
    report = run_benchmark()
    metrics = report.metrics()
    assert metrics["tasks"] >= 40
    assert metrics["exact_correctness"] > 0.85, report.to_markdown()
    assert metrics["invalid_derivation_rate"] == 0.0
    assert metrics["unit_consistency"] == 1.0


def test_markdown_and_json_render():
    report = run_benchmark(suite="solve")
    markdown = report.to_markdown()
    assert "# ARIA benchmark report" in markdown
    assert report.to_json().startswith("{")
