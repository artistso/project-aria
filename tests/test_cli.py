"""Command-line interface smoke tests."""

import json

import pytest

from aria.cli import main


def test_run_command(capsys):
    assert main(["run", "solve for x in 2x + 4 = 10"]) == 0
    output = capsys.readouterr().out
    assert "x = 3" in output
    assert "trace accuracy" in output


def test_json_output(capsys):
    exit_code = main(["--json", "run", "simplify 2x + 3x"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "solved"


def test_units_law(capsys):
    assert main(["units", "--law", "newton_second"]) == 0
    assert "CONSISTENT" in capsys.readouterr().out


def test_units_inconsistent_returns_nonzero(capsys):
    exit_code = main(["units", "F = m a^2", "--dims", "F=kg*m/s^2", "--dims", "m=kg", "--dims", "a=m/s^2"])
    assert exit_code == 1
    assert "INCONSISTENT" in capsys.readouterr().out


def test_derive_dimension_command(capsys):
    assert main(["derive-dim", "F = m a", "--unknown", "a", "--dims", "F=kg*m/s^2", "--dims", "m=kg"]) == 0
    assert "m / s^2" in capsys.readouterr().out


def test_bad_input_reports_error(capsys):
    assert main(["run", "solve for x in (2x + "]) == 2
    assert "error:" in capsys.readouterr().err


def test_bench_subset(capsys):
    assert main(["bench", "--suite", "solve"]) == 0
    assert "exact_correctness" in capsys.readouterr().out
