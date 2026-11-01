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


def test_dataset_export(tmp_path, capsys):
    target = tmp_path / "sft.jsonl"
    assert main(["dataset", "sft", "--out", str(target), "--limit", "5"]) == 0
    lines = target.read_text().strip().splitlines()
    assert len(lines) == 5
    assert "messages" in lines[0]


def test_reward_command(capsys):
    assert main(["reward", "--completion", "goal: solve\nvar: x\nexpr: 2x + 4 = 10"]) == 0
    out = capsys.readouterr().out
    assert "1.00" in out and "solved" in out


def test_ledger_record_and_report(tmp_path, capsys):
    book = tmp_path / "ledger.jsonl"
    code = main(
        [
            "ledger",
            "record",
            "--question",
            "solve 9x - 3 = 24",
            "--completion",
            "goal: solve\nvar: x\nexpr: 9x - 3 = 24",
            "--ledger",
            str(book),
        ]
    )
    assert code == 0
    assert "linear_equation" in capsys.readouterr().out

    assert main(["ledger", "report", "--ledger", str(book)]) == 0
    out = capsys.readouterr().out
    assert "report card" in out
    assert "linear_equation" in out


def test_ledger_next_prints_a_contract(tmp_path, capsys):
    book = tmp_path / "ledger.jsonl"
    assert main(["ledger", "next", "--ledger", str(book)]) == 0
    out = capsys.readouterr().out
    assert "goal:" in out and "expr:" in out


def test_ledger_json_report(tmp_path, capsys):
    book = tmp_path / "ledger.jsonl"
    main(
        [
            "--json",
            "ledger",
            "record",
            "--question",
            "solve 9x - 3 = 24",
            "--completion",
            "goal: solve\nvar: x\nexpr: 9x - 3 = 24",
            "--ledger",
            str(book),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["skill"] == "linear_equation"
    assert payload["author"] == {"verdict": "verifier", "note": "none"}


def test_ledger_sft_exports_only_proven_work_from_held_rungs(tmp_path, capsys):
    book = tmp_path / "ledger.jsonl"
    good = (
        ("solve 9x - 3 = 24", "goal: solve\nvar: x\nexpr: 9x - 3 = 24"),
        ("solve 2x + 7 = 15", "goal: solve\nvar: x\nexpr: 2x + 7 = 15"),
        ("solve 6x + 1 = 19", "goal: solve\nvar: x\nexpr: 6x + 1 = 19"),
    )
    for question, completion in good[:2]:
        main(["ledger", "record", "--question", question, "--completion", completion, "--ledger", str(book)])
    capsys.readouterr()

    out = tmp_path / "ledger.jsonl.sft"
    assert main(["ledger", "sft", "--out", str(out), "--ledger", str(book)]) == 0
    assert out.read_text() == "\n"
    capsys.readouterr()

    question, completion = good[2]
    main(["ledger", "record", "--question", question, "--completion", completion, "--ledger", str(book)])
    main(
        [
            "ledger", "record", "--question", "solve 9/x - 3 = 24",
            "--completion", "goal: solve\nvar: x\nexpr: 9/x - 3 = 24",
            "--ledger", str(book),
        ]
    )
    capsys.readouterr()
    assert main(["ledger", "sft", "--out", str(out), "--ledger", str(book)]) == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    assert all("9/x" not in line for line in lines)


def test_geometry_shape_json(capsys):
    assert main(["--json", "geometry", "shape", "cube", "--size", "2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["volume"] == 8.0  # edge 2
    assert data["euler_characteristic"] == 2


def test_geometry_list_and_penrose(capsys):
    assert main(["geometry", "list"]) == 0
    assert "icosahedron" in capsys.readouterr().out
    assert main(["geometry", "penrose", "schwarzschild"]) == 0
    assert "singularity" in capsys.readouterr().out


def test_geometry_check_exit_code(capsys):
    assert main(["geometry", "check"]) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_memory_add_find_and_audit(tmp_path, capsys):
    store = tmp_path / "memory.jsonl"
    assert (
        main(
            [
                "memory", "add", "the Schwarzschild horizon is a null surface at r = 2M",
                "--tags", "penrose,horizon",
                "--source", "aria/geometry/causal.py",
                "--confidence", "verified",
                "--memory", str(store),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "memory", "add", "a dodecahedron probably tiles the plane",
                "--confidence", "stated",
                "--memory", str(store),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["memory", "find", "horizon", "--memory", str(store)]) == 0
    assert "null surface" in capsys.readouterr().out

    assert main(["memory", "audit", "--memory", str(store)]) == 0
    out = capsys.readouterr().out
    assert "tiles the plane" in out
    assert "1 memory item(s) need backing" in out


def test_memory_context_and_export(tmp_path, capsys):
    store = tmp_path / "memory.jsonl"
    main(
        [
            "memory", "add", "the Schwarzschild singularity is spacelike",
            "--tags", "penrose",
            "--source", "aria/geometry/causal.py",
            "--confidence", "verified",
            "--memory", str(store),
        ]
    )
    capsys.readouterr()
    assert main(["memory", "context", "singularity", "--memory", str(store)]) == 0
    assert "What ARIA remembers" in capsys.readouterr().out

    target = tmp_path / "memory.md"
    assert main(["memory", "export", "--out", str(target), "--memory", str(store)]) == 0
    assert target.read_text().startswith("# What ARIA remembers")


def test_memory_correction_command_preserves_and_replaces_memory(tmp_path, capsys):
    store = tmp_path / "memory.jsonl"
    assert main(
        ["memory", "add", "a cube has six faces", "--source", "user", "--memory", str(store)]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "memory", "correct", "1", "a cube has six faces and eight vertices",
            "--source", "geometry check", "--confidence", "verified", "--memory", str(store),
        ]
    ) == 0
    assert "corrects memory #1" in capsys.readouterr().out
    assert main(["memory", "find", "cube six faces", "--memory", str(store)]) == 0
    out = capsys.readouterr().out
    assert "eight vertices" in out
    assert "(fact" not in out


def test_memory_cli_records_and_corrects_subjective_preferences(tmp_path, capsys):
    store = tmp_path / "memory.jsonl"
    assert main(
        [
            "--json", "memory", "add", "I like this name, not the other one",
            "--kind", "preference", "--source", "user-stated",
            "--confidence", "stated", "--memory", str(store),
        ]
    ) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["kind"] == "preference"
    assert first["confidence"] == "stated"

    assert main(
        [
            "--json", "memory", "correct", str(first["id"]),
            "Use the preferred name Aria",
            "--source", "user correction", "--memory", str(store),
        ]
    ) == 0
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["kind"] == "preference"
    assert corrected["supersedes"] == first["id"]
    assert len(store.read_text().splitlines()) == 2


def test_memory_context_budget_error_is_recoverable_json(tmp_path, capsys):
    store = tmp_path / "memory.jsonl"
    main(["memory", "add", "one triangle fact", "--tags", "triangle", "--memory", str(store)])
    capsys.readouterr()
    code = main(
        ["--json", "memory", "context", "triangle", "--max-chars", "1", "--memory", str(store)]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["recoverable"] is True
    assert payload["source_entries"][0]["text"] == "one triangle fact"


def test_memory_import_from_geometry(tmp_path, capsys):
    store = tmp_path / "memory.jsonl"
    assert main(["memory", "import", "--from", "geometry", "--memory", str(store)]) == 0
    capsys.readouterr()
    assert main(["memory", "digest", "--memory", str(store)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["entries"] >= 18
    assert data["unverified"] == []
    assert data["by_confidence"]["verified"] == data["entries"]
