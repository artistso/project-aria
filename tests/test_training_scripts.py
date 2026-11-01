"""Training scripts' data and legacy-objective gates, without model downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import grpo_verifier  # noqa: E402
import sft_qlora  # noqa: E402


def test_grpo_refuses_to_train_without_explicit_scalar_opt_in(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["grpo_verifier.py"])
    with pytest.raises(SystemExit) as raised:
        grpo_verifier.main()
    assert raised.value.code == 2
    assert "not ARIA's curriculum" in capsys.readouterr().err


def test_grpo_dry_run_remains_available_without_training_opt_in(tmp_path, monkeypatch, capsys):
    data = tmp_path / "rl.jsonl"
    data.write_text(json.dumps({"prompt": "solve 2x + 4 = 10"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["grpo_verifier.py", "--data", str(data), "--dry-run"],
    )
    assert grpo_verifier.main() == 0
    out = capsys.readouterr().out
    assert "rl prompts: 1" in out
    assert "wrong process" in out


def test_sft_dry_run_requires_message_records_and_an_explicit_model(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        sft_qlora.build_parser().parse_args(["--dry-run"])

    data = tmp_path / "bad.jsonl"
    data.write_text(json.dumps({"prompt": "no message field"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sft_qlora.py", "--model", "./model", "--data", str(data), "--dry-run"],
    )
    with pytest.raises(ValueError, match="messages"):
        sft_qlora.main()


def test_sft_dry_run_rejects_empty_export(tmp_path, monkeypatch):
    data = tmp_path / "empty.jsonl"
    data.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sft_qlora.py", "--model", "./model", "--data", str(data), "--dry-run"],
    )
    with pytest.raises(ValueError, match="empty SFT dataset"):
        sft_qlora.main()
